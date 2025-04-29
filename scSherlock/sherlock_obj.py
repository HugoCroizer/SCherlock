import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Callable
from scipy.stats import binom, percentileofscore
from adpbulk import ADPBulk
import logging
from dataclasses import dataclass
from enum import Enum
from joblib import Parallel, delayed
import scipy
import numba as nb
from sklearn.preprocessing import StandardScaler
from statsmodels.distributions.empirical_distribution import ECDF
import gc
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ScSherlock')


class ScoringMethod(Enum):
    """Enum for scoring methods used in ScSherlock"""
    DIFF = "diff"  # Maximized difference between sensitivity and FPR
    SENS_FPR_ZERO = "sensFPRzero"  # Sensitivity at zero FPR
    SENS_PPV_99 = "sensPPV99"  # Sensitivity at PPV > 99%


class ParameterEstimation(Enum):
    """Enum for parameter estimation methods"""
    PATIENT_MEDIAN = "patient_median"  # Estimated per patient, median taken
    MEAN = "mean"  # Estimated on entire dataset


class AggregationMethod(Enum):
    """Enum for score aggregation methods"""
    MEAN = "mean"  # Mean of scores across k values
    MAX = "max"  # Maximum score across k values


@dataclass
class ScSherlockConfig:
    """Configuration class for ScSherlock parameters"""
    k_values: List[int] = None
    scoring_method: ScoringMethod = ScoringMethod.DIFF
    aggregation_method: AggregationMethod = AggregationMethod.MEAN
    parameter_estimation: ParameterEstimation = ParameterEstimation.PATIENT_MEDIAN
    max_genes_kept: int = 100
    min_patients: int = 3
    min_reads: int = 10
    min_cells: int = 10
    score_cutoff: float = 0.5
    n_simulations: int = 1000
    random_seed: int = 0
    sparse_step: int = 5  # Step size for sparse sampling optimization
    promising_threshold: float = 0.1  # Threshold for identifying promising genes
    n_jobs: int = 1
    batch_size: int=5 # Number of cell types to process in parallel
    
    def __post_init__(self):
        """Initialize default values"""
        if self.k_values is None:
            self.k_values = [1, 10, 25]


class ScSherlock:
    """
    ScSherlock: Single-Cell marker gene identification algorithm
    
    This class implements the ScSherlock algorithm for identifying cell type-specific 
    marker genes from single-cell RNA sequencing data.
    """
    
    def __init__(self, adata, column_ctype: str, column_patient: str, config: Optional[ScSherlockConfig] = None):
        """
        Initialize ScSherlock with data and configuration
        
        Args:
            adata: AnnData object containing single-cell gene expression data
            column_ctype: Column name in adata.obs for cell type annotations
            column_patient: Column name in adata.obs for patient IDs
            config: Configuration object with algorithm parameters (optional)
        """
        self.adata = adata
        self.column_ctype = column_ctype
        self.column_patient = column_patient
        self.column_patient = self.simplify_patient_ids()
        self.config = config or ScSherlockConfig()
        
        # Validate inputs
        self._validate_inputs()
        logger.info("Pre-filtering genes...")
        #self._prefilter_genes(self.config.min_cells, self.config.min_reads)
        # Internal state
        self.cell_types = self.adata.obs[self.column_ctype].unique()
        self.theoretical_scores = None
        self.expr_proportions = None
        self.processed_scores = None
        self.aggregated_scores = None
        self.sorted_table = None
        self.filtered_scores = None
        self.empirical_scores = None
        self.aggregated_empirical_scores = None
        self.sorted_empirical_table = None
        self.top_markers = None
        self.method_run = None
        
        logger.info(f"ScSherlock initialized with {len(self.cell_types)} cell types and {self.adata.shape} data matrix")
    
    def _validate_inputs(self):
        """Validate input data and parameters"""
        if self.column_ctype not in self.adata.obs.columns:
            raise ValueError(f"Cell type column '{self.column_ctype}' not found in adata.obs")
        
        if self.column_patient not in self.adata.obs.columns:
            raise ValueError(f"Patient column '{self.column_patient}' not found in adata.obs")
        
        # Check for empty cell types
        for ctype in self.adata.obs[self.column_ctype].unique():
            if sum(self.adata.obs[self.column_ctype] == ctype) < 10:
                logger.warning(f"Cell type '{ctype}' has fewer than 10 cells")

    def _prefilter_genes(self, min_cells=10, min_counts=10):
        """Pre-filter genes to reduce computation time
        
        Args:
            min_cells: Minimum number of cells expressing the gene
            min_counts: Minimum count threshold
            
        Returns:
            Filtered AnnData object
        """
        logger.info(f"Original dataset: {self.adata.shape[0]} cells, {self.adata.shape[1]} genes")
        
    
        
        # Filter genes based on minimum number of cells and counts
        sc.pp.filter_genes(self.adata, min_counts=min_counts)
        sc.pp.filter_genes(self.adata, min_cells=min_cells)
        
        logger.info(f"Filtered dataset: {self.adata.shape[0]} cells, {self.adata.shape[1]} genes")
        logger.info(f"Removed {self.adata.shape[1] - self.adata.shape[1]} genes with low expression")

    def simplify_patient_ids(self, prefix="P"):
        """
        Create a simplified mapping of patient IDs and update the patient column reference.
        
        This method creates a new column in adata.obs with simplified patient IDs
        (e.g., P1, P2, ..., PN) to avoid issues with complex patient identifiers.
        
        Args:
            prefix (str): Prefix to use for simplified patient IDs (default: "P")
            
        Returns:
            str: The name of the new simplified patient column
        """
        # Create a mapping from original patient IDs to simplified ones
        unique_patients = self.adata.obs[self.column_patient].unique()
        patient_mapping = {patient: f"{prefix}{i+1}" for i, patient in enumerate(unique_patients)}
        
        # Create a new column name for the simplified IDs
        simplified_column = f"{self.column_patient}_simplified"
        
        # Add the new column to the AnnData object
        self.adata.obs[simplified_column] = self.adata.obs[self.column_patient].map(patient_mapping)
        
        # Store the original column name and update the reference
        self.original_patient_column = self.column_patient
        self.column_patient = simplified_column
        
        logger.info(f"Created simplified patient IDs in column '{simplified_column}'")
        logger.info(f"Patient column reference updated from '{self.original_patient_column}' to '{self.column_patient}'")
        
        return simplified_column
    
    def run(self, method: str = "empiric") -> Dict[str, str]:
        """
        Run the complete ScSherlock algorithm pipeline
        
        Returns:
            Dict[str, str]: Dictionary of top marker genes for each cell type
        """
        if method not in ["theoric", "empiric"]:
            raise ValueError('Method must be either "theoric" or "empiric"')
        # if theoric model was already run 
        if self.method_run == None:
            # Set random se
            # Set random seed for reproducibility
            np.random.seed(self.config.random_seed)
            
            # Step 1: Calculate theoretical scores based on binomial distributions
            logger.info("Calculating theoretical scores...")
            self.theoretical_scores, self.expr_proportions = self._calculate_theoretical_scores_parallel()
            
            # Step 2: Apply multiple category correction to theoretical scores
            logger.info("Applying multi-category correction...")
            self.processed_scores = self._apply_multi_category_correction(self.theoretical_scores)

            # Step 3: Aggregate scores across k values
            logger.info("Aggregating scores...")
            self.aggregated_scores = self._aggregate_scores(self.processed_scores)

        
            # Step 4: Sort scores and prepare for filtering
            logger.info("Sorting scores...")
            self.sorted_table = self._sort_scores(self.aggregated_scores, self.processed_scores, self.expr_proportions)
            
            if method == "theoric":
                logger.info("Identifying top markers...")
                self.top_markers = self._construct_top_marker_list(self.sorted_table)
                self.method_run = "theoric"
                return self.top_markers

            # Step 5: Filter genes based on scores and expression criteria
            logger.info("Filtering genes...")
            self.filtered_scores = self._filter_genes(self.sorted_table)

            # Step 6: Calculate empirical scores through simulation
            logger.info("Calculating empirical scores...")
            #self.empirical_scores = self._calculate_empirical_scores(self.filtered_scores)
            self.empirical_scores = self.empirical_scores_optimized_batch_parallel_only_ctype()
            
             # Step 7: Aggregate empirical scores across k values
            logger.info("Aggregating empirical scores...")
            self.aggregated_empirical_scores = self._aggregate_scores(self.empirical_scores)

            # Step 8: Sort empirical scores
            logger.info("Sorting empirical scores...")
            self.sorted_empirical_table = self._sort_scores(
                self.aggregated_empirical_scores, 
                self.empirical_scores, 
                self.expr_proportions
            )
            # Step 9: Construct final list of top markers
            logger.info("Identifying top markers...")
            self.top_markers = self._construct_top_marker_list(self.sorted_empirical_table)
            self.method_run = "empiric"

            logger.info(f"ScSherlock completed. Found markers for {len(self.top_markers)}/{len(self.cell_types)} cell types")
            return self.top_markers
        
        elif self.method_run == 'theoric':
            if method == "theoric":
                logger.info(f"ScSherlock already run with theoric method. Found markers for {len(self.top_markers)}/{len(self.cell_types)} cell types")
                return self.top_markers
            else:       
                logger.info(f"Skipping theorical model as it was already run. Running empiric model")
                # Step 5: Filter genes based on scores and expression criteria
                logger.info("Filtering genes...")
                self.filtered_scores = self._filter_genes(self.sorted_table)

                # Step 6: Calculate empirical scores through simulation
                logger.info("Calculating empirical scores...")
                #self.empirical_scores = self._calculate_empirical_scores(self.filtered_scores)
                self.empirical_scores = self.empirical_scores_optimized_batch_parallel_only_ctype(self.filtered_scores)

                # Step 7: Aggregate empirical scores across k values
                logger.info("Aggregating empirical scores...")
                self.aggregated_empirical_scores = self._aggregate_scores(self.empirical_scores)

                # Step 8: Sort empirical scores
                logger.info("Sorting empirical scores...")
                self.sorted_empirical_table = self._sort_scores(
                    self.aggregated_empirical_scores, 
                    self.empirical_scores, 
                    self.expr_proportions
                )
                # Step 9: Construct final list of top markers
                logger.info("Identifying top markers...")
                self.top_markers = self._construct_top_marker_list(self.sorted_empirical_table)
                self.method_run = "empiric"

                logger.info(f"ScSherlock completed. Found markers for {len(self.top_markers)}/{len(self.cell_types)} cell types")
                return self.top_markers
        else:
            logger.info(f"ScSherlock already run with empiric method. Found markers for {len(self.top_markers)}/{len(self.cell_types)} cell types")
    

    def _estimate_binomial_parameters(self) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
        """
        Estimate parameters for binomial distributions
        
        Returns:
            Tuple containing:
            - counts_per_ctype: Counts per cell type
            - counts_per_ctype_complement: Counts for complement of each cell type
            - count_proportions_per_ctype: Expression proportions for each cell type
            - count_proportions_per_ctype_complement: Expression proportions for complements
        """
        self.adata.obs[self.column_ctype] = self.adata.obs[self.column_ctype].astype('category')
        cat_list = self.adata.obs[self.column_ctype].cat.categories
        
        if self.config.parameter_estimation == ParameterEstimation.PATIENT_MEDIAN:
            # Method 1: Patient-specific estimation with median across patients
            
            # Generate pseudobulk data
            adata_agg = ADPBulk(self.adata, [self.column_patient, self.column_ctype])
            pseudobulk_matrix = adata_agg.fit_transform()
            sample_meta = adata_agg.get_meta()
            
            # Create multi-index for cell type and patient
            tuples = list(zip(sample_meta[self.column_ctype], sample_meta[self.column_patient]))
            index = pd.MultiIndex.from_tuples(tuples, names=[self.column_ctype, self.column_patient])
            pseudobulk_matrix.set_index(index, inplace=True)
            
            # Calculate total counts and cell numbers
            total_counts = pseudobulk_matrix.sum(axis=1)
            n_cells = self.adata.obs.groupby(by=[self.column_ctype, self.column_patient]).size()
            
            # Calculate median counts per cell type
            counts_per_ctype = (total_counts/n_cells).groupby(level=self.column_ctype).median()
            
            # Calculate median counts for complements
            counts_per_ctype_complement = pd.Series({
                ctype: ((total_counts.drop(ctype).groupby(level=self.column_patient).sum()) / 
                        (n_cells.drop(ctype).groupby(level=self.column_patient).sum())).median()
                for ctype in cat_list
            })
            
            # Calculate normalized proportions
            count_proportions_per_ctype = (
                pseudobulk_matrix.div(total_counts.values, axis=0)
                .groupby(level=self.column_ctype)
                .median()
                .T
            )
            
            # Calculate normalized proportions for complements
            count_proportions_per_ctype_complement = pd.DataFrame({
                ctype: pseudobulk_matrix.drop(ctype)
                       .groupby(level=self.column_patient)
                       .sum()
                       .div(total_counts.drop(ctype).groupby(level=self.column_patient).sum(), axis=0)
                       .median()
                for ctype in cat_list
            })
            
        elif self.config.parameter_estimation == ParameterEstimation.MEAN:
            # Method 2: Direct estimation on the entire dataset
            
            counts_per_ctype, counts_per_ctype_complement = {}, {}
            count_proportions_per_ctype, count_proportions_per_ctype_complement = {}, {}
            
            for ctype in cat_list:
                # Subset data for target and non-target cells
                adata_A = self.adata[self.adata.obs[self.column_ctype]==ctype].copy()
                adata_nA = self.adata[self.adata.obs[self.column_ctype]!=ctype].copy()
                
                # Calculate sums and averages for target cells
                sums_A = adata_A.X.sum(axis=0)
                total_count_A = sums_A.sum()
                norm_sums_A = sums_A/total_count_A
                n_A = total_count_A/adata_A.shape[0]
                
                # Calculate sums and averages for non-target cells
                sums_nA = adata_nA.X.sum(axis=0)
                total_count_nA = sums_nA.sum()
                norm_sums_nA = sums_nA/total_count_nA
                n_nA = total_count_nA/adata_nA.shape[0]
                
                # Store results
                counts_per_ctype[ctype] = n_A
                counts_per_ctype_complement[ctype] = n_nA
                count_proportions_per_ctype[ctype] = norm_sums_A.tolist()[0]
                count_proportions_per_ctype_complement[ctype] = norm_sums_nA.tolist()[0]
                
            # Convert dictionaries to pandas objects
            counts_per_ctype = pd.Series(counts_per_ctype)
            counts_per_ctype_complement = pd.Series(counts_per_ctype_complement)
            count_proportions_per_ctype = pd.DataFrame(count_proportions_per_ctype, index=self.adata.var.index)
            count_proportions_per_ctype_complement = pd.DataFrame(
                count_proportions_per_ctype_complement, 
                index=self.adata.var.index
            )
        
        return (
            counts_per_ctype, 
            counts_per_ctype_complement, 
            count_proportions_per_ctype, 
            count_proportions_per_ctype_complement
        )
    
    def _apply_multi_category_correction(self, computed_scores: Dict) -> Dict:
        """
        Adjust marker scores across different cell types by normalizing them to sum to 1
        
        Args:
            computed_scores: Dictionary of scores by cell type
            
        Returns:
            Dict: Normalized scores
        """
        # Calculate total sum across all categories
        markers_sum = sum(computed_scores.values())
        
        # Normalize each category's scores and handle NaN values
        processed_scores = {
            ctype: (score / markers_sum).fillna(0)
            for ctype, score in computed_scores.items()
        }
        
        return processed_scores
    
    def _aggregate_scores(self, processed_scores: Dict) -> Dict:
        """
        Aggregate scores across different k values
        
        Args:
            processed_scores: Dictionary of processed scores by cell type
            
        Returns:
            Dict: Aggregated scores
        """
        # Define aggregation function based on method
        if self.config.aggregation_method == AggregationMethod.MEAN:
            agg_func = lambda df: df.mean(axis=1)
        elif self.config.aggregation_method == AggregationMethod.MAX:
            agg_func = lambda df: df.max(axis=1)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.config.aggregation_method}")
        
        # Apply aggregation to each cell type
        aggregated_scores = {
            ctype: agg_func(scores) if not scores.empty else pd.DataFrame([])
            for ctype, scores in processed_scores.items()
        }
        
        return aggregated_scores
    
    def _sort_scores(self, aggregated_scores: Dict, computed_scores: Dict, expr_proportions: Dict) -> Dict:
        """
        Sort scores by aggregated value and expression proportion
        
        Args:
            aggregated_scores: Dictionary of aggregated scores
            computed_scores: Dictionary of computed scores
            expr_proportions: Dictionary of expression proportions
            
        Returns:
            Dict: Sorted scores
        """
        sorted_scores = {
            ctype: computed_scores[ctype].assign(
                aggregated=aggregated_scores[ctype],
                exp_prop=expr_proportions[ctype]
            ).sort_values(by=['aggregated', 'exp_prop'], ascending=False)
            if not aggregated_scores[ctype].empty else pd.DataFrame([])
            for ctype in aggregated_scores
        }
        
        return sorted_scores
    
    def _filter_genes(self, sorted_table: Dict) -> Dict:
        """
        Filter genes based on expression criteria
        
        Args:
            sorted_table: Dictionary of sorted scores
            
        Returns:
            Dict: Filtered scores
        """
        # Count patients expressing each gene by cell type
        patient_agg = ADPBulk(
            self.adata, 
            [self.column_patient, self.column_ctype], 
            name_delim="--", 
            group_delim="::"
        )
        patient_matrix = patient_agg.fit_transform()
        patient_matrix.index = patient_matrix.index.get_level_values(0).str.split('--', expand=True)
        ctype_n_patients = (patient_matrix > 0).groupby(level=0).sum()
        ctype_n_patients.index = ctype_n_patients.index.str.split('::').str[-1]
        
        # Count reads per gene by cell type
        reads_agg = ADPBulk(self.adata, self.column_ctype, group_delim="::")
        ctype_n_reads = reads_agg.fit_transform()
        ctype_n_reads.index = ctype_n_reads.index.str.split('::').str[-1]
        
        # Apply filters to each cell type
        filtered_scores = {}
        for ctype, scores in sorted_table.items():
            if scores.empty:
                filtered_scores[ctype] = pd.DataFrame([])
                continue
                
            # Apply filtering criteria
            valid_genes = (
                (scores['aggregated'] > self.config.score_cutoff) &
                (ctype_n_patients.loc[ctype, scores.index] >= self.config.min_patients) &
                (ctype_n_reads.loc[ctype, scores.index] >= self.config.min_reads)
            )
            
            # Limit number of genes
            filtered = scores[valid_genes]
            filtered_scores[ctype] = (
                filtered.iloc[:self.config.max_genes_kept] 
                if len(filtered) > self.config.max_genes_kept 
                else filtered
            )
            
        return filtered_scores
    
    
    def empirical_scores_optimized_batch_parallel_only_ctype(self, n_sim=1000):
            """
            Calculate empirical marker gene scores through simulation using optimized statistical libraries
            with parallel processing only on cell types.
            
            Args:
                n_sim: Number of simulations to run (default: 1000)
                    
            Returns:
                Dictionary of empirical scores by cell type
            """
            # Get instance variables
            filtered_scores = self.filtered_scores
            adata = self.adata
            column_ctype = self.column_ctype
            column_patient = self.column_patient
            k_values = self.config.k_values
            scoring = self.config.scoring_method
            seed = 42
            n_jobs=self.config.n_jobs
            
            # Set random seed for reproducibility
            np.random.seed(seed)
            
            # Get all unique cell types
            cell_types = adata.obs[column_ctype].unique()
            
            # Collect all genes from filtered_scores across all cell types
            all_genes = []
            for ctype in cell_types:
                if not filtered_scores[ctype].empty:
                    all_genes.extend(filtered_scores[ctype].index.tolist())
            all_genes = list(set(all_genes))  # Remove duplicates
            
            # Create gene index mapping (gene_name -> position in adata.var_names)
            gene_indices = {gene: i for i, gene in enumerate(adata.var_names) if gene in all_genes}
            gene_names = list(gene_indices.keys())
            
            # Dictionary to store scores for all cell types
            scores_all = {}
            
            # Determine if data is sparse
            is_sparse = scipy.sparse.issparse(adata.X)
            
            # Define a worker function to parallelize the cell type processing
            def process_cell_type(ctype):
                # Skip cell types with no filtered genes
                if filtered_scores[ctype].empty:
                    return ctype, pd.DataFrame([])
                    
                # Initialize score matrix (rows=genes, columns=k_values)
                scores_ctype = pd.DataFrame(np.zeros((len(gene_names), len(k_values))), 
                                        index=gene_names, columns=k_values)
                
                # Get indices of target and non-target cells
                target_indices = np.where(adata.obs[column_ctype] == ctype)[0]
                nontarget_indices = np.where(adata.obs[column_ctype] != ctype)[0]
                
                # Get the gene indices as a list for easier slicing
                gene_idx_list = [gene_indices[gene] for gene in gene_names]
                
                # Process k values sequentially
                for k_idx, k in enumerate(k_values):
                    # Initialize arrays for expression sums
                    target_sums = np.zeros((n_sim, len(gene_names)))
                    nontarget_sums = np.zeros((n_sim, len(gene_names)))
                    
                    # Process in manageable batches to reduce memory usage
                    batch_size = min(100, n_sim)  # Adjust based on available memory
                    
                    # Create a local random state for thread safety
                    local_random = np.random.RandomState(seed + k_idx)
                    
                    # Calculate batch counts and indices
                    if k > 1:
                        for batch_start in range(0, n_sim, batch_size):
                            batch_end = min(batch_start + batch_size, n_sim)
                            
                            # Sample k target cells for each simulation in this batch
                            for sim_idx in range(batch_start, batch_end):
                                # Sample k cells for this simulation
                                sampled_cells = local_random.choice(target_indices, k, replace=True)
                                
                                # Extract and sum expression for these cells
                                for cell_idx in sampled_cells:
                                    if is_sparse:
                                        expr = adata.X[cell_idx, gene_idx_list].toarray().flatten()
                                    else:
                                        expr = adata.X[cell_idx, gene_idx_list]
                                    target_sums[sim_idx] += expr
                            
                            # Sample k non-target cells for each simulation in this batch
                            for sim_idx in range(batch_start, batch_end):
                                # Sample k cells for this simulation
                                sampled_cells = local_random.choice(nontarget_indices, k, replace=True)
                                
                                # Extract and sum expression for these cells
                                for cell_idx in sampled_cells:
                                    if is_sparse:
                                        expr = adata.X[cell_idx, gene_idx_list].toarray().flatten()
                                    else:
                                        expr = adata.X[cell_idx, gene_idx_list]
                                    nontarget_sums[sim_idx] += expr
                    else:  # k == 1
                        for batch_start in range(0, n_sim, batch_size):
                            batch_end = min(batch_start + batch_size, n_sim)
                            
                            # Sample target cells for each simulation in this batch
                            for sim_idx in range(batch_start, batch_end):
                                # Sample 1 cell for this simulation
                                cell_idx = local_random.choice(target_indices)
                                
                                # Extract expression for this cell
                                if is_sparse:
                                    target_sums[sim_idx] = adata.X[cell_idx, gene_idx_list].toarray().flatten()
                                else:
                                    target_sums[sim_idx] = adata.X[cell_idx, gene_idx_list]
                            
                            # Sample non-target cells for each simulation in this batch
                            for sim_idx in range(batch_start, batch_end):
                                # Sample 1 cell for this simulation
                                cell_idx = local_random.choice(nontarget_indices)
                                
                                # Extract expression for this cell
                                if is_sparse:
                                    nontarget_sums[sim_idx] = adata.X[cell_idx, gene_idx_list].toarray().flatten()
                                else:
                                    nontarget_sums[sim_idx] = adata.X[cell_idx, gene_idx_list]
                    
                    # Determine cutoff value for CDF calculation (99th percentile of the largest gene)
                    cutoff_k = int(max(
                        np.percentile(target_sums, 99, axis=0).max(),
                        np.percentile(nontarget_sums, 99, axis=0).max(),
                        1
                    ))
                    
                    # Calculate alpha and beta matrices using vectorized operations
                    alpha = np.zeros((len(gene_names), cutoff_k))
                    beta = np.zeros((len(gene_names), cutoff_k))
                    
                    # Use statsmodels ECDF for faster CDF computation
                    for g in range(len(gene_names)):
                        # Get expression values for current gene
                        target_expr = target_sums[:, g]
                        nontarget_expr = nontarget_sums[:, g]
                        
                        # Compute ECDFs
                        ecdf_target = ECDF(target_expr)
                        ecdf_nontarget = ECDF(nontarget_expr)
                        
                        # Evaluate at each threshold level
                        thresholds = np.arange(cutoff_k)
                        alpha[g, :] = ecdf_target(thresholds)
                        beta[g, :] = ecdf_nontarget(thresholds)


                    
                    # Calculate scores based on specified scoring method
                    if scoring == ScoringMethod.DIFF:
                        scores = compute_diff_scores(alpha, beta)
                    elif scoring == ScoringMethod.SENS_FPR_ZERO:
                        scores = compute_sensFPRzero_scores(alpha, beta)
                    elif scoring == ScoringMethod.SENS_PPV_99:
                        scores = compute_sensPPV99_scores(alpha, beta)
                        
                    # Assign scores to the dataframe
                    scores_ctype.iloc[:, k_idx] = scores
                gc.collect()
                return ctype, scores_ctype
            
            # Process cell types in parallel
            results = Parallel(n_jobs=n_jobs, verbose=False)(
                delayed(process_cell_type)(ctype)
                for ctype in cell_types
            )
            
            # Collect results
            for ctype, scores_ctype in results:
                scores_all[ctype] = scores_ctype
            
            # Apply multi-category correction to normalize scores
            corr_scores = self.multi_cat_correction_for_optimized(scores_all)
            
            # Filter scores to include only genes from filtered_scores
            scores = {}
            for ctype in cell_types:
                if filtered_scores[ctype].empty:
                    scores[ctype] = pd.DataFrame([])
                else:
                    # Extract scores for filtered genes and clip values to max of 1
                    genes_to_keep = filtered_scores[ctype].index
                    scores[ctype] = corr_scores[ctype].loc[genes_to_keep].clip(upper=1)
            
            return scores
    
    def _calculate_theoretical_scores_parallel(self) -> Tuple[Dict, Dict]:
        """
        Calculate theoretical scores using joblib for parallelization with reduced memory footprint
        
        Returns:
            Tuple[Dict, Dict]: Scores and expression proportions
        """
        from joblib import Parallel, delayed
        import gc
        
        # Estimate binomial distribution parameters
        parameters = self._estimate_binomial_parameters()
        
        def process_cell_type(ctype):
            """Process a single cell type"""
            # Extract only what we need from parameters to avoid reference to large objects
            param0_ctype = int(parameters[0][ctype])
            param1_ctype = int(parameters[1][ctype]) 
            param2_ctype = parameters[2][ctype]
            param3_ctype = parameters[3][ctype]
            
            # Get gene names once to avoid copying self.adata.var.index multiple times
            gene_names = list(self.adata.var.index)
            k_values = list(self.config.k_values)
            n_genes = len(gene_names)
            scoring_method = self.config.scoring_method  # Store locally to avoid referencing self
            
            # Initialize score matrix using numpy for better memory efficiency
            scores_np = np.zeros((n_genes, len(k_values)))
            
            # Calculate scores for each k value
            for k_idx, k in enumerate(k_values):
                # Calculate cutoffs for computation
                
                # Compute probability values directly to avoid large DataFrame creation
                cutoffs_k = np.maximum(
                    binom.ppf(.99, param0_ctype * k, param2_ctype),
                    binom.ppf(.99, param1_ctype * k, param3_ctype)
                )
                cutoffs_k = np.clip(cutoffs_k, a_min=100, a_max=None)
                
                # Create alpha and beta arrays with minimal size
                max_cutoff = int(np.max(cutoffs_k))
                n_points = min(101, max_cutoff)  # Limit to reduce memory usage
                alpha = np.zeros((n_genes, n_points))
                beta = np.zeros((n_genes, n_points))
                
                # Calculate CDFs for evenly spaced points
                threshold_points = np.linspace(0, max_cutoff, n_points).astype(int)
                for i, l in enumerate(threshold_points):
                    alpha[:, i] = binom.cdf(l, param0_ctype * k, param2_ctype)
                    beta[:, i] = binom.cdf(l, param1_ctype * k, param3_ctype)
                
                # Compute scores based on scoring method
                if scoring_method == ScoringMethod.DIFF:
                    # Max difference between sensitivity and FPR
                    diff = beta - alpha
                    max_indices = np.argmax(diff, axis=1)
                    for i in range(n_genes):
                        scores_np[i, k_idx] = diff[i, max_indices[i]]
                        
                elif scoring_method == ScoringMethod.SENS_FPR_ZERO:
                    # Sensitivity at zero FPR
                    max_indices = np.argmax(beta, axis=1)
                    for i in range(n_genes):
                        scores_np[i, k_idx] = 1 - alpha[i, max_indices[i]]
                        
                elif scoring_method == ScoringMethod.SENS_PPV_99:
                    # Sensitivity at PPV > 99%
                    for i in range(n_genes):
                        ppv = np.nan_to_num((1 - alpha[i]) / (2 - alpha[i] - beta[i]))
                        ppv_exceeds = ppv > 0.99
                        if np.any(ppv_exceeds):
                            first_idx = np.argmax(ppv_exceeds)
                            scores_np[i, k_idx] = 1 - alpha[i, first_idx]
                
                # Delete intermediate arrays to free memory
                del alpha, beta, cutoffs_k
            
            # Convert to DataFrame only at the end
            scores_ctype = pd.DataFrame(
                scores_np,
                index=gene_names, 
                columns=k_values
            )
            
            # Force garbage collection before returning
            gc.collect()
            return ctype, scores_ctype
        
        # Process cell types in batches to reduce peak memory usage
        batch_size = self.config.batch_size
        all_cell_types = list(self.cell_types)
        n_jobs = min(self.config.n_jobs, batch_size)  # Limit jobs per batch
        
        scores = {}
        for batch_start in range(0, len(all_cell_types), batch_size):
            batch_end = min(batch_start + batch_size, len(all_cell_types))
            current_batch = all_cell_types[batch_start:batch_end]
            
            #logger.info(f"Processing batch of {len(current_batch)} cell types with {n_jobs} workers")
            
            # Use memory-efficient options in Parallel
            batch_results = Parallel(
                n_jobs=n_jobs, 
                max_nbytes='50M',  # Limit memory per job
                prefer="threads",   # Use threads for better memory sharing
                verbose=False          # Show progress
            )(delayed(process_cell_type)(ctype) for ctype in current_batch)
            
            # Add batch results to scores dictionary
            for ctype, score in batch_results:
                scores[ctype] = score
            
            # Clean up batch results to free memory
            del batch_results
            gc.collect()
        
        logger.info("Completed theoretical score calculation")
        return scores, parameters[2]

    def multi_cat_correction(self, computed_scores):
        """
        Adjusts marker scores across different cell types by normalizing them to sum to 1.

        Args:
            computed_scores (dict): A dictionary containing marker scores in the form of a 
                                    Pandas DataFrame for each cell type.
        Returns:
            dict: A dictionary where keys are  the cell types from `computed_scores`, 
                and values are Pandas Series with normalized scores. Any NaN values in the result 
                are replaced with 0.
        """
        # Calculate the total sum of all scores across all categories.
        markers_sum = sum(computed_scores.values())
        # Normalize each category's scores by dividing by the total sum.
        # Replace any resulting NaN values with 0 (e.g., to handle division by zero scenarios).
        processed_scores = {
            ctype: (score / markers_sum).fillna(0)
            for ctype, score in computed_scores.items()
        }
        return processed_scores

    def multi_cat_correction_for_optimized(self, computed_scores):
        """
        Adjusts marker scores across different cell types by normalizing them to sum to 1.
        
        Args:
            computed_scores (dict): A dictionary containing marker scores in the form of a
                                Pandas DataFrame for each cell type.
        
        Returns:
            dict: A dictionary where keys are the cell types from `computed_scores`,
                and values are Pandas DataFrames with normalized scores.
        """
        import pandas as pd
        
        # First, create a combined DataFrame with all non-empty scores
        all_scores = {}
        
        # Filter out empty DataFrames
        valid_scores = {k: v for k, v in computed_scores.items() if not v.empty}
        
        # If all DataFrames are empty, return the original dictionary
        if not valid_scores:
            return computed_scores
        
        # For each gene (row), sum the scores across all cell types
        gene_sums = {}
        
        # Process each cell type
        for cell_type, df in valid_scores.items():
            # Iterate through each row of the DataFrame
            for idx, row in df.iterrows():
                gene = idx  # Assuming gene names are the index
                if gene not in gene_sums:
                    gene_sums[gene] = {}
                
                # Extract scores for each column and add to gene_sums
                for col in df.columns:
                    if col not in gene_sums[gene]:
                        gene_sums[gene][col] = 0
                    gene_sums[gene][col] += row[col]
        
        # Now normalize each score by dividing by the total for that gene and column
        processed_scores = {}
        
        for cell_type, df in valid_scores.items():
            # Create a copy of the original DataFrame
            normalized_df = df.copy()
            
            # Normalize each value
            for idx, row in df.iterrows():
                gene = idx
                for col in df.columns:
                    if gene in gene_sums and col in gene_sums[gene] and gene_sums[gene][col] != 0:
                        normalized_df.at[idx, col] = row[col] / gene_sums[gene][col]
                    else:
                        normalized_df.at[idx, col] = 0
            
            processed_scores[cell_type] = normalized_df
        
        # Add empty DataFrames back to the result
        for cell_type, df in computed_scores.items():
            if df.empty and cell_type not in processed_scores:
                processed_scores[cell_type] = df
        
        return processed_scores

    def _construct_top_marker_list(self, sorted_emp_table: Dict) -> Dict[str, str]:
        """
        Identify top marker for each cell type
        
        Args:
            sorted_emp_table: Dictionary of sorted empirical scores
            
        Returns:
            Dict[str, str]: Dictionary mapping cell types to top marker genes
        """
        # Get top scoring marker for each cell type that meets the cutoff
        top_markers = {
            ctype: table.index[0]  # First gene is highest scoring
            for ctype, table in sorted_emp_table.items()
            if not table.empty and table['aggregated'].max() >= self.config.score_cutoff
        }
        return top_markers
    
    def visualize_marker(self, gene: str, cell_type: str = None):
        """
        Visualize the expression distribution of a marker gene
        
        Args:
            gene: Gene name to visualize
            cell_type: Cell type to highlight (optional)
        """
        if gene not in self.adata.var_names:
            raise ValueError(f"Gene '{gene}' not found in the dataset")
            
        # Get the cell type this gene is a marker for (if not specified)
        if cell_type is None:
            for ctype, marker in self.top_markers.items():
                if marker == gene:
                    cell_type = ctype
                    break
            if cell_type is None:
                logger.warning(f"Gene '{gene}' is not a top marker for any cell type")
        
        # Extract gene expression data
        gene_expression = self.adata[:, gene].X.toarray()
        categories = self.adata.obs[self.column_ctype]
        
        # Create DataFrame for visualization
        data = {'Gene': gene_expression.flatten(), 'Category': categories}
        data_df = pd.DataFrame(data)
        
        # Create boxen plot
        plt.figure(figsize=(10, 6))
        sns.boxenplot(x='Category', y='Gene', data=data_df)
        plt.xticks(rotation=90)
        plt.xlabel('Cell type')
        plt.ylabel('Expression counts')
        
        if cell_type:
            plt.title(f"Expression of marker gene '{gene}' for {cell_type} cells")
        else:
            plt.title(f"Expression of gene '{gene}' across cell types")
            
        plt.tight_layout()
        plt.show()
    
    def get_results(self) -> Dict:
        """
        Get complete results from the ScSherlock analysis
        
        Returns:
            Dict: Dictionary with all results
        """
        return {
            'top_markers': self.top_markers,
            'sorted_emp_table': self.sorted_empirical_table,
            'filtered_scores': self.filtered_scores,
            'theoretical_scores': self.theoretical_scores,
            'empirical_scores': self.empirical_scores
        }
    
    def export_markers(self, output_file: str = None) -> pd.DataFrame:
        """
        Export marker genes to a DataFrame or CSV file
        
        Args:
            output_file: Path to output CSV file (optional)
                
        Returns:
            pd.DataFrame: DataFrame of marker genes with scores
        
        Raises:
            ValueError: If no markers identified (run ScSherlock.run() first)
        """
        if self.top_markers is None:
            raise ValueError("No markers identified. Run ScSherlock.run() first")
        
        if self.method_run is None:
            raise ValueError("Model method not set. Run ScSherlock.run() first")
        
        # Determine which tables to use based on the method that was run
        if self.method_run == "empiric":
            if self.sorted_empirical_table is None or self.empirical_scores is None:
                raise ValueError("Empirical scores not available. Run with method='empiric'")
            sorted_table = self.sorted_empirical_table
            scores_table = self.empirical_scores
        else:  # theoric
            if self.sorted_table is None or self.theoretical_scores is None:
                raise ValueError("Theoretical scores not available. Run with method='theoric'")
            sorted_table = self.sorted_table
            scores_table = self.theoretical_scores
                
        # Compile marker information
        results = []
        for ctype, gene in self.top_markers.items():
            # Get scores for this gene
            if ctype in self.theoretical_scores and gene in self.theoretical_scores[ctype].index:
                theoretical_score = self.theoretical_scores[ctype].loc[gene].mean()
            else:
                theoretical_score = float('nan')
                
            # Get empirical score if available
            if self.method_run == "empiric" and ctype in self.empirical_scores and gene in self.empirical_scores[ctype].index:
                empirical_score = self.empirical_scores[ctype].loc[gene].mean()
            else:
                empirical_score = float('nan')
                
            # Get aggregated score and expression proportion
            aggregated_score = sorted_table[ctype].loc[gene, 'aggregated']
            exp_prop = sorted_table[ctype].loc[gene, 'exp_prop']
            
            results.append({
                'cell_type': ctype,
                'marker_gene': gene,
                'theoretical_score': theoretical_score,
                'empirical_score': empirical_score if self.method_run == "empiric" else float('nan'),
                'aggregated_score': aggregated_score,
                'expression_proportion': exp_prop,
                'model_used': self.method_run
            })
        
        # Create DataFrame
        markers_df = pd.DataFrame(results)
        
        # Reorder columns for better readability
        col_order = [
            'cell_type', 'marker_gene', 'model_used', 'aggregated_score',
            'theoretical_score', 'empirical_score', 'expression_proportion'
        ]
        markers_df = markers_df[col_order]
        
        # Save to file if specified
        if output_file:
            markers_df.to_csv(output_file, index=False)
            logger.info(f"Exported {len(markers_df)} markers to {output_file} using {self.method_run} model")
                
        return markers_df
    
    def plot_marker_heatmap(self, n_genes=1, cutoff=0, groupby=None,remove_ctype_no_marker=False, cmap='viridis', standard_scale='var', 
                        use_raw=False, save=None, show=None, **kwargs):
        """
        Create a heatmap visualization of the identified marker genes
        
        This method creates a matrix plot showing the expression of top marker genes
        across all cell types. It orders the genes to match their corresponding cell types.
        
        Args:
            n_genes (int): Number of top genes to display for each cell type (default: 1)
            cutoff (float): Minimum score cutoff for including genes (default: 0)
            groupby (str): Column in adata.obs for grouping cells. Defaults to self.column_ctype if None.
            cmap (str): Colormap for the heatmap (default: 'viridis')
            standard_scale (str): Scale the data ('var', 'group', or None) (default: 'var')
            use_raw (bool): Whether to use raw data for plotting (default: False)
            save (str or bool): If True or a str, save the figure (default: None)
            show (bool): Whether to show the plot (default: None)
            **kwargs: Additional arguments passed to sc.pl.matrixplot
            
        Returns:
            matplotlib.axes.Axes: The axes object containing the plot
        
        Raises:
            ValueError: If no markers are available (run ScSherlock.run() first) or if no 
                    cells are found for the given cell types
        """
        # Check which method was run and select the appropriate table
        if self.method_run is None:
            raise ValueError("No markers available. Run ScSherlock.run() first")
        
        # Select appropriate sorted table based on which method was run
        if self.method_run == "empiric":
            if self.sorted_empirical_table is None:
                raise ValueError("Empirical scores not available. Run with method='empiric'")
            sorted_table = self.sorted_empirical_table
        else:  # theoric
            if self.sorted_table is None:
                raise ValueError("Theoretical scores not available. Run with method='theoric'")
            sorted_table = self.sorted_table
        
        # Use class's column_ctype if groupby not specified
        if groupby is None:
            groupby = self.column_ctype
        
        # Get cell types that have valid markers
        cell_to_genes = {}
        for ctype, table in sorted_table.items():
            if isinstance(table, pd.DataFrame) and not table.empty:
                # Only include cell types that have marker genes meeting the cutoff
                valid_genes = table[table['aggregated'] >= cutoff]
                if not valid_genes.empty:
                    # Get up to n_genes top markers
                    cell_to_genes[ctype] = valid_genes.index[:min(n_genes, len(valid_genes))].tolist()
        
        # Check if any markers were found
        if not cell_to_genes:
            logger.warning(f"No markers found that meet the score cutoff criteria ({cutoff})")
            return None
        
        # Filter adata to only include cell types that have markers
        mask = self.adata.obs[groupby].isin(cell_to_genes.keys())
       
    
        # Get unique cell types in the filtered data to preserve the order
        if remove_ctype_no_marker:
            cell_types = self.adata[mask].obs[groupby].cat.categories.tolist()
        else:
            cell_types = self.adata.obs[groupby].cat.categories.tolist()
        # Filter cell types to only those that appear in the data
        cell_types = [ct for ct in cell_types if ct in self.adata.obs[groupby].unique()]
        
        # Create an ordered list of genes based on the cell type order
        ordered_genes = []
        for cell_type in cell_types:
            if cell_type in cell_to_genes:
                ordered_genes.extend(cell_to_genes[cell_type])
        
        # Now plot with genes in the same order as cell types, showing only cell types with markers
        total_genes = len(ordered_genes)
        logger.info(f"Plotting {total_genes} genes for {len(cell_types)} cell types using {self.method_run} model")
        
        # Create the plot with the ordered genes
        return sc.pl.matrixplot(
            self.adata if not remove_ctype_no_marker else self.adata[mask],
            var_names=ordered_genes,
            groupby=groupby,
            cmap=cmap,
            use_raw=use_raw,
            standard_scale=standard_scale,
            categories_order=cell_types,  # Ensure cell types are in the desired order
            save=save,
            show=show,
            **kwargs
            )
    
    def plot_marker_violins(self, markers=None, n_markers=5, figsize=(12, 10), 
                        sort_by_expression=True, jitter=0.4, alpha=0.2):
        """
        Create violin plots showing expression distribution of marker genes across cell types
        
        Args:
            markers: List of marker genes to plot (if None, uses top markers)
            n_markers: Number of top markers to include if markers=None
            figsize: Figure size (width, height)
            sort_by_expression: Whether to sort cell types by median expression
            jitter: Amount of jitter for data points
            alpha: Transparency of points
            
        Returns:
            Figure with violin plots for each marker gene
        """
        # Get markers to plot
        if markers is None:
            if self.top_markers is None:
                raise ValueError("No markers available. Run ScSherlock.run() first")
                
            # Take n_markers from top scores
            sorted_table = self.sorted_empirical_table if self.method_run == "empiric" else self.sorted_table
            all_markers = []
            for ctype, table in sorted_table.items():
                if not table.empty:
                    top_n = table.index[:min(n_markers, len(table))].tolist()
                    all_markers.extend([(gene, ctype) for gene in top_n])
            
            # Sort by score and take top n_markers
            all_markers.sort(key=lambda x: sorted_table[x[1]].loc[x[0], 'aggregated'], reverse=True)
            markers = [gene for gene, _ in all_markers[:n_markers]]
        
        # Create figure
        n_markers = len(markers)
        ncols = min(2, n_markers)
        nrows = (n_markers + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        # Plot each marker
        for i, marker in enumerate(markers):
            if i < len(axes):
                ax = axes[i]
                # Find which cell type this is a marker for
                for ctype, genes in self.top_markers.items():
                    gene_list = [genes] if isinstance(genes, str) else genes
                    if marker in gene_list:
                        marker_cell_type = ctype
                        break
                else:
                    marker_cell_type = None
                    
                sc.pl.violin(self.adata, marker, groupby=self.column_ctype, 
                            ax=ax, show=False, jitter=jitter, alpha=alpha, use_raw=False)
                
                if marker_cell_type:
                    ax.set_title(f"{marker} (marker for {marker_cell_type})")
                
                # Rotate x-axis labels
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        
        # Hide any unused axes
        for i in range(n_markers, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    

    def plot_marker_scores_comparison(self, n_markers=10, figsize=(10, 6)):
        """
        Create a comparison plot of marker scores across different scoring methods
        
        Args:
            n_markers: Number of top markers to show
            figsize: Figure size
            
        Returns:
            Figure with comparison of marker scores
        """
        if self.top_markers is None or self.method_run is None:
            raise ValueError("No markers available. Run ScSherlock.run() first")
        
        # Export markers to get scores
        markers_df = self.export_markers()
        
        # Sort by aggregated score and take top n
        markers_df = markers_df.sort_values('aggregated_score', ascending=False).head(n_markers)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set up bar width and positions
        bar_width = 0.25
        index = np.arange(len(markers_df))
        
        # Create bars for each score type
        ax.bar(index, markers_df['theoretical_score'], bar_width, 
            label='Theoretical Score', color='steelblue')
        
        if 'empirical_score' in markers_df.columns and not markers_df['empirical_score'].isna().all():
            ax.bar(index + bar_width, markers_df['empirical_score'], bar_width,
                label='Empirical Score', color='coral')
        
        ax.bar(index + (2 * bar_width), markers_df['aggregated_score'], bar_width,
            label='Aggregated Score', color='forestgreen')
        
        # Add labels and title
        ax.set_xlabel('Marker Genes')
        ax.set_ylabel('Score')
        ax.set_title('Comparison of Marker Scores')
        ax.set_xticks(index + bar_width)
        ax.set_xticklabels([f"{gene}\n({ctype})" for gene, ctype in 
                        zip(markers_df['marker_gene'], markers_df['cell_type'])], rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_marker_radar(self, n_markers=5, figsize=(10, 10)):
        """
        Create a radar plot visualizing multiple metrics for top marker genes
        
        Args:
            n_markers: Number of top markers to include
            figsize: Figure size
            
        Returns:
            Figure with radar plot
        """
        if self.top_markers is None:
            raise ValueError("No markers available. Run ScSherlock.run() first")
        
        # Get marker metrics
        markers_df = self.export_markers()
        
        # Sort by aggregated score and take top n
        markers_df = markers_df.sort_values('aggregated_score', ascending=False).head(n_markers)
        
        # Define metrics to include in radar plot
        metrics = ['theoretical_score', 'empirical_score', 'aggregated_score', 'expression_proportion']
        
        # Normalize metrics to 0-1 scale for radar plot
        markers_norm = markers_df.copy()
        for metric in metrics:
            if metric in markers_norm.columns and not markers_norm[metric].isna().all():
                markers_norm[metric] = (markers_norm[metric] - markers_norm[metric].min()) / (markers_norm[metric].max() - markers_norm[metric].min() + 1e-10)
        
        # Create radar plot
        fig = plt.figure(figsize=figsize)
        
        # Calculate angles for radar plot
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Create subplot
        ax = plt.subplot(111, polar=True)
        
        # Add metric labels
        metric_labels = ['Theoretical Score', 'Empirical Score', 'Aggregated Score', 'Expression Proportion']
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        
        # Plot each marker
        for _, row in markers_norm.iterrows():
            values = [row[metric] if metric in row and not np.isnan(row[metric]) else 0 for metric in metrics]
            values += values[:1]  # Close the loop
            
            ax.plot(angles, values, linewidth=2, label=f"{row['marker_gene']} ({row['cell_type']})")
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_title('Marker Gene Metrics Comparison', y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        return fig
    
    def plot_corr_theoric_empiric(self, figsize=(10, 8), cell_types=None, min_genes=1, sort_by='correlation'):
        """
        Plot correlation between theoretical and empirical scores for each cell type
        
        Args:
            figsize (tuple): Figure size as (width, height), default is (10, 8)
            cell_types (list): List of cell types to include. If None, includes all cell types
            min_genes (int): Minimum number of genes required to calculate correlation for a cell type
            sort_by (str): How to sort the cell types. Options are:
                        - 'correlation': Sort by correlation value (default)
                        - 'gene_count': Sort by number of genes
                        - 'name': Sort alphabetically by cell type name
                        - 'abs_correlation': Sort by absolute correlation value
        
        Returns:
            matplotlib.figure.Figure: The figure containing the correlation bar chart
            
        Raises:
            ValueError: If empirical or theoretical scores are not available
        """
        if self.empirical_scores is None or self.theoretical_scores is None:
            raise ValueError("Both theoretical and empirical scores must be available. Run with method='empiric'")
        
        # Create a DataFrame for comparison
        comparison_data = []
        
        # Filter cell types if specified
        if cell_types is None:
            cell_types = list(self.cell_types)
        else:
            # Ensure all requested cell types exist
            for ct in cell_types:
                if ct not in self.cell_types:
                    logger.warning(f"Cell type '{ct}' not found in dataset")
            cell_types = [ct for ct in cell_types if ct in self.cell_types]
        
        # Collect data for all genes in filtered cell types
        for ctype in cell_types:
            # Get scores from both theoretical and empirical calculations
            if (ctype in self.theoretical_scores and ctype in self.empirical_scores and 
                not self.theoretical_scores[ctype].empty and not self.empirical_scores[ctype].empty):
                
                # Get common genes
                genes = set(self.theoretical_scores[ctype].index) & set(self.empirical_scores[ctype].index)
                
                for gene in genes:
                    # Get aggregated scores (mean across k values)
                    theo_score = self.theoretical_scores[ctype].loc[gene].mean()
                    emp_score = self.empirical_scores[ctype].loc[gene].mean()
                    
                    # Add to comparison data
                    comparison_data.append({
                        'gene': gene,
                        'cell_type': ctype,
                        'theoretical_score': theo_score,
                        'empirical_score': emp_score
                    })
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate correlation for each cell type
        correlations = []
        for ctype in cell_types:
            # Get genes with both scores
            ctype_df = comparison_df[comparison_df['cell_type'] == ctype]
            
            if len(ctype_df) >= min_genes:  # Need enough genes for meaningful correlation
                corr = np.corrcoef(ctype_df['theoretical_score'], ctype_df['empirical_score'])[0, 1]
                correlations.append({
                    'cell_type': ctype, 
                    'correlation': corr, 
                    'abs_correlation': abs(corr),
                    'gene_count': len(ctype_df)
                })
        
        # Create correlation DataFrame
        corr_df = pd.DataFrame(correlations)
        
        # Sort based on selected method
        if sort_by == 'correlation':
            corr_df = corr_df.sort_values('correlation', ascending=False)
        elif sort_by == 'gene_count':
            corr_df = corr_df.sort_values('gene_count', ascending=False)
        elif sort_by == 'name':
            corr_df = corr_df.sort_values('cell_type')
        elif sort_by == 'abs_correlation':
            corr_df = corr_df.sort_values('abs_correlation', ascending=False)
        else:
            logger.warning(f"Unknown sort_by value: {sort_by}. Defaulting to 'correlation'")
            corr_df = corr_df.sort_values('correlation', ascending=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot correlation bars
        bars = ax.barh(np.arange(len(corr_df)), corr_df['correlation'], color='skyblue')
        
        # Color-code the bars by correlation value
        for i, bar in enumerate(bars):
            corr = corr_df.iloc[i]['correlation']
            if corr > 0.7:
                bar.set_color('darkgreen')
            elif corr > 0.5:
                bar.set_color('forestgreen')
            elif corr > 0.3:
                bar.set_color('mediumseagreen')
            elif corr > 0:
                bar.set_color('lightgreen')
            elif corr > -0.3:
                bar.set_color('lightcoral')
            elif corr > -0.5:
                bar.set_color('indianred')
            elif corr > -0.7:
                bar.set_color('firebrick')
            else:
                bar.set_color('darkred')
        
        # Add cell type labels
        ax.set_yticks(np.arange(len(corr_df)))
        ax.set_yticklabels([f"{row['cell_type']} (n={row['gene_count']})" for _, row in corr_df.iterrows()])
        
        # Add value labels to the bars
        for i, v in enumerate(corr_df['correlation']):
            ax.text(v + np.sign(v)*0.02, i, f"{v:.2f}", va='center', fontweight='bold',
                    color='black' if v >= 0 else 'white')
        
        # Add a vertical line at x=0
        ax.axvline(x=0, color='gray', linestyle='--')
        
        # Set axis limits, labels, and title
        ax.set_xlim(-1, 1)
        ax.set_xlabel('Pearson Correlation')
        ax.set_title('Correlation between Theoretical and Empirical Scores by Cell Type')
        
        # Add summary statistics as text
        stats_text = (
            f"Mean correlation: {corr_df['correlation'].mean():.3f}\n"
            f"Median correlation: {corr_df['correlation'].median():.3f}\n"
            f"Cell types with correlation > 0.5: {(corr_df['correlation'] > 0.5).sum()}/{len(corr_df)}"
        )
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        
        return fig
    
    def get_scores(self, cell_type: Optional[Union[str, List[str]]] = None, method: str = "empiric") -> Dict:
        """
        Get aggregated scores from the ScSherlock analysis
        
        Args:
            cell_type (str or List[str], optional): If provided, return results only for this cell type
                                                or list of cell types
            method (str): Specify which method's scores to return: "theoric" or "empiric" (default: "empiric")
            
        Returns:
            Dict: Dictionary with only the aggregated scores for each cell type, sorted by decreasing value
            
        Raises:
            ValueError: If specified cell_type does not exist in the dataset or method is invalid,
                        or if the specified method hasn't been run yet
        """
        # Validate method parameter
        if method not in ["theoric", "empiric"]:
            raise ValueError('Method must be either "theoric" or "empiric"')
        
        # Get the appropriate aggregated scores based on the method
        if method == "theoric":
            # Check if theoretical scores have been calculated
            if self.aggregated_scores is None:
                raise ValueError("Theoretical scores haven't been calculated yet. Run with method='theoric' first")
            agg_scores = self.aggregated_scores
        else:  # method == "empiric"
            # Check if empirical scores have been calculated
            if self.aggregated_empirical_scores is None:
                raise ValueError("Empirical scores haven't been calculated yet. Run with method='empiric' first")
            agg_scores = self.aggregated_empirical_scores
        
        # Create a dictionary to hold the results
        results = {}
        
        # Process cell_type parameter if provided
        if cell_type is not None:
            cell_types_to_include = []
            
            # Convert single cell type to list for uniform processing
            if isinstance(cell_type, str):
                cell_types_to_include = [cell_type]
            else:
                cell_types_to_include = list(cell_type)
            
            # Validate that all specified cell types exist
            for ct in cell_types_to_include:
                if ct not in self.cell_types:
                    raise ValueError(f"Cell type '{ct}' not found in the dataset")
            
            # Filter and sort scores by cell type
            for ct in cell_types_to_include:
                if ct in agg_scores and not isinstance(agg_scores[ct], pd.DataFrame) and not agg_scores[ct].empty:
                    # Extract only the aggregated scores and sort in descending order
                    results[ct] = agg_scores[ct].sort_values(ascending=False)
        else:
            # Sort aggregated scores for all cell types
            for ct, scores in agg_scores.items():
                if not isinstance(scores, pd.DataFrame) and not scores.empty:
                    # Extract only the aggregated scores and sort in descending order
                    results[ct] = scores.sort_values(ascending=False)
        
        return results

    def create_proportion_based_hierarchy(self, annotation_columns, min_proportion=0.05):
        """
        Create hierarchy relationships based on co-occurrence proportions.
        
        Args:
            annotation_columns (list): List of column names in adata.obs for hierarchy levels
            min_proportion (float): Minimum proportion threshold for including relationships
            
        Returns:
            dict: Hierarchical structure of cell types
        """
        relationships = {}
        
        # Process each pair of adjacent hierarchy levels
        for i in range(len(annotation_columns) - 1):
            parent_col = annotation_columns[i]
            child_col = annotation_columns[i+1]
            
            # Get unique values for parent and child levels
            parent_values = self.adata.obs[parent_col].unique()
            child_values = self.adata.obs[child_col].unique()
            
            # Create relationships for this level pair
            level_relationships = defaultdict(list)
            
            # For each parent value, find related child values
            for parent_val in parent_values:
                # Get cells with this parent value
                parent_mask = self.adata.obs[parent_col] == parent_val
                parent_cells = sum(parent_mask)
                
                if parent_cells == 0:
                    continue
                
                # Check each child value
                for child_val in child_values:
                    # Count cells with both annotations
                    both = sum(parent_mask & (self.adata.obs[child_col] == child_val))
                    
                    if both == 0:
                        continue
                    
                    # Calculate proportion of parent cells with this child annotation
                    proportion = both / parent_cells
                    
                    # If meets minimum proportion, add to relationships
                    if proportion >= min_proportion:
                        level_relationships[parent_val].append({
                            'id': child_val,
                            'proportion': float(proportion)
                        })
            
            # Store relationships for this level pair
            if i == 0:
                relationships = {parent: sorted(children, key=lambda x: x['proportion'], reverse=True) 
                            for parent, children in level_relationships.items()}
            else:
                # Integrate with existing hierarchy
                for parent_level in list(relationships.keys()):
                    for child_item in relationships[parent_level]:
                        child_id = child_item['id']
                        # If this child has its own children, add them
                        if child_id in level_relationships:
                            child_item['children'] = sorted(
                                level_relationships[child_id], 
                                key=lambda x: x['proportion'], 
                                reverse=True
                            )
        
        return relationships

    def create_hierarchy_graph(self, annotation_columns, min_proportion=0.05, max_children=None):
        """
        Create a hierarchy graph based on co-occurrence proportions.
        
        Args:
            annotation_columns (list): List of column names in adata.obs for hierarchy levels
            min_proportion (float): Minimum proportion threshold for including relationships
            max_children (int, optional): Maximum number of children to include for each node
            
        Returns:
            networkx.DiGraph: Directed graph representing cell type hierarchy
        """
        # Create proportion-based relationships
        relationships = self.create_proportion_based_hierarchy(annotation_columns, min_proportion)
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add root node
        G.add_node('root')
        
        # Add first level nodes connected to root
        for parent in relationships:
            G.add_edge('root', parent)
            
            # Helper function to recursively add child nodes
            def add_children(parent_node, children):
                # Limit number of children if specified
                if max_children is not None and len(children) > max_children:
                    # Take only the top N children by proportion
                    children = sorted(children, key=lambda x: x['proportion'], reverse=True)[:max_children]
                
                for child_item in children:
                    child_id = child_item['id']
                    proportion = child_item['proportion']
                    
                    # Create node name and add edge
                    child_node = f"{parent_node}_{child_id}"
                    G.add_edge(parent_node, child_node)
                    
                    # Store attributes
                    G.nodes[child_node]['label'] = child_id
                    G.nodes[child_node]['proportion'] = proportion
                    G.edges[parent_node, child_node]['weight'] = proportion
                    
                    # Recursively add grandchildren if any
                    if 'children' in child_item:
                        add_children(child_node, child_item['children'])
            
            # Add children for this parent
            add_children(parent, relationships[parent])
        
        self.G = G

    def visualize_hierarchy(self, min_proportion=0.05, max_children=None, output_file=None, figsize=(20, 12)):
        """
        Visualize the cell hierarchy using a hierarchical graph layout.
        
        Args:
            annotation_columns (list): List of column names in adata.obs for hierarchy levels
            min_proportion (float): Minimum proportion threshold for including relationships
            max_children (int, optional): Maximum number of children to include for each node
            output_file (str, optional): Path to save the figure
            figsize (tuple): Figure size (width, height)
            
        Returns:
            matplotlib.figure.Figure: Figure object with the hierarchy visualization
        """
        if hasattr(self, 'G') and self.G is not None:
            G = self.G
        else:
            raise ValueError("No hierarchical graph present in ScSherlock object. Ensure hierarchy creation is successful.")

            

        plt.figure(figsize=figsize)
        
        try:
            # Use dot layout for hierarchical trees
            pos = graphviz_layout(G, prog='twopi')
            
            # Get node depths for coloring
            node_depth = {}
            for node in G.nodes():
                node_depth[node] = 0 if node == 'root' else node.count('_') + 1
            
            max_depth = max(node_depth.values())
            
            # Prepare node colors
            node_colors = [plt.cm.viridis(depth/max_depth) for node, depth in node_depth.items()]
            
            # Prepare edge widths
            edge_widths = [G.edges[edge].get('weight', 0.5) * 3 + 0.5 for edge in G.edges()]
            
            # Prepare node labels
            labels = {}
            for node in G.nodes():
                if node == 'root':
                    labels[node] = 'ROOT'
                elif 'label' in G.nodes[node]:
                    label = G.nodes[node]['label']
                    if 'proportion' in G.nodes[node] and node != 'root':
                        prop = G.nodes[node]['proportion']
                        labels[node] = f"{label}\n({prop:.1%})"
                    else:
                        labels[node] = label
                else:
                    parts = node.split('_')
                    labels[node] = parts[-1]
            
            # Draw the graph
            nx.draw(G, pos, 
                    node_color=node_colors,
                    node_size=800,
                    width=edge_widths,
                    with_labels=True,
                    labels=labels,
                    font_size=9,
                    font_weight='bold',
                    arrows=True,
                    edge_color='gray',
                    alpha=0.9)
            
            plt.title('Cell Type Hierarchy', fontsize=15)
            plt.axis('off')
            
            # Save if output file is specified
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
            
            fig = plt.gcf()
            return fig
            
        except Exception as e:
            logger.error(f"Error with graphviz_layout: {e}")
            logger.error("Make sure Graphviz is installed:")
            logger.error("  Ubuntu/Debian: sudo apt-get install graphviz graphviz-dev")
            logger.error("  macOS: brew install graphviz")
            logger.error("  Windows: Download from https://graphviz.org/download/")
            logger.error("Then install pygraphviz: pip install pygraphviz")
            return None

    def visualize_hierarchy_marker(self, cutoff=0.5, max_children=None, output_file=None, figsize=(20, 12)):
        """
        Visualize the cell hierarchy with marker gene information.
        
        This method creates a hierarchy visualization where:
        - Cell types with no entry in sorted_empirical_table are shown in grey
        - Cell types with no marker genes passing the cutoff are shown in red
        - Cell types with marker genes show the marker name alongside the cell type
        
        Args:
            annotation_columns (list): List of column names in adata.obs for hierarchy levels
            cutoff (float): Score cutoff threshold for marker genes
            max_children (int, optional): Maximum number of children to include for each node
            output_file (str, optional): Path to save the figure
            figsize (tuple): Figure size (width, height)
            
        Returns:
            matplotlib.figure.Figure: Figure object with the hierarchy visualization
        """
        # Check if marker analysis has been run
        if self.sorted_empirical_table is None and self.sorted_table is None:
            raise ValueError("No marker analysis results available. Run ScSherlock.run() first")
            
        # Determine which sorted table to use based on what's available
        if self.sorted_empirical_table is not None:
            sorted_table = self.sorted_empirical_table
            logger.info("Using empirical marker scores for visualization")
        else:
            sorted_table = self.sorted_table
            logger.info("Using theoretical marker scores for visualization")
        
        if hasattr(self, 'G') and self.G is not None:
            G = self.G
        else:
            raise ValueError("No hierarchical graph present in ScSherlock object. Ensure hierarchy creation is successful.")

            
        
        plt.figure(figsize=figsize)
        
        try:
            # Use dot layout for hierarchical trees
            pos = graphviz_layout(G, prog='twopi')
            
            # Get node depths for coloring
            node_depth = {}
            for node in G.nodes():
                node_depth[node] = 0 if node == 'root' else node.count('_') + 1
            
            max_depth = max(node_depth.values())
            
            # Prepare node colors based on marker genes status
            node_colors = []
            for node in G.nodes():
                if node == 'root':
                    # Root node is always blue
                    node_colors.append('royalblue')
                    continue
                    
                # Extract cell type name from node
                if 'label' in G.nodes[node]:
                    cell_type = G.nodes[node]['label']
                else:
                    parts = node.split('_')
                    cell_type = parts[-1]
                
                # Check if cell type has markers
                if cell_type in sorted_table:
                    # Check if there are markers passing the cutoff
                    if not sorted_table[cell_type].empty and sorted_table[cell_type]['aggregated'].max() >= cutoff:
                        # Has marker passing cutoff - use a gradient based on depth
                        node_colors.append(plt.cm.viridis(node_depth[node]/max_depth))
                    else:
                        # Has entry but no marker passing cutoff - red
                        node_colors.append('red')
                else:
                    # No entry in sorted_table - grey
                    node_colors.append('grey')
            
            # Prepare edge widths
            edge_widths = [G.edges[edge].get('weight', 0.5) * 3 + 0.5 for edge in G.edges()]
            
            # Prepare node labels with marker information
            labels = {}
            for node in G.nodes():
                if node == 'root':
                    labels[node] = 'ROOT'
                    continue
                    
                # Extract cell type name
                if 'label' in G.nodes[node]:
                    cell_type = G.nodes[node]['label']
                else:
                    parts = node.split('_')
                    cell_type = parts[-1]
                
                # Check if cell type has markers
                if cell_type in sorted_table and not sorted_table[cell_type].empty:
                    # Get marker genes passing the cutoff
                    markers_passing = sorted_table[cell_type][sorted_table[cell_type]['aggregated'] >= cutoff]
                    
                    if not markers_passing.empty:
                        # Get top marker gene
                        top_marker = markers_passing.index[0]
                        labels[node] = f"{cell_type}\n[{top_marker}]"
                    else:
                        # No markers passing cutoff
                        labels[node] = f"{cell_type}\n[no marker]"
                else:
                    # No entry in sorted_table
                    labels[node] = f"{cell_type}"
                    
            
            # Draw the graph
            nx.draw(G, pos, 
                    node_color=node_colors,
                    node_size=1000,  # Larger nodes for better readability
                    width=edge_widths,
                    with_labels=True,
                    labels=labels,
                    font_size=9,
                    font_weight='bold',
                    arrows=True,
                    edge_color='gray',
                    alpha=0.9)
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', label='With marker gene',
                        markerfacecolor=plt.cm.viridis(0.5), markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label='No marker passing cutoff',
                        markerfacecolor='red', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label='Cell type not analyzed',
                        markerfacecolor='grey', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label='Root',
                        markerfacecolor='royalblue', markersize=10)
            ]
            plt.legend(handles=legend_elements, loc='lower right', fontsize=9)
            
            plt.title(f'Cell Type Hierarchy with Marker Genes (cutoff={cutoff})', fontsize=15)
            plt.axis('off')
            
            # Save if output file is specified
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
            
            fig = plt.gcf()
            plt.show()
            return fig
            
        except Exception as e:
            logger.error(f"Error with visualization: {e}")
            logger.error("Make sure Graphviz is installed:")
            logger.error("  Ubuntu/Debian: sudo apt-get install graphviz graphviz-dev")
            logger.error("  macOS: brew install graphviz")
            logger.error("  Windows: Download from https://graphviz.org/download/")
            logger.error("Then install pygraphviz: pip install pygraphviz")
            return None


# Keep the Numba-optimized scoring functions
@nb.jit(nopython=True)
def compute_diff_scores(alpha, beta):
    """
    Compute the maximum difference between beta and alpha.
    """
    n_genes, n_thresholds = alpha.shape
    scores = np.zeros(n_genes)
    
    for g in range(n_genes):
        max_diff = -1.0
        for t in range(n_thresholds):
            diff = beta[g, t] - alpha[g, t]
            if diff > max_diff:
                max_diff = diff
        scores[g] = max_diff
        
    return scores

@nb.jit(nopython=True)
def compute_sensFPRzero_scores(alpha, beta):
    """
    Compute sensitivity at point of zero false positive rate.
    """
    n_genes, n_thresholds = alpha.shape
    scores = np.zeros(n_genes)
    
    for g in range(n_genes):
        max_beta_idx = 0
        max_beta = -1.0
        for t in range(n_thresholds):
            if beta[g, t] > max_beta:
                max_beta = beta[g, t]
                max_beta_idx = t
        scores[g] = 1.0 - alpha[g, max_beta_idx]
        
    return scores

@nb.jit(nopython=True)
def compute_sensPPV99_scores(alpha, beta):
    """
    Compute sensitivity at point where positive predictive value > 99%.
    """
    n_genes, n_thresholds = alpha.shape
    scores = np.zeros(n_genes)
    
    for g in range(n_genes):
        max_idx = -1
        for t in range(n_thresholds):
            # Calculate PPV, handling division by zero
            denominator = 2.0 - alpha[g, t] - beta[g, t]
            if denominator == 0:
                ppv = 0.0
            else:
                ppv = (1.0 - alpha[g, t]) / denominator
                
            if ppv > 0.99 and max_idx == -1:
                max_idx = t
                
        if max_idx != -1:
            scores[g] = 1.0 - alpha[g, max_idx]
        else:
            scores[g] = 0.0
            
    return scores