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
    
    def __init__(self, adata, column_patient: str, config: Optional[ScSherlockConfig] = None):
        """
        Initialize ScSherlock with data and configuration
        
        Args:
            adata: AnnData object containing single-cell gene expression data
            column_patient: Column name in adata.obs for patient IDs
            config: Configuration object with algorithm parameters (optional)
        """
        self.adata = adata
        self.column_patient = column_patient
        self.config = config or ScSherlockConfig()
        self.column_patient = self.simplify_patient_ids()        
        # Validate inputs
        self._validate_inputs()
        
        # Internal state
        self.theoretical_scores = {}
        self.expr_proportions = {}
        self.processed_scores = {}
        self.aggregated_scores = {}
        self.sorted_table = {}
        self.filtered_scores = {}
        self.empirical_scores = {}
        self.aggregated_empirical_scores = {}
        self.sorted_empirical_table = {}
        self.top_markers = {}
        self.method_run = {}
        
        # Hierarchy graph
        self.G = None
        
        logger.info(f"ScSherlock initialized with {self.adata.shape} data matrix")
    
    def _validate_inputs(self):
        """Validate input data and parameters"""
        if self.column_patient not in self.adata.obs.columns:
            raise ValueError(f"Patient column '{self.column_patient}' not found in adata.obs")

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
        
        return simplified_column


    def run(self, column_ctype: str, method: str = "empiric", bootstrap: bool = False) -> Dict[str, str]:
        """
        Run the complete ScSherlock algorithm pipeline
        
        Args:
            column_ctype: Column name in adata.obs for cell type annotations
            method: Method to use, either "theoric" or "empiric"
            bootstrap: Whether to perform bootstrap validation of markers (only for empiric method)
                
        Returns:
            Dict[str, str]: Dictionary of top marker genes for each cell type
        """
        if method not in ["theoric", "empiric"]:
            raise ValueError('Method must be either "theoric" or "empiric"')
        
        # Validate cell type column
        if column_ctype not in self.adata.obs.columns:
            raise ValueError(f"Cell type column '{column_ctype}' not found in adata.obs")
            
        # Check for empty cell types
        cell_types = self.adata.obs[column_ctype].unique()
        for ctype in cell_types:
            if sum(self.adata.obs[column_ctype] == ctype) < 10:
                logger.warning(f"Cell type '{ctype}' has fewer than 10 cells")
                
        # Initialize dictionaries for this annotation if needed
        if column_ctype not in self.method_run:
            self.theoretical_scores[column_ctype] = None
            self.expr_proportions[column_ctype] = None
            self.processed_scores[column_ctype] = None
            self.aggregated_scores[column_ctype] = None
            self.sorted_table[column_ctype] = None
            self.filtered_scores[column_ctype] = None
            self.empirical_scores[column_ctype] = None
            self.aggregated_empirical_scores[column_ctype] = None
            self.sorted_empirical_table[column_ctype] = None
            self.top_markers[column_ctype] = None
            self.method_run[column_ctype] = None
        
        # If this annotation's theoric model was already run 
        if self.method_run[column_ctype] is None:
            # Set random seed for reproducibility
            np.random.seed(self.config.random_seed)
            
            # Step 1: Calculate theoretical scores based on binomial distributions
            logger.info(f"Calculating theoretical scores for {column_ctype}...")
            self.theoretical_scores[column_ctype], self.expr_proportions[column_ctype] = self._calculate_theoretical_scores_parallel(column_ctype)
            
            # Step 2: Apply multiple category correction to theoretical scores
            logger.info("Applying multi-category correction...")
            self.processed_scores[column_ctype] = self._apply_multi_category_correction(self.theoretical_scores[column_ctype])

            # Step 3: Aggregate scores across k values
            logger.info("Aggregating scores...")
            self.aggregated_scores[column_ctype] = self._aggregate_scores(self.processed_scores[column_ctype])
            
            # Step 4: Sort scores and prepare for filtering
            logger.info("Sorting scores...")
            self.sorted_table[column_ctype] = self._sort_scores(
                self.aggregated_scores[column_ctype], 
                self.processed_scores[column_ctype], 
                self.expr_proportions[column_ctype]
            )
            
            if method == "theoric":
                logger.info("Identifying top markers...")
                self.top_markers[column_ctype] = self._construct_top_marker_list(self.sorted_table[column_ctype])
                self.method_run[column_ctype] = "theoric"
                return self.top_markers[column_ctype]

            # Step 5: Filter genes based on scores and expression criteria
            logger.info("Filtering genes...")
            self.filtered_scores[column_ctype] = self._filter_genes(self.sorted_table[column_ctype], column_ctype)

            # Step 6: Calculate empirical scores through simulation
            logger.info("Calculating empirical scores...")
            self.empirical_scores[column_ctype] = self.empirical_scores_optimized_batch_parallel_only_ctype(column_ctype)
            
            # Step 7: Aggregate empirical scores across k values
            logger.info("Aggregating empirical scores...")
            self.aggregated_empirical_scores[column_ctype] = self._aggregate_scores(self.empirical_scores[column_ctype])

            # Step 8: Sort empirical scores
            logger.info("Sorting empirical scores...")
            self.sorted_empirical_table[column_ctype] = self._sort_scores(
                self.aggregated_empirical_scores[column_ctype], 
                self.empirical_scores[column_ctype], 
                self.expr_proportions[column_ctype]
            )

            # Step 9: Perform bootstrap validation if requested
            if bootstrap and method == "empiric":
                logger.info("Performing bootstrap validation...")
                self._perform_bootstrap_validation(column_ctype)

            # Step 10: Construct final list of top markers
            logger.info("Identifying top markers...")
            self.top_markers[column_ctype] = self._construct_top_marker_list(self.sorted_empirical_table[column_ctype])
            self.method_run[column_ctype] = "empiric"

            logger.info(f"ScSherlock completed. Found markers for {len(self.top_markers[column_ctype])}/{len(cell_types)} cell types")
            return self.top_markers[column_ctype]
        
        elif self.method_run[column_ctype] == 'theoric':
            if method == "theoric":
                logger.info(f"ScSherlock already run with theoric method for {column_ctype}. Found markers for {len(self.top_markers[column_ctype])}/{len(cell_types)} cell types")
                return self.top_markers[column_ctype]
            else:       
                logger.info(f"Skipping theorical model as it was already run. Running empiric model for {column_ctype}")
                # Step 5: Filter genes based on scores and expression criteria
                logger.info("Filtering genes...")
                self.filtered_scores[column_ctype] = self._filter_genes(self.sorted_table[column_ctype], column_ctype)

                # Step 6: Calculate empirical scores through simulation
                logger.info("Calculating empirical scores...")
                self.empirical_scores[column_ctype] = self.empirical_scores_optimized_batch_parallel_only_ctype(column_ctype)

                # Step 7: Aggregate empirical scores across k values
                logger.info("Aggregating empirical scores...")
                self.aggregated_empirical_scores[column_ctype] = self._aggregate_scores(self.empirical_scores[column_ctype])

                # Step 8: Sort empirical scores
                logger.info("Sorting empirical scores...")
                self.sorted_empirical_table[column_ctype] = self._sort_scores(
                    self.aggregated_empirical_scores[column_ctype], 
                    self.empirical_scores[column_ctype], 
                    self.expr_proportions[column_ctype]
                )

                # Step 9: Perform bootstrap validation if requested
                if bootstrap and method == "empiric":
                    logger.info("Performing bootstrap validation...")
                    self._perform_bootstrap_validation(column_ctype)

                # Step 10: Construct final list of top markers
                logger.info("Identifying top markers...")
                self.top_markers[column_ctype] = self._construct_top_marker_list(self.sorted_empirical_table[column_ctype])
                self.method_run[column_ctype] = "empiric"

                logger.info(f"ScSherlock completed. Found markers for {len(self.top_markers[column_ctype])}/{len(cell_types)} cell types")
                return self.top_markers[column_ctype]
        else:
            logger.info(f"ScSherlock already run with empiric method for {column_ctype}. Found markers for {len(self.top_markers[column_ctype])}/{len(cell_types)} cell types")
            return self.top_markers[column_ctype]


    def _estimate_binomial_parameters(self, column_ctype: str) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
        """
        Estimate parameters for binomial distributions
        
        Args:
            column_ctype: Column name in adata.obs for cell type annotations
            
        Returns:
            Tuple containing:
            - counts_per_ctype: Counts per cell type
            - counts_per_ctype_complement: Counts for complement of each cell type
            - count_proportions_per_ctype: Expression proportions for each cell type
            - count_proportions_per_ctype_complement: Expression proportions for complements
        """
        self.adata.obs[column_ctype] = self.adata.obs[column_ctype].astype('category')
        cat_list = self.adata.obs[column_ctype].cat.categories
        
        if self.config.parameter_estimation == ParameterEstimation.PATIENT_MEDIAN:
            # Method 1: Patient-specific estimation with median across patients
            
            # Generate pseudobulk data
            adata_agg = ADPBulk(self.adata, [self.column_patient, column_ctype])
            pseudobulk_matrix = adata_agg.fit_transform()
            sample_meta = adata_agg.get_meta()
            
            # Create multi-index for cell type and patient
            tuples = list(zip(sample_meta[column_ctype], sample_meta[self.column_patient]))
            index = pd.MultiIndex.from_tuples(tuples, names=[column_ctype, self.column_patient])
            pseudobulk_matrix.set_index(index, inplace=True)
            
            # Calculate total counts and cell numbers
            total_counts = pseudobulk_matrix.sum(axis=1)
            n_cells = self.adata.obs.groupby(by=[column_ctype, self.column_patient]).size()
            
            # Calculate median counts per cell type
            counts_per_ctype = (total_counts/n_cells).groupby(level=column_ctype).median()
            
            # Calculate median counts for complements
            counts_per_ctype_complement = pd.Series({
                ctype: ((total_counts.drop(ctype).groupby(level=self.column_patient).sum()) / 
                        (n_cells.drop(ctype).groupby(level=self.column_patient).sum())).median()
                for ctype in cat_list
            })
            
            # Calculate normalized proportions
            count_proportions_per_ctype = (
                pseudobulk_matrix.div(total_counts.values, axis=0)
                .groupby(level=column_ctype)
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
                adata_A = self.adata[self.adata.obs[column_ctype]==ctype].copy()
                adata_nA = self.adata[self.adata.obs[column_ctype]!=ctype].copy()
                
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
    
    def _filter_genes(self, sorted_table: Dict, column_ctype: str) -> Dict:
        """
        Filter genes based on expression criteria
        
        Args:
            sorted_table: Dictionary of sorted scores
            column_ctype: Column name in adata.obs for cell type annotations
            
        Returns:
            Dict: Filtered scores
        """
        # Count patients expressing each gene by cell type
        patient_agg = ADPBulk(
            self.adata,
            [self.column_patient, column_ctype],
            name_delim="--",
            group_delim="::"
        )
        patient_matrix = patient_agg.fit_transform()

        # Split index and determine which part contains the cell type
        split_indices = [idx.split('--') for idx in patient_matrix.index]
        sample_splits = split_indices[:5]  # Look at a few samples to determine the pattern

        # Check which part contains the cell type column name
        if any(column_ctype in part for part in [split[0] for split in sample_splits]):
            # Cell type is in the first part
            cell_type_position = 0
            patient_position = 1
        elif any(column_ctype in part for part in [split[1] for split in sample_splits]):
            # Cell type is in the second part
            cell_type_position = 1
            patient_position = 0
        else:
            # Fallback - assume default positions
            print("Warning: Could not determine position of cell type in index")
            cell_type_position = 1  # Assume cell type is second
            patient_position = 0    # Assume patient is first

        # Create a MultiIndex with consistent ordering
        patient_matrix.index = pd.MultiIndex.from_tuples(
            [(split[patient_position], split[cell_type_position]) 
            for split in split_indices],
            names=['patient', 'cell_type']
        )

        # Now group by cell type, which is consistently at level 1
        ctype_n_patients = (patient_matrix > 0).groupby(level='cell_type').sum()

        # Clean up the cell type names
        ctype_n_patients.index = ctype_n_patients.index.str.split('::').str[-1]
        # Count reads per gene by cell type
        reads_agg = ADPBulk(self.adata, column_ctype, group_delim="::")
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
    

    
    def empirical_scores_optimized_batch_parallel_only_ctype(self, column_ctype: str, n_sim=1000):
        """
        Calculate empirical marker gene scores through simulation using optimized statistical libraries
        with parallel processing only on cell types.
        
        Args:
            column_ctype: Column name in adata.obs for cell type annotations
            n_sim: Number of simulations to run (default: 1000)
                
        Returns:
            Dictionary of empirical scores by cell type
        """
        # Get instance variables
        filtered_scores = self.filtered_scores[column_ctype]
        adata = self.adata
        k_values = self.config.k_values
        scoring = self.config.scoring_method
        seed = 42
        n_jobs = self.config.n_jobs
        
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
    
    def _calculate_theoretical_scores_parallel(self, column_ctype: str) -> Tuple[Dict, Dict]:
        """
        Calculate theoretical scores using joblib for parallelization with reduced memory footprint
        
        Args:
            column_ctype: Column name in adata.obs for cell type annotations
            
        Returns:
            Tuple[Dict, Dict]: Scores and expression proportions
        """
        from joblib import Parallel, delayed
        import gc
        
        # Estimate binomial distribution parameters
        parameters = self._estimate_binomial_parameters(column_ctype)
        
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
        all_cell_types = list(self.adata.obs[column_ctype].unique())
        n_jobs = min(self.config.n_jobs, batch_size)  # Limit jobs per batch
        
        scores = {}
        for batch_start in range(0, len(all_cell_types), batch_size):
            batch_end = min(batch_start + batch_size, len(all_cell_types))
            current_batch = all_cell_types[batch_start:batch_end]
            
            # Use memory-efficient options in Parallel
            batch_results = Parallel(
                n_jobs=n_jobs, 
                max_nbytes='50M',  # Limit memory per job
                prefer="threads",   # Use threads for better memory sharing
                verbose=False       # Show progress
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
        
        # create a combined DataFrame with all non-empty scores
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
                gene = idx  # gene names are the index
                if gene not in gene_sums:
                    gene_sums[gene] = {}
                
                # Extract scores for each column and add to gene_sums
                for col in df.columns:
                    if col not in gene_sums[gene]:
                        gene_sums[gene][col] = 0
                    gene_sums[gene][col] += row[col]
        
        # normalize each score by dividing by the total for that gene and column
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
    
    def visualize_marker(self, column_ctype: str, gene: str, cell_type: str = None):
        """
        Visualize the expression distribution of a marker gene
        
        Args:
            column_ctype: Column name in adata.obs for cell type annotations
            gene: Gene name to visualize
            cell_type: Cell type to highlight (optional)
        """
        if gene not in self.adata.var_names:
            raise ValueError(f"Gene '{gene}' not found in the dataset")
            
        # Get the cell type this gene is a marker for (if not specified)
        if cell_type is None and column_ctype in self.top_markers and self.top_markers[column_ctype] is not None:
            for ctype, marker in self.top_markers[column_ctype].items():
                if marker == gene:
                    cell_type = ctype
                    break
            if cell_type is None:
                logger.warning(f"Gene '{gene}' is not a top marker for any cell type in annotation {column_ctype}")
        
        # Extract gene expression data
        gene_expression = self.adata[:, gene].X.toarray()
        categories = self.adata.obs[column_ctype]
        
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

    def get_results(self, column_ctype: str = None) -> Dict:
        """
        Get complete results from the ScSherlock analysis
        
        Args:
            column_ctype: Column name in adata.obs for cell type annotations (optional)
                        If None, returns results for all annotations
        
        Returns:
            Dict: Dictionary with all results for the specified annotation
        """
        if column_ctype is not None:
            # Check if the annotation has been processed
            if column_ctype not in self.method_run:
                raise ValueError(f"No analysis exists for {column_ctype}. Run ScSherlock.run() with this annotation first")
                
            return {
                'top_markers': self.top_markers.get(column_ctype),
                'sorted_emp_table': self.sorted_empirical_table.get(column_ctype),
                'filtered_scores': self.filtered_scores.get(column_ctype),
                'theoretical_scores': self.theoretical_scores.get(column_ctype),
                'empirical_scores': self.empirical_scores.get(column_ctype),
                'method_run': self.method_run.get(column_ctype)
            }
        else:
            # Return all results organized by annotation
            results = {}
            for annotation in self.method_run.keys():
                results[annotation] = {
                    'top_markers': self.top_markers.get(annotation),
                    'sorted_emp_table': self.sorted_empirical_table.get(annotation),
                    'filtered_scores': self.filtered_scores.get(annotation),
                    'theoretical_scores': self.theoretical_scores.get(annotation),
                    'empirical_scores': self.empirical_scores.get(annotation),
                    'method_run': self.method_run.get(annotation)
                }
            return results
    
    def export_markers(self, column_ctype: str, output_file: str = None) -> pd.DataFrame:
        """
        Export marker genes to a DataFrame or CSV file
        
        Args:
            column_ctype: Column name in adata.obs for cell type annotations
            output_file: Path to output CSV file (optional)
                
        Returns:
            pd.DataFrame: DataFrame of marker genes with scores
        
        Raises:
            ValueError: If no markers identified (run ScSherlock.run() first)
        """
        if column_ctype not in self.top_markers or self.top_markers[column_ctype] is None:
            raise ValueError(f"No markers identified for {column_ctype}. Run ScSherlock.run() with this annotation first")
        
        if column_ctype not in self.method_run or self.method_run[column_ctype] is None:
            raise ValueError(f"Model method not set for {column_ctype}. Run ScSherlock.run() with this annotation first")
        
        # Determine which tables to use based on the method that was run
        if self.method_run[column_ctype] == "empiric":
            if column_ctype not in self.sorted_empirical_table or column_ctype not in self.empirical_scores:
                raise ValueError(f"Empirical scores not available for {column_ctype}. Run with method='empiric'")
            sorted_table = self.sorted_empirical_table[column_ctype]
            scores_table = self.empirical_scores[column_ctype]
        else:  # theoric
            if column_ctype not in self.sorted_table or column_ctype not in self.theoretical_scores:
                raise ValueError(f"Theoretical scores not available for {column_ctype}. Run with method='theoric'")
            sorted_table = self.sorted_table[column_ctype]
            scores_table = self.theoretical_scores[column_ctype]
                
        # Compile marker information
        results = []
        for ctype, gene in self.top_markers[column_ctype].items():
            # Get scores for this gene
            if ctype in self.theoretical_scores[column_ctype] and gene in self.theoretical_scores[column_ctype][ctype].index:
                theoretical_score = self.theoretical_scores[column_ctype][ctype].loc[gene].mean()
            else:
                theoretical_score = float('nan')
                
            # Get empirical score if available
            if self.method_run[column_ctype] == "empiric" and ctype in self.empirical_scores[column_ctype] and gene in self.empirical_scores[column_ctype][ctype].index:
                empirical_score = self.empirical_scores[column_ctype][ctype].loc[gene].mean()
            else:
                empirical_score = float('nan')
                
            # Get aggregated score and expression proportion
            aggregated_score = sorted_table[ctype].loc[gene, 'aggregated']
            exp_prop = sorted_table[ctype].loc[gene, 'exp_prop']
            
            results.append({
                'cell_type': ctype,
                'marker_gene': gene,
                'theoretical_score': theoretical_score,
                'empirical_score': empirical_score if self.method_run[column_ctype] == "empiric" else float('nan'),
                'aggregated_score': aggregated_score,
                'expression_proportion': exp_prop,
                'model_used': self.method_run[column_ctype],
                'annotation': column_ctype  # Add annotation information
            })
        
        # Create DataFrame
        markers_df = pd.DataFrame(results)
        
        # Reorder columns for better readability
        col_order = [
            'annotation', 'cell_type', 'marker_gene', 'model_used', 'aggregated_score',
            'theoretical_score', 'empirical_score', 'expression_proportion'
        ]
        markers_df = markers_df[col_order]
        
        # Save to file if specified
        if output_file:
            markers_df.to_csv(output_file, index=False)
            logger.info(f"Exported {len(markers_df)} markers for {column_ctype} to {output_file} using {self.method_run[column_ctype]} model")
                
        return markers_df
    
    def plot_marker_heatmap(self, column_ctype, n_genes=1, cutoff=0, remove_ctype_no_marker=False, cmap='viridis', 
                        standard_scale='var', use_raw=False, save=None, show=None, 
                        dataset_column=None, score_type='weighted_score', **kwargs):
        """
        Create a heatmap visualization of the identified marker genes
        
        This method creates a matrix plot showing the expression of top marker genes
        across all cell types. It orders the genes to match their corresponding cell types.
        
        Args:
            column_ctype: Column name in adata.obs for cell type annotations
            n_genes (int): Number of top genes to display for each cell type (default: 1)
            cutoff (float): Minimum score cutoff for including genes (default: 0)
            remove_ctype_no_marker (bool): Whether to remove cell types with no markers (default: False)
            cmap (str): Colormap for the heatmap (default: 'viridis')
            standard_scale (str): Scale the data ('var', 'group', or None) (default: 'var')
            use_raw (bool): Whether to use raw data for plotting (default: False)
            save (str or bool): If True or a str, save the figure (default: None)
            show (bool): Whether to show the plot (default: None)
            dataset_column (str, optional): If provided, use markers from dataset analysis using this column
            score_type (str, optional): Which score to use from dataset analysis ('avg_score' or 'weighted_score')
            **kwargs: Additional arguments passed to sc.pl.matrixplot
                
        Returns:
            matplotlib.axes.Axes: The axes object containing the plot
        
        Raises:
            ValueError: If no markers are available (run ScSherlock.run() first) or if no 
                    cells are found for the given cell types
        """
        # Check if dataset analysis should be used
        using_dataset_analysis = False
        dataset_key = None
        
        if dataset_column is not None:
            dataset_key = f"{column_ctype}_by_{dataset_column}"
            if dataset_key in self.__dict__ and 'combined_tables' in self.__dict__[dataset_key]:
                using_dataset_analysis = True
                sorted_table = self.__dict__[dataset_key]['combined_tables']
                logger.info(f"Using dataset analysis results with {score_type}")
        
        # Validate score_type
        if using_dataset_analysis and score_type not in ['avg_score', 'weighted_score']:
            logger.warning(f"Invalid score_type '{score_type}'. Defaulting to 'weighted_score'")
            score_type = 'weighted_score'
        
        # Check if the annotation is available if not using dataset analysis
        if not using_dataset_analysis:
            if column_ctype not in self.method_run:
                raise ValueError(f"No markers available for {column_ctype}. Run ScSherlock.run() with this annotation first")
            
            # Select appropriate sorted table based on which method was run
            if self.method_run[column_ctype] == "empiric":
                if column_ctype not in self.sorted_empirical_table or self.sorted_empirical_table[column_ctype] is None:
                    raise ValueError(f"Empirical scores not available for {column_ctype}. Run with method='empiric'")
                sorted_table = self.sorted_empirical_table[column_ctype]
            else:  # theoric
                if column_ctype not in self.sorted_table or self.sorted_table[column_ctype] is None:
                    raise ValueError(f"Theoretical scores not available for {column_ctype}. Run with method='theoric'")
                sorted_table = self.sorted_table[column_ctype]
        
        # Get cell types that have valid markers
        cell_to_genes = {}
        for ctype, table in sorted_table.items():
            if using_dataset_analysis:
                # Dataset analysis handling
                if not table.empty:
                    # Only include cell types that have marker genes meeting the cutoff
                    valid_genes = table[table[score_type] >= cutoff]
                    if not valid_genes.empty:
                        # Get up to n_genes top markers
                        cell_to_genes[ctype] = valid_genes.index[:min(n_genes, len(valid_genes))].tolist()
            else:
                # Standard analysis handling
                if isinstance(table, pd.DataFrame) and not table.empty:
                    # Only include cell types that have marker genes meeting the cutoff
                    valid_genes = table[table['aggregated'] >= cutoff]
                    if not valid_genes.empty:
                        # Get up to n_genes top markers
                        cell_to_genes[ctype] = valid_genes.index[:min(n_genes, len(valid_genes))].tolist()
        
        # Check if any markers were found
        if not cell_to_genes:
            logger.warning(f"No markers found for {column_ctype} that meet the score cutoff criteria ({cutoff})")
            return None
        
        # Filter adata to only include cell types that have markers
        mask = self.adata.obs[column_ctype].isin(cell_to_genes.keys())

        # Get unique cell types in the filtered data to preserve the order
        if remove_ctype_no_marker:
            cell_types = self.adata[mask].obs[column_ctype].cat.categories.tolist()
        else:
            cell_types = self.adata.obs[column_ctype].cat.categories.tolist()
        # Filter cell types to only those that appear in the data
        cell_types = [ct for ct in cell_types if ct in self.adata.obs[column_ctype].unique()]
        
        # Create an ordered list of genes based on the cell type order
        ordered_genes = []
        for cell_type in cell_types:
            if cell_type in cell_to_genes:
                ordered_genes.extend(cell_to_genes[cell_type])
        
        # Now plot with genes in the same order as cell types, showing only cell types with markers
        total_genes = len(ordered_genes)
        
        # Log what type of scores we're using
        if using_dataset_analysis:
            logger.info(f"Plotting {total_genes} genes for {len(cell_types)} cell types using dataset analysis with {score_type}")
        else:
            method_used = self.method_run[column_ctype]
            logger.info(f"Plotting {total_genes} genes for {len(cell_types)} cell types using {method_used} model")
        
        # Create the plot with the ordered genes
        return sc.pl.matrixplot(
            self.adata if not remove_ctype_no_marker else self.adata[mask],
            var_names=ordered_genes,
            groupby=column_ctype,
            cmap=cmap,
            use_raw=use_raw,
            standard_scale=standard_scale,
            categories_order=cell_types,  # Ensure cell types are in the desired order
            save=save,
            show=show,
            **kwargs
            )
    
    
    def plot_corr_theoric_empiric(self, column_ctype, figsize=(10, 8), cell_types=None, min_genes=1, sort_by='correlation'):
        """
        Plot correlation between theoretical and empirical scores for each cell type
        
        Args:
            column_ctype: Column name in adata.obs for cell type annotations
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
        if column_ctype not in self.empirical_scores or column_ctype not in self.theoretical_scores:
            raise ValueError(f"Both theoretical and empirical scores must be available for {column_ctype}. Run with method='empiric'")
            
        if self.empirical_scores[column_ctype] is None or self.theoretical_scores[column_ctype] is None:
            raise ValueError(f"Both theoretical and empirical scores must be available for {column_ctype}. Run with method='empiric'")
        
        # Create a DataFrame for comparison
        comparison_data = []
        
        # Filter cell types if specified
        if cell_types is None:
            cell_types = list(self.adata.obs[column_ctype].unique())
        else:
            # Ensure all requested cell types exist
            for ct in cell_types:
                if ct not in self.adata.obs[column_ctype].unique():
                    logger.warning(f"Cell type '{ct}' not found in dataset for annotation {column_ctype}")
            cell_types = [ct for ct in cell_types if ct in self.adata.obs[column_ctype].unique()]
        
        # Collect data for all genes in filtered cell types
        for ctype in cell_types:
            # Get scores from both theoretical and empirical calculations
            if (ctype in self.theoretical_scores[column_ctype] and ctype in self.empirical_scores[column_ctype] and 
                not self.theoretical_scores[column_ctype][ctype].empty and not self.empirical_scores[column_ctype][ctype].empty):
                
                # Get common genes
                genes = set(self.theoretical_scores[column_ctype][ctype].index) & set(self.empirical_scores[column_ctype][ctype].index)
                
                for gene in genes:
                    # Get aggregated scores (mean across k values)
                    theo_score = self.theoretical_scores[column_ctype][ctype].loc[gene].mean()
                    emp_score = self.empirical_scores[column_ctype][ctype].loc[gene].mean()
                    
                    # Add to comparison data
                    comparison_data.append({
                        'gene': gene,
                        'cell_type': ctype,
                        'theoretical_score': theo_score,
                        'empirical_score': emp_score
                    })
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Check if we have any data
        if comparison_df.empty:
            logger.warning(f"No genes with both theoretical and empirical scores found for {column_ctype}")
            return None
        
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
        
        # Check if we have any correlations
        if not correlations:
            logger.warning(f"No cell types had enough genes (>={min_genes}) for correlation in {column_ctype}")
            return None
        
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
        ax.set_title(f'Correlation between Theoretical and Empirical Scores for {column_ctype}')
        
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
    
    def get_scores(self, column_ctype: str, cell_type: Optional[Union[str, List[str]]] = None, method: str = "empiric") -> Dict:
        """
        Get aggregated scores from the ScSherlock analysis
        
        Args:
            column_ctype (str): Column name in adata.obs for cell type annotations
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
        
        # Check if the annotation has been processed
        if column_ctype not in self.method_run:
            raise ValueError(f"No analysis exists for {column_ctype}. Run ScSherlock.run() with this annotation first")
            
        # Get the appropriate aggregated scores based on the method
        if method == "theoric":
            # Check if theoretical scores have been calculated
            if column_ctype not in self.aggregated_scores or self.aggregated_scores[column_ctype] is None:
                raise ValueError(f"Theoretical scores haven't been calculated for {column_ctype}. Run with method='theoric' first")
            agg_scores = self.aggregated_scores[column_ctype]
        else:  # method == "empiric"
            # Check if empirical scores have been calculated
            if column_ctype not in self.aggregated_empirical_scores or self.aggregated_empirical_scores[column_ctype] is None:
                raise ValueError(f"Empirical scores haven't been calculated for {column_ctype}. Run with method='empiric' first")
            agg_scores = self.aggregated_empirical_scores[column_ctype]
        
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
            cell_types_in_data = self.adata.obs[column_ctype].unique()
            for ct in cell_types_to_include:
                if ct not in cell_types_in_data:
                    raise ValueError(f"Cell type '{ct}' not found in the dataset for annotation {column_ctype}")
            
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

    def create_proportion_based_hierarchy(self, annotation_columns, min_proportion=0):
        """
        Create hierarchy relationships based on co-occurrence proportions.
        
        Args:
            annotation_columns (list): List of column names in adata.obs for hierarchy levels
            min_proportion (float): Minimum proportion threshold for including relationships
                
        Returns:
            dict: Hierarchical structure of cell types
        """
        # Initialize the hierarchy structure
        relationships = {}
        
        # Process the first level to create the base structure
        if len(annotation_columns) > 0:
            # Get unique values for the first level
            top_level = annotation_columns[0]
            top_values = self.adata.obs[top_level].astype('category').cat.categories.tolist()
            
            # Initialize with empty lists for each top-level value
            for val in top_values:
                relationships[val] = []
        
        # Process each level starting from the first
        for i in range(len(annotation_columns) - 1):
            parent_col = annotation_columns[i]
            child_col = annotation_columns[i+1]
            
            # Get unique values for parent and child levels
            parent_values = self.adata.obs[parent_col].unique()
            child_values = self.adata.obs[child_col].unique()
            
            # Store temporary relationships for this level
            level_relationships = {}
            
            # For each parent value, find related child values
            for parent_val in parent_values:
                # Get cells with this parent value
                parent_mask = self.adata.obs[parent_col] == parent_val
                parent_cells = sum(parent_mask)
                
                if parent_cells == 0:
                    continue
                
                # Initialize children array for this parent
                children_list = []
                
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
                        children_list.append({
                            'id': child_val,
                            'proportion': float(proportion)
                        })
                
                # Sort children by proportion
                children_list = sorted(children_list, key=lambda x: x['proportion'], reverse=True)
                
                # Store in level relationships
                if children_list:
                    level_relationships[parent_val] = children_list
            
            # Now, integrate these relationships into our hierarchy
            # For the first level, it's straightforward
            if i == 0:
                for parent_val, children in level_relationships.items():
                    relationships[parent_val] = children
            else:
                # For subsequent levels, we need to find the right place in the hierarchy
                # A recursive function to update deeper levels
                def add_children_to_level(data_structure, current_level, target_level, parent_value, children_to_add):
                    # If we're at the right level and found the right parent
                    if current_level == target_level and data_structure.get('id') == parent_value:
                        data_structure['children'] = children_to_add
                        return True
                    
                    # If we're not at the target level yet but have children, go deeper
                    if 'children' in data_structure:
                        for child in data_structure['children']:
                            if add_children_to_level(child, current_level + 1, target_level, parent_value, children_to_add):
                                return True
                    
                    return False
                
                # For each parent-child relationship at this level
                for parent_val, children in level_relationships.items():
                    # Find the parent in the existing hierarchy
                    found = False
                    
                    # Start searching from the top
                    if i == 1:  # Level 1 parents are at the top level
                        for top_val, top_children in relationships.items():
                            for child in top_children:
                                if child['id'] == parent_val:
                                    child['children'] = children
                                    found = True
                                    break
                            if found:
                                break
                    else:  # Level 2+ parents are deeper
                        for top_val, top_children in relationships.items():
                            for child in top_children:
                                if add_children_to_level(child, 1, i, parent_val, children):
                                    found = True
                                    break
                            if found:
                                break
                    
                    if not found:
                        logger.warning(f"Could not find parent {parent_val} in the existing hierarchy")
        
        return relationships

    def create_hierarchy_graph(self, annotation_columns, min_proportion=0.05, max_children=None, store_graph=True):
        """
        Create a hierarchy graph based on co-occurrence proportions.
        
        Args:
            annotation_columns (list): List of column names in adata.obs for hierarchy levels
            min_proportion (float): Minimum proportion threshold for including relationships
            max_children (int, optional): Maximum number of children to include for each node
            store_graph (bool): Whether to store the graph in self.G (default: True)
            
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
            
            # Store annotation information for first level nodes
            G.nodes[parent]['annotation'] = annotation_columns[0]
            G.nodes[parent]['level'] = 0
            
            # Helper function to recursively add child nodes
            def add_children(parent_node, children, level):
                # Check if children list is empty
                if not children:
                    return
                    
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
                    
                    # Store attributes, including annotation information
                    G.nodes[child_node]['label'] = child_id
                    G.nodes[child_node]['proportion'] = proportion
                    
                    # Check if we're still within the annotation columns range
                    if level + 1 < len(annotation_columns):
                        G.nodes[child_node]['annotation'] = annotation_columns[level + 1]
                    else:
                        # If beyond the provided annotation columns, use the last one
                        G.nodes[child_node]['annotation'] = annotation_columns[-1]
                        
                    G.nodes[child_node]['level'] = level + 1
                    G.edges[parent_node, child_node]['weight'] = proportion
                    
                    # Recursively add grandchildren if any (no need to check level here)
                    if 'children' in child_item:
                        add_children(child_node, child_item['children'], level + 1)
            
            # Add children for this parent, starting at level 0
            add_children(parent, relationships[parent], 0)
        
        # Store graph in instance if requested
        if store_graph:
            self.G = G
            
        return G
    
    def verify_hierarchy_depth(self, annotation_columns):
        """
        Utility method to verify the hierarchy depth matches the number of annotation columns
        
        Args:
            annotation_columns (list): List of column names in adata.obs for hierarchy levels
            
        Returns:
            dict: Information about the hierarchy depth
        """
        if not hasattr(self, 'G') or self.G is None:
            logger.error("No hierarchical graph exists. Call create_hierarchy_graph first.")
            return None
        
        # Get all nodes except the root
        nodes = [n for n in self.G.nodes() if n != 'root']
        
        # Calculate depth for each node
        depths = {}
        for node in nodes:
            # The depth is the number of underscores plus 1
            # (root_level1_level2_level3 has depth 3)
            depth = node.count('_')
            if depth not in depths:
                depths[depth] = []
            depths[depth].append(node)
        
        # Get the maximum depth
        max_depth = max(depths.keys()) if depths else 0
        
        # Check if the maximum depth matches the number of annotation levels
        expected_max_depth = len(annotation_columns)
        
        # Print detailed results
        logger.info("\n=== HIERARCHY DEPTH VERIFICATION ===")
        logger.info(f"Number of annotation columns: {len(annotation_columns)}")
        logger.info(f"Maximum depth found in hierarchy: {max_depth}")
        
        if max_depth < expected_max_depth:
            logger.warning(f"Hierarchy depth ({max_depth}) is less than expected ({expected_max_depth})")
            logger.warning("Some annotation levels may not be properly represented in the hierarchy")
        else:
            logger.info("Hierarchy depth matches or exceeds the number of annotation levels ✓")
        
        # Count nodes at each depth
        for depth in sorted(depths.keys()):
            count = len(depths[depth])
            sample_nodes = depths[depth][:3]  # Show up to 3 sample nodes
            if depth < len(annotation_columns):
                annotation = annotation_columns[depth]
            else:
                annotation = "beyond defined annotations"
                
            logger.info(f"Depth {depth} ({annotation}): {count} nodes")
            logger.info(f"  Sample nodes: {', '.join(sample_nodes)}")
            if len(depths[depth]) > 3:
                logger.info(f"  ... and {len(depths[depth]) - 3} more")
        
        # Return depth information
        return {
            "expected_depth": expected_max_depth,
            "actual_depth": max_depth,
            "depth_counts": {d: len(nodes) for d, nodes in depths.items()},
            "is_complete": max_depth >= expected_max_depth
        }


    # 2. Add a debug method to check the hierarchy structure
    def debug_hierarchy_structure(self, relationships, max_depth=5):
        """
        Utility method to debug the hierarchy structure created by create_proportion_based_hierarchy
        
        Args:
            relationships: The hierarchy dictionary returned by create_proportion_based_hierarchy
            max_depth: Maximum depth to print (to avoid excessive output for large hierarchies)
        """
        logger.info("\n=== HIERARCHY STRUCTURE DEBUGGING ===")
        logger.info(f"Top level nodes: {len(relationships)}")
        
        def print_node(node_id, children, depth=0, path=""):
            if depth > max_depth:
                logger.info(f"{'  ' * depth}... (max depth reached)")
                return
                
            current_path = f"{path}/{node_id}" if path else node_id
            children_count = len(children) if children else 0
            logger.info(f"{'  ' * depth}→ {node_id} ({children_count} children) - Path: {current_path}")
            
            if children:
                for child in children[:3]:  # Print only first 3 children to avoid clutter
                    child_id = child['id']
                    proportion = child['proportion']
                    logger.info(f"{'  ' * (depth+1)}· {child_id} (proportion: {proportion:.2f})")
                    
                    if 'children' in child:
                        print_node(child_id, child['children'], depth + 2, current_path)
                
                if len(children) > 3:
                    logger.info(f"{'  ' * (depth+1)}... ({len(children) - 3} more children)")
        
        for parent, children in list(relationships.items())[:5]:  # Show only first 5 top-level nodes
            print_node(parent, children)
            
        if len(relationships) > 5:
            logger.info(f"... ({len(relationships) - 5} more top-level nodes)")
        
        logger.info("\n=== HIERARCHY DEPTH ANALYSIS ===")
        
        # Analyze the maximum depth of each branch
        max_depths = {}
        
        def calculate_depth(node_id, children, current_depth=1):
            max_child_depth = current_depth
            
            if children:
                for child in children:
                    child_id = child['id']
                    if 'children' in child and child['children']:
                        child_depth = calculate_depth(child_id, child['children'], current_depth + 1)
                        max_child_depth = max(max_child_depth, child_depth)
            
            return max_child_depth
        
        for parent, children in relationships.items():
            max_depths[parent] = calculate_depth(parent, children)
        
        # Report the maximum depths
        depths = list(max_depths.values())
        if depths:
            logger.info(f"Maximum depth found: {max(depths)}")
            logger.info(f"Average depth: {sum(depths)/len(depths):.1f}")
            logger.info(f"Distribution of maximum depths:")
            for depth in range(1, max(depths) + 1):
                count = sum(1 for d in depths if d == depth)
                if count > 0:
                    logger.info(f"  Depth {depth}: {count} branches ({count/len(depths)*100:.1f}%)")
        else:
            logger.info("No data available for depth analysis")

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

    def visualize_hierarchy_marker(self, column_ctype=None, cutoff=0.5, output_file=None, figsize=(20, 12), 
                              show_score=True, dataset_column=None, score_type='weighted_score'):
        """
        Visualize the cell hierarchy with marker gene information.
        
        This method creates a hierarchy visualization where:
        - Cell types with no entry in sorted_empirical_table are shown in grey
        - Cell types with no marker genes passing the cutoff are shown in red
        - Cell types with marker genes are colored according to their score (from 0 to 1)
        
        Args:
            column_ctype (str, optional): Column name in adata.obs for cell type annotations.
                                        If None, uses markers from the annotation level the cell type belongs to.
            cutoff (float): Score cutoff threshold for marker genes
            output_file (str, optional): Path to save the figure
            figsize (tuple): Figure size (width, height)
            show_score (bool, optional): Whether to show score values in node labels. Default is True.
            dataset_column (str, optional): If provided, use markers from dataset analysis using this column
            score_type (str, optional): Which score to use from dataset analysis ('avg_score' or 'weighted_score')
                
        Returns:
            matplotlib.figure.Figure: Figure object with the hierarchy visualization
        """
        # Check if dataset analysis should be used
        using_dataset_analysis = False
        dataset_keys = {}  # Will store {annotation: dataset_key} for all annotations with dataset analysis
        
        if dataset_column is not None:
            # If column_ctype is specified, check just that one
            if column_ctype is not None:
                dataset_key = f"{column_ctype}_by_{dataset_column}"
                if dataset_key in self.__dict__ and 'combined_tables' in self.__dict__[dataset_key]:
                    using_dataset_analysis = True
                    dataset_keys[column_ctype] = dataset_key
                    logger.info(f"Using dataset analysis results for {column_ctype} with {score_type}")
            else:
                # If column_ctype is not specified, look for all annotations with dataset analysis
                for attr in self.__dict__:
                    if attr.endswith(f"_by_{dataset_column}") and 'combined_tables' in self.__dict__[attr]:
                        annotation = attr.split(f"_by_{dataset_column}")[0]
                        dataset_keys[annotation] = attr
                        using_dataset_analysis = True
                        logger.info(f"Found dataset analysis for {annotation} with {score_type}")
        
        # Validate score_type
        if using_dataset_analysis and score_type not in ['avg_score', 'weighted_score']:
            logger.warning(f"Invalid score_type '{score_type}'. Defaulting to 'weighted_score'")
            score_type = 'weighted_score'
        
        # Check if any marker analysis has been run
        if not using_dataset_analysis and not self.sorted_empirical_table and not self.sorted_table:
            raise ValueError("No marker analysis results available. Run ScSherlock.run() first")
        
        # Find available annotations with marker data
        available_annotations = {}  # {annotation: {"empiric": table} or {"theoric": table}}
        
        if using_dataset_analysis:
            # Use dataset analysis for annotations with it available
            for annotation, dataset_key in dataset_keys.items():
                available_annotations[annotation] = {
                    "method": "dataset", 
                    "table": self.__dict__[dataset_key]['combined_tables']
                }
                logger.info(f"Using dataset analysis results for {annotation}")
        
        # For annotations without dataset analysis, check standard analysis results
        for ann in self.sorted_empirical_table.keys():
            if ann not in available_annotations and self.sorted_empirical_table[ann] is not None:
                available_annotations[ann] = {"method": "empiric", "table": self.sorted_empirical_table[ann]}
                logger.info(f"Found empirical marker scores for {ann}")
        
        # Then check theoretical scores for any missing annotations
        for ann in self.sorted_table.keys():
            if ann not in available_annotations and self.sorted_table[ann] is not None:
                available_annotations[ann] = {"method": "theoric", "table": self.sorted_table[ann]}
                logger.info(f"Found theoretical marker scores for {ann}")
        
        if not available_annotations:
            raise ValueError("No marker analysis results available. Run ScSherlock.run() first")
        
        # If column_ctype is specified, verify it has marker data
        if column_ctype is not None and column_ctype not in available_annotations:
            raise ValueError(f"No marker analysis results available for {column_ctype}. Run ScSherlock.run() first")
        
        # Check if the hierarchy graph exists
        if not hasattr(self, 'G') or self.G is None:
            raise ValueError("No hierarchical graph present in ScSherlock object. Call create_hierarchy_graph first.")
            
        G = self.G
        
        # Map nodes to their annotation level based on node attributes or structure
        node_annotations = {}  # {node_id: annotation_level}
        
        # Check if nodes have annotation information stored
        annotation_attr_found = False
        for node in G.nodes():
            if node != 'root' and 'annotation' in G.nodes[node]:
                annotation_attr_found = True
                node_annotations[node] = G.nodes[node]['annotation']
        
        # If no annotation attributes found, try to infer from node structure and available annotations
        if not annotation_attr_found:
            logger.info("No annotation attributes found in graph nodes, inferring from structure")
            
            # Try to infer annotations from node structure
            for node in G.nodes():
                if node == 'root':
                    continue
                    
                # Extract cell type name
                if 'label' in G.nodes[node]:
                    cell_type = G.nodes[node]['label']
                else:
                    parts = node.split('_')
                    cell_type = parts[-1]
                
                # Check which annotation this cell type belongs to
                for ann, data in available_annotations.items():
                    if cell_type in data['table']:
                        node_annotations[node] = ann
                        break
        
        # Create a mapping of cell types to their markers from appropriate annotation
        cell_markers = {}  # {node_id: (annotation, marker, score)}
        
        # If a specific column_ctype is provided, only use markers from that annotation
        if column_ctype is not None:
            ann_data = available_annotations[column_ctype]
            method = ann_data["method"]
            sorted_table = ann_data["table"]
            
            logger.info(f"Using {method} marker scores for {column_ctype}")
            
            # Process each cell type in this annotation
            if method == "dataset":
                # Special handling for dataset analysis results
                for ctype, table in sorted_table.items():
                    if table.empty:
                        continue
                        
                    # Get markers passing the cutoff using the specified score_type
                    markers_passing = table[table[score_type] >= cutoff]
                    
                    if not markers_passing.empty:
                        # Get the top marker and its score
                        top_marker = markers_passing.index[0]
                        top_score = markers_passing.loc[top_marker, score_type]
                        
                        # Store this marker for nodes with this cell type
                        for node, cell_type in [(n, G.nodes[n].get('label', n.split('_')[-1])) 
                                        for n in G.nodes() if n != 'root']:
                            if cell_type == ctype:
                                cell_markers[node] = (column_ctype, top_marker, top_score)
            else:
                # Original handling for standard analysis results
                for ctype, table in sorted_table.items():
                    if not isinstance(table, pd.DataFrame) or table.empty:
                        continue
                        
                    # Get markers passing the cutoff
                    markers_passing = table[table['aggregated'] >= cutoff]
                    
                    if not markers_passing.empty:
                        # Get the top marker and its score
                        top_marker = markers_passing.index[0]
                        top_score = markers_passing.loc[top_marker, 'aggregated']
                        
                        # Store this marker for nodes with this cell type
                        for node, cell_type in [(n, G.nodes[n].get('label', n.split('_')[-1])) 
                                        for n in G.nodes() if n != 'root']:
                            if cell_type == ctype:
                                cell_markers[node] = (column_ctype, top_marker, top_score)
        else:
            # No specific annotation provided, use markers from the annotation the cell type belongs to
            for node in G.nodes():
                if node == 'root':
                    continue
                    
                # Extract cell type name
                if 'label' in G.nodes[node]:
                    cell_type = G.nodes[node]['label']
                else:
                    parts = node.split('_')
                    cell_type = parts[-1]
                
                # Determine annotation for this node
                node_ann = node_annotations.get(node)
                
                if node_ann and node_ann in available_annotations:
                    # Get marker data for this annotation
                    ann_data = available_annotations[node_ann]
                    method = ann_data["method"]
                    sorted_table = ann_data["table"]
                    
                    # Check if this cell type has markers in its annotation
                    if method == "dataset":
                        # Dataset analysis handling
                        if cell_type in sorted_table and not sorted_table[cell_type].empty:
                            # Get markers passing the cutoff using the specified score_type
                            markers_passing = sorted_table[cell_type][sorted_table[cell_type][score_type] >= cutoff]
                            
                            if not markers_passing.empty:
                                # Get the top marker and its score
                                top_marker = markers_passing.index[0]
                                top_score = markers_passing.loc[top_marker, score_type]
                                
                                # Store this marker for this node
                                cell_markers[node] = (node_ann, top_marker, top_score)
                    else:
                        # Standard analysis handling
                        if cell_type in sorted_table and isinstance(sorted_table[cell_type], pd.DataFrame) and not sorted_table[cell_type].empty:
                            # Get markers passing the cutoff
                            markers_passing = sorted_table[cell_type][sorted_table[cell_type]['aggregated'] >= cutoff]
                            
                            if not markers_passing.empty:
                                # Get the top marker and its score
                                top_marker = markers_passing.index[0]
                                top_score = markers_passing.loc[top_marker, 'aggregated']
                                
                                # Store this marker for this node
                                cell_markers[node] = (node_ann, top_marker, top_score)
    
        plt.figure(figsize=figsize)
        
        try:
            # Use dot layout for hierarchical trees
            pos = graphviz_layout(G, prog='twopi')
            
            # Get node depths for coloring
            node_depth = {}
            for node in G.nodes():
                node_depth[node] = 0 if node == 'root' else node.count('_') + 1
            
            max_depth = max(node_depth.values())
            
            # Import colormap for score-based coloring
            from matplotlib.colors import LinearSegmentedColormap
            
            # Create a custom colormap from blue (low score) to red (high score)
            score_cmap = plt.cm.get_cmap('viridis')
            
            # Prepare node colors based on marker gene status
            node_colors = []
            node_state = []  # Store state for legend (0: root, 1: with marker, 2: no marker computed, 3: no marker passing cutoff)
            
            for node in G.nodes():
                if node == 'root':
                    # Root node is always blue
                    node_colors.append('royalblue')
                    node_state.append(0)
                    continue
                    
                # Check if node has a marker
                if node in cell_markers:
                    # Has marker passing cutoff - use score for coloring
                    _, _, score = cell_markers[node]
                    node_colors.append(score_cmap(score))
                    node_state.append(1)
                else:
                    # Extract cell type name for further checking
                    if 'label' in G.nodes[node]:
                        cell_type = G.nodes[node]['label']
                    else:
                        parts = node.split('_')
                        cell_type = parts[-1]
                    
                    # Determine if the cell type has been computed but no markers pass threshold
                    has_computed = False
                    node_ann = node_annotations.get(node)
                    
                    if node_ann and node_ann in available_annotations:
                        ann_data = available_annotations[node_ann]
                        method = ann_data["method"]
                        sorted_table = ann_data["table"]
                        
                        if method == "dataset":
                            # Dataset analysis handling
                            if cell_type in sorted_table and not sorted_table[cell_type].empty:
                                has_computed = True
                        else:
                            # Standard analysis handling
                            if cell_type in sorted_table and isinstance(sorted_table[cell_type], pd.DataFrame) and not sorted_table[cell_type].empty:
                                has_computed = True
                    
                    if has_computed:
                        # Computed but no markers passing cutoff - red
                        node_colors.append('red')
                        node_state.append(3)
                    else:
                        # No computation or no entry - grey
                        node_colors.append('grey')
                        node_state.append(2)
            
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
                
                # Check if node has a marker
                if node in cell_markers:
                    # Get marker information
                    ann, marker, score = cell_markers[node]
                    if show_score:
                        if using_dataset_analysis:
                            # For dataset analysis, show different info
                            node_ann = node_annotations.get(node)
                            if node_ann in available_annotations and available_annotations[node_ann]["method"] == "dataset":
                                # Get the number of datasets for this marker
                                sorted_table = available_annotations[node_ann]["table"]
                                if cell_type in sorted_table and marker in sorted_table[cell_type].index:
                                    num_datasets = int(sorted_table[cell_type].loc[marker, 'num_datasets'])
                                    labels[node] = f"{cell_type}\n[{marker}]\n{score:.2f} ({num_datasets}d)"
                                else:
                                    labels[node] = f"{cell_type}\n[{marker}]\n{score:.2f}"
                            else:
                                labels[node] = f"{cell_type}\n[{marker}]\n{score:.2f}"
                        else:
                            labels[node] = f"{cell_type}\n[{marker}]\n{score:.2f}"
                    else:
                        labels[node] = f"{cell_type}\n[{marker}]"
                else:
                    # Determine if computed but no markers pass threshold
                    has_computed = False
                    node_ann = node_annotations.get(node)
                    
                    if node_ann and node_ann in available_annotations:
                        ann_data = available_annotations[node_ann]
                        method = ann_data["method"]
                        sorted_table = ann_data["table"]
                        
                        if method == "dataset":
                            # Dataset analysis handling
                            if cell_type in sorted_table and not sorted_table[cell_type].empty:
                                has_computed = True
                        else:
                            # Standard analysis handling
                            if cell_type in sorted_table and isinstance(sorted_table[cell_type], pd.DataFrame) and not sorted_table[cell_type].empty:
                                has_computed = True
                    
                    if has_computed:
                        labels[node] = f"{cell_type}\n[no marker > {cutoff}]"
                    else:
                        labels[node] = f"{cell_type}\n[no data]"
            
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
            
            # Add a colorbar for the score scale
            sm = plt.cm.ScalarMappable(cmap=score_cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=plt.gca(), orientation='horizontal', pad=0.05, fraction=0.05, shrink=0.8)
            cbar.set_label(f'Marker Gene Score ({score_type if using_dataset_analysis else "standard"})')
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', label='With marker gene (colored by score)',
                        markerfacecolor=score_cmap(0.7), markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label='No marker data computed',
                        markerfacecolor='grey', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label='No marker passing cutoff',
                        markerfacecolor='red', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label='Root',
                        markerfacecolor='royalblue', markersize=10)
            ]
            plt.legend(handles=legend_elements, loc='lower right', fontsize=9)
            
            # Set title based on whether a specific annotation was used
            if using_dataset_analysis:
                plt.title(f'Cell Type Hierarchy with Dataset Markers for {column_ctype}\n({score_type}, cutoff={cutoff})', fontsize=15)
            elif column_ctype is not None:
                plt.title(f'Cell Type Hierarchy with Marker Genes for {column_ctype} (cutoff={cutoff})', fontsize=15)
            else:
                annotations_used = ", ".join(available_annotations.keys())
                plt.title(f'Cell Type Hierarchy with Annotation-Specific Marker Genes\n(Annotations: {annotations_used}, cutoff={cutoff})', fontsize=15)
            
            plt.axis('off')
            
            # Save if output file is specified
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
            
            fig = plt.gcf()
            
        except Exception as e:
            logger.error(f"Error generating hierarchy visualization: {e}")
            raise

    def _perform_bootstrap_validation(self, column_ctype: str):
        """
        Perform bootstrap validation for marker genes by subsetting data and recomputing empirical scores.
        
        Steps:
        1. Get all potential markers identified (not applying cutoff yet)
        2. Subset the AnnData object to include only those genes
        3. Split the subset into two balanced datasets based on annotations
        4. Recompute empirical scores separately on each subset
        5. Record the scores from both subsets and set bootstrap_pass if the gene is a marker in both
        subsets regardless of the cutoff
        
        Args:
            column_ctype: Column name in adata.obs for cell type annotations
            
        Returns:
            dict: Dictionary of bootstrap validation results
        """
        from sklearn.model_selection import StratifiedGroupKFold
        import copy
        
        logger.info("Starting bootstrap validation of markers...")
        
        # Step 1: Get all markers identified in the empirical score calculation (without cutoff)
        # First collect all genes from empirical_scores regardless of their score
        all_marker_genes = set()
        for ctype, scores_df in self.empirical_scores[column_ctype].items():
            if not scores_df.empty:
                all_marker_genes.update(scores_df.index.tolist())
        
        if not all_marker_genes:
            logger.warning(f"No marker genes found for {column_ctype}")
            return {}
        
        logger.info(f"Found {len(all_marker_genes)} potential marker genes for bootstrap validation")
        
        # Step 2: Subset the AnnData object to include only those genes
        adata_subset = self.adata[:, list(all_marker_genes)].copy()
        
        # Step 3: Split this subset into two balanced datasets
        fold_generator = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=self.config.random_seed)
        
        # Prepare data for stratified split
        X = np.zeros(adata_subset.shape[0])  # Dummy X data
        y = adata_subset.obs[column_ctype].astype('category').cat.codes.values  # Cell type codes
        groups = adata_subset.obs[self.column_patient].values  # Patient IDs for grouping
        
        # Get indices for both folds
        for fold1_idx, fold2_idx in fold_generator.split(X, y, groups):
            subset1_idx = fold1_idx
            subset2_idx = fold2_idx
            break  # We only need one split
        
        # Create two AnnData subsets
        adata_subset1 = adata_subset[subset1_idx].copy()
        adata_subset2 = adata_subset[subset2_idx].copy()
        
        logger.info(f"Created two balanced subsets: {adata_subset1.shape[0]} and {adata_subset2.shape[0]} cells")
        
        # Step 4: Recompute empirical scores separately on each subset
        # Create config with fewer simulations for faster bootstrap validation
        config_copy = copy.deepcopy(self.config)
        config_copy.n_simulations = max(100, self.config.n_simulations // 5)  # Fewer simulations
        
        # Initialize ScSherlock objects for both subsets
        logger.info("Initializing ScSherlock for subset 1...")
        sc_subset1 = ScSherlock(adata_subset1, self.original_patient_column, config_copy)
        
        logger.info("Initializing ScSherlock for subset 2...")
        sc_subset2 = ScSherlock(adata_subset2, self.original_patient_column, config_copy)
        
        # Run empirical score calculation on both subsets
        logger.info("Computing empirical scores on subset 1...")
        sc_subset1.run(column_ctype, method="empiric")
        
        logger.info("Computing empirical scores on subset 2...")
        sc_subset2.run(column_ctype, method="empiric")
        
        # Step 5: Record scores from both subsets and set bootstrap_pass if gene is a marker in both
        bootstrap_results = {}
        
        # Check each cell type
        for ctype, scores_df in self.sorted_empirical_table[column_ctype].items():
            if scores_df.empty:
                continue
            
            # Initialize bootstrap columns if they don't exist
            if 'bootstrap_pass' not in scores_df.columns:
                self.sorted_empirical_table[column_ctype][ctype]['bootstrap_pass'] = False
            if 'bootstrap_1_score' not in scores_df.columns:
                self.sorted_empirical_table[column_ctype][ctype]['bootstrap_1_score'] = np.nan
            if 'bootstrap_2_score' not in scores_df.columns:
                self.sorted_empirical_table[column_ctype][ctype]['bootstrap_2_score'] = np.nan
            
            # Initialize results for this cell type
            bootstrap_results[ctype] = []
            
            # Check each marker gene (consider all genes from sorted_empirical_table)
            for gene in scores_df.index:
                # Check if this gene is in both subset results
                in_subset1 = (ctype in sc_subset1.sorted_empirical_table[column_ctype] and 
                            not sc_subset1.sorted_empirical_table[column_ctype][ctype].empty and
                            gene in sc_subset1.sorted_empirical_table[column_ctype][ctype].index)
                
                in_subset2 = (ctype in sc_subset2.sorted_empirical_table[column_ctype] and 
                            not sc_subset2.sorted_empirical_table[column_ctype][ctype].empty and
                            gene in sc_subset2.sorted_empirical_table[column_ctype][ctype].index)
                
                # Get scores if available
                score1 = (sc_subset1.sorted_empirical_table[column_ctype][ctype].loc[gene, 'aggregated'] 
                        if in_subset1 else np.nan)
                score2 = (sc_subset2.sorted_empirical_table[column_ctype][ctype].loc[gene, 'aggregated'] 
                        if in_subset2 else np.nan)
                
                # A marker passes if it's identified as a marker in both subsets regardless of cutoff
                bootstrap_pass = in_subset1 and in_subset2
                
                # Update the bootstrap columns
                self.sorted_empirical_table[column_ctype][ctype].loc[gene, 'bootstrap_pass'] = bootstrap_pass
                self.sorted_empirical_table[column_ctype][ctype].loc[gene, 'bootstrap_1_score'] = score1
                self.sorted_empirical_table[column_ctype][ctype].loc[gene, 'bootstrap_2_score'] = score2
                
                # Record result with scores
                bootstrap_results[ctype].append({
                    'gene': gene,
                    'bootstrap_pass': bootstrap_pass,
                    'bootstrap_1_score': score1,
                    'bootstrap_2_score': score2,
                    'original_score': scores_df.loc[gene, 'aggregated']
                })
                
                # Detailed log for each gene
                msg = f"Marker {gene} for {ctype}: "
                if bootstrap_pass:
                    msg += f"PASSED bootstrap validation"
                else:
                    reason = []
                    if not in_subset1:
                        reason.append("not a marker in subset 1")
                    if not in_subset2:
                        reason.append("not a marker in subset 2")
                    msg += f"FAILED bootstrap validation ({', '.join(reason)})"
                
                msg += f" (original: {scores_df.loc[gene, 'aggregated']:.3f}, subset1: {score1 if not np.isnan(score1) else 'NA'}, subset2: {score2 if not np.isnan(score2) else 'NA'})"
                logger.info(msg)
        
        # Log summary
        for ctype, results in bootstrap_results.items():
            total = len(results)
            passed = sum(1 for r in results if r['bootstrap_pass'])
            
            # Calculate mean scores
            original_scores = [r['original_score'] for r in results]
            subset1_scores = [r['bootstrap_1_score'] for r in results if not np.isnan(r['bootstrap_1_score'])]
            subset2_scores = [r['bootstrap_2_score'] for r in results if not np.isnan(r['bootstrap_2_score'])]
            
            logger.info(f"Cell type {ctype}:")
            logger.info(f"  {passed}/{total} markers passed bootstrap validation ({passed/total*100:.1f}%)")
            logger.info(f"  Mean scores - Original: {np.mean(original_scores):.3f}, Subset1: {np.mean(subset1_scores):.3f}, Subset2: {np.mean(subset2_scores):.3f}")

    def run_dataset(self, column_ctype: str, column_dataset: str, min_shared: int = 1, method: str = "empiric", bootstrap: bool = False) -> Dict[str, Dict[str, List[str]]]:
        """
        Run ScSherlock on each dataset separately and identify shared marker genes across datasets.
        
        This method:
        1. Splits the AnnData object by the dataset column
        2. Runs ScSherlock on each dataset separately
        3. Identifies markers that are shared across multiple datasets
        4. Stores markers with dataset-specific scores in the ScSherlock object
        
        Args:
            column_ctype: Column name in adata.obs for cell type annotations
            column_dataset: Column name in adata.obs for dataset annotations
            min_shared: Minimum number of datasets that must share a marker (default: 1)
            method: Method to use, either "theoric" or "empiric"
            bootstrap: Whether to perform bootstrap validation of markers
                
        Returns:
            Dict[str, Dict[str, List[str]]]: Dictionary mapping cell types to datasets and their shared markers
        """
        if method not in ["theoric", "empiric"]:
            raise ValueError('Method must be either "theoric" or "empiric"')
        
        # Validate column names
        if column_ctype not in self.adata.obs.columns:
            raise ValueError(f"Cell type column '{column_ctype}' not found in adata.obs")
        
        if column_dataset not in self.adata.obs.columns:
            raise ValueError(f"Dataset column '{column_dataset}' not found in adata.obs")
            
        # Get all datasets
        datasets = self.adata.obs[column_dataset].unique()
        logger.info(f"Found {len(datasets)} datasets: {', '.join(datasets)}")
        
        # Get all cell types
        cell_types = self.adata.obs[column_ctype].unique()
        logger.info(f"Found {len(cell_types)} cell types in the combined data")
        
        # Dictionary to store results for each dataset
        dataset_markers = {}
        
        # Dictionary to track marker scores per dataset
        dataset_scores = {}
        
        # Dictionary to track all potential markers (not just top ones)
        all_potential_markers = {ctype: set() for ctype in cell_types}
        
        # Dictionary to track the full results for each dataset
        dataset_results = {}
        
        # Initialize dictionaries to store results, similar to the run method
        dataset_key = f"{column_ctype}_by_{column_dataset}"
        if dataset_key not in self.__dict__:
            self.__dict__[dataset_key] = {}
        
        # Create subdictionaries for storing results
        self.__dict__[dataset_key]['datasets'] = datasets
        self.__dict__[dataset_key]['dataset_markers'] = {}
        self.__dict__[dataset_key]['dataset_scores'] = {}
        self.__dict__[dataset_key]['shared_markers'] = {}
        self.__dict__[dataset_key]['marker_counts'] = {}
        self.__dict__[dataset_key]['sorted_tables'] = {}
        self.__dict__[dataset_key]['method'] = method
        self.__dict__[dataset_key]['all_potential_markers'] = all_potential_markers
        
        # Dictionary to track marker counts across datasets
        marker_counts = {ctype: {} for ctype in cell_types}
        
        # Run ScSherlock on each dataset
        for dataset in datasets:
            logger.info(f"Processing dataset: {dataset}")
            
            # Subset the AnnData object for this dataset
            adata_subset = self.adata[self.adata.obs[column_dataset] == dataset].copy()
            
            # Check if dataset has enough cells
            if adata_subset.shape[0] < self.config.min_cells:
                logger.warning(f"Dataset {dataset} has only {adata_subset.shape[0]} cells, which is below the minimum threshold of {self.config.min_cells}. Skipping.")
                continue
            
            # Check which cell types are present in this dataset
            dataset_cell_types = adata_subset.obs[column_ctype].unique()
            logger.info(f"Dataset {dataset} has {len(dataset_cell_types)} cell types")
            
            # Skip if no cell types or too few cells
            if len(dataset_cell_types) == 0:
                logger.warning(f"Dataset {dataset} has no cell types. Skipping.")
                continue
            
            # Create a new ScSherlock instance for this dataset
            sc_subset = ScSherlock(adata_subset, self.column_patient, self.config)
            
            # Run the pipeline
            try:
                # Run ScSherlock on this dataset
                top_markers = sc_subset.run(column_ctype, method=method, bootstrap=bootstrap)
                
                # Store top markers
                dataset_markers[dataset] = top_markers
                self.__dict__[dataset_key]['dataset_markers'][dataset] = top_markers
                
                # Store full results for the dataset
                if method == "empiric":
                    dataset_results[dataset] = sc_subset.sorted_empirical_table[column_ctype]
                    self.__dict__[dataset_key]['sorted_tables'][dataset] = sc_subset.sorted_empirical_table[column_ctype]
                else:
                    dataset_results[dataset] = sc_subset.sorted_table[column_ctype]
                    self.__dict__[dataset_key]['sorted_tables'][dataset] = sc_subset.sorted_table[column_ctype]
                    
                # Collect all scored markers, not just top ones
                dataset_scores[dataset] = {}
                
                for ctype, table in dataset_results[dataset].items():
                    if isinstance(table, pd.DataFrame) and not table.empty:
                        # Store scores for all genes in this cell type
                        dataset_scores[dataset][ctype] = table['aggregated'].to_dict()
                        
                        # Collect all potential markers without any score filtering
                        all_potential_markers[ctype].update(table.index)
                        
                        # For the top marker, update marker counts
                        if ctype in top_markers:
                            marker = top_markers[ctype]
                            if ctype not in marker_counts:
                                marker_counts[ctype] = {}
                            
                            if marker not in marker_counts[ctype]:
                                marker_counts[ctype][marker] = []
                            
                            marker_counts[ctype][marker].append(dataset)
                
                # Store dataset scores
                self.__dict__[dataset_key]['dataset_scores'][dataset] = dataset_scores[dataset]
                    
                logger.info(f"Successfully processed dataset {dataset}. Found markers for {len(top_markers)} cell types.")
            except Exception as e:
                logger.error(f"Error processing dataset {dataset}: {e}")
                continue
        
        # Identify shared markers across datasets - without applying score cutoffs
        shared_markers = {ctype: {} for ctype in cell_types}
        
        for ctype in cell_types:
            # Get all potential markers for this cell type
            potential_markers = all_potential_markers[ctype]
            
            # For each potential marker, count datasets where it appears with any score
            for marker in potential_markers:
                marker_datasets = []
                
                for dataset in datasets:
                    if (dataset in dataset_scores and 
                        ctype in dataset_scores[dataset] and 
                        marker in dataset_scores[dataset][ctype]):
                        marker_datasets.append(dataset)
                
                # Include marker if it appears in at least min_shared datasets
                if len(marker_datasets) >= min_shared:
                    shared_markers[ctype][marker] = marker_datasets
        
        # Store marker counts and shared markers
        self.__dict__[dataset_key]['marker_counts'] = marker_counts
        self.__dict__[dataset_key]['shared_markers'] = shared_markers
        
        # Create combined tables with dataset-specific scores - without score filtering
        combined_tables = {}
        
        for ctype in cell_types:
            if ctype not in shared_markers or not shared_markers[ctype]:
                continue
                
            # Get all shared markers for this cell type
            shared_markers_list = list(shared_markers[ctype].keys())
            
            # Create a dataframe with all the shared markers
            combined_df = pd.DataFrame(index=shared_markers_list)
            
            # Initialize dataset score columns
            for dataset in datasets:
                column_name = f"score_{dataset}"
                combined_df[column_name] = np.nan
            
            # Add scores from each dataset
            for marker in shared_markers_list:
                for dataset in datasets:
                    if (dataset in dataset_scores and 
                        ctype in dataset_scores[dataset] and 
                        marker in dataset_scores[dataset][ctype]):
                        combined_df.loc[marker, f"score_{dataset}"] = dataset_scores[dataset][ctype][marker]
            
            # Calculate score columns
            score_columns = [col for col in combined_df.columns if col.startswith('score_')]
            
            # Calculate average score
            combined_df['avg_score'] = combined_df[score_columns].mean(axis=1)
            
            # Calculate actual number of datasets by counting non-NaN values
            combined_df['num_datasets'] = combined_df[score_columns].count(axis=1)
            
            # Create datasets list string from actual present datasets
            combined_df['datasets'] = combined_df.apply(
                lambda row: ', '.join([
                    col.replace('score_', '') 
                    for col in score_columns 
                    if not pd.isna(row[col])
                ]), 
                axis=1
            )
            # Calculate weighted score that considers both average score and dataset coverage
            combined_df['weighted_score'] = combined_df['avg_score'] * (combined_df['num_datasets'] / len(datasets))

            # Sort by weighted score instead of just average score and number of datasets
            combined_df = combined_df.sort_values(['weighted_score'], ascending=[False])
            # Sort by number of datasets and then by average score
            #combined_df = combined_df.sort_values(['num_datasets', 'avg_score'], ascending=[False, False])
            
            # Store the combined table
            combined_tables[ctype] = combined_df
        
        # Store the combined tables
        self.__dict__[dataset_key]['combined_tables'] = combined_tables
        
        # Log summary of shared markers
        total_shared = sum(len(markers) for markers in shared_markers.values())
        logger.info(f"Found {total_shared} markers shared across at least {min_shared} datasets (no score cutoff applied).")
        
        for ctype, markers in shared_markers.items():
            if markers:
                logger.info(f"Cell type {ctype}: {len(markers)} shared markers")
        # Update standard tables with the dataset analysis results
        logger.info("Updating standard tables with dataset analysis results...")
        
        # For each cell type, update the tables
        cells_updated = 0
        markers_updated = 0
        
        # Check if column_ctype exists in method_run
        if column_ctype not in self.method_run:
            self.method_run[column_ctype] = method  # Use the current method
        
        # Initialize tables if they don't exist
        if column_ctype not in self.sorted_empirical_table or self.sorted_empirical_table[column_ctype] is None:
            self.sorted_empirical_table[column_ctype] = {}
            self.aggregated_empirical_scores[column_ctype] = {}
            self.empirical_scores[column_ctype] = {}
        
        # Get the target tables based on method
        if method == 'empiric':
            target_sorted_table = self.sorted_empirical_table
            target_aggregated_scores = self.aggregated_empirical_scores
        else:  # theoric
            target_sorted_table = self.sorted_table
            target_aggregated_scores = self.aggregated_scores
        
        # For each cell type, update the tables
        for cell_type, table in combined_tables.items():
            # Create a new table for this cell type
            if cell_type not in target_sorted_table[column_ctype]:
                target_sorted_table[column_ctype][cell_type] = pd.DataFrame()
            
            # For each marker, update or add to the table
            for marker in table.index:
                avg_score = table.loc[marker, 'avg_score']
                num_datasets = table.loc[marker, 'num_datasets']
                
                # Create a row for the target table
                if marker not in target_sorted_table[column_ctype][cell_type].index:
                    # Add new marker
                    new_row = pd.Series({
                        'aggregated': avg_score,
                        'exp_prop': 0.0,  # Default value
                        'dataset_score': avg_score,
                        'num_datasets': num_datasets,
                        'datasets': table.loc[marker, 'datasets']
                    })
                    
                    # Update the table
                    target_sorted_table[column_ctype][cell_type] = pd.concat([
                        target_sorted_table[column_ctype][cell_type],
                        pd.DataFrame([new_row], index=[marker])
                    ])
                else:
                    # Update existing marker
                    target_sorted_table[column_ctype][cell_type].loc[marker, 'dataset_score'] = avg_score
                    target_sorted_table[column_ctype][cell_type].loc[marker, 'num_datasets'] = num_datasets
                    target_sorted_table[column_ctype][cell_type].loc[marker, 'datasets'] = table.loc[marker, 'datasets']
                    
                    # Only update aggregated score if method is empiric and running in dataset mode
                    if method == 'empiric':
                        target_sorted_table[column_ctype][cell_type].loc[marker, 'aggregated'] = avg_score
                
                markers_updated += 1
            
            # Sort the table by aggregated score (descending)
            if not target_sorted_table[column_ctype][cell_type].empty:
                target_sorted_table[column_ctype][cell_type] = target_sorted_table[column_ctype][cell_type].sort_values(
                    'aggregated', ascending=False
                )
            
            # Also update aggregated scores table
            if cell_type not in target_aggregated_scores[column_ctype]:
                target_aggregated_scores[column_ctype][cell_type] = pd.Series()
            
            for marker in table.index:
                avg_score = table.loc[marker, 'avg_score']
                target_aggregated_scores[column_ctype][cell_type][marker] = avg_score
            
            cells_updated += 1
        
        # Create top markers dictionary based on dataset analysis
        if column_ctype not in self.top_markers or self.top_markers[column_ctype] is None:
            self.top_markers[column_ctype] = {}
        
        # Update top markers based on the sorted tables
        for cell_type, table in target_sorted_table[column_ctype].items():
            if not table.empty:
                # Get the top marker (first row after sorting)
                top_marker = table.index[0]
                self.top_markers[column_ctype][cell_type] = top_marker
        
        logger.info(f"Updated standard tables with {cells_updated} cell types, {markers_updated} markers")

        # Return the shared markers
        return shared_markers


    def export_dataset_markers(self, column_ctype: str, column_dataset: str, output_file: str = None) -> pd.DataFrame:
        """
        Export shared marker genes to a DataFrame or CSV file.
        
        Args:
            column_ctype: Column name in adata.obs for cell type annotations
            column_dataset: Column name in adata.obs for dataset annotations
            output_file: Path to output CSV file (optional)
                
        Returns:
            pd.DataFrame: DataFrame of shared marker genes with dataset information and scores
        """
        # Check if the dataset analysis has been run
        dataset_key = f"{column_ctype}_by_{column_dataset}"
        if dataset_key not in self.__dict__ or 'combined_tables' not in self.__dict__[dataset_key]:
            raise ValueError(f"Dataset analysis for {column_ctype} by {column_dataset} not found. Run run_dataset() first.")
        
        # Get the combined tables
        combined_tables = self.__dict__[dataset_key]['combined_tables']
        
        # Compile marker information
        results = []
        
        for cell_type, table in combined_tables.items():
            for marker in table.index:
                # Extract dataset-specific scores
                dataset_scores = {}
                for col in table.columns:
                    if col.startswith('score_'):
                        dataset = col.replace('score_', '')
                        score = table.loc[marker, col]
                        if not pd.isna(score):
                            dataset_scores[dataset] = score
                
                # Create a row for this marker
                row = {
                    'cell_type': cell_type,
                    'marker_gene': marker,
                    'num_datasets': int(table.loc[marker, 'num_datasets']),
                    'datasets': table.loc[marker, 'datasets'],
                    'avg_score': table.loc[marker, 'avg_score']
                }
                
                # Add dataset-specific scores
                for dataset, score in dataset_scores.items():
                    row[f"score_{dataset}"] = score
                    
                results.append(row)
        
        # Create DataFrame
        if not results:
            logger.warning("No shared markers found to export")
            return pd.DataFrame()
            
        markers_df = pd.DataFrame(results)
        
        # Sort by cell type, number of datasets (descending), and average score (descending)
        markers_df = markers_df.sort_values(['cell_type', 'num_datasets', 'avg_score'], ascending=[True, False, False])
        
        # Save to file if specified
        if output_file:
            markers_df.to_csv(output_file, index=False)
            logger.info(f"Exported {len(markers_df)} shared markers to {output_file}")
                
        return markers_df

    def visualize_dataset_markers(self, column_ctype: str, column_dataset: str, min_datasets: int = None, max_markers: int = 5, 
                                figsize: Tuple[int, int] = (12, 10), output_file: str = None):
        """
        Visualize shared marker genes across datasets with their scores.
        
        Args:
            column_ctype: Column name in adata.obs for cell type annotations
            column_dataset: Column name in adata.obs for dataset annotations
            min_datasets: Minimum number of datasets for visualization filtering (default: None)
            max_markers: Maximum number of markers to show per cell type (default: 5)
            figsize: Figure size as (width, height) (default: (12, 10))
            output_file: Path to save the figure (optional)
            
        Returns:
            matplotlib.figure.Figure: Figure object with the visualization
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.colors import LinearSegmentedColormap
        
        # Check if the dataset analysis has been run
        dataset_key = f"{column_ctype}_by_{column_dataset}"
        if dataset_key not in self.__dict__ or 'combined_tables' not in self.__dict__[dataset_key]:
            raise ValueError(f"Dataset analysis for {column_ctype} by {column_dataset} not found. Run run_dataset() first.")
        
        # Get the combined tables
        combined_tables = self.__dict__[dataset_key]['combined_tables']
        datasets = self.__dict__[dataset_key]['datasets']
        
        # Extract data for plotting
        plot_data = []
        
        for cell_type, table in combined_tables.items():
            for marker in table.index:
                # Apply minimum datasets filter if specified
                num_datasets = table.loc[marker, 'num_datasets']
                if min_datasets is not None and num_datasets < min_datasets:
                    continue
                    
                # Create a base entry for this marker
                entry = {
                    'cell_type': cell_type,
                    'marker_gene': marker,
                    'num_datasets': num_datasets,
                    'avg_score': table.loc[marker, 'avg_score']
                }
                
                # Add dataset-specific scores
                for dataset in datasets:
                    score_col = f"score_{dataset}"
                    if score_col in table.columns:
                        score = table.loc[marker, score_col]
                        if not pd.isna(score):
                            entry[score_col] = score
                            
                plot_data.append(entry)
        
        # If no data to plot, return None
        if not plot_data:
            logger.warning(f"No markers found that are shared across {min_datasets or 'multiple'} datasets")
            return None
            
        # Convert to DataFrame
        plot_df = pd.DataFrame(plot_data)
        
        # Sort by number of datasets (descending), average score (descending), and cell type
        plot_df = plot_df.sort_values(['num_datasets', 'avg_score', 'cell_type'], ascending=[False, False, True])
        
        # Get top markers per cell type
        top_markers = []
        for cell_type, group in plot_df.groupby('cell_type'):
            # Take top markers for this cell type
            top = group.head(max_markers)
            top_markers.append(top)
        
        # Combine top markers
        if top_markers:
            plot_df = pd.concat(top_markers)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot horizontal bar chart for number of datasets
        ax = sns.barplot(
            data=plot_df,
            y='marker_gene',
            x='num_datasets',
            hue='cell_type',
            palette='viridis',
            dodge=False
        )
        
        # Annotate bars with values and average scores
        for i, p in enumerate(ax.patches):
            width = p.get_width()
            marker = plot_df.iloc[i % len(plot_df)]['marker_gene']
            cell_type = plot_df.iloc[i % len(plot_df)]['cell_type']
            avg_score = plot_df.iloc[i % len(plot_df)]['avg_score']
            
            ax.text(
                width + 0.1,
                p.get_y() + p.get_height()/2,
                f'{int(width)} | {avg_score:.2f}',
                ha='left',
                va='center',
                fontsize=9
            )
        
        # Set labels and title
        plt.xlabel('Number of Datasets')
        plt.ylabel('Marker Gene')
        plt.title('Shared Marker Genes Across Datasets (# datasets | avg score)')
        
        # Adjust legend
        plt.legend(title='Cell Type', loc='upper right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if specified
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {output_file}")
        
        # Return figure object
        return plt.gcf()

    def get_dataset_analysis_results(self, column_ctype: str, column_dataset: str):
        """
        Get the complete results from the dataset analysis.
        
        Args:
            column_ctype: Column name in adata.obs for cell type annotations
            column_dataset: Column name in adata.obs for dataset annotations
            
        Returns:
            dict: Dictionary containing all dataset analysis results
        """
        # Check if the dataset analysis has been run
        dataset_key = f"{column_ctype}_by_{column_dataset}"
        if dataset_key not in self.__dict__:
            raise ValueError(f"Dataset analysis for {column_ctype} by {column_dataset} not found. Run run_dataset() first.")
        
        # Return the complete results
        return self.__dict__[dataset_key]

    def get_global_markers(self, column_ctype: str, column_dataset: str, min_shared: int = 2, min_score: float = 0.5):
        """
        Get global marker genes that are shared across multiple datasets with high scores.
        
        Args:
            column_ctype: Column name in adata.obs for cell type annotations
            column_dataset: Column name in adata.obs for dataset annotations
            min_shared: Minimum number of datasets that must share a marker (default: 2)
            min_score: Minimum average score threshold (default: 0.5)
            
        Returns:
            pd.DataFrame: DataFrame of global marker genes with dataset information and scores
        """
        # Check if the dataset analysis has been run
        dataset_key = f"{column_ctype}_by_{column_dataset}"
        if dataset_key not in self.__dict__ or 'combined_tables' not in self.__dict__[dataset_key]:
            raise ValueError(f"Dataset analysis for {column_ctype} by {column_dataset} not found. Run run_dataset() first.")
        
        # Get the combined tables
        combined_tables = self.__dict__[dataset_key]['combined_tables']
        
        # Compile global marker information
        global_markers = []
        
        for cell_type, table in combined_tables.items():
            # Filter by minimum datasets and score
            filtered_table = table[(table['num_datasets'] >= min_shared) & (table['avg_score'] >= min_score)]
            
            # Skip if no markers remain
            if filtered_table.empty:
                continue
                
            # Add each marker to the global list
            for marker in filtered_table.index:
                global_markers.append({
                    'cell_type': cell_type,
                    'marker_gene': marker,
                    'num_datasets': filtered_table.loc[marker, 'num_datasets'],
                    'avg_score': filtered_table.loc[marker, 'avg_score'],
                    'datasets': filtered_table.loc[marker, 'datasets']
                })
        
        # Convert to DataFrame
        if not global_markers:
            logger.warning(f"No global markers found that meet criteria (min_shared={min_shared}, min_score={min_score})")
            return pd.DataFrame()
            
        global_df = pd.DataFrame(global_markers)
        
        # Sort by cell type, number of datasets (descending), and average score (descending)
        global_df = global_df.sort_values(['cell_type', 'num_datasets', 'avg_score'], ascending=[True, False, False])
        
        return global_df

    def create_global_marker_dict(self, column_ctype: str, column_dataset: str, min_shared: int = 2, min_score: float = 0.5):
        """
        Create a dictionary mapping cell types to their top global marker genes.
        
        This is useful for visualization with standard ScSherlock methods like plot_marker_heatmap.
        
        Args:
            column_ctype: Column name in adata.obs for cell type annotations
            column_dataset: Column name in adata.obs for dataset annotations
            min_shared: Minimum number of datasets that must share a marker (default: 2)
            min_score: Minimum average score threshold (default: 0.5)
            
        Returns:
            dict: Dictionary mapping cell types to their top global marker genes
        """
        # Get global markers
        global_markers_df = self.get_global_markers(column_ctype, column_dataset, min_shared, min_score)
        
        # If no global markers found, return empty dictionary
        if global_markers_df.empty:
            return {}
        
        # Create dictionary mapping cell types to their best marker
        top_markers = {}
        
        for cell_type, group in global_markers_df.groupby('cell_type'):
            # Take the top marker (first row after sorting)
            top_marker = group.iloc[0]['marker_gene']
            top_markers[cell_type] = top_marker
        
        return top_markers

    def compare_dataset_vs_global_markers(self, column_ctype: str, column_dataset: str, min_shared: int = 2, min_score: float = 0.5,
                                        figsize: Tuple[int, int] = (12, 8), output_file: str = None):
        """
        Compare dataset-specific markers with global markers.
        
        Args:
            column_ctype: Column name in adata.obs for cell type annotations
            column_dataset: Column name in adata.obs for dataset annotations
            min_shared: Minimum number of datasets for global markers (default: 2)
            min_score: Minimum average score threshold (default: 0.5)
            figsize: Figure size as (width, height) (default: (12, 8))
            output_file: Path to save the figure (optional)
            
        Returns:
            matplotlib.figure.Figure: Figure object with the comparison visualization
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Check if the dataset analysis has been run
        dataset_key = f"{column_ctype}_by_{column_dataset}"
        if dataset_key not in self.__dict__:
            raise ValueError(f"Dataset analysis for {column_ctype} by {column_dataset} not found. Run run_dataset() first.")
        
        # Get dataset-specific markers
        dataset_markers = self.__dict__[dataset_key]['dataset_markers']
        
        # Get global markers
        global_markers = self.create_global_marker_dict(column_ctype, column_dataset, min_shared, min_score)
        
        # Create a comparison DataFrame
        comparison_data = []
        
        # Process each dataset
        for dataset, markers in dataset_markers.items():
            # For each cell type in this dataset
            for cell_type, marker in markers.items():
                # Check if this cell type has a global marker
                global_marker = global_markers.get(cell_type)
                
                # Record if the dataset and global markers match
                is_match = (marker == global_marker) if global_marker else False
                
                comparison_data.append({
                    'dataset': dataset,
                    'cell_type': cell_type,
                    'dataset_marker': marker,
                    'global_marker': global_marker if global_marker else "None",
                    'is_match': is_match
                })
        
        # If no data to plot, return None
        if not comparison_data:
            logger.warning("No comparison data available")
            return None
            
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate match percentage for each dataset
        match_percent = comparison_df.groupby('dataset')['is_match'].mean() * 100
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot match percentage by dataset
        ax = sns.barplot(x=match_percent.index, y=match_percent.values, palette='viridis')
        
        # Add value labels
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(
                p.get_x() + p.get_width()/2,
                height + 1,
                f'{height:.1f}%',
                ha='center',
                fontsize=10
            )
        
        # Set labels and title
        plt.xlabel('Dataset')
        plt.ylabel('Marker Match Percentage')
        plt.title(f'Agreement Between Dataset-Specific and Global Markers\n(global markers shared across ≥{min_shared} datasets with avg score ≥{min_score})')
        
        # Add horizontal line at 100%
        plt.axhline(y=100, color='red', linestyle='--', alpha=0.7)
        
        # Set y-axis limit
        plt.ylim(0, 110)
        
        # Rotate x-axis labels if needed
        if len(match_percent) > 5:
            plt.xticks(rotation=45, ha='right')
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure if specified
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison to {output_file}")
        
        # Return figure object
        return plt.gcf()

    def get_marker(self, column_ctype: str, method: str = "empiric", dataset_column: str = None, 
             score_type: str = "aggregated", n_top_genes: int = 1, min_score: float = 0.0) -> pd.DataFrame:
        """
        Get marker genes and their scores for specified cell types.
        
        Args:
            column_ctype: Column name in adata.obs for cell type annotations
            method: Method to use, either "theoric" or "empiric" (default: "empiric")
            dataset_column: If provided, use markers from dataset analysis using this column (default: None)
            score_type: Which score to use (default: "aggregated")
                For regular analysis: "aggregated" or any column in the sorted tables
                For dataset analysis: "avg_score", "weighted_score", or "num_datasets"
            n_top_genes: Number of top genes to return for each cell type (default: 1)
            min_score: Minimum score threshold for including markers (default: 0.0)
                
        Returns:
            pd.DataFrame: DataFrame with marker genes and their scores
            
        Raises:
            ValueError: If specified method or dataset_column is not valid
        """
        # Validate method parameter
        if method not in ["theoric", "empiric"]:
            raise ValueError('Method must be either "theoric" or "empiric"')
        
        # Check if we're using dataset analysis
        using_dataset_analysis = False
        if dataset_column is not None:
            dataset_key = f"{column_ctype}_by_{dataset_column}"
            if dataset_key in self.__dict__ and 'combined_tables' in self.__dict__[dataset_key]:
                using_dataset_analysis = True
                
                # Validate score_type for dataset analysis
                valid_score_types = ["avg_score", "weighted_score", "num_datasets"]
                if score_type not in valid_score_types:
                    logger.warning(f"Invalid score_type '{score_type}' for dataset analysis. Using 'weighted_score' instead.")
                    score_type = "weighted_score"
            else:
                raise ValueError(f"Dataset analysis for {column_ctype} by {dataset_column} not found. Run run_dataset() first.")
        
        # Check if the annotation has been processed
        if not using_dataset_analysis:
            if column_ctype not in self.method_run:
                raise ValueError(f"No analysis exists for {column_ctype}. Run ScSherlock.run() with this annotation first.")
                
            # Check if the requested method has been run
            if method == "empiric" and (column_ctype not in self.sorted_empirical_table or self.sorted_empirical_table[column_ctype] is None):
                raise ValueError(f"Empirical scores haven't been calculated for {column_ctype}. Run with method='empiric' first.")
                
            if method == "theoric" and (column_ctype not in self.sorted_table or self.sorted_table[column_ctype] is None):
                raise ValueError(f"Theoretical scores haven't been calculated for {column_ctype}. Run with method='theoric' first.")
        
        # Prepare results DataFrame
        results = []
        
        # Get markers based on dataset analysis or regular analysis
        if using_dataset_analysis:
            # Get combined tables from dataset analysis
            combined_tables = self.__dict__[dataset_key]['combined_tables']
            
            # Process each cell type
            for cell_type, table in combined_tables.items():
                if table.empty:
                    continue
                    
                # Filter by minimum score
                if score_type in table.columns:
                    filtered_table = table[table[score_type] >= min_score]
                else:
                    logger.warning(f"Score type '{score_type}' not found in table for {cell_type}. Using all markers.")
                    filtered_table = table
                    
                # Skip if no markers remain
                if filtered_table.empty:
                    continue
                    
                # Sort by specified score type if it exists
                if score_type in filtered_table.columns:
                    filtered_table = filtered_table.sort_values(score_type, ascending=False)
                    
                # Get top N genes
                top_genes = filtered_table.head(n_top_genes)
                
                # Add to results
                for marker, row in top_genes.iterrows():
                    # Get scores for all available columns
                    marker_data = {
                        'cell_type': cell_type,
                        'marker_gene': marker,
                    }
                    
                    # Add all available scores
                    for col in row.index:
                        if col.startswith('score_') or col in ['avg_score', 'weighted_score', 'num_datasets']:
                            marker_data[col] = row[col]
                    
                    # Add datasets list if available
                    if 'datasets' in row:
                        marker_data['datasets'] = row['datasets']
                        
                    results.append(marker_data)
        else:
            # Get appropriate tables based on the method
            if method == "empiric":
                sorted_tables = self.sorted_empirical_table[column_ctype]
            else:  # method == "theoric"
                sorted_tables = self.sorted_table[column_ctype]
                
            # Process each cell type
            for cell_type, table in sorted_tables.items():
                if not isinstance(table, pd.DataFrame) or table.empty:
                    continue
                    
                # Validate score_type for regular analysis
                if score_type not in table.columns:
                    if score_type != "aggregated":
                        logger.warning(f"Score type '{score_type}' not found in table for {cell_type}. Using 'aggregated' instead.")
                    score_type = "aggregated"
                    
                # Filter by minimum score
                filtered_table = table[table[score_type] >= min_score]
                
                # Skip if no markers remain
                if filtered_table.empty:
                    continue
                    
                # Sort by score type
                filtered_table = filtered_table.sort_values(score_type, ascending=False)
                
                # Get top N genes
                top_genes = filtered_table.head(n_top_genes)
                
                # Add to results
                for marker, row in top_genes.iterrows():
                    # Get basic data
                    marker_data = {
                        'cell_type': cell_type,
                        'marker_gene': marker,
                        'score': row[score_type]
                    }
                    
                    # Add method used
                    marker_data['method'] = method
                    
                    # Add expression proportion if available
                    if 'exp_prop' in row:
                        marker_data['expression_proportion'] = row['exp_prop']
                        
                    # Add bootstrap results if available
                    if 'bootstrap_pass' in row:
                        marker_data['bootstrap_pass'] = row['bootstrap_pass']
                        
                    if 'bootstrap_1_score' in row:
                        marker_data['bootstrap_1_score'] = row['bootstrap_1_score']
                        
                    if 'bootstrap_2_score' in row:
                        marker_data['bootstrap_2_score'] = row['bootstrap_2_score']
                        
                    results.append(marker_data)
        
        # If no results found, return empty DataFrame with appropriate columns
        if not results:
            if using_dataset_analysis:
                columns = ['cell_type', 'marker_gene', 'avg_score', 'weighted_score', 'num_datasets']
            else:
                columns = ['cell_type', 'marker_gene', 'score', 'method']
            return pd.DataFrame(columns=columns)
        
        # Convert to DataFrame
        markers_df = pd.DataFrame(results)
        
        # Sort by cell type
        markers_df = markers_df.sort_values('cell_type')
        
        return markers_df
    

    def plot_dataset_heatmap(self, column_ctype: str, column_dataset: str, min_datasets: int = 2, 
                        min_average_score: float = 0.0, n_top_markers: int = None,
                        display_score: bool = True, figsize: Tuple[int, int] = (14, 10), 
                        output_file: str = None, cmap: str = 'viridis'):
        """
        Create a heatmap of marker gene scores across datasets.
        
        Args:
            column_ctype: Column name in adata.obs for cell type annotations
            column_dataset: Column name in adata.obs for dataset annotations
            min_datasets: Minimum number of datasets for marker inclusion (default: 2)
            min_average_score: Minimum average score threshold for including markers (default: 0.0)
            n_top_markers: Limit to top N markers per cell type (default: None, include all)
            display_score: Whether to display score values in heatmap cells (default: True)
            figsize: Figure size as (width, height) (default: (14, 10))
            output_file: Path to save the figure (optional)
            cmap: Colormap for the heatmap (default: 'viridis')
            
        Returns:
            matplotlib.figure.Figure: Figure object with the heatmap
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Check if the dataset analysis has been run
        dataset_key = f"{column_ctype}_by_{column_dataset}"
        if dataset_key not in self.__dict__ or 'combined_tables' not in self.__dict__[dataset_key]:
            raise ValueError(f"Dataset analysis for {column_ctype} by {column_dataset} not found. Run run_dataset() first.")
        
        # Get the combined tables
        combined_tables = self.__dict__[dataset_key]['combined_tables']
        datasets = self.__dict__[dataset_key]['datasets']
        
        # Create a DataFrame for the heatmap
        heatmap_data = []
        
        # Process each cell type
        for cell_type, table in combined_tables.items():
            # Filter by minimum datasets and average score
            filtered_table = table[(table['num_datasets'] >= min_datasets) & 
                                (table['avg_score'] >= min_average_score)]
            
            # Skip if no markers remain
            if filtered_table.empty:
                continue
                
            # Limit to top N markers if specified
            if n_top_markers is not None and len(filtered_table) > n_top_markers:
                filtered_table = filtered_table.head(n_top_markers)
                
            # Get dataset score columns
            score_cols = [col for col in filtered_table.columns if col.startswith('score_')]
            
            # Process each marker
            for marker in filtered_table.index:
                # Create a base row for this marker
                base_row = {
                    'cell_type': cell_type,
                    'marker_gene': marker,
                    'num_datasets': filtered_table.loc[marker, 'num_datasets'],
                    'avg_score': filtered_table.loc[marker, 'avg_score']
                }
                
                # Add score for each dataset
                for col in score_cols:
                    dataset = col.replace('score_', '')
                    score = filtered_table.loc[marker, col]
                    
                    # Create a row for this dataset-marker combination
                    row = base_row.copy()
                    row['dataset'] = dataset
                    row['score'] = score
                    
                    # Add to heatmap data if score is not NaN
                    if not pd.isna(score):
                        heatmap_data.append(row)
        
        # If no data to plot, return None
        if not heatmap_data:
            logger.warning(f"No markers found that meet the criteria (min_datasets={min_datasets}, min_average_score={min_average_score})")
            return None
            
        # Convert to DataFrame
        heatmap_df = pd.DataFrame(heatmap_data)
        
        # Create marker labels with cell type and additional information
        heatmap_df['marker_label'] = heatmap_df.apply(
            lambda row: f"{row['marker_gene']} ({row['cell_type']}, {int(row['num_datasets'])}d, {row['avg_score']:.2f}a)", 
            axis=1
        )
        
        # Create pivot table for the heatmap
        pivot_df = heatmap_df.pivot_table(
            index='marker_label',
            columns='dataset',
            values='score',
            aggfunc='mean'
        )
        
        # Sort the pivot table by cell type, num_datasets, and marker_gene
        # First create a sorting key
        sort_df = heatmap_df.drop_duplicates('marker_label').set_index('marker_label')
        sort_df['sort_key'] = sort_df.apply(
            lambda row: f"{row['cell_type']}_{100-row['num_datasets']:03d}_{-row['avg_score']:0.3f}_{row['marker_gene']}", 
            axis=1
        )
        sort_order = sort_df.sort_values('sort_key').index
        
        # Apply the sort order
        pivot_df = pivot_df.reindex(sort_order)
        
        # Create the figure
        plt.figure(figsize=figsize)
        
        # Plot the heatmap
        ax = sns.heatmap(
            pivot_df,
            cmap=cmap,
            linewidths=0.5,
            linecolor='white',
            square=False,
            annot=display_score,  # Use display_score parameter
            fmt=".2f" if display_score else "",
            cbar_kws={"shrink": 0.8, "label": "Score"}
        )
        
        # Set labels and title
        title = f'Marker Gene Scores Across Datasets\n'
        title += f'(markers in ≥{min_datasets} datasets'
        if min_average_score > 0:
            title += f' with avg score ≥{min_average_score:.2f}'
        title += ')'
        
        plt.title(title)
        plt.xlabel('Dataset')
        plt.ylabel('Marker Gene (Cell Type, #Datasets, AvgScore)')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Tight layout
        plt.tight_layout()
        
        # Save figure if specified
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved heatmap to {output_file}")
        
        # Return figure object
        #return plt.gcf()

    def create_hierarchy_graph_unsupervised(self, annotation_column, max_depth=3, 
                                    linkage_method="ward", distance_metric="correlation", 
                                    min_proportion=0.05, store_graph=True):
        """
        Create a hierarchy graph based on transcriptional similarity of cell clusters.
        
        This method:
        1. Creates pseudobulk profiles for each cluster in the annotation column
        2. Performs hierarchical clustering on these profiles
        3. Extracts a true hierarchy directly from the clustering tree
        4. Adds depth columns to the AnnData object
        5. Returns a graph that's compatible with all visualization functions
        
        Args:
            annotation_column (str): Column name in adata.obs for cell type annotations
            max_depth (int): Maximum depth of the hierarchy (default: 3)
            linkage_method (str): Method for hierarchical clustering (default: "ward")
                                Options: "single", "complete", "average", "weighted", "ward"
            distance_metric (str): Distance metric for hierarchical clustering (default: "correlation")
                                Options: "euclidean", "correlation", "cosine"
            min_proportion (float): Minimum proportion for edge weights (default: 0.05)
            store_graph (bool): Whether to store the graph in self.G (default: True)
            
        Returns:
            networkx.DiGraph: Directed graph representing cell type hierarchy
        """
        import scipy.cluster.hierarchy as sch
        import scipy.spatial.distance as ssd
        import numpy as np
        import pandas as pd
        import networkx as nx
        
        logger.info(f"Creating unsupervised hierarchy based on {annotation_column} with max depth {max_depth}")
        
        # Validate input
        if annotation_column not in self.adata.obs.columns:
            raise ValueError(f"Annotation column '{annotation_column}' not found in adata.obs")
        
        # Step 1: Create pseudobulk profiles for each cluster
        logger.info("Creating pseudobulk profiles...")
        adata_agg = ADPBulk(self.adata, annotation_column)
        pseudobulk_matrix = adata_agg.fit_transform()
        sample_meta = adata_agg.get_meta()
        
        # Ensure annotation column is in sample_meta
        if annotation_column not in sample_meta.columns:
            raise ValueError(f"Failed to create pseudobulk profiles with {annotation_column}")
        
        # Set index to annotation column
        pseudobulk_matrix.set_index(sample_meta[annotation_column], inplace=True)
        
        # Store original cell type names
        cell_types = list(pseudobulk_matrix.index)
        n_samples = len(cell_types)
        
        # Step 2: Perform hierarchical clustering
        logger.info(f"Performing hierarchical clustering using {linkage_method} linkage and {distance_metric} distance...")
        
        # Compute distance matrix
        if distance_metric == "correlation":
            # For correlation, higher values mean more similar, so we convert to distance
            corr_matrix = pseudobulk_matrix.T.corr()
            dist_matrix = 1 - corr_matrix
            # Handle any NA values
            dist_matrix = dist_matrix.fillna(1.0)
            # Convert to condensed form
            condensed_dist = ssd.squareform(dist_matrix)
        elif distance_metric == "cosine":
            # Compute pairwise cosine distances
            condensed_dist = ssd.pdist(pseudobulk_matrix, metric="cosine")
        else:
            # Default to euclidean
            condensed_dist = ssd.pdist(pseudobulk_matrix, metric="euclidean")
        
        # Perform hierarchical clustering
        Z = sch.linkage(condensed_dist, method=linkage_method)
        
        # We need to ensure that each parent layer has fewer nodes than child layers
        # Start with a small number of clusters at the top level and increase for deeper levels
        
        # Calculate number of clusters for each depth
        # For max_depth=3: [2, 4, 8] or similar
        n_clusters_by_depth = {}
        
        # Start with 2 clusters at top level (or adjust based on data size)
        base_clusters = max(2, len(cell_types) // (2**(max_depth+1)))
        
        # Ensure we have at least one more cluster at each deeper level
        for depth in range(1, max_depth + 1):
            # Number of clusters increases with depth but stays below number of cell types
            n_clusters_by_depth[depth] = min(
                # Increase exponentially with depth
                base_clusters * (2**(depth-1)),
                # But never exceed the number of cell types
                max(depth, len(cell_types) - (max_depth - depth))
            )
        
        # For debugging
        logger.info(f"Number of clusters by depth: {n_clusters_by_depth}")
        
        # Verify that we have increasing numbers of clusters with depth
        for depth in range(1, max_depth):
            if n_clusters_by_depth[depth] >= n_clusters_by_depth[depth+1]:
                # Adjust to ensure increasing clusters with depth
                n_clusters_by_depth[depth+1] = n_clusters_by_depth[depth] + 1
                logger.info(f"Adjusted clusters at depth {depth+1} to {n_clusters_by_depth[depth+1]}")
        
        # Create cluster assignments at each depth using the calculated number of clusters
        cluster_labels = {}
        for depth in range(1, max_depth + 1):
            # Use the pre-calculated number of clusters for this depth
            n_clusters = n_clusters_by_depth[depth]
            labels = sch.fcluster(Z, n_clusters, criterion='maxclust')
            
            # Store labels with cell types as index
            cluster_labels[depth] = pd.Series(labels, index=cell_types)
            
            # Log the number of unique clusters at this depth
            unique_clusters = len(np.unique(labels))
            logger.info(f"Depth {depth}: {unique_clusters} unique clusters (target: {n_clusters})")
        
        # Add depth columns to AnnData object
        for depth in range(1, max_depth + 1):
            depth_column = f"{annotation_column}_depth{depth}"
            # Create cluster names in format "Cluster_depth_id"
            cluster_names = {
                cell_type: f"Cluster_{depth}_{cluster_labels[depth][cell_type]}" 
                for cell_type in cell_types
            }
            
            # Add to AnnData
            self.adata.obs[depth_column] = self.adata.obs[annotation_column].map(
                lambda x: cluster_names.get(x, f"Unknown_{depth}")
            )
            # Convert to categorical
            self.adata.obs[depth_column] = self.adata.obs[depth_column].astype('category')
            logger.info(f"Added column {depth_column} to AnnData.obs")
        
        # Step 4: Create the hierarchy graph in the exact same format as create_hierarchy_graph
        logger.info("Building compatible networkx graph from hierarchy...")
        G = nx.DiGraph()
        
        # Create a consistent node naming scheme for compatibility
        # In the original function, nodes are named with the actual value
        
        # Start with root node
        G.add_node('root')
        
        # For each depth level
        column_list = [f"{annotation_column}_depth{d}" for d in range(1, max_depth + 1)] + [annotation_column]
        
        # Process each level in the hierarchy
        for i, column in enumerate(column_list):
            # Get unique values at this level
            if column in self.adata.obs.columns:
                unique_values = self.adata.obs[column].unique()
                level = i  # This becomes the 'level' attribute
                
                # Add nodes for each unique value
                for value in unique_values:
                    # Add the node if it doesn't exist
                    if not G.has_node(value):
                        G.add_node(value)
                        # Set node attributes exactly as in the original function
                        G.nodes[value]['level'] = level
                        G.nodes[value]['annotation'] = column
                        G.nodes[value]['label'] = value  # The label is the value itself
                    
                    # Connect to parent nodes
                    if i > 0:
                        # Get the parent column
                        parent_column = column_list[i-1]
                        
                        # Find all cells with this value
                        mask = self.adata.obs[column] == value
                        
                        if any(mask):
                            # Get the parent values for these cells
                            parent_values = self.adata.obs.loc[mask, parent_column].unique()
                            
                            for parent in parent_values:
                                # Calculate proportion
                                parent_mask = self.adata.obs[parent_column] == parent
                                child_in_parent = sum(mask & parent_mask)
                                parent_total = sum(parent_mask)
                                
                                proportion = child_in_parent / parent_total if parent_total > 0 else 0
                                
                                # Add edge regardless of proportion for connectivity
                                # But use the proportion for edge weight and node attribute
                                G.add_edge(parent, value)
                                G.edges[parent, value]['weight'] = max(proportion, 0.01)  # Ensure minimum weight
                                
                                # Add proportion to node for visualization
                                if 'proportion' not in G.nodes[value] or proportion > G.nodes[value]['proportion']:
                                    G.nodes[value]['proportion'] = proportion
                        else:
                            logger.warning(f"No cells found for value {value} in column {column}")
                    # For the first level, connect to root
                    elif i == 0:
                        G.add_edge('root', value)
                        # Default weight for top level is 1.0 / number of top-level nodes
                        weight = 1.0 / len(unique_values)
                        G.edges['root', value]['weight'] = weight
                        G.nodes[value]['proportion'] = weight
        
        # Clean up the graph - remove isolated nodes
        isolated = [n for n in G.nodes() if G.degree(n) == 0 and n != 'root']
        G.remove_nodes_from(isolated)
        
        # Check for disconnected components
        if not nx.is_weakly_connected(G):
            logger.warning("The hierarchy graph is not fully connected. Some nodes may be isolated.")
        
        # Store graph in instance if requested
        if store_graph:
            self.G = G
            logger.info("Stored hierarchy graph in ScSherlock.G")
        
        logger.info(f"Created hierarchy graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
        return G

    


    def plot_cluster_dendrogram(self, annotation_column, max_depth=3, 
                            linkage_method="ward", distance_metric="correlation",
                            figsize=(12, 8), color_threshold=None, output_file=None):
        """
        Plot the hierarchical clustering dendrogram for the pseudobulk profiles.
        
        Args:
            annotation_column (str): Column name in adata.obs for cell type annotations
            max_depth (int): Maximum depth to show in the dendrogram (default: 3)
            linkage_method (str): Method for hierarchical clustering (default: "ward")
            distance_metric (str): Distance metric for hierarchical clustering (default: "correlation")
            figsize (tuple): Figure size (width, height)
            color_threshold (float, optional): The threshold to use for coloring clusters
            output_file (str, optional): Path to save the figure
                
        Returns:
            matplotlib.figure.Figure: Figure object with the dendrogram
        """
        import matplotlib.pyplot as plt
        import scipy.cluster.hierarchy as sch
        import scipy.spatial.distance as ssd
        import pandas as pd
        
        # Create pseudobulk profiles
        adata_agg = ADPBulk(self.adata, annotation_column)
        pseudobulk_matrix = adata_agg.fit_transform()
        sample_meta = adata_agg.get_meta()
        
        # Set index to annotation column
        pseudobulk_matrix.set_index(sample_meta[annotation_column], inplace=True)
        
        # Compute distance matrix
        if distance_metric == "correlation":
            corr_matrix = pseudobulk_matrix.T.corr()
            dist_matrix = 1 - corr_matrix
            dist_matrix = dist_matrix.fillna(1.0)
            condensed_dist = ssd.squareform(dist_matrix)
        elif distance_metric == "cosine":
            condensed_dist = ssd.pdist(pseudobulk_matrix, metric="cosine")
        else:
            condensed_dist = ssd.pdist(pseudobulk_matrix, metric="euclidean")
        
        # Perform hierarchical clustering
        Z = sch.linkage(condensed_dist, method=linkage_method)
        
        # Plot dendrogram
        plt.figure(figsize=figsize)
        
        # Define color threshold based on max_depth if not specified
        if color_threshold is None:
            # Calculate automatic color threshold based on max_depth
            n_clusters = max(1, min(len(pseudobulk_matrix) - max_depth, len(pseudobulk_matrix) // (2**(max_depth-1))))
            # Use fcluster to get the threshold that would give n_clusters
            threshold = sch.fcluster(Z, n_clusters, criterion='maxclust')
            # Get unique cluster assignments
            unique_clusters = pd.Series(threshold).unique()
            if len(unique_clusters) > 1:
                # Use the minimum threshold that gives the desired number of clusters
                color_threshold = min(Z[-(n_clusters-1):, 2])
            else:
                color_threshold = 0
        
        # Plot the dendrogram
        dendrogram = sch.dendrogram(
            Z,
            labels=pseudobulk_matrix.index,
            orientation='right',
            leaf_font_size=10,
            color_threshold=color_threshold
        )
        
        # Add labels and title
        plt.title(f'Hierarchical Clustering of {annotation_column}\n({linkage_method} linkage, {distance_metric} distance)', fontsize=14)
        plt.xlabel('Distance', fontsize=12)
        plt.tight_layout()
        
        # Save if output file is specified
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved dendrogram to {output_file}")
        
        return plt.gcf()


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