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
from enum import Enum, auto
import time
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import scipy
import numba as nb
from sklearn.preprocessing import StandardScaler
from statsmodels.distributions.empirical_distribution import ECDF
import scipy
import gc

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
            #self.theoretical_scores, self.expr_proportions = self._calculate_theoretical_scores_parallel2()
            self.theoretical_scores, self.expr_proportions = self._calculate_theoretical_scores()
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
            #self.empirical_scores = self.empirical_scores_optimized_batch_parallel_only_ctype()
            self.empirical_scores = self.empirical_scores_v0(self.filtered_scores, self.adata, self.column_ctype, self.column_patient, self.config.k_values, 'diff', n_sim=self.config.n_simulations)

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
                self.empirical_scores = self.empirical_scores_v0_optimized(self.filtered_scores)

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
    
    def _calculate_theoretical_scores_optimized(self) -> Tuple[Dict, Dict]:
        """
        Calculate theoretical scores based on binomial distribution with optimization
        
        Returns:
            Tuple[Dict, Dict]: Scores and expression proportions
        """
        # Sparse sampling parameters
        sparse_step = 10
        promising_threshold = 0.1

        # Estimate binomial distribution parameters
        parameters = self._estimate_binomial_parameters()

        scores = {}
        # Calculate scores for each cell type
        for ctype in self.cell_types:
            # Initialize score matrix (rows=genes, columns=k_values)
            scores_ctype = pd.DataFrame(
                np.zeros((self.adata.shape[1], len(self.config.k_values))),
                index=self.adata.var.index, 
                columns=self.config.k_values
            )
            
            # Calculate scores for each k value
            for k in self.config.k_values:
                # Calculate cutoffs for computation
                cutoffs_k = pd.DataFrame([
                    # Target cells
                    binom.ppf(.99, int(parameters[0][ctype] * k), parameters[2][ctype]),
                    # Non-target cells
                    binom.ppf(.99, int(parameters[1][ctype] * k), parameters[3][ctype])
                ], columns=scores_ctype.index).max().clip(lower=100)
                
                # Determine the cutoff value
                cutoff_k = int(cutoffs_k.max())
                
                # OPTIMIZATION: Start with sparse sampling
                sparse_points = np.arange(0, cutoff_k, sparse_step)
                if len(sparse_points) == 0:  # Ensure at least one point
                    sparse_points = np.array([0])
                
                # Initialize arrays for sparse CDFs
                alpha_sparse = np.zeros(shape=(self.adata.shape[1], len(sparse_points)))
                beta_sparse = np.zeros(shape=(self.adata.shape[1], len(sparse_points)))
                
                # Calculate sparse CDFs using binomial CDF
                for i, l in enumerate(sparse_points):
                    alpha_sparse[:, i] = binom.cdf(
                        l, int(parameters[0][ctype] * k), parameters[2][ctype]
                    )
                    beta_sparse[:, i] = binom.cdf(
                        l, int(parameters[1][ctype] * k), parameters[3][ctype]
                    )
                
                # Identify promising genes based on scoring method
                promising_genes = np.ones(self.adata.shape[1], dtype=bool)  # Default all genes to promising
                
                if self.config.scoring_method == ScoringMethod.DIFF:
                    # Max difference between specificity (beta) and false negative rate (alpha)
                    sparse_diff = beta_sparse - alpha_sparse
                    max_sparse_diff = np.max(sparse_diff, axis=1)
                    promising_genes = max_sparse_diff > promising_threshold
                    
                elif self.config.scoring_method == ScoringMethod.SENS_FPR_ZERO:
                    # Sensitivity at point of zero false positive rate
                    max_beta = np.max(beta_sparse, axis=1)
                    promising_genes = max_beta > (1 - promising_threshold)
                    
                elif self.config.scoring_method == ScoringMethod.SENS_PPV_99:
                    # PPV calculation for sparse points
                    ppv_sparse = np.nan_to_num((1-alpha_sparse)/(2-alpha_sparse-beta_sparse))
                    # Check if any value exceeds 0.99
                    promising_genes = np.max(ppv_sparse > 0.99, axis=1) > 0
                
                # Initialize full arrays for alpha and beta
                alpha = np.zeros(shape=(self.adata.shape[1], cutoff_k))
                beta = np.zeros(shape=(self.adata.shape[1], cutoff_k))
                
                # Copy sparse results into the full arrays to avoid recalculation
                for i, l in enumerate(sparse_points):
                    alpha[:, l] = alpha_sparse[:, i]
                    beta[:, l] = beta_sparse[:, i]
                
                # Calculate remaining points only for promising genes
                remaining_points = np.setdiff1d(np.arange(cutoff_k), sparse_points)
                
                for l in remaining_points:
                    # Calculate alpha, but only for promising genes
                    if np.any(promising_genes):
                        if promising_genes.all():  # If all genes are promising, calculate for all
                            alpha[:, l] = binom.cdf(l, int(parameters[0][ctype] * k), parameters[2][ctype])
                            beta[:, l] = binom.cdf(l, int(parameters[1][ctype] * k), parameters[3][ctype])
                        else:
                            # Get indices of promising genes
                            promising_indices = np.where(promising_genes)[0]
                            
                            # Calculate for promising genes only
                            alpha[promising_indices, l] = binom.cdf(
                                l, 
                                int(parameters[0][ctype] * k), 
                                parameters[2][ctype][promising_indices]
                            )
                            beta[promising_indices, l] = binom.cdf(
                                l, 
                                int(parameters[1][ctype] * k), 
                                parameters[3][ctype][promising_indices]
                            )
                
                # Calculate scores based on the scoring method
                if self.config.scoring_method == ScoringMethod.DIFF:
                    # Max difference between specificity (beta) and false negative rate (alpha)
                    scores_ctype[k] = (beta-alpha)[np.arange(self.adata.shape[1]), np.argmax(beta-alpha, axis=1)]
                elif self.config.scoring_method == ScoringMethod.SENS_FPR_ZERO:
                    # Sensitivity at point of zero false positive rate
                    scores_ctype[k] = (1-alpha)[np.arange(self.adata.shape[1]), np.argmax(beta, axis=1)]
                elif self.config.scoring_method == ScoringMethod.SENS_PPV_99:
                    # Positive predictive value calculation
                    ppv = np.nan_to_num((1-alpha)/(2-alpha-beta))
                    # Sensitivity at point where PPV > 99%
                    scores_ctype[k] = ((1-alpha)[np.arange(self.adata.shape[1]), 
                                        np.argmax(ppv>0.99, axis=1)]) * (np.sum(ppv>0.99, axis=1)>0)
            
            scores[ctype] = scores_ctype
        
        return scores, parameters[2]
    
    def _calculate_theoretical_scores_optimized_memory(self) -> Tuple[Dict, Dict]:
        """
        Calculate theoretical scores based on binomial distribution with memory-efficient optimization
        
        Returns:
            Tuple[Dict, Dict]: Scores and expression proportions
        """
        # Sparse sampling parameters
        sparse_step = 10
        promising_threshold = 0.1

        # Get total number of genes for reporting
        total_genes = self.adata.shape[1]
        print(f"Calculating theoretical scores for {total_genes} genes")

        # Estimate binomial distribution parameters
        parameters = self._estimate_binomial_parameters()
        
        scores = {}
        # Process each cell type
        for ctype in self.cell_types:
            print(f"Processing cell type: {ctype}")
            
            # Process genes in batches to reduce memory usage
            batch_size = 1000  # Adjust based on available memory
            gene_batches = [(i, min(i + batch_size, total_genes)) 
                            for i in range(0, total_genes, batch_size)]
            
            print(f"Processing in {len(gene_batches)} batches of {batch_size} genes")
            
            # Initialize score matrix (rows=genes, columns=k_values)
            scores_ctype = pd.DataFrame(
                np.zeros((total_genes, len(self.config.k_values))),
                index=self.adata.var.index, 
                columns=self.config.k_values
            )
            
            # Calculate scores for each k value
            for k in self.config.k_values:
                print(f"  Processing k={k}")
                
                # Calculate cutoffs for computation (for all genes at once)
                cutoffs_k = pd.DataFrame([
                    # Target cells
                    binom.ppf(.99, int(parameters[0][ctype] * k), parameters[2][ctype]),
                    # Non-target cells
                    binom.ppf(.99, int(parameters[1][ctype] * k), parameters[3][ctype])
                ], columns=scores_ctype.index).max().clip(lower=100)
                
                # Determine the cutoff value
                cutoff_k = int(cutoffs_k.max())
                print(f"  Cutoff value: {cutoff_k}")
                
                # OPTIMIZATION: Use sparse sampling for all genes
                sparse_points = np.arange(0, cutoff_k, sparse_step)
                if len(sparse_points) == 0:  # Ensure at least one point
                    sparse_points = np.array([0])
                print(f"  Using {len(sparse_points)} sparse points")
                
                # Process each batch of genes
                for batch_start, batch_end in gene_batches:
                    batch_size_actual = batch_end - batch_start
                    # print(f"  Processing gene batch {batch_start}-{batch_end}")
                    
                    # Initialize arrays for sparse CDFs for this batch
                    alpha_sparse = np.zeros(shape=(batch_size_actual, len(sparse_points)))
                    beta_sparse = np.zeros(shape=(batch_size_actual, len(sparse_points)))
                    
                    # Get parameters for this batch
                    p2_batch = parameters[2][ctype][batch_start:batch_end]
                    p3_batch = parameters[3][ctype][batch_start:batch_end]
                    
                    # Calculate sparse CDFs using binomial CDF
                    for i, l in enumerate(sparse_points):
                        alpha_sparse[:, i] = binom.cdf(
                            l, int(parameters[0][ctype] * k), p2_batch
                        )
                        beta_sparse[:, i] = binom.cdf(
                            l, int(parameters[1][ctype] * k), p3_batch
                        )
                    
                    # Identify promising genes based on scoring method
                    if self.config.scoring_method == ScoringMethod.DIFF:
                        # Max difference between specificity (beta) and false negative rate (alpha)
                        sparse_diff = beta_sparse - alpha_sparse
                        max_sparse_diff = np.max(sparse_diff, axis=1)
                        promising_genes = max_sparse_diff > promising_threshold
                        
                    elif self.config.scoring_method == ScoringMethod.SENS_FPR_ZERO:
                        # Sensitivity at point of zero false positive rate
                        max_beta = np.max(beta_sparse, axis=1)
                        promising_genes = max_beta > (1 - promising_threshold)
                        
                    elif self.config.scoring_method == ScoringMethod.SENS_PPV_99:
                        # PPV calculation for sparse points
                        ppv_sparse = np.nan_to_num((1-alpha_sparse)/(2-alpha_sparse-beta_sparse))
                        # Check if any value exceeds 0.99
                        promising_genes = np.max(ppv_sparse > 0.99, axis=1) > 0
                    
                    # Count promising genes
                    promising_count = np.sum(promising_genes)
                    # print(f"  Promising genes in batch: {promising_count}/{batch_size_actual}")
                    
                    # If no promising genes in this batch, we can skip further processing
                    if promising_count == 0:
                        continue
                    
                    # Initialize full arrays for alpha and beta (only for promising genes)
                    if promising_count == batch_size_actual:
                        # All genes are promising, process the whole batch
                        alpha = np.zeros(shape=(batch_size_actual, cutoff_k))
                        beta = np.zeros(shape=(batch_size_actual, cutoff_k))
                        
                        # Copy sparse results into the full arrays
                        for i, l in enumerate(sparse_points):
                            alpha[:, l] = alpha_sparse[:, i]
                            beta[:, l] = beta_sparse[:, i]
                        
                        # Calculate remaining points for all genes in batch
                        remaining_points = np.setdiff1d(np.arange(cutoff_k), sparse_points)
                        for l in remaining_points:
                            alpha[:, l] = binom.cdf(l, int(parameters[0][ctype] * k), p2_batch)
                            beta[:, l] = binom.cdf(l, int(parameters[1][ctype] * k), p3_batch)
                        
                        # Calculate scores based on the scoring method
                        if self.config.scoring_method == ScoringMethod.DIFF:
                            diff = beta - alpha
                            max_indices = np.argmax(diff, axis=1)
                            scores_batch = np.array([diff[i, max_indices[i]] for i in range(batch_size_actual)])
                            scores_ctype.iloc[batch_start:batch_end, scores_ctype.columns.get_loc(k)] = scores_batch
                            
                        elif self.config.scoring_method == ScoringMethod.SENS_FPR_ZERO:
                            max_indices = np.argmax(beta, axis=1)
                            scores_batch = np.array([1-alpha[i, max_indices[i]] for i in range(batch_size_actual)])
                            scores_ctype.iloc[batch_start:batch_end, scores_ctype.columns.get_loc(k)] = scores_batch
                            
                        elif self.config.scoring_method == ScoringMethod.SENS_PPV_99:
                            ppv = np.nan_to_num((1-alpha)/(2-alpha-beta))
                            scores_batch = np.zeros(batch_size_actual)
                            
                            for i in range(batch_size_actual):
                                ppv_exceeds = ppv[i, :] > 0.99
                                if np.any(ppv_exceeds):
                                    first_idx = np.argmax(ppv_exceeds)
                                    scores_batch[i] = 1 - alpha[i, first_idx]
                            
                            scores_ctype.iloc[batch_start:batch_end, scores_ctype.columns.get_loc(k)] = scores_batch
                    
                    else:
                        # Only some genes are promising, process selectively
                        promising_indices = np.where(promising_genes)[0]
                        # print(f"  Processing {len(promising_indices)} promising genes")
                        
                        # Allocate arrays only for promising genes
                        alpha_promising = np.zeros(shape=(promising_count, cutoff_k))
                        beta_promising = np.zeros(shape=(promising_count, cutoff_k))
                        
                        # Copy sparse results for promising genes
                        for i, l in enumerate(sparse_points):
                            alpha_promising[:, l] = alpha_sparse[promising_indices, i]
                            beta_promising[:, l] = beta_sparse[promising_indices, i]
                        
                        # Get parameters for promising genes
                        p2_promising = p2_batch[promising_indices]
                        p3_promising = p3_batch[promising_indices]
                        
                        # Calculate remaining points for promising genes
                        remaining_points = np.setdiff1d(np.arange(cutoff_k), sparse_points)
                        for l in remaining_points:
                            alpha_promising[:, l] = binom.cdf(l, int(parameters[0][ctype] * k), p2_promising)
                            beta_promising[:, l] = binom.cdf(l, int(parameters[1][ctype] * k), p3_promising)
                        
                        # Calculate scores based on the scoring method (only for promising genes)
                        scores_promising = np.zeros(promising_count)
                        
                        if self.config.scoring_method == ScoringMethod.DIFF:
                            diff = beta_promising - alpha_promising
                            max_indices = np.argmax(diff, axis=1)
                            scores_promising = np.array([diff[i, max_indices[i]] for i in range(promising_count)])
                            
                        elif self.config.scoring_method == ScoringMethod.SENS_FPR_ZERO:
                            max_indices = np.argmax(beta_promising, axis=1)
                            scores_promising = np.array([1-alpha_promising[i, max_indices[i]] for i in range(promising_count)])
                            
                        elif self.config.scoring_method == ScoringMethod.SENS_PPV_99:
                            ppv = np.nan_to_num((1-alpha_promising)/(2-alpha_promising-beta_promising))
                            
                            for i in range(promising_count):
                                ppv_exceeds = ppv[i, :] > 0.99
                                if np.any(ppv_exceeds):
                                    first_idx = np.argmax(ppv_exceeds)
                                    scores_promising[i] = 1 - alpha_promising[i, first_idx]
                        
                        # Assign scores for promising genes to the full scores matrix
                        for i, idx in enumerate(promising_indices):
                            scores_ctype.iloc[batch_start + idx, scores_ctype.columns.get_loc(k)] = scores_promising[i]
            
            scores[ctype] = scores_ctype
        
        # Check for and report any anomalies in scores
        for ctype in scores:
            zero_scores = (scores[ctype] == 0).all(axis=1).sum()
            if zero_scores > 0:
                print(f"Warning: {ctype} has {zero_scores} genes with all zero scores")
        
        return scores, parameters[2]
        

    def _calculate_theoretical_scores_parallel2(self) -> Tuple[Dict, Dict]:
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
            
            logger.info(f"Processing batch of {len(current_batch)} cell types with {n_jobs} workers")
            
            # Use memory-efficient options in Parallel
            batch_results = Parallel(
                n_jobs=n_jobs, 
                max_nbytes='50M',  # Limit memory per job
                prefer="threads",   # Use threads for better memory sharing
                verbose=10          # Show progress
            )(delayed(process_cell_type)(ctype) for ctype in current_batch)
            
            # Add batch results to scores dictionary
            for ctype, score in batch_results:
                scores[ctype] = score
            
            # Clean up batch results to free memory
            del batch_results
            gc.collect()
        
        logger.info("Completed theoretical score calculation")
        return scores, parameters[2]

    def _calculate_theoretical_scores_parallel(self) -> Tuple[Dict, Dict]:
        """
        Calculate theoretical scores using joblib for parallelization
        
        Returns:
            Tuple[Dict, Dict]: Scores and expression proportions
        """
        from joblib import Parallel, delayed
        
        # Estimate binomial distribution parameters
        parameters = self._estimate_binomial_parameters()
        
        def process_cell_type(ctype):
            """Process a single cell type"""
            # Initialize score matrix (rows=genes, columns=k_values)
            scores_ctype = pd.DataFrame(
                np.zeros((self.adata.shape[1], len(self.config.k_values))),
                index=self.adata.var.index,
                columns=self.config.k_values
            )
            
            # Calculate scores for each k value
            for k in self.config.k_values:
                # Calculate cutoffs for computation
                cutoffs_k = pd.DataFrame([
                    # Target cells
                    binom.ppf(.99, int(parameters[0][ctype] * k), parameters[2][ctype]),
                    # Non-target cells
                    binom.ppf(.99, int(parameters[1][ctype] * k), parameters[3][ctype])
                ], columns=scores_ctype.index).max().clip(lower=100)
                
                # Calculate alpha (CDF for target) and beta (CDF for non-target)
                alpha = (cutoffs_k.values[:, None] * np.arange(101) // 100)
                beta = alpha.copy()
                
                # Calculate CDFs
                for l in np.arange(101):
                    alpha[:, l] = binom.cdf(alpha[:, l], int(parameters[0][ctype] * k), parameters[2][ctype])
                    beta[:, l] = binom.cdf(beta[:, l], int(parameters[1][ctype] * k), parameters[3][ctype])
                
                # Compute scores based on scoring method
                if self.config.scoring_method == ScoringMethod.DIFF:
                    # Max difference between sensitivity and FPR
                    scores_ctype[k] = (beta - alpha)[np.arange(self.adata.shape[1]), np.argmax(beta - alpha, axis=1)]
                elif self.config.scoring_method == ScoringMethod.SENS_FPR_ZERO:
                    # Sensitivity at zero FPR
                    scores_ctype[k] = (1 - alpha)[np.arange(self.adata.shape[1]), np.argmax(beta, axis=1)]
                elif self.config.scoring_method == ScoringMethod.SENS_PPV_99:
                    # Sensitivity at PPV > 99%
                    ppv = np.nan_to_num((1 - alpha) / (2 - alpha - beta))
                    scores_ctype[k] = (
                        (1 - alpha)[np.arange(self.adata.shape[1]), np.argmax(ppv > 0.99, axis=1)] *
                        (np.sum(ppv > 0.99, axis=1) > 0)
                    )
            gc.collect()
            return ctype, scores_ctype
        
        # Determine number of jobs (cores)
        n_jobs = self.config.n_jobs  
        
        # Process in parallel
        logger.info(f"Starting parallel processing with joblib")
        results = Parallel(n_jobs=n_jobs, mmap_mode='r')(
            delayed(process_cell_type)(ctype) for ctype in self.cell_types
        )
        gc.collect()
        # Collect results
        scores = {ctype: score for ctype, score in results}
        
        return scores, parameters[2]

    
    def _calculate_theoretical_scores(self) -> Tuple[Dict, Dict]:
        """
        Calculate theoretical scores based on binomial distribution
        
        Returns:
            Tuple[Dict, Dict]: Scores and expression proportions
        """
        # Estimate binomial distribution parameters
        parameters = self._estimate_binomial_parameters()

        scores = {}
        # Calculate scores for each cell type
        for ctype in self.cell_types:
            # Initialize score matrix (rows=genes, columns=k_values)
            scores_ctype = pd.DataFrame(
                np.zeros((self.adata.shape[1], len(self.config.k_values))),
                index=self.adata.var.index, 
                columns=self.config.k_values
            )
            
            # Calculate scores for each k value
            for k in self.config.k_values:
                # Calculate cutoffs for computation
                cutoffs_k = pd.DataFrame([
                    # Target cells
                    binom.ppf(.99, int(parameters[0][ctype] * k), parameters[2][ctype]),
                    # Non-target cells
                    binom.ppf(.99, int(parameters[1][ctype] * k), parameters[3][ctype])
                ], columns=scores_ctype.index).max().clip(lower=100)
                
                # Calculate alpha (CDF for target) and beta (CDF for non-target)
                alpha = (cutoffs_k.values[:, None] * np.arange(101) // 100)
                beta = alpha.copy()
                
                # Calculate CDFs
                for l in np.arange(101):
                    alpha[:, l] = binom.cdf(alpha[:, l], int(parameters[0][ctype] * k), parameters[2][ctype])
                    beta[:, l] = binom.cdf(beta[:, l], int(parameters[1][ctype] * k), parameters[3][ctype])
                
                # Compute scores based on scoring method
                if self.config.scoring_method == ScoringMethod.DIFF:
                    # Max difference between sensitivity and FPR
                    scores_ctype[k] = (beta - alpha)[np.arange(self.adata.shape[1]), np.argmax(beta - alpha, axis=1)]
                    
                elif self.config.scoring_method == ScoringMethod.SENS_FPR_ZERO:
                    # Sensitivity at zero FPR
                    scores_ctype[k] = (1 - alpha)[np.arange(self.adata.shape[1]), np.argmax(beta, axis=1)]
                    
                elif self.config.scoring_method == ScoringMethod.SENS_PPV_99:
                    # Sensitivity at PPV > 99%
                    ppv = np.nan_to_num((1 - alpha) / (2 - alpha - beta))
                    scores_ctype[k] = (
                        (1 - alpha)[np.arange(self.adata.shape[1]), np.argmax(ppv > 0.99, axis=1)] *
                        (np.sum(ppv > 0.99, axis=1) > 0)
                    )
            
            scores[ctype] = scores_ctype
            
        return scores, parameters[2]


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
    

    import numpy as np
    import pandas as pd
    import scipy
    import numba as nb
    from statsmodels.distributions.empirical_distribution import ECDF

    # Numba-optimized function for computing CDFs for all genes at a specific threshold
    @nb.jit(nopython=True, parallel=True)
    def compute_cdfs_at_threshold(expression_matrix, threshold):
        """
        Compute empirical CDF at a specific threshold for all genes.
        
        Args:
            expression_matrix: NumPy array of shape (n_simulations, n_genes)
            threshold: Expression threshold to evaluate
            
        Returns:
            NumPy array of CDF values for each gene
        """
        n_sim, n_genes = expression_matrix.shape
        result = np.zeros(n_genes)
        
        for g in nb.prange(n_genes):
            count = 0
            for s in range(n_sim):
                if expression_matrix[s, g] <= threshold:
                    count += 1
            result[g] = count / n_sim
            
        return result

    # Numba-optimized function for computing scores
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

    def empirical_scores_optimized_batch(self, n_sim=1000):
        """
        Calculate empirical marker gene scores through simulation using optimized statistical libraries.
        
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
        
        # Iterate through each cell type
        for ctype in cell_types:
            # Skip cell types with no filtered genes
            if filtered_scores[ctype].empty:
                scores_all[ctype] = pd.DataFrame([])
                continue
                
            # Initialize score matrix (rows=genes, columns=k_values)
            scores_ctype = pd.DataFrame(np.zeros((len(gene_names), len(k_values))), 
                                    index=gene_names, columns=k_values)
            
            # Get indices of target and non-target cells
            target_indices = np.where(adata.obs[column_ctype] == ctype)[0]
            nontarget_indices = np.where(adata.obs[column_ctype] != ctype)[0]
            
            # Get the gene indices as a list for easier slicing
            gene_idx_list = [gene_indices[gene] for gene in gene_names]
            
            # Calculate scores for each k value
            for k_idx, k in enumerate(k_values):
                # For all k values, perform random sampling consistently
                # Initialize arrays for expression sums
                target_sums = np.zeros((n_sim, len(gene_names)))
                nontarget_sums = np.zeros((n_sim, len(gene_names)))
                
                # Process in manageable batches to reduce memory usage
                batch_size = min(100, n_sim)  # Adjust based on available memory
                
                # Calculate batch counts and indices
                if k > 1:
                    for batch_start in range(0, n_sim, batch_size):
                        batch_end = min(batch_start + batch_size, n_sim)
                        batch_size_actual = batch_end - batch_start
                        
                        # Sample k target cells for each simulation in this batch
                        for sim_idx in range(batch_start, batch_end):
                            # Sample k cells for this simulation
                            sampled_cells = np.random.choice(target_indices, k, replace=True)
                            
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
                            sampled_cells = np.random.choice(nontarget_indices, k, replace=True)
                            
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
                            cell_idx = np.random.choice(target_indices)
                            
                            # Extract expression for this cell
                            if is_sparse:
                                target_sums[sim_idx] = adata.X[cell_idx, gene_idx_list].toarray().flatten()
                            else:
                                target_sums[sim_idx] = adata.X[cell_idx, gene_idx_list]
                        
                        # Sample non-target cells for each simulation in this batch
                        for sim_idx in range(batch_start, batch_end):
                            # Sample 1 cell for this simulation
                            cell_idx = np.random.choice(nontarget_indices)
                            
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
                    scores_ctype.iloc[:, k_idx] = compute_diff_scores(alpha, beta)
                elif scoring == 'sensFPRzero':
                    scores_ctype.iloc[:, k_idx] = compute_sensFPRzero_scores(alpha, beta)
                elif scoring == 'sensPPV99':
                    scores_ctype.iloc[:, k_idx] = compute_sensPPV99_scores(alpha, beta)
            
            # Store scores for this cell type
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
                    elif scoring == 'sensFPRzero':
                        scores = compute_sensFPRzero_scores(alpha, beta)
                    elif scoring == 'sensPPV99':
                        scores = compute_sensPPV99_scores(alpha, beta)
                        
                    # Assign scores to the dataframe
                    scores_ctype.iloc[:, k_idx] = scores
                gc.collect()
                return ctype, scores_ctype
            
            # Process cell types in parallel
            results = Parallel(n_jobs=n_jobs)(
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
    def empirical_scores_optimized_batch_parallel(self, n_sim=1000):
        """
        Calculate empirical marker gene scores through simulation using optimized statistical libraries
        with parallel processing.
        
        Args:
            n_sim: Number of simulations to run (default: 1000)
            n_jobs: Number of jobs for parallel processing. -1 means using all processors (default: -1)
                
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
            
            # Define a worker function for the k-value parallelization
            def process_k_value(k_idx_k):
                k_idx, k = k_idx_k
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
                elif scoring == 'sensFPRzero':
                    scores = compute_sensFPRzero_scores(alpha, beta)
                elif scoring == 'sensPPV99':
                    scores = compute_sensPPV99_scores(alpha, beta)
                    
                return k_idx, scores
            
            # Process k values in parallel
            k_idx_scores = Parallel(n_jobs=n_jobs)(
                delayed(process_k_value)(k_idx_k) 
                for k_idx_k in enumerate(k_values)
            )
            
            # Assign results to the scores dataframe
            for k_idx, scores in k_idx_scores:
                scores_ctype.iloc[:, k_idx] = scores
            
            return ctype, scores_ctype
        
        # Process cell types in parallel
        results = Parallel(n_jobs=n_jobs)(
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

    def empirical_scores_optimized(self, n_sim=1000):
        """
        Calculate empirical marker gene scores through simulation without gene batching.
        
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
        
        print(f"Starting empirical score calculation with {n_sim} simulations")
        print(f"Scoring method: {scoring}")
        
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
        print(f"Processing {len(all_genes)} genes")
        
        # Create gene index mapping (gene_name -> position in adata.var_names)
        gene_indices = {gene: i for i, gene in enumerate(adata.var_names) if gene in all_genes}
        gene_names = list(gene_indices.keys())
        
        # Dictionary to store scores for all cell types
        scores_all = {}
        
        # Determine if data is sparse
        is_sparse = scipy.sparse.issparse(adata.X)
        
        # Iterate through each cell type
        for ctype in cell_types:
            print(f"Processing cell type: {ctype}")
            # Skip cell types with no filtered genes
            if filtered_scores[ctype].empty:
                scores_all[ctype] = pd.DataFrame([])
                continue
                
            # Initialize score matrix (rows=genes, columns=k_values)
            scores_ctype = pd.DataFrame(np.zeros((len(gene_names), len(k_values))), 
                                    index=gene_names, columns=k_values)
            
            # Get indices of target and non-target cells
            target_indices = np.where(adata.obs[column_ctype] == ctype)[0]
            nontarget_indices = np.where(adata.obs[column_ctype] != ctype)[0]
            print(f"  Target cells: {len(target_indices)}, Non-target cells: {len(nontarget_indices)}")
            
            # Get the gene indices as a list for easier slicing
            gene_idx_list = [gene_indices[gene] for gene in gene_names]
            
            # Calculate scores for each k value
            for k_idx, k in enumerate(k_values):
                print(f"  Processing k={k}")
                # Initialize arrays for expression sums
                target_sums = np.zeros((n_sim, len(gene_names)))
                nontarget_sums = np.zeros((n_sim, len(gene_names)))
                
                # Process all simulations at once for each k value
                if k > 1:
                    # Sample cells for all simulations at once
                    for sim_idx in range(n_sim):
                        # Sample k target cells
                        sampled_target_cells = np.random.choice(target_indices, k, replace=True)
                        
                        # Extract and sum expression for these cells
                        for cell_idx in sampled_target_cells:
                            if is_sparse:
                                expr = adata.X[cell_idx, gene_idx_list].toarray().flatten()
                            else:
                                expr = adata.X[cell_idx, gene_idx_list]
                            target_sums[sim_idx] += expr
                        
                        # Sample k non-target cells
                        sampled_nontarget_cells = np.random.choice(nontarget_indices, k, replace=True)
                        
                        # Extract and sum expression for these cells
                        for cell_idx in sampled_nontarget_cells:
                            if is_sparse:
                                expr = adata.X[cell_idx, gene_idx_list].toarray().flatten()
                            else:
                                expr = adata.X[cell_idx, gene_idx_list]
                            nontarget_sums[sim_idx] += expr
                else:  # k == 1
                    # For k=1, sample cells for all simulations at once
                    sampled_target_cells = np.random.choice(target_indices, n_sim, replace=True)
                    sampled_nontarget_cells = np.random.choice(nontarget_indices, n_sim, replace=True)
                    
                    # Extract expression for all sampled cells
                    for sim_idx in range(n_sim):
                        if is_sparse:
                            target_sums[sim_idx] = adata.X[sampled_target_cells[sim_idx], gene_idx_list].toarray().flatten()
                            nontarget_sums[sim_idx] = adata.X[sampled_nontarget_cells[sim_idx], gene_idx_list].toarray().flatten()
                        else:
                            target_sums[sim_idx] = adata.X[sampled_target_cells[sim_idx], gene_idx_list]
                            nontarget_sums[sim_idx] = adata.X[sampled_nontarget_cells[sim_idx], gene_idx_list]
                
                # Check if we have any data
                if target_sums.size == 0 or nontarget_sums.size == 0:
                    print("  WARNING: Empty expression data for this cell type")
                    continue
                    
                # Check for NaN values before computing percentiles
                if np.isnan(target_sums).any() or np.isnan(nontarget_sums).any():
                    print("  WARNING: NaN values detected in expression data")
                    # Replace NaNs with zeros for now
                    target_sums = np.nan_to_num(target_sums)
                    nontarget_sums = np.nan_to_num(nontarget_sums)
                
                # Determine cutoff value for CDF calculation (99th percentile of the largest gene)
                cutoff_k = int(max(
                    np.percentile(target_sums, 99, axis=0).max(),
                    np.percentile(nontarget_sums, 99, axis=0).max(),
                    1
                ))
                print(f"  Cutoff value: {cutoff_k}")
                
                # Calculate alpha and beta matrices using vectorized operations
                alpha = np.zeros((len(gene_names), cutoff_k))
                beta = np.zeros((len(gene_names), cutoff_k))
                
                print(f"  Computing CDFs for {len(gene_names)} genes at {cutoff_k} thresholds")
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
                
                print(f"  Computing scores using method: {scoring}")
                # Calculate scores based on specified scoring method
                if scoring == ScoringMethod.DIFF:
                    # Use the standalone function
                    scores_ctype.iloc[:, k_idx] = compute_diff_scores(alpha, beta)
                    print("dif ussed")
                    
                elif scoring == 'sensFPRzero':
                    # Use the standalone function
                    scores_ctype.iloc[:, k_idx] = compute_sensFPRzero_scores(alpha, beta)
                    
                elif scoring == 'sensPPV99':
                    # Use the standalone function
                    scores_ctype.iloc[:, k_idx] = compute_sensPPV99_scores(alpha, beta)

                # Check for zeros in scores
                num_zeros = (scores_ctype.iloc[:, k_idx] == 0).sum()
                print(f"  Number of zero scores: {num_zeros} out of {len(gene_names)}")
                
                # Print some example scores
                sample_scores = scores_ctype.iloc[:5, k_idx].values
                print(f"  Sample scores: {sample_scores}")
            
            # Store scores for this cell type
            scores_all[ctype] = scores_ctype
            print(scores_ctype)
        
        print("Applying multi-category correction")
        self.score_all_to_remove = scores_all
        # Apply multi-category correction to normalize scores
        corr_scores = self.multi_cat_correction_for_optimized(scores_all)
        print(corr_scores)
        # Filter scores to include only genes from filtered_scores
        scores = {}
        for ctype in cell_types:
            if filtered_scores[ctype].empty:
                scores[ctype] = pd.DataFrame([])
            else:
                # Extract scores for filtered genes and clip values to max of 1
                genes_to_keep = filtered_scores[ctype].index
                try:
                    scores[ctype] = corr_scores[ctype].loc[genes_to_keep].clip(upper=1)
                    # Print a sample of scores
                    sample_score = scores[ctype].iloc[:5, 0].values if not scores[ctype].empty else []
                    print(f"Cell type {ctype} - sample final scores: {sample_score}")
                except KeyError as e:
                    print(f"KeyError for cell type {ctype}: {e}")
                    missing_genes = [gene for gene in genes_to_keep if gene not in corr_scores[ctype].index]
                    print(f"Missing genes: {missing_genes[:5]}...")
                    scores[ctype] = pd.DataFrame([])
        print(scores)
        return scores
    
    def empirical_cdf_v0(self, df, value):
        #empirical cdf computation (all genes must be evaluated on the same number of pointd)
        return df.apply(lambda col: percentileofscore(col, value, kind='rank') / 100)
    

    
    def empirical_scores_v0_optimized_parallel(self, filtered_scores):
        """
        Calculate empirical marker gene scores through simulation with sparse sampling optimization,
        using parallel processing for improved performance.
        
        Args:
            filtered_scores: Dictionary of filtered scores by cell type
                
        Returns:
            Dictionary of empirical scores by cell type
        """
        from joblib import Parallel, delayed
        import time
        
        start_time = time.time()
        logger.info("Starting parallel empirical score calculation")
        
        # Initialize an empty list to collect genes of interest
        gene_list = []
        
        # Set random seed for reproducibility
        np.random.seed(self.config.random_seed)
        
        # Collect all genes from filtered_scores across all cell types
        for ctype in self.adata.obs[self.column_ctype].unique():
            if not filtered_scores[ctype].empty:
                gene_list += filtered_scores[ctype].index.tolist()
        
        # Remove duplicates while preserving order
        gene_list = list(dict.fromkeys(gene_list))
        logger.info(f"Computing empirical scores for {len(gene_list)} genes")
        
        # Create a subset of the AnnData object containing only genes of interest
        self.adata = self.adata[:, gene_list]
        
        # Define a function to process a single cell type
        def process_cell_type(ctype):
            logger.debug(f"Processing empirical scores for cell type: {ctype}")
            cell_start_time = time.time()
            
            # Initialize score matrix (rows=genes, columns=k_values)
            scores_ctype = np.zeros(shape=(self.adata.shape[1], len(self.config.k_values)))
            scores_ctype = pd.DataFrame(scores_ctype, index=self.adata.var.index, columns=self.config.k_values)
            
            # Extract expression data for target and non-target cells
            expr_A = sc.get.obs_df(self.adata[self.adata.obs[self.column_ctype]==ctype], 
                                self.adata.var.index.tolist())
            expr_nA = sc.get.obs_df(self.adata[self.adata.obs[self.column_ctype]!=ctype], 
                                self.adata.var.index.tolist())
            
            # Calculate scores for each k value
            for k in self.config.k_values:
                if k > 1:
                    # Sample k cells from target population n_sim times
                    sampled_A = np.random.choice(expr_A.shape[0], (self.config.n_simulations, k), replace=True)
                    k_sums_A = pd.DataFrame(expr_A.values[sampled_A].sum(axis=1), columns=expr_A.columns)
                    
                    # Sample k cells from non-target population n_sim times
                    sampled_nA = np.random.choice(expr_nA.shape[0], (self.config.n_simulations, k), replace=True)
                    k_sums_nA = pd.DataFrame(expr_nA.values[sampled_nA].sum(axis=1), columns=expr_nA.columns)
                else:
                    # For k=1, no aggregation needed, use raw expression values
                    k_sums_A = expr_A
                    k_sums_nA = expr_nA
                
                # Determine cutoff value for CDF calculation (99th percentile of the largest gene)
                cutoff_k = np.max([k_sums_A.quantile(0.99).max(),
                                k_sums_nA.quantile(0.99).max(), 1])
                cutoff_k = int(cutoff_k)
                
                # Use sparse sampling with step size from config
                sparse_points = np.arange(0, cutoff_k, self.config.sparse_step)
                if len(sparse_points) == 0:  # Ensure at least one point
                    sparse_points = np.array([0])
                
                # Initialize arrays for sparse CDFs
                alpha_sparse = np.zeros(shape=(self.adata.shape[1], len(sparse_points)))
                beta_sparse = np.zeros(shape=(self.adata.shape[1], len(sparse_points)))
                
                # Calculate sparse CDFs
                for i, l in enumerate(sparse_points):
                    alpha_sparse[:, i] = self.empirical_cdf_v0(k_sums_A, l).values
                    beta_sparse[:, i] = self.empirical_cdf_v0(k_sums_nA, l).values
                
                # Identify promising genes based on scoring method
                promising_genes = np.ones(self.adata.shape[1], dtype=bool)  # Default all genes to promising
                
                if self.config.scoring_method == ScoringMethod.DIFF:
                    sparse_diff = beta_sparse - alpha_sparse
                    max_sparse_diff = np.max(sparse_diff, axis=1)
                    promising_genes = max_sparse_diff > self.config.promising_threshold
                    
                elif self.config.scoring_method == ScoringMethod.SENS_FPR_ZERO:
                    # For sensFPRzero, we want genes where beta can reach high values
                    max_beta = np.max(beta_sparse, axis=1)
                    promising_genes = max_beta > (1 - self.config.promising_threshold)
                    
                elif self.config.scoring_method == ScoringMethod.SENS_PPV_99:
                    # For sensPPV99, we want genes where PPV can exceed 0.99
                    # Calculate sparse PPV
                    ppv_sparse = np.nan_to_num((1-alpha_sparse)/(2-alpha_sparse-beta_sparse))
                    # Check if any value exceeds 0.99
                    promising_genes = np.max(ppv_sparse > 0.99, axis=1) > 0
                
                # Initialize full arrays for alpha and beta
                alpha = np.zeros(shape=(self.adata.shape[1], int(cutoff_k)))
                beta = np.zeros(shape=(self.adata.shape[1], int(cutoff_k)))
                
                # Copy sparse results into the full arrays to avoid recalculation
                for i, l in enumerate(sparse_points):
                    alpha[:, l] = alpha_sparse[:, i]
                    beta[:, l] = beta_sparse[:, i]
                
                # Calculate remaining points only for promising genes
                remaining_points = np.setdiff1d(np.arange(cutoff_k), sparse_points)
                
                for l in remaining_points:
                    # Calculate alpha, but only for promising genes
                    if np.any(promising_genes):
                        if promising_genes.all():  # If all genes are promising, calculate for all
                            alpha[:, l] = self.empirical_cdf_v0(k_sums_A, l).values
                            beta[:, l] = self.empirical_cdf_v0(k_sums_nA, l).values
                        else:  # Otherwise calculate only for promising genes
                            # Get indices of promising genes
                            promising_indices = np.where(promising_genes)[0]
                            
                            # Calculate for promising genes only
                            promising_gene_names = self.adata.var.index[promising_indices]
                            k_sums_A_promising = k_sums_A[promising_gene_names]
                            k_sums_nA_promising = k_sums_nA[promising_gene_names]
                            
                            alpha_promising = self.empirical_cdf_v0(k_sums_A_promising, l).values
                            beta_promising = self.empirical_cdf_v0(k_sums_nA_promising, l).values
                            
                            # Assign values back to the right positions
                            alpha[promising_indices, l] = alpha_promising
                            beta[promising_indices, l] = beta_promising
                
                # Calculate scores based on the scoring method
                if self.config.scoring_method == ScoringMethod.DIFF:
                    scores_ctype[k] = (beta-alpha)[np.arange(self.adata.shape[1]), np.argmax(beta-alpha, axis=1)]
                elif self.config.scoring_method == ScoringMethod.SENS_FPR_ZERO:
                    scores_ctype[k] = (1-alpha)[np.arange(self.adata.shape[1]), np.argmax(beta, axis=1)]
                elif self.config.scoring_method == ScoringMethod.SENS_PPV_99:
                    ppv = np.nan_to_num((1-alpha)/(2-alpha-beta))
                    scores_ctype[k] = ((1-alpha)[np.arange(self.adata.shape[1]), 
                                    np.argmax(ppv>0.99, axis=1)]) * (np.sum(ppv>0.99, axis=1)>0)
            
            cell_elapsed_time = time.time() - cell_start_time
            logger.debug(f"Completed empirical scores for {ctype} in {cell_elapsed_time:.2f} seconds")
            return ctype, pd.DataFrame(scores_ctype)
        
        # Get list of cell types to process
        cell_types_to_process = [ctype for ctype in self.adata.obs[self.column_ctype].unique()]
        
        # Process cell types in parallel using joblib
        n_jobs = self.config.n_jobs 
        logger.info(f"Starting parallel processing with joblib")
        
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_cell_type)(ctype) for ctype in cell_types_to_process
        )
        
        # Convert results to dictionary
        scores_all = {ctype: score for ctype, score in results}
        
        # Apply multi-category correction and filter
        corr_scores = self._apply_multi_category_correction(scores_all)
        scores = {}
        for ctype in self.adata.obs[self.column_ctype].unique():
            if filtered_scores[ctype].empty:
                scores[ctype] = pd.DataFrame([])
            else:
                scores[ctype] = corr_scores[ctype].loc[filtered_scores[ctype].index].clip(upper=1)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Completed parallel empirical score calculation in {elapsed_time:.2f} seconds")
        
        return scores
    
    def empirical_scores_v0_optimized(self, filtered_scores):
        """
        Calculate empirical marker gene scores through simulation with sparse sampling optimization.
        Uses config parameters for sparse_step and promising_threshold.
        
        Args:
            filtered_scores: Dictionary of filtered scores by cell type
                
        Returns:
            Dictionary of empirical scores by cell type
        """
        # Initialize an empty list to collect genes of interest
        gene_list = []
        
        # Set random seed for reproducibility
        np.random.seed(self.config.random_seed)
        
        # Dictionary to store scores for all cell types
        scores_all = {}
        
        # Collect all genes from filtered_scores across all cell types
        for ctype in self.adata.obs[self.column_ctype].unique():
            if not filtered_scores[ctype].empty:
                gene_list += filtered_scores[ctype].index.tolist()
        logger.info(f"Computing empirical scores for {len(gene_list)} genes")
        
        # Create a subset of the AnnData object containing only genes of interest
        adata_subset = self.adata[:, gene_list].copy()
        
        # Iterate through each cell type
        for ctype in self.adata.obs[self.column_ctype].unique():
            # Initialize score matrix (rows=genes, columns=k_values)
            scores_ctype = np.zeros(shape=(adata_subset.shape[1], len(self.config.k_values)))
            scores_ctype = pd.DataFrame(scores_ctype, index=adata_subset.var.index, columns=self.config.k_values)
            
            # Extract expression data for target and non-target cells
            expr_A = sc.get.obs_df(adata_subset[adata_subset.obs[self.column_ctype]==ctype], 
                                adata_subset.var.index.tolist())
            expr_nA = sc.get.obs_df(adata_subset[adata_subset.obs[self.column_ctype]!=ctype], 
                                adata_subset.var.index.tolist())
            
            # Calculate scores for each k value
            for k in self.config.k_values:
                if k > 1:
                    # Sample k cells from target population n_sim times
                    sampled_A = np.random.choice(expr_A.shape[0], (self.config.n_simulations, k), replace=True)
                    k_sums_A = pd.DataFrame(expr_A.values[sampled_A].sum(axis=1), columns=expr_A.columns)
                    
                    # Sample k cells from non-target population n_sim times
                    sampled_nA = np.random.choice(expr_nA.shape[0], (self.config.n_simulations, k), replace=True)
                    k_sums_nA = pd.DataFrame(expr_nA.values[sampled_nA].sum(axis=1), columns=expr_nA.columns)
                else:
                    # For k=1, no aggregation needed, use raw expression values
                    k_sums_A = expr_A
                    k_sums_nA = expr_nA
                
                # Determine cutoff value for CDF calculation (99th percentile of the largest gene)
                cutoff_k = np.max([k_sums_A.quantile(0.99).max(),
                                k_sums_nA.quantile(0.99).max(), 1])
                cutoff_k = int(cutoff_k)
                
                # Use sparse sampling with step size from config
                sparse_points = np.arange(0, cutoff_k, self.config.sparse_step)
                if len(sparse_points) == 0:  # Ensure at least one point
                    sparse_points = np.array([0])
                
                # Initialize arrays for sparse CDFs
                alpha_sparse = np.zeros(shape=(adata_subset.shape[1], len(sparse_points)))
                beta_sparse = np.zeros(shape=(adata_subset.shape[1], len(sparse_points)))
                
                # Calculate sparse CDFs
                for i, l in enumerate(sparse_points):
                    alpha_sparse[:, i] = self.empirical_cdf_v0(k_sums_A, l).values
                    beta_sparse[:, i] = self.empirical_cdf_v0(k_sums_nA, l).values
                
                # Identify promising genes based on scoring method
                promising_genes = np.ones(adata_subset.shape[1], dtype=bool)  # Default all genes to promising
                
                if self.config.scoring_method == ScoringMethod.DIFF:
                    sparse_diff = beta_sparse - alpha_sparse
                    max_sparse_diff = np.max(sparse_diff, axis=1)
                    promising_genes = max_sparse_diff > self.config.promising_threshold
                    
                elif self.config.scoring_method == ScoringMethod.SENS_FPR_ZERO:
                    # For sensFPRzero, we want genes where beta can reach high values
                    max_beta = np.max(beta_sparse, axis=1)
                    promising_genes = max_beta > (1 - self.config.promising_threshold)
                    
                elif self.config.scoring_method == ScoringMethod.SENS_PPV_99:
                    # For sensPPV99, we want genes where PPV can exceed 0.99
                    # Calculate sparse PPV
                    ppv_sparse = np.nan_to_num((1-alpha_sparse)/(2-alpha_sparse-beta_sparse))
                    # Check if any value exceeds 0.99
                    promising_genes = np.max(ppv_sparse > 0.99, axis=1) > 0
                
                # Initialize full arrays for alpha and beta
                alpha = np.zeros(shape=(adata_subset.shape[1], int(cutoff_k)))
                beta = np.zeros(shape=(adata_subset.shape[1], int(cutoff_k)))
                
                # Copy sparse results into the full arrays to avoid recalculation
                for i, l in enumerate(sparse_points):
                    alpha[:, l] = alpha_sparse[:, i]
                    beta[:, l] = beta_sparse[:, i]
                
                # Calculate remaining points only for promising genes
                remaining_points = np.setdiff1d(np.arange(cutoff_k), sparse_points)
                
                for l in remaining_points:
                    # Calculate alpha, but only for promising genes
                    if np.any(promising_genes):
                        if promising_genes.all():  # If all genes are promising, calculate for all
                            alpha[:, l] = self.empirical_cdf_v0(k_sums_A, l).values
                            beta[:, l] = self.empirical_cdf_v0(k_sums_nA, l).values
                        else:  # Otherwise calculate only for promising genes
                            # Get indices of promising genes
                            promising_indices = np.where(promising_genes)[0]
                            
                            # Calculate for promising genes only
                            promising_gene_names = adata_subset.var.index[promising_indices]
                            k_sums_A_promising = k_sums_A[promising_gene_names]
                            k_sums_nA_promising = k_sums_nA[promising_gene_names]
                            
                            alpha_promising = self.empirical_cdf_v0(k_sums_A_promising, l).values
                            beta_promising = self.empirical_cdf_v0(k_sums_nA_promising, l).values
                            
                            # Assign values back to the right positions
                            alpha[promising_indices, l] = alpha_promising
                            beta[promising_indices, l] = beta_promising
                
                # Calculate scores based on the scoring method
                if self.config.scoring_method == ScoringMethod.DIFF:
                    scores_ctype[k] = (beta-alpha)[np.arange(adata_subset.shape[1]), np.argmax(beta-alpha, axis=1)]
                elif self.config.scoring_method == ScoringMethod.SENS_FPR_ZERO:
                    scores_ctype[k] = (1-alpha)[np.arange(adata_subset.shape[1]), np.argmax(beta, axis=1)]
                elif self.config.scoring_method == ScoringMethod.SENS_PPV_99:
                    ppv = np.nan_to_num((1-alpha)/(2-alpha-beta))
                    scores_ctype[k] = ((1-alpha)[np.arange(adata_subset.shape[1]), 
                                    np.argmax(ppv>0.99, axis=1)]) * (np.sum(ppv>0.99, axis=1)>0)
            
            scores_all[ctype] = pd.DataFrame(scores_ctype)
        
        # Apply multi-category correction and filter
        corr_scores = self._apply_multi_category_correction(scores_all)
        scores = {}
        for ctype in self.adata.obs[self.column_ctype].unique():
            if filtered_scores[ctype].empty:
                scores[ctype] = pd.DataFrame([])
            else:
                scores[ctype] = corr_scores[ctype].loc[filtered_scores[ctype].index].clip(upper=1)
        
        return scores

    def empirical_scores_v0(self, filtered_scores, adata, column_ctype, column_patient, k_values, scoring, seed=0, n_sim=1000):
        """
        Calculate empirical marker gene scores through simulation.
        
        This function computes empirical scores for candidate marker genes by randomly sampling cells
        and evaluating their expression patterns across different cell types.
        
        Args:
            filtered_scores: Dictionary of filtered scores by cell type
            adata: AnnData object containing gene expression data
            column_ctype: Column name in adata.obs for cell type annotations
            column_patient: Column name in adata.obs for patient IDs
            k_values: List of k values (sampling sizes) to test
            scoring: Scoring method ('diff', 'sensFPRzero', or 'sensPPV99')
            seed: Random seed for reproducibility (default: 0)
            n_sim: Number of simulations to run (default: 1000)
            
        Returns:
            Dictionary of empirical scores by cell type
        """
        # Initialize an empty list to collect genes of interest
        gene_list = []
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Dictionary to store scores for all cell types
        scores_all = {}
        
        # Collect all genes from filtered_scores across all cell types
        for ctype in adata.obs[column_ctype].unique():
            if not filtered_scores[ctype].empty:
                gene_list += filtered_scores[ctype].index.tolist()
        
        # Create a subset of the AnnData object containing only genes of interest
        adata_subset = adata[:, gene_list].copy()
        
        # Iterate through each cell type
        for ctype in adata.obs[column_ctype].unique():
            # Initialize score matrix (rows=genes, columns=k_values)
            scores_ctype = np.zeros(shape=(adata_subset.shape[1], len(k_values)))
            scores_ctype = pd.DataFrame(scores_ctype, index=adata_subset.var.index, columns=k_values)
            
            # Extract expression data for target cells (cells of current cell type)
            expr_A = sc.get.obs_df(adata_subset[adata_subset.obs[column_ctype]==ctype], 
                                adata_subset.var.index.tolist())
            
            # Extract expression data for non-target cells (cells of other cell types)
            expr_nA = sc.get.obs_df(adata_subset[adata_subset.obs[column_ctype]!=ctype], 
                                adata_subset.var.index.tolist())
            
            # Calculate scores for each k value
            for k in k_values:
                # For k > 1, perform random sampling to simulate k-cell aggregation
                if k > 1:
                    # Sample k cells from target population n_sim times
                    sampled_A = np.random.choice(expr_A.shape[0], (n_sim, k), replace=True)
                    # Sum expression values across k cells for each simulation
                    k_sums_A = pd.DataFrame(expr_A.values[sampled_A].sum(axis=1), columns=expr_A.columns)
                    
                    # Sample k cells from non-target population n_sim times
                    sampled_nA = np.random.choice(expr_nA.shape[0], (n_sim, k), replace=True)
                    # Sum expression values across k cells for each simulation
                    k_sums_nA = pd.DataFrame(expr_nA.values[sampled_nA].sum(axis=1), columns=expr_nA.columns)
                else:
                    # For k=1, no aggregation needed, use raw expression values
                    k_sums_A = expr_A
                    k_sums_nA = expr_nA
                
                # Determine cutoff value for CDF calculation (99th percentile of the largest gene)
                cutoff_k = np.max([k_sums_A.quantile(0.99).max(),
                                k_sums_nA.quantile(0.99).max(), 1])
                
                # Calculate empirical CDF (alpha) for target cells at each count level
                alpha = np.zeros(shape=(adata_subset.shape[1], int(cutoff_k)))
                for l in np.arange(int(cutoff_k)):
                    alpha[:,l] = self.empirical_cdf_v0(k_sums_A, l).values
                
                # Calculate empirical CDF (beta) for non-target cells at each count level
                beta = np.zeros(shape=(adata_subset.shape[1], int(cutoff_k)))
                for l in np.arange(int(cutoff_k)):
                    beta[:,l] = self.empirical_cdf_v0(k_sums_nA, l).values
                
                # Calculate scores based on the specified scoring method
                if scoring == 'diff':
                    # Maximum difference between specificity (beta) and false negative rate (alpha)
                    scores_ctype[k] = (beta-alpha)[np.arange(adata_subset.shape[1]), 
                                                np.argmax(beta-alpha, axis=1)]
                elif scoring == 'sensFPRzero':
                    # Sensitivity (1-alpha) at point of zero false positive rate (max beta)
                    scores_ctype[k] = (1-alpha)[np.arange(adata_subset.shape[1]), 
                                            np.argmax(beta, axis=1)]
                elif scoring == 'sensPPV99':
                    # Positive predictive value calculation
                    ppv = np.nan_to_num((1-alpha)/(2-alpha-beta))
                    # Sensitivity at point where PPV > 99%
                    scores_ctype[k] = ((1-alpha)[np.arange(adata_subset.shape[1]), 
                                                np.argmax(ppv>0.99, axis=1)]) * (np.sum(ppv>0.99, axis=1)>0)
            
            # Store scores for this cell type
            scores_all[ctype] = pd.DataFrame(scores_ctype)
        
        # Apply multi-category correction to normalize scores
        corr_scores = self.multi_cat_correction(scores_all)
        
        # Filter scores to include only genes from filtered_scores
        scores = {}
        for ctype in adata.obs[column_ctype].unique():
            if filtered_scores[ctype].empty:
                scores[ctype] = pd.DataFrame([])
            else:
                # Extract scores for filtered genes and clip values to max of 1
                scores[ctype] = corr_scores[ctype].loc[filtered_scores[ctype].index].clip(upper=1)
        
        return scores
    
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
    
    def plot_marker_heatmap(self, n_genes=1, cutoff=0, groupby=None, cmap='viridis', standard_scale='var', 
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
        adata_filtered = self.adata[mask]
        
        if adata_filtered.n_obs == 0:
            raise ValueError(f"No cells found for the given cell types in {groupby}")
        
        # Get unique cell types in the filtered data to preserve the order
        cell_types = adata_filtered.obs[groupby].cat.categories.tolist()
        
        # Filter cell types to only those that appear in the data
        cell_types = [ct for ct in cell_types if ct in adata_filtered.obs[groupby].unique()]
        
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
            adata_filtered, 
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
                            ax=ax, show=False, jitter=jitter, alpha=alpha, raw=False)
                
                if marker_cell_type:
                    ax.set_title(f"{marker} (marker for {marker_cell_type})")
                
                # Rotate x-axis labels
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        
        # Hide any unused axes
        for i in range(n_markers, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    

    def plot_marker_roc(self, markers=None, n_markers=5, figsize=(12, 10), k=None):
        """
        Plot ROC curves for marker genes showing their sensitivity/specificity tradeoffs
        
        Args:
            markers: Dictionary mapping cell types to marker genes
            n_markers: Number of markers to show per cell type if markers=None
            figsize: Figure size
            k: Which k value to use for ROC calculation (if None, uses the first k value)
            
        Returns:
            Figure with ROC curves for each marker
        """
        from sklearn.metrics import roc_curve, auc
        
        # Get markers to plot
        if markers is None:
            if self.top_markers is None:
                raise ValueError("No markers available. Run ScSherlock.run() first")
            markers = self.top_markers
        
        # If k is not specified, use the first k value
        if k is None:
            k = self.config.k_values[0]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot ROC curve for each marker
        for ctype, genes in markers.items():
            gene_list = [genes] if isinstance(genes, str) else genes
            gene_list = gene_list[:n_markers]  # Limit to n_markers
            
            for gene in gene_list:
                # Get expression values
                if gene not in self.adata.var_names:
                    continue
                    
                gene_expr = self.adata[:, gene].X.toarray().flatten()
                y_true = (self.adata.obs[self.column_ctype] == ctype).astype(int).values
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_true, gene_expr)
                roc_auc = auc(fpr, tpr)
                
                # Plot ROC curve
                ax.plot(fpr, tpr, lw=2, label=f'{gene} ({ctype}, AUC = {roc_auc:.2f})')
        
        # Add diagonal line
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        # Set axes labels and title
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve for Marker Genes')
        ax.legend(loc="lower right")
        
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
    

    def plot_theoretical_vs_empirical(self, n_genes=20, figsize=(12, 10), cell_types=None):
        """
        Create plots comparing theoretical vs empirical scores for marker genes
        
        This method produces multiple visualizations to compare theoretical and empirical scores:
        1. Scatter plot of theoretical vs empirical scores
        2. Bar plot of top genes showing both scores
        3. Distribution plots showing the correlation
        
        Args:
            n_genes: Number of top genes to include in the visualizations
            figsize: Figure size as (width, height)
            cell_types: List of cell types to include (if None, includes all)
            
        Returns:
            matplotlib.figure.Figure: The figure containing the visualizations
        
        Raises:
            ValueError: If empirical scores are not available (must run with method='empiric')
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
                        'empirical_score': emp_score,
                        'abs_difference': abs(theo_score - emp_score)
                    })
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by difference
        comparison_df = comparison_df.sort_values('abs_difference', ascending=False)
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        
        # Define subplot grid: 2 rows, 2 columns
        gs = fig.add_gridspec(2, 2)
        
        # 1. Overall scatter plot (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        sc = ax1.scatter(comparison_df['theoretical_score'], comparison_df['empirical_score'], 
                alpha=0.7, c=comparison_df['abs_difference'], cmap='viridis')
        
        # Add diagonal line (perfect agreement)
        max_val = max(comparison_df['theoretical_score'].max(), comparison_df['empirical_score'].max())
        ax1.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
        
        ax1.set_xlabel('Theoretical Score')
        ax1.set_ylabel('Empirical Score')
        ax1.set_title('Theoretical vs Empirical Scores')
        
        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax1)
        cbar.set_label('Absolute Difference')
        
        # 2. Bar plot for top genes with largest differences (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Get top n genes with largest differences
        top_diff_genes = comparison_df.head(n_genes)
        
        # Create index for bars
        x = np.arange(len(top_diff_genes))
        width = 0.35
        
        # Create bars
        ax2.bar(x - width/2, top_diff_genes['theoretical_score'], width, label='Theoretical')
        ax2.bar(x + width/2, top_diff_genes['empirical_score'], width, label='Empirical')
        
        # Add gene names
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{row['gene']}\n({row['cell_type']})" for _, row in top_diff_genes.iterrows()], 
                        rotation=90, ha='right')
        
        ax2.set_ylabel('Score')
        ax2.set_title(f'Top {n_genes} Genes with Largest Score Differences')
        ax2.legend()
        
        # 3. Correlation heatmap by cell type (bottom left)
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Calculate correlation for each cell type
        correlations = []
        for ctype in cell_types:
            # Get genes with both scores
            ctype_df = comparison_df[comparison_df['cell_type'] == ctype]
            
            if len(ctype_df) >= 5:  # Need enough genes for meaningful correlation
                corr = np.corrcoef(ctype_df['theoretical_score'], ctype_df['empirical_score'])[0, 1]
                correlations.append({'cell_type': ctype, 'correlation': corr, 'gene_count': len(ctype_df)})
        
        # Create correlation DataFrame
        corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
        
        # Plot correlation bars
        ax3.barh(np.arange(len(corr_df)), corr_df['correlation'], color='skyblue')
        ax3.set_yticks(np.arange(len(corr_df)))
        ax3.set_yticklabels([f"{row['cell_type']} (n={row['gene_count']})" for _, row in corr_df.iterrows()])
        
        ax3.set_xlabel('Correlation')
        ax3.set_title('Correlation between Theoretical and Empirical Scores by Cell Type')
        ax3.axvline(x=0, color='gray', linestyle='--')
        ax3.set_xlim(-1, 1)
        
        # 4. Distribution plot of differences (bottom right)
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Calculate differences
        differences = comparison_df['theoretical_score'] - comparison_df['empirical_score']
        
        # Plot histogram of differences
        ax4.hist(differences, bins=30, alpha=0.7, color='cornflowerblue')
        ax4.axvline(x=0, color='red', linestyle='--')
        
        # Add statistics
        mean_diff = differences.mean()
        median_diff = differences.median()
        ax4.text(0.05, 0.95, f"Mean: {mean_diff:.3f}\nMedian: {median_diff:.3f}", 
                transform=ax4.transAxes, va='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax4.set_xlabel('Theoretical - Empirical Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Score Differences')
        
        # Add overall statistics to figure title
        overall_corr = np.corrcoef(comparison_df['theoretical_score'], comparison_df['empirical_score'])[0, 1]
        fig.suptitle(f'Theoretical vs Empirical Score Comparison\nOverall Correlation: {overall_corr:.3f}', 
                    fontsize=16)
        
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        return fig
    
    import numpy as np
import pandas as pd
import scipy
import numba as nb
from statsmodels.distributions.empirical_distribution import ECDF

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