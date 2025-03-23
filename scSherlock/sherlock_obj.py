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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SCherlock')


class ScoringMethod(Enum):
    """Enum for scoring methods used in SCherlock"""
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
class SCherlockConfig:
    """Configuration class for SCherlock parameters"""
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
    
    def __post_init__(self):
        """Initialize default values"""
        if self.k_values is None:
            self.k_values = [1, 10, 25]


class SCherlock:
    """
    SCherlock: Single-Cell marker gene identification algorithm
    
    This class implements the SCherlock algorithm for identifying cell type-specific 
    marker genes from single-cell RNA sequencing data.
    """
    
    def __init__(self, adata, column_ctype: str, column_patient: str, config: Optional[SCherlockConfig] = None):
        """
        Initialize SCherlock with data and configuration
        
        Args:
            adata: AnnData object containing single-cell gene expression data
            column_ctype: Column name in adata.obs for cell type annotations
            column_patient: Column name in adata.obs for patient IDs
            config: Configuration object with algorithm parameters (optional)
        """
        self.adata = adata
        self.column_ctype = column_ctype
        self.column_patient = column_patient
        self.config = config or SCherlockConfig()
        
        # Validate inputs
        self._validate_inputs()
        logger.info("Pre-filtering genes...")
        self.adata = self._prefilter_genes(self.config.min_cells, self.config.min_reads)
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
        
        logger.info(f"SCherlock initialized with {len(self.cell_types)} cell types and {self.adata.shape} data matrix")
    
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
        
        # Create a copy to avoid modifying the original
        adata_filtered = self.adata.copy()
        
        # Filter genes based on minimum number of cells and counts
        sc.pp.filter_genes(adata_filtered, min_counts=min_counts)
        sc.pp.filter_genes(adata_filtered, min_cells=min_cells)
        
        logger.info(f"Filtered dataset: {adata_filtered.shape[0]} cells, {adata_filtered.shape[1]} genes")
        logger.info(f"Removed {self.adata.shape[1] - adata_filtered.shape[1]} genes with low expression")
        
        return adata_filtered
    
    def run(self) -> Dict[str, str]:
        """
        Run the complete SCherlock algorithm pipeline
        
        Returns:
            Dict[str, str]: Dictionary of top marker genes for each cell type
        """
        # Set random seed for reproducibility
        np.random.seed(self.config.random_seed)
        
        # Step 1: Calculate theoretical scores based on binomial distributions
        logger.info("Calculating theoretical scores...")
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
        
        # Step 5: Filter genes based on scores and expression criteria
        logger.info("Filtering genes...")
        self.filtered_scores = self._filter_genes(self.sorted_table)

        # Step 6: Calculate empirical scores through simulation
        logger.info("Calculating empirical scores...")
        #self.empirical_scores = self._calculate_empirical_scores(self.filtered_scores)
        self.empirical_scores = self.empirical_scores_v0(self.filtered_scores, self.adata, self.column_ctype, self.column_patient, k_values=self.config.k_values, scoring='diff', seed=0, n_sim=1000)

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
        

        logger.info(f"SCherlock completed. Found markers for {len(self.top_markers)}/{len(self.cell_types)} cell types")
        return self.top_markers
    
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
    
    def empirical_cdf_v0(self, df, value):
        #empirical cdf computation (all genes must be evaluated on the same number of pointd)
        return df.apply(lambda col: percentileofscore(col, value, kind='rank') / 100)

    def empirical_scores_v0(self, filtered_scores, adata, column_ctype, column_patient, k_values, scoring, seed=0, n_sim=1000):
        #original stable empirical score computation: slow (calculated for all values up to 99th percentile of largest gene)
        gene_list = []
        np.random.seed(seed)
        scores_all = {}
        for ctype in adata.obs[column_ctype].unique():
            if not filtered_scores[ctype].empty:
                gene_list += filtered_scores[ctype].index.tolist()
        adata_subset=adata[:,gene_list].copy()
        for ctype in adata.obs[column_ctype].unique():
            scores_ctype = np.zeros(shape=(adata_subset.shape[1], len(k_values)))
            scores_ctype = pd.DataFrame(scores_ctype, index=adata_subset.var.index, columns=k_values)
            expr_A = sc.get.obs_df(adata_subset[adata_subset.obs[column_ctype]==ctype], adata_subset.var.index.tolist())
            expr_nA = sc.get.obs_df(adata_subset[adata_subset.obs[column_ctype]!=ctype], adata_subset.var.index.tolist())
            for k in k_values:
                if k > 1:
                    sampled_A= np.random.choice(expr_A.shape[0], (n_sim, k), replace=True)
                    k_sums_A = pd.DataFrame(expr_A.values[sampled_A].sum(axis=1), columns=expr_A.columns)
                    sampled_nA= np.random.choice(expr_nA.shape[0], (n_sim, k), replace=True)
                    k_sums_nA = pd.DataFrame(expr_nA.values[sampled_nA].sum(axis=1), columns=expr_nA.columns)
                else:
                    k_sums_A = expr_A
                    k_sums_nA = expr_nA
                cutoff_k=np.max([k_sums_A.quantile(0.99).max(),
                                k_sums_nA.quantile(0.99).max(), 1])
                alpha = np.zeros(shape=(adata_subset.shape[1], int(cutoff_k)))
                for l in np.arange(int(cutoff_k)):
                    alpha[:,l]=self.empirical_cdf_v0(k_sums_A, l).values
                beta = np.zeros(shape=(adata_subset.shape[1], int(cutoff_k)))
                for l in np.arange(int(cutoff_k)):
                    beta[:,l]=self.empirical_cdf_v0(k_sums_nA, l).values
                if scoring=='diff':
                    scores_ctype[k]=(beta-alpha)[np.arange(adata_subset.shape[1]), np.argmax(beta-alpha,axis=1)]
                elif scoring=='sensFPRzero':
                    scores_ctype[k]=(1-alpha)[np.arange(adata_subset.shape[1]), np.argmax(beta,axis=1)]
                elif scoring=='sensPPV99':
                    ppv = np.nan_to_num((1-alpha)/(2-alpha-beta))
                    scores_ctype[k]=((1-alpha)[np.arange(adata_subset.shape[1]), np.argmax(ppv>0.99,axis=1)])*(np.sum(ppv>0.99,axis=1)>0)
            scores_all[ctype] = pd.DataFrame(scores_ctype)
        corr_scores = self.multi_cat_correction(scores_all)
        scores = {}
        for ctype in adata.obs[column_ctype].unique():
            if filtered_scores[ctype].empty:
                scores[ctype]=pd.DataFrame([])
            else:
                scores[ctype]=corr_scores[ctype].loc[filtered_scores[ctype].index].clip(upper=1)
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


    def _construct_top_marker_list(self, sorted_emp_table: Dict) -> Dict[str, str]:
        """
        Identify top marker for each cell type
        
        Args:
            sorted_emp_table: Dictionary of sorted empirical scores
            
        Returns:
            Dict[str, str]: Dictionary mapping cell types to top marker genes
        """
        # Get top scoring marker for each cell type that meets the cutoff
        print(sorted_emp_table)
        top_markers = {
            ctype: table.index[0]  # First gene is highest scoring
            for ctype, table in sorted_emp_table.items()
            if not table.empty and table['aggregated'].max() >= self.config.score_cutoff
        }
        print(top_markers)
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
        Get complete results from the SCherlock analysis
        
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
        """
        if self.top_markers is None:
            raise ValueError("No markers identified. Run SCherlock.run() first")
            
        # Compile marker information
        results = []
        for ctype, gene in self.top_markers.items():
            # Get scores for this gene
            theoretical_score = self.theoretical_scores[ctype].loc[gene].mean()
            empirical_score = self.empirical_scores[ctype].loc[gene].mean()
            aggregated_score = self.sorted_empirical_table[ctype].loc[gene, 'aggregated']
            exp_prop = self.sorted_empirical_table[ctype].loc[gene, 'exp_prop']
            
            results.append({
                'cell_type': ctype,
                'marker_gene': gene,
                'theoretical_score': theoretical_score,
                'empirical_score': empirical_score,
                'aggregated_score': aggregated_score,
                'expression_proportion': exp_prop
            })
        
        # Create DataFrame
        markers_df = pd.DataFrame(results)
        
        # Save to file if specified
        if output_file:
            markers_df.to_csv(output_file, index=False)
            logger.info(f"Exported markers to {output_file}")
            
        return markers_df
    
    def plot_marker_heatmap(self, n_genes=1, cutoff=0, groupby=None, cmap='viridis', standard_scale='var', 
                            use_raw=False, save=None, show=None, **kwargs):
        """
        Create a heatmap visualization of the identified marker genes
        
        This method creates a matrix plot showing the expression of top marker genes
        across all cell types. It orders the genes to match their corresponding cell types.
        
        Args:
            n_genes (int): Number of top genes to display for each cell type (default: 1)
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
            ValueError: If no markers are available (run SCherlock.run() first) or if no 
                    cells are found for the given cell types
        """
        if self.sorted_empirical_table is None:
            raise ValueError("No markers available. Run SCherlock.run() first")
        
        # Use class's column_ctype if groupby not specified
        if groupby is None:
            groupby = self.column_ctype
        
        # Get cell types that have valid markers
        cell_to_genes = {}
        for ctype, table in self.sorted_empirical_table.items():
            if isinstance(table, pd.DataFrame) and not table.empty:
                # Only include cell types that have marker genes meeting the cutoff
                valid_genes = table[table['aggregated'] >= cutoff]
                if not valid_genes.empty:
                    # Get up to n_genes top markers
                    cell_to_genes[ctype] = valid_genes.index[:min(n_genes, len(valid_genes))].tolist()
        
        # Check if any markers were found
        if not cell_to_genes:
            logger.warning("No markers found that meet the score cutoff criteria")
            return None
        
        # Filter adata to only include cell types that have markers
        mask = self.adata.obs[groupby].isin(cell_to_genes.keys())
        adata_filtered = self.adata[mask].copy()
        
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
        logger.info(f"Plotting {total_genes} genes for {len(cell_types)} cell types")
        
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