# SCherlock: Single-Cell marker gene identification algorithm

## Overview
<img src="images/logo.png" align="right" width="150px" />
SCherlock is a statistical algorithm for identifying cell type-specific marker genes from single-cell RNA sequencing data. The algorithm uses probabilistic models to identify genes that are specifically expressed in target cell types but not in other cell types, making them reliable markers for cell type identification.  
Documentation is available here : 

## Features

- Statistically rigorous marker gene identification
- Support for multiple cell types and patient samples
- Theoretical and empirical validation of marker predictions
- Flexible scoring methods to prioritize sensitivity or specificity
- Visualization tools for marker gene evaluation

## Installation

```bash
pip install scherlock
```

## Requirements

- Python 3.6+
- scanpy
- numpy
- pandas
- scipy
- matplotlib
- seaborn

## Quick Start

```python
import scanpy as sc
from scherlock import SCherlock, SCherlockConfig, ScoringMethod

# Load your data
adata = sc.read("your_data.h5ad")

# Configure SCherlock
config = SCherlockConfig(
    k_values=[1, 10, 25],              # Cell aggregation levels to evaluate
    scoring_method=ScoringMethod.DIFF, # Scoring method for marker evaluation
    max_genes_kept=100,                # Maximum number of genes to keep per cell type
    min_patients=3,                    # Minimum number of patients expressing the gene
    min_reads=10,                      # Minimum number of reads for a gene
    min_cells=10                       # Minimum number of cells expressing the gene
)

# Initialize SCherlock
scherlock = SCherlock(
    adata=adata,
    column_ctype="cell_type",          # Column in adata.obs for cell type annotations
    column_patient="patient_id",       # Column in adata.obs for patient IDs
    config=config
)

# Run the algorithm
top_markers = scherlock.run()

# Export results
markers_df = scherlock.export_markers("markers.csv")

# Visualize top marker
scherlock.visualize_marker("TOP_MARKER_GENE")

# Generate heatmap of markers
scherlock.plot_marker_heatmap(n_genes=5)
```

## How SCherlock Works

SCherlock identifies marker genes through a multi-step process:

1. **Pre-filtering**: Removes genes with low expression and those unlikely to be markers
2. **Parameter Estimation**: Estimates binomial distribution parameters for each cell type
3. **Theoretical Score Calculation**: Calculates marker scores based on binomial distributions
4. **Multi-category Correction**: Normalizes scores across different cell types
5. **Empirical Validation**: Validates marker predictions using random sampling
6. **Marker Selection**: Selects top markers based on aggregated scores

### Scoring Methods

SCherlock supports three scoring methods:

- **DIFF**: Maximized difference between sensitivity and false positive rate
- **SENS_FPR_ZERO**: Sensitivity at zero false positive rate
- **SENS_PPV_99**: Sensitivity at positive predictive value > 99%

### Performance Optimizations

SCherlock includes several optimizations to improve performance:

- **Early Gene Filtering**: Quickly removes genes with no potential to be markers
- **Sparse Sampling**: Evaluates only a subset of expression values for initial screening
- **Parallelization**: Process multiple k values in parallel (optional)

## Advanced Usage

### Customizing Parameter Estimation

```python
from scherlock import ParameterEstimation

config = SCherlockConfig(
    parameter_estimation=ParameterEstimation.PATIENT_MEDIAN,  # Estimate per patient
    # or
    parameter_estimation=ParameterEstimation.MEAN  # Estimate on entire dataset
)
```

### Customizing Score Aggregation

```python
from scherlock import AggregationMethod

config = SCherlockConfig(
    aggregation_method=AggregationMethod.MEAN,  # Mean of scores across k values
    # or
    aggregation_method=AggregationMethod.MAX  # Maximum score across k values
)
```

### Accessing Intermediate Results

```python
# Get all results including intermediate calculations
results = scherlock.get_results()

# Access specific components
theoretical_scores = results['theoretical_scores']
empirical_scores = results['empirical_scores']
```

## Citation

If you use SCherlock in your research, please cite:

```
[Citation information will be added here]
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

[Acknowledgements and funding information]