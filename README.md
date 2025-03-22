# scSherlock

## Overview
<img src="images/logo.png" align="right" width="100px" />
scSherlock is a statistical algorithm for identifying cell type-specific marker genes from single-cell RNA sequencing data. The algorithm uses probabilistic models to identify genes that are specifically expressed in target cell types but not in other cell types, making them reliable markers for cell type identification.  

Documentation is available here : 

## Features

- Statistically rigorous marker gene identification
- Support for multiple cell types and patient samples
- Theoretical and empirical validation of marker predictions
- Flexible scoring methods to prioritize sensitivity or specificity
- Visualization tools for marker gene evaluation

## Installation

```bash
pip install scsherlock
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

## Citation

If you use SCherlock in your research, please cite:

```
[Citation information will be added here]
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

