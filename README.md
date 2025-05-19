# SCherlock

## Overview
<img src="images/logo.png" align="right" width="125px" />
SCherlock is a statistical algorithm for identifying cell type-specific marker genes from single-cell RNA sequencing data. The algorithm uses probabilistic models to identify genes that are specifically expressed in target cell types but not in other cell types, making them reliable markers for cell type identification.  

Documentation is available here : https://SCherlock.readthedocs.io/en/latest/  

## Installation

```bash
pip install scherlock
```

## Quick Start

```python
import scanpy as sc
from scherlock import SCherlock, SCherlockConfig, ScoringMethod

# Load your data
adata = sc.read("your_data.h5ad")

# Initialize SCherlock
scherlock = SCherlock(
    adata=adata,
    column_patient="patient_id",       # Column in adata.obs for patient IDs
    config=config
)

# Run the algorithm
top_markers = scherlock.run(column_ctype='cell_type')

# Plot marker heatmap
scherlock.plot_marker_heatmap(n_genes=1, column_ctype=cell_type_column, cutoff=0)

# Get top 3 markers for each cell type with a score > 0.5
scherlock.get_marker(column_ctype='cell_type', n_top_genes=3, min_score=0.5)

```

## Citation

If you use SCherlock in your research, please cite:

```
[Citation information will be added here]
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

