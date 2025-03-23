.. scSherlock documentation master file

=======================================
scSherlock
=======================================

.. image:: ../images/logo.png
   :width: 200px
   :align: right
   :alt: scSherlock logo

**scSherlock** is a robust statistical approach for identifying marker genes that leverages both theoretical probability distributions and empirical validation through simulation. The method evaluates gene expression patterns across multiple aggregation levels to identify genes that reliably distinguish target cell types from others. SCherlock incorporates patient-level information to ensure markers are consistent across biological replicates, and employs multiple scoring strategies to optimize for different marker characteristics. 

.. note::
   This project is under active development.


Installation
===========

You can install scSherlock via pip:

.. code-block:: bash

   pip install scSherlock

Quick Start
==========

.. code-block:: python
   
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

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   installation/index
   quickstart
   principle/index
   tutorials/index
   examples/index
   paper

.. toctree::
   :maxdepth: 1
   :caption: Development
   
   contributing
   changelog

Indices and tables
=================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`