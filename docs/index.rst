.. scSherlock documentation master file

=======================================
scSherlock - Single-cell data analysis
=======================================

.. image:: ../images/logo.png
   :width: 200px
   :align: center
   :alt: scSherlock logo

**scSherlock**: A powerful toolkit for discovering cell-cell interactions involved in tissue development.

.. note::
   This project is under active development.

Main Features
============

* Analyze single-cell RNA sequencing data
* Identify and visualize cell-cell interactions
* Integrate with popular analysis frameworks like scanpy
* Generate interpretable visualizations of interaction networks

Installation
===========

You can install scSherlock via pip:

.. code-block:: bash

   pip install scSherlock

Quick Start
==========

.. code-block:: python

   import scSherlock as scs
   
   # Load your data
   adata = scs.read_h5ad("your_data.h5ad")
   
   # Run analysis
   scs.run_analysis(adata)
   
   # Visualize results
   scs.plot_interactions(adata)

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   installation/index
   quickstart
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