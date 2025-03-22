Installation
===========

This guide covers different methods to install scSherlock, with Conda being the recommended approach.

Conda Installation (Recommended)
-------------------------------

The recommended way to install scSherlock is using Conda, which helps manage dependencies efficiently:

1. First, make sure you have Conda installed (either `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or `Anaconda <https://www.anaconda.com/products/distribution>`_)

2. Create a new environment for scSherlock:

   .. code-block:: bash

      conda create -n scSherlock python=3.10
      conda activate scSherlock

3. Install required dependencies:

   .. code-block:: bash

      conda install -c conda-forge  pandas numpy matplotlib plotly scipy shapely
      conda install -c conda-forge jupyterlab  # for running tutorial notebooks

4. Install scSherlock using pip:

   .. code-block:: bash

      pip install scSherlock

Pip Installation
--------------

If you prefer using pip, you can install scSherlock directly:

.. code-block:: bash

   pip install scSherlock

Note that you'll need to ensure all dependencies are properly installed, which may be more challenging than using Conda.


Troubleshooting
-------------

Common installation issues and their solutions:


Platform-Specific Notes
^^^^^^^^^^^^^^^^^^^^^

**Windows Users**:
   Shapely and other geospatial libraries might require additional steps. Using Conda is strongly recommended.

**Mac M1/M2 Users**:
   Make sure to use the arm64 version of Conda for best performance.

Verifying Installation
-------------------

To verify scSherlock is correctly installed, run:

.. code-block:: python

   import scSherlock
   print(scSherlock.__version__)

If this runs without errors, your installation is successful.