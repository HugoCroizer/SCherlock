===============================================================================
SCherlock: A Statistical Method for Identifying Cell Type-Specific Marker Genes
===============================================================================

.. image:: ../images/logo.png
   :align: right
   :width: 150px

Overview
========

The identification of cell type-specific marker genes is a crucial task in single-cell RNA sequencing (scRNA-seq) analysis. Here, we present SCherlock, a robust statistical approach for identifying marker genes that leverages both theoretical probability distributions and empirical validation through simulation. Our method evaluates gene expression patterns across multiple aggregation levels to identify genes that reliably distinguish target cell types from others. SCherlock incorporates patient-level information to ensure markers are consistent across biological replicates, and employs multiple scoring strategies to optimize for different marker characteristics. We demonstrate that SCherlock efficiently identifies biologically relevant marker genes with high sensitivity and specificity.

Methods
=======

Overview
--------

SCherlock identifies marker genes through a multi-step process that combines theoretical statistical modeling with empirical validation. The algorithm evaluates genes across varying levels of cell aggregation (k) to identify markers that are robust at different resolutions.

Notation
--------

We denote the expression matrix as :math:`X \in \mathbb{R}^{n \times m}`, where :math:`n` is the number of cells and :math:`m` is the number of genes. Each cell belongs to a cell type :math:`c \in C`, where :math:`C` is the set of all cell types. For each cell type :math:`c`, we aim to identify genes that are specifically expressed in cells of type :math:`c` but not in cells of types :math:`C \setminus \{c\}`.

Binomial Parameter Estimation
----------------------------

For each cell type :math:`c` and its complement :math:`\bar{c}`, we estimate the following parameters:

1. **Average counts per cell**: :math:`\lambda_c` and :math:`\lambda_{\bar{c}}`
2. **Expression proportion per gene**: :math:`p_{c,g}` and :math:`p_{\bar{c},g}` for each gene :math:`g`

These parameters can be estimated in two ways:

1. **Patient-median estimation**: For each patient :math:`i`, we compute:

   .. math::
      \lambda_{c,i} = \frac{\sum_{j \in c_i} \sum_{g} X_{j,g}}{|c_i|}
   
   .. math::
      p_{c,i,g} = \frac{\sum_{j \in c_i} X_{j,g}}{\sum_{j \in c_i} \sum_{g'} X_{j,g'}}
   
   where :math:`c_i` represents cells of type :math:`c` from patient :math:`i`. The final parameters are then the median across patients:
   
   .. math::
      \lambda_c = \text{median}_i(\lambda_{c,i})
   
   .. math::
      p_{c,g} = \text{median}_i(p_{c,i,g})

2. **Mean estimation**: Parameters are estimated directly on the pooled data:

   .. math::
      \lambda_c = \frac{\sum_{j \in c} \sum_{g} X_{j,g}}{|c|}
   
   .. math::
      p_{c,g} = \frac{\sum_{j \in c} X_{j,g}}{\sum_{j \in c} \sum_{g'} X_{j,g'}}

Theoretical Score Calculation
----------------------------

For each cell type :math:`c`, gene :math:`g`, and aggregation level :math:`k`, we model the number of reads as binomial random variables:

.. math::
   R_{c,g,k} \sim \text{Binom}(k \cdot \lambda_c, p_{c,g})

.. math::
   R_{\bar{c},g,k} \sim \text{Binom}(k \cdot \lambda_{\bar{c}}, p_{\bar{c},g})

We then compute the cumulative distribution functions (CDFs) for these distributions:

.. math::
   \alpha_{c,g,k}(t) = P(R_{c,g,k} \leq t) = \sum_{i=0}^{t} \binom{k \cdot \lambda_c}{i} p_{c,g}^i (1-p_{c,g})^{k \cdot \lambda_c - i}

.. math::
   \beta_{c,g,k}(t) = P(R_{\bar{c},g,k} \leq t) = \sum_{i=0}^{t} \binom{k \cdot \lambda_{\bar{c}}}{i} p_{\bar{c},g}^i (1-p_{\bar{c},g})^{k \cdot \lambda_{\bar{c}} - i}

To determine an appropriate range for threshold evaluation, we compute:

.. math::
   q_{c,g,k} = \text{max}(F^{-1}_{c,g,k}(0.99), F^{-1}_{\bar{c},g,k}(0.99), 100)

where :math:`F^{-1}` represents the inverse CDF (quantile function).

We evaluate thresholds :math:`t \in [0, q_{c,g,k}]` and compute scores based on the chosen scoring method:

1. **Maximum difference (DIFF)**:

   .. math::
      S^{\text{diff}}_{c,g,k} = \max_t [\beta_{c,g,k}(t) - \alpha_{c,g,k}(t)]

2. **Sensitivity at zero false positive rate (SENS_FPR_ZERO)**:

   .. math::
      S^{\text{sensFPR0}}_{c,g,k} = 1 - \alpha_{c,g,k}(t^*) \text{ where } t^* = \arg\max_t \beta_{c,g,k}(t)

3. **Sensitivity at positive predictive value > 99% (SENS_PPV_99)**:

   .. math::
      S^{\text{sensPPV99}}_{c,g,k} = (1 - \alpha_{c,g,k}(t^*)) \cdot \mathbf{1}_{\exists t: \text{PPV}(t) > 0.99} \text{ where } t^* = \arg\max_t [\text{PPV}(t) > 0.99]
   
   .. math::
      \text{PPV}(t) = \frac{1 - \alpha_{c,g,k}(t)}{2 - \alpha_{c,g,k}(t) - \beta_{c,g,k}(t)}

Multi-Category Correction
------------------------

To handle genes that may mark multiple cell types, we normalize scores across all cell types:

.. math::
   \hat{S}_{c,g,k} = \frac{S_{c,g,k}}{\sum_{c' \in C} S_{c',g,k}}

Score Aggregation
---------------

We aggregate scores across different :math:`k` values using either:

1. **Mean aggregation**:

   .. math::
      S^{\text{agg}}_{c,g} = \frac{1}{|K|} \sum_{k \in K} \hat{S}_{c,g,k}

2. **Maximum aggregation**:

   .. math::
      S^{\text{agg}}_{c,g} = \max_{k \in K} \hat{S}_{c,g,k}

Empirical Validation
------------------

We validate theoretical scores through Monte Carlo simulation. For each cell type :math:`c`, gene :math:`g`, and aggregation level :math:`k`:

1. We randomly sample :math:`k` cells with replacement from the target population and compute the sum of expression values for gene :math:`g`. This is repeated :math:`n_{\text{sim}}` times to obtain the distribution :math:`\{R^{\text{sim}}_{c,g,k,i}\}_{i=1}^{n_{\text{sim}}}`.

2. Similarly, we sample from the non-target population to obtain :math:`\{R^{\text{sim}}_{\bar{c},g,k,i}\}_{i=1}^{n_{\text{sim}}}`.

3. We compute empirical CDFs:

   .. math::
      \hat{\alpha}_{c,g,k}(t) = \frac{1}{n_{\text{sim}}} \sum_{i=1}^{n_{\text{sim}}} \mathbf{1}_{R^{\text{sim}}_{c,g,k,i} \leq t}
   
   .. math::
      \hat{\beta}_{c,g,k}(t) = \frac{1}{n_{\text{sim}}} \sum_{i=1}^{n_{\text{sim}}} \mathbf{1}_{R^{\text{sim}}_{\bar{c},g,k,i} \leq t}

4. Using these empirical CDFs, we compute empirical scores following the same approach as for theoretical scores.

Optimization Techniques
---------------------

To improve computational efficiency, SCherlock employs several optimization strategies:

1. **Sparse Sampling**: For empirical validation, we initially evaluate CDFs at a sparse set of points to identify promising genes:

   .. math::
      T_{\text{sparse}} = \{0, s, 2s, ..., \lfloor q_{c,g,k}/s \rfloor \cdot s\}
   
   where :math:`s` is the sparse step size.

2. **Promising Gene Identification**: We identify promising genes using scoring method-specific criteria:

   - For DIFF: :math:`\max_t [\hat{\beta}_{c,g,k}(t) - \hat{\alpha}_{c,g,k}(t)] > \theta`
   - For SENS_FPR_ZERO: :math:`\max_t \hat{\beta}_{c,g,k}(t) > 1 - \theta`
   - For SENS_PPV_99: :math:`\exists t: \widehat{\text{PPV}}(t) > 0.99`

Final Marker Selection
--------------------

For each cell type :math:`c`, we select genes with the highest empirical scores that meet the following criteria:

- Score exceeds the cutoff threshold
- Expressed in at least :math:`n_{\text{min}}` patients
- Has at least :math:`r_{\text{min}}` reads across cells of type :math:`c`

The final set of markers is:

.. math::
   M_c = \{g \in G : S^{\text{emp,agg}}_{c,g} \geq \sigma \text{ and } \text{patients}(g,c) \geq n_{\text{min}} \text{ and } \text{reads}(g,c) \geq r_{\text{min}}\}

where :math:`\sigma` is the score cutoff, :math:`n_{\text{min}}` is the minimum number of patients, and :math:`r_{\text{min}}` is the minimum number of reads.