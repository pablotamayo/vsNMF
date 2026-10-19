
<h2 > 
Variable sparsity non-negative matrix (vsNMF) and tensor factorization (vsNTF) algorithms and their applications in cancer genomics


</h2>
<br>

**Authors:** Timothy Pham and Pablo Tamayo

**Date:** October 18, 2026

The field of oncology has experienced significant advancements in novel therapies and molecular diagnostics to personalize initial cancer treatments. However, cancer's propensity for evolution and resistance to therapies remains a formidable challenge due to its dynamic nature. Cancer's complex and adaptive behavior surpasses the capabilities of simple models, complicating our comprehension at every level, from basic sciences to clinical practices, highlighting the urgent need for innovative approaches to improve the modeling and classification of cancer. In this context, matrix factorization algorithms have been used to generate transcriptional signatures that summarize the most salient patterns of behaviors in cancer datasets. For example, the standard non-negative matrix factorization algorithm (NMF) generates a dimensionality-reduced representation that encodes a significant amount of data relationships as a mixture of positive components and provides a working summary of the input data. The resulting NMF factor matrices, W and H, display some degree of sparseness; however, this sparseness is often a byproduct rather than an explicit objective of the NMF algorithm, and the user has limited control over the degree of sparseness of the representation and the corresponding balance between local versus global structures in the NMF results. Some sparse versions of NMF have been introduced but still do not allow a fine-grained control on the sparsity constraints. We present here a novel variable sparsity Non-negative Matrix Factorization algorithm (vsNMF), where instead of imposing sparsity as a regularization constraint on the NMF objective function, sparsity is explicitly enforced by a projective step performed for each column of W and row of H after a standard NMF update. In this way, the algorithm takes as input the input matrix, plus two one-dimensional vectors, that specify the sparsity functions that will be imposed on the NMF solution. The more explicit control of sparsity, and the corresponding balance between local vs. global structure, at the level of individual components, allows the generation of NMF solutions that can be tuned to operate anywhere between a generative, parts-based encoding of overlapping features, and a partitional, soft-clustering model. This allows greater flexibility to represent biological subtypes, patterns of pathway activation or deregulation, and higher biological interpretability. These learned low-dimensionality representations can also be used to integrate and merge datasets, or to define groups of similar cancer samples representing putative cellular or tumor states. Adjustable/tunable NMF representations of biological data can greatly help advance our fundamental understanding of cancer biology, provide more interpretable computational models of cancer, and better functional cancer taxonomies and precision oncology paradigms.

## Installation
Requires Python ≥3.11 

## Usage
``` variable sparsity Non-Negative Matrix Factorization

vsNMF(V,
      rdim,
      niter,
      sW,
      sH,
      initW = None,
      initH = None,
      random_seed = 17895314,
      normalize_initial_NMF_input = True,
      save_logs = False,
      verbose = True,
      log_file_path = "../log_file.txt"):MCI(x, y)
```
``` variable sparsity Non-Negative Tensor Factorization

vsNTF(tensor,
      rank,
      n_iter_max=10,
      init = "svd",
      tol = 10e-5,
      sparsity_coefficients = None,
      normalize_initial_tensor_input = True,
      random_state = None,
      verbos e= False,
      return_errors  =False,
      normalize_factors = False,
```

See `vsNMF Examples.ipynb` for examples.
