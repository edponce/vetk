.. highlight:: shell


======
README
======

Vector Embedding Toolkit
========================

`https://code.ornl.gov/VA-OSL/vector-embedding-toolkit`


Workflow
--------

0. Prepare text (e.g., single file with sentences delimited by newline)
1. Run vector embedding software (e.g., word2vec)
2. Load vector models by instantiating VectorEmbedding objects (vocabulary if available)
3. Compute attributes for each vector model (e.g, similarities and point distances between
   vector pairs
4. Instantiate a VectorEmbeddingAnalysis object with VectorEmbedding objects
5. Compute statistics and plots to compare vector embeddings

This work is currently under development for the OSL at ORNL.
