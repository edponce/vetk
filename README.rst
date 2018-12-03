Vector Embedding Toolkit
========================

This work is currently under development for the OSL at ORNL.


Workflow
--------

#. Prepare text (e.g., single file with sentences delimited by newline)
#. Run vector embedding software (e.g., word2vec)
#. Load vector models by instantiating VectorEmbedding objects (vocabulary if available)
#. Compute attributes for each vector model (e.g, similarities and point distances between vector pairs
#. Instantiate a VectorEmbeddingAnalysis object with VectorEmbedding objects
#. Compute statistics and plots to compare vector embeddings


WordEmbedding
-------------

Class that represents a word embedding model.
Contains a numpy matrix of float32 representing the embedding matrix.
Contains a dictionary for the vocabulary and occurrences (true occurrences are
available if loaded from vocabulary file, else each word has value of 1).
A WordEmbedding object can be accessed as a dictionary using either the
word string or word index.


Assumptions
^^^^^^^^^^^

* Embedding matrix and vocabulary are in same order and contain the same number
  of rows (one vector per word).
* Embedding matrix and vocabulary are arranged in decreasing order based on
  word occurrences.


Example 1
^^^^^^^^^

.. code-block:: text

    # This example create a WordEmbedding object and loads a word embedding
    # and vocabulary produced by word2vec software.

    from VETk import WordEmbedding

    # Word embedding and vocabulary files generated from same
    # dataset (100 files) but different number of threads (1, 16, 32)
    fvec1 = '../w2v_outputs/cbow_hs/outputs_1a/mimicnotes_100.vec',
    fvec16 = '../w2v_outputs/cbow_hs/outputs_16a/mimicnotes_100.vec',
    fvec32 = '../w2v_outputs/cbow_hs/outputs_32a/mimicnotes_100.vec',
    fvoc1 = '../w2v_outputs/cbow_hs/outputs_1a/mimicnotes_100.voc',
    fvoc16 = '../w2v_outputs/cbow_hs/outputs_16a/mimicnotes_100.voc',
    fvoc32 = '../w2v_outputs/cbow_hs/outputs_32a/mimicnotes_100.voc',

    # Load word embeddings and vocabulary, each with a unique name
    v1 = ve.WordEmbedding(fvec1, fvoc1, name='100N/1T')
    v16 = ve.WordEmbedding(fvec16, fvoc16, name='100N/16T')
    v32 = ve.WordEmbedding(fvec32, fvoc32, name='100N/32T')

    # Compute attributes requiring vector pairs (similarities, point distances)
    v1.process_attributes(2)
    v16.process_attributes(2)
    v32.process_attributes(2)

    # Compute attributes requiring vector triplets (angle pairs between 3
    vectors)
    v1.process_attributes(3)
    v16.process_attributes(3)
    v32.process_attributes(3)

    # Access vector corresponding to 'word'
    v1['word']

    # Access vectors corresponding to multiple words, 'w1', 'w2', 'w3'
    v1['w1', 'w2', 'w3']

    # Access vector corresponding to word index 10
    v1[10]

    # Access embedding attributes
    v1.get('similarities')


WordEmbeddingAnalysis
---------------------

Class used to plot, analyze, and compare word embeddings.
Embedding can be either a single or list of WordEmbedding objects.
For any method of WordEmbeddingAnalysis, if an embedding is passed as
parameter then it will overwrite internal embedding assigned during
instantiation or other method invocations.


Example 2
^^^^^^^^^

.. code-block:: text

    # This example create a WordEmbeddingAnalysis object and loads multiple
    # WordEmbedding objects to analyze.

    from VETk import WordEmbedding, WordEmbeddingAnalysis

    # Assume v1, v16, and v32 are available from Example 1
    va = WordEmbeddingAnalysis([v1, v16, v32])

    # Generate histogram from given vector attributes
    va.histogram(['similarities', 'point distances'])

    # Run PCA reduction using 2 components and K-means clustering with 8
    # clusters on embedding matrix
    va.reduction_clustering(reduce='pca', cluster='kmeans', dim=2, num_clusters=8)
