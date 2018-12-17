"""WordEmbedding

Classes:
    :class:`WordEmbedding`

Todo:
    * For analogy tests or topN methods, add a flag for caching. Caching can be
      implemented using a dictionary of a given size, that maps words (not
      vectors).
    * Support binary embeddings.
"""
# https://medium.com/@luckylwk/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
# https://docs.scipy.org/doc/scipy/reference/stats.html
# https://stackoverflow.com/questions/10374930/matplotlib-annotating-a-3d-scatter-plot
# https://www.datacamp.com/community/tutorials/apache-spark-python
# https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
# http://scikit-learn.org/stable/modules/model_evaluation.html


# Notes:
#There is no "k-means algorithm". There is MacQueens algorithm for k-means, the
#Lloyd/Forgy algorithm for k-means, the Hartigan-Wong method, ...
#
#There also isn't "the" EM-algorithm. It is a general scheme of repeatedly
#expecting the likelihoods and then maximizing the model. The most popular
#variant of EM is also known as "Gaussian Mixture Modeling" (GMM), where the
#model are multivariate Gaussian distributions.


import os
import sys
import math
import re
import numpy
import collections
import copy
import scipy


__all__ = ['WordEmbedding']


class WordEmbedding:
    """Represents a vector embedding model

    Notes:
        * Only supports word2vec vector/vocabulary file formats in ASCII
        * Datatype of vectors is set to float32 for computing efficiency
        * Vocabulary should not contain class delimiter (default is blank)

    .. code: python

        v.set_processing_count(k=1, frac=1)
        v.set_processing_count(k=2, frac=1)
        v.set_processing_count(k=3, frac=0.25)
    """
    FPPrecision = 6
    ColWidth = 10
    Delim = ' '

    def __init__(self, **kwargs):
        self.dtype = kwargs.get('dtype', numpy.float32)
        self.label = kwargs.get('label', "")

        self._fileVocabulary = ""
        self.vocabulary = collections.OrderedDict()

        self._fileVectors = ""
        self.vectors = numpy.empty(shape=0, dtype=self.dtype)

        self.fileVocabulary = kwargs.get('vocabulary', "")
        self.fileVectors = kwargs.get('vectors', "")

        self.similarities = numpy.empty(shape=0, dtype=self.dtype)
        self.distances = numpy.empty(shape=0, dtype=self.dtype)
        self.point_distances = numpy.empty(shape=0, dtype=self.dtype)
        self.angle_pairs = numpy.empty(shape=0, dtype=self.dtype)
        self.angle_triplets = numpy.empty(shape=0, dtype=self.dtype)

        self._vectorProcessingCount = 0 # number of vectors to process
        self._pairProcessingCount = 0 # number of vector pairs to process
        self._tripletProcessingCount = 0 # number of vector triplets to process


    def __str__(self):
        return "Label: {label}:\n" \
               "Files:\n" \
               "    Vectors: {vectors}\n" \
               "    Vocabulary: {vocabulary}\n" \
               "Sizes:\n" \
               "    Vectors: {vec_size}\n" \
               "    Vocabulary: {vocab_size}" \
               .format(label=self.label,
                       vectors=os.path.abspath(self.fileVectors),
                       vocabulary=os.path.abspath(self.fileVocabulary),
                       vec_size=self.vectors.shape,
                       vocab_size=len(self.vocabulary))

    def __len__(self):
        return self.vectors.shape[0]

    def __getitem__(self, *keys):
        """Use dictionary syntax to access vector/vocabulary.

        Args:
            keys (str, int, slice, iterable): If string, then consider it as a
                vocabulary word and return vector. If key is integer, then
                consider it as an index and return tuple with vocabulary word
                and vector. Key can be a slice or an iterable of integers or
                strings.

        Returns:
            numpy.ndarray, tuple: Vector or tuple with word and vector.

        Raises:
            KeyError: If *keys* is not found.
        """
        data = []
        for key in keys:
            try:
                # Using list(zip()) runs the generator fully. To return a
                # generator need to remove list().
                # Note: slices always return a list
                if isinstance(key, slice):
                    if key != slice(None):
                        return list(zip(list(self.vocabulary)[key], self.vectors[key]))
                    else:
                        return list(zip(self.vocabulary, self.vectors))

                else:
                    # Ensure 'key' is a single level iterable to allow loop processing.
                    # When an iterable or multiple keys are passed, the arguments are
                    # automatically organized as a tuple of tuple of values
                    # ((arg1,arg2),)
                    if not hasattr(key, '__iter__') or isinstance(key, str):
                        key = [key]

                    # str: returns a list of vectors [v1,v2,...]
                    # int: returns a list of tuples [(w1, v1),(w2,v2),...]
                    if all(isinstance(k, str) for k in key):
                        vocab = list(self.vocabulary)
                        data = [self.vectors[vocab.index(k)] for k in key]
                    elif all(isinstance(k, int) for k in key):
                        vocab = list(self.vocabulary)
                        data = [(vocab[k], self.vectors[k]) for k in key]
                    else:
                        raise KeyError('Invalid key \'{}\' for {} '
                                       'object'.format(key, repr(self)))
            except (IndexError, ValueError) as ex:
                print(ex)

        # Return the items when there is only a single item
        # This is important for supporting vector arithmetic.
        if len(data) == 1: data = data[0]

        return data

    @property
    def fileVectors(self):
        return self._fileVectors

    @fileVectors.setter
    def fileVectors(self, file):
        """Load data from given vector file."""
        if not file:
            self._fileVectors = ""
            if self.vectors.size > 0:
                self.vectors = numpy.empty(shape=0, dtype=self.dtype)
        else:
            self._fileVectors = file
            self.load_vectors(file)

    @property
    def fileVocabulary(self):
        return self._fileVocabulary

    @fileVocabulary.setter
    def fileVocabulary(self, file):
        """Assign and load data from given vocabulary file."""
        if not file:
            self._fileVocabulary = ""
            if len(self.vocabulary) > 0:
                self.vocabulary = collections.OrderedDict()
        else:
            self._fileVocabulary = file
            self.load_vocabulary(file)

    def load_embedding_model(self, vectors="", vocab="", n=1):
        """Load an embedding model and vocabulary.

        Currently, only supports word2vec ASCII output.

        Args:
            vectors: See :meth:`load_vectors`.
            vocab: See :meth:`load_vocab`.
            n: See :meth:`load_vectors` or :meth:`load_vocabulary`.
        """
        # Vocabulary file has precedence over vector file because if a
        # vocabulary file is provided then vocabulary is extracted from it,
        # else the vocabulary is extracted from vector file.
        self.load_vocabulary(vocab, n)
        self.load_vectors(vectors, n)

    def load_vectors(self, vectors, n=1):
        """Load an embedding model from word2vec output.

        Currently only supports ASCII format.
        If vocabulary is empty then the words in vector file will be used to
        populate the vocabulary (occurrences will be set to 0).

        .. code: python

            # Process entire file
            v.load_vectors("...")
            v.load_vectors("...", n=1)

            # Process first 10 lines
            v.load_vectors("...", n=10)

            # Process lines 10 through 20, [10,20]
            v.load_vectors("...", n=slice(10,21))

            # Process every other line from lines 10 through 20, [10,20]
            v.load_vectors("...", n=slice(10,21,2))

        Args:
            vectors (str, numpy.ndarray): If string, then consider it as a
                vector file based on original word2vec format, words and
                vectors. Vectors are ordered by occurrences in decreasing order
                which is the default behavior of original word2vec. If
                numpy.ndarray, then a deep copy is performed.

            n (float, int, slice): If float (0,1], then it is used as a
                fraction corresponding to first n*N words. If int, then it is
                used to select the first n words, N[:floor(n)]. If slice, then
                it is used as a range of words, N[n1:n2:ns].

        Raises:
            ValueError: If *vectors* is not a valid type.
        """
        if isinstance(vectors, str):
            self._fileVectors = vectors
            self.vectors = numpy.empty(shape=0, dtype=self.dtype)

            # Calculate lines to process
            if isinstance(n, slice):
                line_begin = n.start
                line_end = n.stop
                line_step = n.step if n.step else 1
            else:
                # Get number of lines from file only if n = (0,1).
                # If n = 1, then set line_end = -1 to process all lines.
                # Subtract 1 from file lines to ignore line with embedding dimensions.
                nlines = n if n > 1 else self._get_nlines_from_file(self._fileVectors) - 1
                line_begin = 0
                line_end = -1 if n == 1 else self._processing_count(nlines, n)
                line_step = 1

            if line_end != 0:
                try:
                    with open(self._fileVectors) as fd:
                        dims = tuple(int(dim) for dim in fd.readline().strip().split())
                        if line_end > 0:
                            self.vectors = numpy.empty(shape=(line_end - line_begin, dims[1]), dtype=self.dtype)
                        else:
                            self.vectors = numpy.empty(shape=dims, dtype=self.dtype)

                        if len(self.vocabulary) == 0:
                            line_curr = line_begin
                            for i, line in enumerate(fd):
                                # Do not stop if line_end is negative
                                if line_end > 0 and i >= line_end: break
                                if i < line_begin: continue
                                if i == line_curr:
                                    word, vector = line.strip().split(maxsplit=1)
                                    self.vectors[i][:] = numpy.fromstring(vector, sep=' ', dtype=self.dtype)
                                    self.vocabulary[word] = 0
                                    line_curr += line_step
                        else:
                            line_curr = line_begin
                            for i, line in enumerate(fd):
                                # Do not stop if line_end is negative
                                if line_end > 0 and i >= line_end: break
                                if i < line_begin: continue
                                if i == line_curr:
                                    word, vector = line.strip().split(maxsplit=1)
                                    self.vectors[i][:] = numpy.fromstring(vector, sep=' ', dtype=self.dtype)
                                    line_curr += line_step
                except UnicodeDecodeError as ex:
                    with open(self._fileVectors, 'rb') as fd:
                        dims = tuple(int(dim) for dim in fd.readline().strip().split())
                        if line_end > 0:
                            self.vectors = numpy.empty(shape=(line_end - line_begin, dims[1]), dtype=self.dtype)
                        else:
                            self.vectors = numpy.empty(shape=dims, dtype=self.dtype)

                        if len(self.vocabulary) == 0:
                            line_len = self.vectors.shape[1] * numpy.dtype(self.dtype).itemsize
                            line_curr = line_begin
                            vector = b''
                            i = -1
                            for line_part in fd:
                                # First pass for current line
                                if not vector:
                                    word, vector = line_part.split(maxsplit=1)
                                    continue

                                # Accumulate a vector in bytes
                                vector += line_part
                                if len(vector) < line_len:
                                    continue
                                i += 1

                                # Do not stop if line_end is negative
                                if line_end > 0 and i >= line_end: break
                                if i < line_begin:
                                    vector = b''
                                    continue
                                if i == line_curr:
                                    self.vectors[i][:] = numpy.frombuffer(vector[:-1], dtype=self.dtype)
                                    self.vocabulary[word] = 0
                                    line_curr += line_step
                                    vector = b''
        elif isinstance(vectors, numpy.ndarray):
            self._fileVectors = ""

            # Calculate lines to process
            lines = n if isinstance(n, slice) else slice(0, self._processing_count(vectors.shape[0], n), 1)

            if lines.stop > 0:
                self.vectors = copy.deepcopy(vectors[lines])
            else:
                self.vectors = numpy.empty(shape=0, dtype=self.dtype)
        else:
            self._fileVectors = ""
            self.vectors = numpy.empty(shape=0, dtype=self.dtype)
            raise ValueError('ERROR: Invalid vectors type \'{}\', allowed values are '
                             'string and numpy.ndarray'.format(type(vectors)))

        self.set_processing_count(k=1, frac=1)
        self.set_processing_count(k=2, frac=1)
        self.set_processing_count(k=3, frac=1)

        if len(self.vocabulary) > 0 and not self._is_consistent():
            print('WARN: Vector and vocabulary are inconsistent in size')

    def load_vocabulary(self, vocab, n=1):
        """Load vocabulary from file or given object.

        .. code: python

            # Process entire file
            v.load_vocabulary("...")
            v.load_vocabulary("...", n=1)

            # Process first 10 lines
            v.load_vocabulary("...", n=10)

            # Process lines 10 through 20, [10,20]
            v.load_vocabulary("...", n=slice(10,21))

            # Process every other line from lines 10 through 20, [10,20]
            v.load_vocabulary("...", n=slice(10,21,2))

        Args:
            vocab (str, collections.OrderedDict): If string, then consider it
                as a vocabulary file with 2 columns, words and occurrences.
                Words are ordered by occurrences in decreasing order which is
                the default behavior of original word2vec. If
                collections.OrderedDict, then a deep copy is performed.

            n (float, int, slice): If float (0,1], then it is used as a
                fraction corresponding to first n*N words. If int, then it is
                used to select the first n words, N[:floor(n)]. If slice, then
                it is used as a range of words, N[n1:n2:ns].

        Raises:
            ValueError: If *vocab* is not a valid type.
        """
        if isinstance(vocab, str):
            self._fileVocabulary= vocab
            self.vocabulary = collections.OrderedDict()

            # Calculate lines to process
            if isinstance(n, slice):
                line_begin = n.start
                line_end = n.stop
                line_step = n.step if n.step else 1
            else:
                # Get number of lines from file only if n = (0,1).
                # If n = 1, then set line_end = -1 to process all lines.
                nlines = n if n > 1 else self._get_nlines_from_file(self._fileVocabulary)
                line_begin = 0
                line_end = -1 if n == 1 else self._processing_count(nlines, n)
                line_step = 1

            if line_end != 0:
                with open(self._fileVocabulary) as fd:
                    line_curr = line_begin
                    for i, line in enumerate(fd):
                        # Do not stop if line_end is negative
                        if line_end > 0 and i >= line_end: break
                        if i < line_begin: continue
                        if i == line_curr:
                            word, count = line.strip().split(maxsplit=1)
                            self.vocabulary[word] = int(count)
                            line_curr += line_step
        elif isinstance(vocab, collections.OrderedDict):
            self._fileVocabulary= vocab
            self.vocabulary = collections.OrderedDict()

            # Calculate lines to process
            if isinstance(n, slice):
                line_begin = n.start
                line_end = n.stop
                line_step = n.step if n.step else 1
            else:
                line_begin = 0
                line_end = self._processing_count(len(vocab), n)
                line_step = 1

            if line_end != 0:
                if line_end == len(vocab):
                    self.vocabulary = copy.deepcopy(vocab)
                else:
                    line_curr = line_begin
                    for i, (word, count) in enumerate(vocab.items()):
                        if i >= line_end: break
                        if i < line_begin: continue
                        if i == line_curr:
                            self.vocabulary[word] = count
                            line_curr += line_step
        else:
            self._fileVocabulary = ""
            self.vocabulary = collections.OrderedDict()
            raise ValueError('Invalid vocabulary type \'{}\', allowed values '
                             'are string and collections.OrderedDict'
                             .format(type(vocab)))

        self.set_processing_count(k=1, frac=1)

        if self.vectors.size > 0 and not self._is_consistent():
            print('WARN: Vector and vocabulary are inconsistent in size')

    def _get_nlines_from_file(self, file, sizehint=1024 * 1024):
        """Calculate number of lines in a given file.

        Args:
            file (str): Input file.

        Returns:
            int: Number of lines in *file*.
        """
        N = 0
        with open(file) as fd:
            buf = fd.read(sizehint)
            while buf:
                N += buf.count('\n')
                buf = fd.read(sizehint)
        return N

    def write_vectors(self, file="", n=1):
        """Write/print vectors from an embedding model.

        Format follows word2vec output (ASCII).

        Args:
            file (str, optional): Input file.
        """
        # Calculate lines to process
        if isinstance(n, slice):
            line_begin = n.start
            line_end = n.stop
            line_step = n.step if n.step else 1
        else:
            line_begin = 0
            line_end = self._processing_count(self.vectors.shape[0], n)
            line_step = 1

        if line_end == 0: return

        if file:
            with open(file, 'w') as fd:
                fmt = ' '.join(['{}'] + self.vectors.shape[1] *
                               ['{:.' + str(self.FPPrecision) + 'f}']) + '\n'
                fd.write('{} {}\n'.format(n, self.vectors.shape[1]))
                line_curr = line_begin
                for i, (word, vector) in enumerate(zip(self.vocabulary.keys(),
                                                       self.vectors)):
                    if i >= line_end: break
                    if i < line_begin: continue
                    if i == line_curr:
                        fd.write(fmt.format(word, *vector))
                        line_curr += line_step
        else:
            fmt = ' '.join(['{}'] + self.vectors.shape[1] *
                           ['{:.' + str(self.FPPrecision) + 'f}'])
            print('{} {}'.format(n, self.vectors.shape[1]))
            line_curr = line_begin
            for i, (word, vector) in enumerate(zip(self.vocabulary.keys(),
                                                   self.vectors)):
                if i >= line_end: break
                if i < line_begin: continue
                if i == line_curr:
                    print(fmt.format(word, *vector))
                    line_curr += line_step

    def write_vocabulary(self, file="", n=1):
        """Write/print vocabulary from an embedding model.

        Format follows word2vec output where words are ordered by occurrences
        in decreasing order.

        Args:
            file (str, optional): Input file.
        """
        # Calculate lines to process
        if isinstance(n, slice):
            line_begin = n.start
            line_end = n.stop
            line_step = n.step if n.step else 1
        else:
            line_begin = 0
            line_end = self._processing_count(len(self.vocabulary), n)
            line_step = 1

        if line_end == 0: return

        if file:
            with open(file, 'w') as fd:
                line_curr = line_begin
                for i, (word, count) in enumerate(self.vocabulary.items()):
                    if i >= line_end: break
                    if i < line_begin: continue
                    if i == line_curr:
                        fd.write('{} {}\n'.format(word, count))
                        line_curr += line_step
        else:
            line_curr = line_begin
            for i, (word, count) in enumerate(self.vocabulary.items()):
                if i >= line_end: break
                if i < line_begin: continue
                if i == line_curr:
                    print('{} {}'.format(word, count))
                    line_curr += line_step

    def clear(self):
        """Reset object to initial state."""
        self.label = ""
        self.fileVectors = ""
        self.fileVocabulary = ""

        self.similarities = numpy.empty(shape=0, dtype=self.dtype)
        self.distances = numpy.empty(shape=0, dtype=self.dtype)
        self.point_distances = numpy.empty(shape=0, dtype=self.dtype)
        self.angle_pairs = numpy.empty(shape=0, dtype=self.dtype)
        self.angle_triplets = numpy.empty(shape=0, dtype=self.dtype)

        self._vectorProcessingCount = 0
        self._pairProcessingCount = 0
        self._tripletProcessingCount = 0

    def _is_consistent(self):
        """Check size consistency between vectors and vocabulary.

        Returns:
            bool: True if vector and vocabulary sizes match, False otherwise.
        """
        return self.vectors.shape[0] == len(self.vocabulary)

    def _flat_size(self, k, n):
        """Calculate number of combinations with k items.

        C(n,k) = n!/(k!(n-k)!))

        Returns:
            int: Number of combinations.
        """
        N = 0
        if k > 0:
            N = math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
        return N

    def _processing_count(self, n, frac=1):
        """
        Calculate number of data to processed based on either:
            * Fraction - n = (0,1]
            * Selection - n > 1
        """
        N = 0
        if frac > 0 and frac <= 1:
            N = math.ceil(frac * (n))
        elif frac > 1:
            N = min(int(frac), n)
        return N

    def set_processing_count(self, k, frac):
        """Set fraction/number of data elements to use.

        Notes:
            * k = 1 - vectors/vocabulary processing
            * k = 2 - vector pairs processing
            * k = 3 - vector triplets processing

        For k = 1:
            * Reduces vectors data (keeps same data ordering)
            * Reduces vocabulary only if size is consistent with vectors
        """
        n = self._processing_count(self.vectors.shape[0], frac)
        if k == 1:
            if self.vectors.shape[0] > n:
                if self._is_consistent():
                    self.vocabulary = collections.OrderedDict(list(self.vocabulary.items())[:n])
                self.vectors = self.vectors[:n]
            self._vectorProcessingCount = self.vectors.shape[0]
        elif k == 2:
            self._pairProcessingCount = n
        elif k == 3:
            self._tripletProcessingCount = n

    def _word_pairs(self, k, index=False):
        """Find word/index pairs/triplets.

        For k = 2:
            * Find words/index pairs that generate the lower/upper triangle of a symmetric matrix
        For k = 3:
            * Find word/index triplets
        """
        pairs = []
        if index:
            if k == 2:
                pairs = [(row, col) for row in range(self._pairProcessingCount)
                                    for col in range(row)]
            elif k == 3:
                pairs = [(x, y, z) for x in range(self._tripletProcessingCount - 2)
                                   for y in range(x + 1, self._tripletProcessingCount - 1)
                                   for z in range(y + 1, self._tripletProcessingCount)]
        else:
            vocab = list(self.vocabulary.keys())
            if k == 2:
                pairs = [(vocab[row], vocab[col]) for row in range(self._pairProcessingCount) for col in range(row)]
            elif k == 3:
                pairs = [(vocab[x], vocab[y], vocab[z])
                            for x in range(self._tripletProcessingCount - 2)
                            for y in range(x + 1, self._tripletProcessingCount - 1)
                            for z in range(y + 1, self._tripletProcessingCount)]
        return pairs

    def scale_vectors(self):
        """Scale vectors to zero mean and unit variance."""
        sklearn.preprocessing.scale(self.vectors, copy=False)

    def get(self, key):
        """
        Returns data corresponding to attribute represented by given key string
        Key is used in lower-case form and spaces are replaced with
        underscores, this allows using 'angle pairs' for self.angle_pairs

        Notes:
            * Pre/trailing whitespaces are ignored
        """
        return self.__dict__[re.sub('\s+', '_', key.strip()).lower()]

    def process_attributes(self, k):
        """Compute vector pair/triplet attributes.

        If k = 2:
            * Calculate vector pair attributes between all vectors and represent as a
              flat array of the row-based lower triangular matrix.
            * Cosine similarities
            * Cosine distances
            * Euclidean point distances
            * Angles (radians)

        If k = 3:
            * Computes the angle between 3 points.
            * O(n!/(k!(n-k)!)) complexity, so choose subset of vector triplets.
        """
        # Calculate linear size of lower/upper half of data matrix
        flatSize = self._flat_size(k=k, n=self._pairProcessingCount)

        if k == 2:
            # Used to consider only unique data from symmetric matrix
            self.similarities = numpy.empty(shape=flatSize, dtype=self.dtype)
            self.distances = numpy.empty(shape=flatSize, dtype=self.dtype)
            self.point_distances = numpy.empty(shape=flatSize, dtype=self.dtype)
            self.angle_pairs = numpy.empty(shape=flatSize, dtype=self.dtype)

            idx = 0
            for row in range(self._pairProcessingCount):
                for col in range(row):
                    va = self.vectors[row]
                    vb = self.vectors[col]
                    similarity = numpy.dot(va, vb) / (numpy.linalg.norm(va) *
                                 numpy.linalg.norm(vb))

                    # Check bounds due to numerical errors
                    if similarity > 1.0:
                        similarity = 1.0
                    elif similarity < -1.0:
                        similarity = -1.0

                    self.similarities[idx] = similarity
                    self.distances[idx] = 1 - self.similarities[idx]
                    self.point_distances[idx] = numpy.linalg.norm(va - vb)
                    self.angle_pairs[idx] = numpy.arccos(self.similarities[idx])
                    idx += 1
        elif k == 3:
            self.angle_triplets = numpy.empty(shape=flatSize, dtype=self.dtype)

            idx = 0
            for x in range(self._tripletProcessingCount - 2):
                for y in range(x + 1, self._tripletProcessingCount - 1):
                    for z in range(y + 1, self._tripletProcessingCount):
                        va = self.vectors[x] - self.vectors[y]
                        vb = self.vectors[z] - self.vectors[y]
                        similarity = numpy.dot(va, vb) / (numpy.linalg.norm(va)
                                     * numpy.linalg.norm(vb))

                        # Check bounds due to numerical errors
                        if similarity > 1.0:
                            similarity = 1.0
                        elif similarity < -1.0:
                            similarity = -1.0

                        self.angle_triplets[idx] = numpy.arccos(similarity)
                        idx += 1

    def write_stats(self, file=""):
        """Write/print statistics data.

        Args:
            file (str, optional): Output file.
        """
        # Headers and labels should not contain delimiter.
        headers = ['data', 'n', 'min', 'max', 'mean', 'var', 'stddev', 'skew', 'kurtosis', 'percent_25', 'percent_50', 'percent_75']
        labels = ['similarities', 'distances', 'angle_pairs', 'point_distances', 'angle_triplets']

        statistics = []
        percentiles = []
        for i, d in enumerate(self.get(key) for key in labels):
            if len(d) == 0: continue
            statistics.append(scipy.stats.describe(d))
            percentiles.append(numpy.percentile(d, [25, 50, 75]))

        k = 2  # number of initial columns that are not floating-point
        if file:
            with open(file, 'w') as fd:
                gfmt =  '{}'
                fpfmt = '{:.' + str(self.FPPrecision) + 'f}'
                hfmt = self.Delim.join(len(headers)*[gfmt]) + '\n'
                dfmt = self.Delim.join(k*[gfmt] + (len(headers)-k)*[fpfmt]) + '\n'
                fd.write(hfmt.format(*headers))
                for label, st, percentile in zip(labels, statistics, percentiles):
                    fd.write(dfmt.format(label, st.nobs, *st.minmax, st.mean,
                    st.variance, numpy.sqrt(st.variance), st.skewness, st.kurtosis, *percentile))
        else:
            gfmt =  '{:<' + str(self.ColWidth) + '}'
            fpfmt = '{:<' + str(self.ColWidth) + '.' + str(self.FPPrecision) + 'f}'
            hfmt = self.Delim.join(len(headers)*[gfmt])
            dfmt = self.Delim.join(k*[gfmt] + (len(headers)-k)*[fpfmt])
            print(hfmt.format(*headers))
            for label, st, percentile in zip(labels, statistics, percentiles):
                print(dfmt.format(label, st.nobs, *st.minmax, st.mean,
                st.variance, numpy.sqrt(st.variance), st.skewness, st.kurtosis, *percentile))

    def write_attributes(self, k, file="", index=False):
        """Write/print attributes data.

        Args:
            file (str, optional): Output file.
        """
        # Headers and vocabulary should not contain delimiter.
        if k == 2:
            headers = ['word1', 'word2', 'similarities', 'distances', 'angle_pairs', 'point_distances']
        elif k == 3:
            headers = ['word1', 'word2', 'word3', 'angle_triplets']

        if file:
            with open(file, 'w') as fd:
                gfmt =  '{}'
                fpfmt = '{:<' + str(self.ColWidth) + '.' + str(self.FPPrecision) + 'f}'
                hfmt = self.Delim.join(len(headers)*[gfmt]) + '\n'
                dfmt = self.Delim.join(k*[gfmt] + (len(headers)-k)*[fpfmt]) + '\n'
                fd.write(hfmt.format(*headers))
                for data in zip(self._word_pairs(k, index=index), *[self.get(key) for key in headers[k:]]):
                    fd.write(dfmt.format(*data[0], *data[1:]))
        else:
            gfmt =  '{:<' + str(self.ColWidth) + '}'
            fpfmt = '{:<' + str(self.ColWidth) + '.' + str(self.FPPrecision) + 'f}'
            hfmt = self.Delim.join(len(headers)*[gfmt])
            dfmt = self.Delim.join(k*[gfmt] + (len(headers)-k)*[fpfmt])
            print(hfmt.format(*headers))
            for data in zip(self._word_pairs(k, index=index), *[self.get(key) for key in headers[k:]]):
                print(dfmt.format(*data[0], *data[1:]))

    def load_attributes(self, file):
        """Load attributes data.

        Notes:
            * Type of processing count (k) is inferred from 'word#' count in headers.
        """
        # Count number of lines for allocating
        with open(file) as fd:
            headers = fd.readline().strip().split(self.Delim)
            for i, _ in enumerate(fd): continue
            nlines = i + 1

        # Identify k based on headers keyword
        for k, h in enumerate(headers):
            if 'word' not in h: break

        if k == 2:
            self._pairProcessingCount = nlines
        elif k == 3:
            self._tripleProcessingCount = nlines

        # Allocate arrays
        for i, h in enumerate(headers):
            if i >= k:
                self.__dict__[h] = numpy.empty(shape=nlines, dtype=self.dtype)

        with open(file) as fd:
            fd.readline()  # skip headers
            for i, line in enumerate(fd):
                data = line.strip().split(self.Delim)
                for j, h in enumerate(headers):
                    if j >= k:
                        self.__dict__[h][i] = self.dtype(data[j])
