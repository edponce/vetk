"""WordEmbedding

Classes:
    :class:`WordEmbedding`
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
        * Preferably, vocabulary should not contain whitespace (default delimiter)
    """
    FPPrecision = 6
    ColWidth = 10
    Delim = ' '


    def __init__(self, label="", **kwargs):
        self._dtype = kwargs.get('dtype', numpy.float32)
        self._label = ""
        self.label = kwargs.get('label', "")
        self._filePrefix = ""
        self.filePrefix = kwargs.get('prefix', "")

        self._vectorProcessingCount = 0 # number of vectors to process
        self._pairProcessingCount = 0 # number of vector pairs to process
        self._tripletProcessingCount = 0 # number of vector triplets to process

        self.vectors = numpy.empty(shape=0, dtype=self._dtype)
        self.vocabulary = collections.OrderedDict()

        self.similarities = numpy.empty(shape=0, dtype=self._dtype)
        self.distances = numpy.empty(shape=0, dtype=self._dtype)
        self.point_distances = numpy.empty(shape=0, dtype=self._dtype)
        self.angle_pairs = numpy.empty(shape=0, dtype=self._dtype)
        self.angle_triplets = numpy.empty(shape=0, dtype=self._dtype)

        # Load data (vocabulary has precedence over vectors)
        self._fileVectors = ""
        self._fileVocabulary = ""
        self.fileVocabulary = kwargs.get('vocabulary', "")
        self.fileVectors = kwargs.get('vectors', "")

    def __str__(self):
        return "Label: {label}:\n" \
               "Files:\n" \
               "    Prefix: {prefix}\n" \
               "    Vocabulary: {vocabulary}\n" \
               "    Vectors: {vectors}\n" \
               "Sizes:\n" \
               "    Vocabulary: {vocab_size}\n" \
               "    Vectors: {vec_size}" \
               .format(label=self.label,
                       prefix=self.filePrefix,
                       vocabulary=self.fileVocabulary,
                       vectors=self.fileVectors,
                       vocab_size=len(self.vocabulary),
                       vec_size=self.vectors.shape)

    def __getitem__(self, *keys):
        '''
        Use dictionary syntax to access vocabulary/vector

        Notes:
            * If key is string, then consider it as a vocabulary word and
              return vector
            * If key is integer, then consider it as an index and return tuple
              with vocabulary word and vector
            * Key can be a slice or an iterable of integers or strings

        Todo:
            * Should invalid keys be simply ignored or raise exception?
        '''
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

        # Re    turn the items when there is only a single item
        # This is important for supporting vector arithmetic.
        if len(data) == 1: data = data[0]

        return data

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        if not isinstance(label, str):
            raise TypeError("label is not a {}".format(str))
        self._label = label

    @property
    def filePrefix(self):
        return self._filePrefix

    @filePrefix.setter
    def filePrefix(self, prefix):
        if not isinstance(prefix, str):
            raise TypeError("filePrefix is not a {}".format(str))
        self._filePrefix = prefix

        # Always end prefix with a dash/underscore because it is prepended to filenames
        # Todo:
        #   * Move this logic to the moment when it is used
        if prefix and prefix[-1] not in ['-', '_']:
            self._filePrefix += '-'

    @property
    def fileVectors(self):
        return self._fileVectors

    @fileVectors.setter
    def fileVectors(self, file):
        """
        Notes:
            * Vectors are loaded only if a new and non-empty string is provided
            * If an empty string is provided, then vectors array is cleared
              and vocabulary (if it was loaded from vector file)
        """
        if not isinstance(file, str):
            raise TypeError("fileVectors is not a {}".format(str))
        if not file:
            self._fileVectors = ''
            if self.vectors.size > 0:
                self.vectors = numpy.empty(shape=0, dtype=self._dtype)

            # Clear vocabulary if it was loaded from vector file
            if not self.fileVocabulary and len(self.vocabulary) > 0:
                self.vocabulary = collections.OrderedDict()
        elif file != self.fileVectors:
            # Argument assignment needs to occur before loading to prevent
            # infinite recursion
            self._fileVectors = file
            self.load_word2vec(file)


    @property
    def fileVocabulary(self):
        return self._fileVocabulary


    @fileVocabulary.setter
    def fileVocabulary(self, arg):
        '''Assign and load data from given vocabulary file

        Notes:
            * Vocabulary is loaded only if a new and non-empty string is provided
            * If an empty string is provided, then vocabulary dictionary is cleared
        '''
        if not arg:
            self._fileVocabulary = ''
            if len(self.vocabulary) > 0:
                self.vocabulary = collections.OrderedDict()
        elif arg != self.fileVocabulary:
            # Argument assignment needs to occur before loading to prevent
            # infinite recursion
            self._fileVocabulary = arg
            self.load_vocabulary(arg)


    def load_word2vec(self, vectors, n=0):
        '''
        Load an embedding model and vocabulary from word2vec output (ASCII format).
        Vocabulary consists of a dictionary word strings as keys and
        occurrences as values.

        Assumes words are ordered by occurrences in decreasing order (which is the default
        behavior of original word2vec).

        To index into vocabulary:
        * list(v.vocabulary)[index]
        To find index of a word:
        * list(v.vocabulary).index('word')

        Notes:
            * n specifies number of lines to load (0 = all)
            * sets 100% processing count for k=1,2
            * sets 25% processing count for k=3
        '''
        if isinstance(vectors, str):
            try:
                with open(vectors) as fd:
                    self.fileVectors = os.path.abspath(vectors)
                    dims = tuple(int(dim) for dim in fd.readline().strip().split())
                    self.vectors = numpy.empty(shape=dims, dtype=self._dtype)
                    if len(self.vocabulary) == 0:
                        self.vocabulary = collections.OrderedDict()
                        for i, line in enumerate(fd):
                            if n > 0 and i >= n: break
                            word, vector = line.strip().split(maxsplit=1)
                            self.vectors[i][:] = numpy.fromstring(vector, sep=' ',
                                                               dtype=self._dtype)
                            self.vocabulary[word] = 1
                    else:
                        for i, line in enumerate(fd):
                            if n > 0 and i >= n: break
                            word, vector = line.strip().split(maxsplit=1)
                            self.vectors[i][:] = numpy.fromstring(vector, sep=' ',
                                                               dtype=self._dtype)
            except Exception as ex:
                self.fileVectors = ''
                self.vectors = numpy.empty(shape=0, dtype=numpy.float32)
                raise Exception('ERROR: {}'.format(ex))
        elif isinstance(vectors, numpy.ndarray):
            self.fileVectors = ''
            self.vectors = copy.deepcopy(vectors)
        else:
            self.fileVectors = ''
            self.vectors = numpy.empty(shape=0, dtype=numpy.float32)
            raise ValueError('ERROR: Invalid vectors type \'{}\', allowed values are '
                             'string and numpy.ndarray'.format(type(vectors)))

        self.set_processing_count(k=1, frac=1)
        self.set_processing_count(k=2, frac=1)
        self.set_processing_count(k=3, frac=0.25)

        if len(self.vocabulary) > 0 and not self._is_consistent():
            print('WARN: Vector and vocabulary are inconsistent in size')


    def load_vocabulary(self, vocab, n=0):
        '''
        Load vocabulary from file containing 2 columns
        * Words
        * Occurrences
        Words are ordered by occurrences in decreasing order (which is the default
        behavior of original word2vec).

        Notes:
            * n specifies number of lines to load (0 = all)
        '''
        if isinstance(vocab, str):
            try:
                with open(vocab) as fd:
                    self.fileVocabulary= os.path.abspath(vocab)
                    self.vocabulary = collections.OrderedDict()
                    for i, line in enumerate(fd):
                        if n > 0 and i >= n: break
                        word, count = line.strip().split(maxsplit=1)
                        self.vocabulary[word] = int(count)
            except Exception as e:
                self.fileVocabulary = ''
                self.vocabulary = collections.OrderedDict()
                raise Exception('ERROR: {}'.format(e))
        elif isinstance(vocab, collections.OrderedDict) and len(vocab) > 0:
            self.vocabulary = copy.deepcopy(vocab)
        else:
            self.fileVocabulary = ''
            self.vocabulary = collections.OrderedDict()
            raise ValueError('Invalid vocabulary type \'{}\', allowed values '
                             'are string and collections.OrderedDict'
                             .format(type(vocabulary)))

        if self.vectors.size > 0 and not self._is_consistent():
            print('WARN: Vector and vocabulary are inconsistent in size')


    def write_vectors(self, file='', n=0):
        '''
        Write an embedding model and vocabulary in word2vec output (ASCII format).
        Vocabulary consists of word objects with index and count attributes, assumes
        words are ordered by occurrences in decreasing order (which is the default
        behavior of original word2vec).

        Notes:
            * n specifies number of lines to load (0 = all)
        '''
        if n == 0:
            n = self.vectors.shape[0]
        else:
            n = min(n, self.vectors.shape[0])

        if file:
            with open(self.filePrefix + file, 'w') as fd:
                fmt = ' '.join(['{}'] + self.vectors.shape[1] *
                               ['{:.' + str(self.FPPrecision) + 'f}']) + '\n'
                fd.write('{} {}\n'.format(n, self.vectors.shape[1]))
                for i, (word, vector) in enumerate(zip(self.vocabulary.keys(),
                                                       self.vectors)):
                    if n > 0 and i >= n: break
                    fd.write(fmt.format(word, *vector))
        else:
            fmt = ' '.join(['{}'] + self.vectors.shape[1] *
                           ['{:.' + str(self.FPPrecision) + 'f}'])
            print('{} {}'.format(n, self.vectors.shape[1]))
            for i, (word, vector) in enumerate(zip(self.vocabulary.keys(),
                                                   self.vectors)):
                if n > 0 and i >= n: break
                print(fmt.format(word, *vector))


    def write_vocabulary(self, file='', n=0):
        '''
        Write vocabulary to file containing 2 columns
        * Words
        * Occurrences
        Words are ordered by occurrences in decreasing order (which is the default
        behavior of original word2vec).

        Notes:
            * n specifies number of lines to write (0 = all)
            * Vocabulary should not contain delimiter
        '''
        if n == 0:
            n = len(self.vocabulary)
        else:
            n = min(n, len(self.vocabulary))

        if file:
            with open(self.filePrefix + file, 'w') as fd:
                for i, (word, count) in enumerate(self.vocabulary.items()):
                    if n > 0 and i >= n: break
                    fd.write('{} {}\n'.format(word, count))
        else:
            for i, (word, count) in enumerate(self.vocabulary.items()):
                if n > 0 and i >= n: break
                print('{} {}'.format(word, count))


    def clear(self):
        '''
        Reset object to initial state
        '''
        self._vectorProcessingCount = 0
        self._pairProcessingCount = 0
        self._tripletProcessingCount = 0

        #self.vectors = numpy.empty(shape=0, dtype=self._dtype)
        #self.vocabulary = collections.OrderedDict()
        self.label = ""
        self.filePrefix = ""
        self.fileVectors = ""
        self.fileVocabulary = ""
        self.similarities = numpy.empty(shape=0, dtype=self._dtype)
        self.distances = numpy.empty(shape=0, dtype=self._dtype)
        self.point_distances = numpy.empty(shape=0, dtype=self._dtype)
        self.angle_pairs = numpy.empty(shape=0, dtype=self._dtype)
        self.angle_triplets = numpy.empty(shape=0, dtype=self._dtype)


    def _is_consistent(self):
        '''
        Check size consistency between vectors and vocabulary
        '''
        return self.vectors.shape[0] == len(self.vocabulary)


    def _flat_size(self, k, n):
        '''
        Calculate number of combinations with k items
        C(n,k) = n!/(k!(n-k)!))
        '''
        N = 0
        if k > 0:
            N = math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
        return N


    def _processing_count(self, n, frac=1):
        '''
        Calculate number of data to processed based on either:
        * Fraction - n = (0,1]
        * Selection - n > 1
        '''
        N = 0
        if frac > 0 and frac <= 1:
            N = math.ceil(frac * (n))
        elif frac > 1:
            N = min(int(frac), n)
        return N


    def set_processing_count(self, k, frac):
        '''
        Set fraction/number of data elements to use:
        * k = 1 - vectors/vocabulary processing
        * k = 2 - vector pairs processing
        * k = 3 - vector triplets processing

        For k = 1:
        * Reduces vectors data (keeps same data ordering)
        * Reduces vocabulary only if size is consistent with vectors
        '''
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
        '''Find word/index pairs/triplets
        For k = 2:
        * Find words/index pairs that generate the lower/upper triangle of a symmetric matrix
        For k = 3:
        * Find word/index triplets
        '''
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
        '''
        Scale vectors to zero mean and unit variance
        '''
        sklearn.preprocessing.scale(self.vectors, copy=False)


    def get(self, key):
        '''
        Returns data corresponding to attribute represented by given key string
        Key is used in lower-case form and spaces are replaced with
        underscores, this allows using 'angle pairs' for self.angle_pairs

        Notes:
            * Pre/trailing whitespaces are ignored
        '''
        return self.__dict__[re.sub('\s+', '_', key.strip()).lower()]


    def process_attributes(self, k):
        '''Compute vector pair/triplet attributes
        If k = 2:
        Calculate vector pair attributes between all vectors and represent as a
        flat array of the row-based lower triangular matrix.
        * Cosine similarities
        * Cosine distances
        * Euclidean point distances
        * Angles (radians)

        If k = 3:
        Computes the angle between 3 points.
        O(n!/(k!(n-k)!)) complexity, so choose subset of vector triplets.
        '''
        # Calculate linear size of lower/upper half of data matrix
        flatSize = self._flat_size(k=k, n=self._pairProcessingCount)

        if k == 2:
            # Used to consider only unique data from symmetric matrix
            self.similarities = numpy.empty(shape=flatSize, dtype=self._dtype)
            self.distances = numpy.empty(shape=flatSize, dtype=self._dtype)
            self.point_distances = numpy.empty(shape=flatSize, dtype=self._dtype)
            self.angle_pairs = numpy.empty(shape=flatSize, dtype=self._dtype)

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
            self.angle_triplets = numpy.empty(shape=flatSize, dtype=self._dtype)

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


    def write_stats(self, file=''):
        '''
        Write/print data statistics

        Notes:
            * Headers and labels should not contain delimiter
        '''
        headers = ['data', 'n', 'min', 'max', 'mean', 'var', 'stddev', 'skew', 'kurtosis', 'percent_25', 'percent_50', 'percent_75']
        labels = ['similarities', 'distances', 'angle_pairs', 'point_distances', 'angle_triplets']

        statistics = []
        percentiles = []
        for i, d in enumerate(self[key] for key in labels):
            if len(d) == 0: continue
            statistics.append(scipy.stats.describe(d))
            percentiles.append(numpy.percentile(d, [25, 50, 75]))

        k = 2  # number of initial columns that are not floating-point
        if file:
            with open(self.filePrefix + file,  'w') as fd:
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


    def write_attributes(self, k, file='', index=False):
        '''
        Write/print data

        Notes:
            * Headers and vocabulary should not contain delimiter
        '''
        if k == 2:
            headers = ['word1', 'word2', 'similarities', 'distances', 'angle_pairs', 'point_distances']
        elif k == 3:
            headers = ['word1', 'word2', 'word3', 'angle_triplets']

        if file:
            with open(self.filePrefix + file, 'w') as fd:
                gfmt =  '{}'
                fpfmt = '{:<' + str(self.ColWidth) + '.' + str(self.FPPrecision) + 'f}'
                hfmt = self.Delim.join(len(headers)*[gfmt]) + '\n'
                dfmt = self.Delim.join(k*[gfmt] + (len(headers)-k)*[fpfmt]) + '\n'
                fd.write(hfmt.format(*headers))
                for data in zip(self._word_pairs(k, index=index), *[self[key] for key in headers[k:]]):
                    fd.write(dfmt.format(*data[0], *data[1:]))
        else:
            gfmt =  '{:<' + str(self.ColWidth) + '}'
            fpfmt = '{:<' + str(self.ColWidth) + '.' + str(self.FPPrecision) + 'f}'
            hfmt = self.Delim.join(len(headers)*[gfmt])
            dfmt = self.Delim.join(k*[gfmt] + (len(headers)-k)*[fpfmt])
            print(hfmt.format(*headers))
            for data in zip(self._word_pairs(k, index=index), *[self[key] for key in headers[k:]]):
                print(dfmt.format(*data[0], *data[1:]))


    def load_attributes(self, file):
        '''Load data

        Notes:
            * Type of processing count (k) is identified from 'word' in headers
        '''
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
                self.__dict__[h] = numpy.empty(shape=nlines, dtype=self._dtype)

        with open(file) as fd:
            fd.readline()  # skip headers
            for i, line in enumerate(fd):
                data = line.strip().split(self.Delim)
                for j, h in enumerate(headers):
                    if j >= k:
                        self.__dict__[h][i] = self._dtype(data[j])
