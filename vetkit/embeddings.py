"""Class representing a vector embedding model of words.

Classes:
    :class:`WordEmbedding`

Todo:
    * Change *filter* parameter/attribute name because of built-in filter().
    * Provide list-like and slice support to vocabulary data structure.
    * Vectors can be filtered by contiguous range, fraction, amount, blacklist,
      whitelist, occurrence values, regex, etc.. Should filtering be performed
      during load, runtime, or both?
        * For very large datasets, filtering during load is necessary.
        * Runtime can be easily implemented with a method to trigger filtering.
        * Current implementation supports filtering only during load.
    * Add a verbosity data member.
    * For functions/methods that support multiprocessing, add a variant
      triggered that is triggered by class member to control threads/processes.
"""
# https://docs.scipy.org/doc/scipy/reference/stats.html


import os
import re
from collections import OrderedDict
from copy import deepcopy
import numpy
import scipy
from .utils import (convert_to_range, n_choose_k, tokenize_file)
from .word2vec_models import (load_vectors_word2vec, load_vocabulary_word2vec,
                              dump_vectors_word2vec, dump_vocabulary_word2vec)
from .glove_models import (load_vectors_glove, load_vocabulary_glove,
                           dump_vectors_glove, dump_vocabulary_glove)


__all__ = ['WordEmbedding']


class WordEmbedding:
    """Represents a vector embedding model of words.

    Args:
        vectors (str, numpy.ndarray, list, tuple, optional): If string, then
            consider it as a file in word2vec format. If numpy.ndarray, then a
            deep copy is performed. If list or tuple, a copy is performed.
            Default is empty numpy.ndarray.
        vocabulary (str, dict, optional): If string, then consider it as a file
            in word2vec format. If dict, then a deep copy is performed. Default
            is empty OrderedDict.
        label (str, optional): Name/identifier of instance. Default is empty
            string.
        model (str, optional): Name of model to use for loading/dumping word
            embeddings. Defaults is 'word2vec'.
        filter (range, slice, list, tuple, float, int, set, dict, None): Values
            representing a range to filter file processing, see
            *utils.convert_to_range()*. Filtering is only performed during load
            operations (runtime or dump are not filtered). Default is 1.0.
        blacklist (bool): If True, consider *filter* as a blacklist.  If False,
            consider *filter* as a whitelist. Only applicable when *filter* is
            a set or dict. Default is False.
        dtype (numpy.ndarray, optional): Type of vector embedding values.
            Default is numpy.float32.

    Notes:
        * Vectors and vocabulary should match and be ordered by word
          occurrences in decreasing order.
        * Only supports word2vec vector/vocabulary file formats
        * Datatype of vectors is set to float32 for computing efficiency

    .. code: python

        # Set embedding model
        v.model = 'word2vec'

        # Load entire dataset
        v.embeddingFilter = 1.
        v.load_embedding_model(...)

        # Load entire dataset
        v.embeddingFilter = None
        v.load_embedding_model(...)

        # Load first half dataset
        v.embeddingFilter = .5
        v.load_embedding_model(...)

        # Process first 10 lines
        v.embeddingFilter = 10
        v.load_embedding_model(...)

        # Process lines 10 through 20, [10,20]
        v.embeddingFilter = range(10,21)
        v.embeddingFilter = slice(10,21)
        v.embeddingFilter = (10,21)
        v.embeddingFilter = [10,21]
        v.load_embedding_model(...)

        # Process every other line from lines 10 through 20, [10,20]
        v.embeddingFilter = range(10,21,2)
        v.embeddingFilter = slice(10,21,2)
        v.embeddingFilter = (10,21,2)
        v.embeddingFilter = [10,21,2]
        v.load_embedding_model(...)
    """
    FPPrecision = 6
    ColWidth = 10
    Delim = ' '

    def __init__(self,
                 vectors=None,
                 vocabulary=None,
                 label="",
                 model='word2vec',
                 filter=1.,
                 blacklist=False,
                 dtype=numpy.float32):
        # Use reset() to indirectly create internal object's data members
        self.reset()

        self.label = label
        self.model = model
        self.embeddingFilter = filter
        self.blacklistFilter = blacklist
        self.dtype = dtype

        self.load_embedding_model(vectors, vocabulary)

    def reset(self):
        """Reset instance to initial state."""
        self.label = ""
        self.model = 'word2vec'
        self._embeddingFilter = 1.
        self.blacklistFilter = False
        self.dtype = numpy.float32
        self.clear()

    def clear(self):
        """Set vectors and vocabulary to initial state."""
        self._clear_vocabulary()
        self._clear_vectors()

    def _clear_vectors(self):
        """Set vectors to initial state."""
        self._fileVectors = ""
        self._vectors = numpy.empty(shape=0, dtype=self.dtype)
        self.similarities = numpy.empty(shape=0, dtype=self.dtype)
        self.distances = numpy.empty(shape=0, dtype=self.dtype)
        self.point_distances = numpy.empty(shape=0, dtype=self.dtype)
        self.angle_pairs = numpy.empty(shape=0, dtype=self.dtype)
        self.angle_triplets = numpy.empty(shape=0, dtype=self.dtype)

    def _clear_vocabulary(self):
        """Set vocabulary to initial state."""
        self._fileVocabulary = ""
        self._vocabulary = OrderedDict()

    def __str__(self):
        return "Label: {}\n" \
               "Model: {}\n" \
               "Vectors:\n" \
               "  File: {}\n" \
               "  Size: {}\n" \
               "Vocabulary:\n" \
               "  File: {}\n" \
               "  Size: {}".format(self.label, self.model,
                                   self.fileVectors, self.vectors.shape,
                                   self.fileVocabulary, len(self.vocabulary))

    def __len__(self):
        return self.vectors.shape[0]

    def __iter__(self):
        """Support iteration via a generator.

        .. code: python

            # Loop through all vocabulary and vector pairs
            v = vetkit.WordEmbedding('vectors.txt')
            for voc, vec, in v:
                print(voc, vec)

            # Create iterator from generator and iterate through data
            it = iter(v)
            voc, vec = next(it)
            voc, vec = next(it)

        Returns:
            generator: Tuple of vocabulary and vectors.
        """
        return ((voc,vec) for voc, vec in zip(self.vocabulary, self.vectors))

    def __getitem__(self, *keys):
        """Use dictionary syntax to access vector/vocabulary.

        Args:
            keys (str, int, range, slice, list, tuple): If string, consider it
                as a vocabulary word and return vector. If integer, range, or
                slice, then consider it as an index and return vocabulary word.
                If list or tuple, *keys* must be all of the same type, integers
                or strings.

        Returns:
            numpy.ndarray: Vectors.
            str or list of str: Words.

        Raises:
            KeyError: If *keys* is invalid in type/value.
        """
        data = []
        for key in keys:
            # range: returns a list of words [w1,w2,...]
            if isinstance(key, range):
                data = list(self.vocabulary)[slice(key.start, key.stop,
                                                   key.step)]
            # slice: returns a list of words [w1,w2,...]
            elif isinstance(key, slice):
                data = list(self.vocabulary)[key]
            else:
                # Ensure 'key' is a single level iterable to allow loop
                # processing. When an iterable or multiple keys are passed, the
                # arguments are automatically organized as a tuple of tuple of
                # values ((arg1,arg2),)
                if not hasattr(key, '__iter__') or isinstance(key, str):
                    key = [key]

                # str: returns an array of vectors [v1,v2,...]
                # int: returns a list of words [w1,w2,...]
                if all(isinstance(k, str) for k in key):
                    vocab = list(self.vocabulary)
                    data = numpy.empty(shape=(len(key), self.vectors.shape[1]),
                                       dtype=self.dtype)
                    for i, k in enumerate(key):
                        data[i][:] = self.vectors[vocab.index(k)]
                elif all(isinstance(k, int) for k in key):
                    vocab = list(self.vocabulary)
                    data = [vocab[k] for k in key]
                else:
                    raise KeyError("invalid key type/value '{}'".format(key))

        # Do not return a list when there is only a single item.
        # This allows vector algebra: v['king'] + v['man'] - # v['queen'].
        return data[0] if len(data) == 1 else data

    @property
    def embeddingFilter(self):
        return self._embeddingFilter

    @embeddingFilter.setter
    def embeddingFilter(self, val):
        if isinstance(val, str):
            # File with tokens
            self._embeddingFilter = tokenize_file(val)
        elif isinstance(val, (set, dict)):
            # Set or dictionary with tokens
            self._embeddingFilter = val
        elif isinstance(val, (list, tuple)) and isinstance(val[0], str):
            # Iterable with tokens
            self._embeddingFilter = set(val)
        else:
            # Assume range-like expression
            self._embeddingFilter = val

    @property
    def fileVectors(self):
        return self._fileVectors

    @property
    def fileVocabulary(self):
        return self._fileVocabulary

    @property
    def vectors(self):
        return self._vectors

    @vectors.setter
    def vectors(self, vectors):
        self.load_embedding_model(vectors=vectors)

    @property
    def vocabulary(self):
        return self._vocabulary

    @vocabulary.setter
    def vocabulary(self, vocab):
        self.load_embedding_model(vocab=vocab)

    def load_embedding_model(self, vectors=None, vocab=None):
        """Wrapper method to load a vector embedding model and vocabulary.

        Args:
            vectors (str, numpy.ndarray, optional): See :class:`WordEmbedding`.
            vocab (str, dict, optional): See :class:`WordEmbedding`.

        Raises:
            UserWarning: If *vectors* and *vocabulary* sizes are inconsistent.
        """
        # Vocabulary data has precedence over vector data because words and
        # occurrences are obtained from vocabulary data. If no vocabulary data
        # is provided then only words are obtained from vector data.
        if vocab is not None:
            self._load_vocabulary(vocab)
        if vectors is not None:
            self._load_vectors(vectors)

        if not self._is_consistent():
            raise UserWarning("vector and vocabulary are inconsistent in size")

    def _load_vectors(self, vectors):
        """Load a vector embedding model from a file or similar data structure.

        Todo:
            * Place numpy.ndarray, list, tuple copy into a function.

        Args:
            vectors: See :class:`WordEmbedding`.

        Raises:
            ValueError: If *v.model* is not a valid value.
            TypeError: If *vectors* is not a valid type.
        """
        self._clear_vectors()
        if isinstance(vectors, str):
            if self.model == 'word2vec':
                self._fileVectors = vectors
                if len(self._vocabulary) == 0:
                    self._fileVocabulary = self._fileVectors
                    self._vectors, self._vocabulary = load_vectors_word2vec(vectors, True, self.embeddingFilter, self.blacklistFilter, self.dtype)
                else:
                    self._vectors, _ = load_vectors_word2vec(vectors, False, self.embeddingFilter, self.blacklistFilter, self.dtype)
            elif self.model == 'glove':
                self._fileVectors = vectors
                if len(self._vocabulary) == 0:
                    self._fileVocabulary = self._fileVectors
                    self._vectors, self._vocabulary = load_vectors_glove(vectors, True, self.embeddingFilter, self.blacklistFilter, self.dtype)
                else:
                    self._vectors, _ = load_vectors_glove(vectors, False, self.embeddingFilter, self.blacklistFilter, self.dtype)
            else:
                raise ValueError("invalid model value '{}'".format(self.model))
        elif isinstance(vectors, numpy.ndarray):
            r = convert_to_range(self.embeddingFilter, vectors)
            self._vectors = deepcopy(vectors[slice(*r)])
        elif isinstance(vectors, (list, tuple)):
            r = convert_to_range(self.embeddingFilter, vectors)
            self._vectors = numpy.array(vectors[slice(*r)])
        else:
            raise TypeError("invalid vectors type '{}'".format(type(vectors)))

    def _load_vocabulary(self, vocab):
        """Load vocabulary from file or given object.

        Todo:
            * Place dict copy into a function.

        Args:
            vocab: See :class:`WordEmbedding`.

        Raises:
            ValueError: If *v.model* is not a valid value.
            TypeError: If *vocab* is not a valid type.
        """
        self._clear_vocabulary()
        if isinstance(vocab, str):
            if self.model == 'word2vec':
                self._fileVocabulary = vocab
                self._vocabulary = load_vocabulary_word2vec(vocab, self.embeddingFilter, self.blacklistFilter)
            elif self.model == 'glove':
                self._fileVocabulary = vocab
                self._vocabulary = load_vocabulary_glove(vocab, self.embeddingFilter, self.blacklistFilter)
            else:
                raise ValueError("invalid model '{}'".format(self.model))
        elif isinstance(vocab, dict):
            r = convert_to_range(self.embeddingFilter, vocab)
            line_curr = r[0]
            for i, (word, count) in enumerate(vocab.items()):
                if i < r[0]: continue
                if i >= r[1]: break
                if i == line_curr:
                    self._vocabulary[word] = count
                    line_curr += r[2]
        else:
            raise TypeError("invalid vocabulary type '{}'".format(type(vocab)))

    def common_words(self, n=10):
        """Find the words with most or least occurrences.

        Args:
            n (int, optional): If positive, then number of most common words.
                If negative, then number of least common words.

        Returns:
            tuple: Common words.
        """
        vocab = tuple(self.vocabulary)
        return vocab[:n] if n >= 0 else vocab[n:]

    def dump_embedding_model(self, vectors=None, vocab=None, binary=False):
        """Wrapper to write/print vectors/vocabulary from an embedding model.

        For *vectors* and *vocab* arguments, if non-empty string, consider it a
        file and write data. If empty string, print data. If None, no action is
        performed.

        Args:
            vectors (str, None, optional): Output file/method for vectors.
                Default is None.
            vocab (str, None, optional): Output file/method for vocabulary.
                Default is None.
            binary (bool, optional): Control written encoding of vectors.
                Default is False.
        """
        if vectors is not None:
            if vectors:
                self._dump_vectors(vectors, binary)
            else:
                self._print_vectors()

        if vocab is not None:
            if vocab:
                self._dump_vocabulary(vocab)
            else:
                self._print_vocabulary()

    def _dump_vectors(self, file, binary=False):
        """Write vectors from an embedding model.

        Todo:
            * Currently, only applies to word2vec model.
        Args:
            file (str): Output file.
            binary (bool, optional): Set write to use binary format. Default is
                False.
        """
        if self.model == 'word2vec':
            dump_vectors_word2vec(file, self.vectors, self.vocabulary, binary)
        elif self.model == 'glove':
            dump_vectors_glove(file, self.vectors, self.vocabulary)
        else:
            raise ValueError("invalid model value '{}'".format(self.model))

    def _dump_vocabulary(self, file):
        """Write vocabulary from an embedding model.

        Args:
            file (str): Output file.
        """
        if self.model == 'word2vec':
            dump_vocabulary_word2vec(file, self.vocabulary)
        elif self.model == 'glove':
            dump_vocabulary_glove(file, self.vocabulary)
        else:
            raise ValueError("invalid model value '{}'".format(self.model))

    def _print_vectors(self):
        """Print vectors from an embedding model."""
        if not self._is_consistent():
            raise UserWarning("vector and vocabulary are inconsistent in size")

        for word, vector in zip(self.vocabulary, self.vectors):
            print(word, vector)

    def _print_vocabulary(self):
        """Print vocabulary from an embedding model."""
        for word, count in self.vocabulary.items():
            print(word, count)

    def _is_consistent(self):
        """Check size consistency between vectors and vocabulary.

        Inconsistency occurs when both vectors and vocabulary contain active
        elements but sizes do not match. Otherwise, they are consistent.

        Returns:
            bool: True if sizes are consistent. False otherwise.
        """
        if self.vectors.size > 0 and len(self.vocabulary) > 0:
            return self.vectors.shape[0] == len(self.vocabulary)
        else:
            return True

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
        return vars(self)[re.sub('\s+', '_', key.strip()).lower()]

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
                pairs = [(row, col) for row in range(len(self.vocabulary))
                                    for col in range(row)]
            elif k == 3:
                pairs = [(x, y, z) for x in range(len(self.vocabulary) - 2)
                                   for y in range(x + 1, len(self.vocabulary) - 1)
                                   for z in range(y + 1, len(self.vocabulary))]
        else:
            vocab = list(self.vocabulary.keys())
            if k == 2:
                pairs = [(vocab[row], vocab[col]) for row in range(len(self.vocabulary)) for col in range(row)]
            elif k == 3:
                pairs = [(vocab[x], vocab[y], vocab[z])
                            for x in range(len(self.vocabulary) - 2)
                            for y in range(x + 1, len(self.vocabulary) - 1)
                            for z in range(y + 1, len(self.vocabulary))]
        return pairs

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
        flatSize = n_choose_k(len(self.vocabulary), k)

        if k == 2:
            # Used to consider only unique data from symmetric matrix
            self.similarities = numpy.empty(shape=flatSize, dtype=self.dtype)
            self.distances = numpy.empty(shape=flatSize, dtype=self.dtype)
            self.point_distances = numpy.empty(shape=flatSize, dtype=self.dtype)
            self.angle_pairs = numpy.empty(shape=flatSize, dtype=self.dtype)

            idx = 0
            for row in range(len(self.vocabulary)):
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
            for x in range(len(self.vocabulary) - 2):
                for y in range(x + 1, len(self.vocabulary) - 1):
                    for z in range(y + 1, len(self.vocabulary)):
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
                newline = os.linesep
                gfmt =  '{}'
                fpfmt = '{:.' + str(self.FPPrecision) + 'f}'
                hfmt = self.Delim.join(len(headers)*[gfmt]) + newline
                dfmt = self.Delim.join(k*[gfmt] + (len(headers)-k)*[fpfmt]) + newline
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
                newline = os.linesep
                gfmt =  '{}'
                fpfmt = '{:<' + str(self.ColWidth) + '.' + str(self.FPPrecision) + 'f}'
                hfmt = self.Delim.join(len(headers)*[gfmt]) + newline
                dfmt = self.Delim.join(k*[gfmt] + (len(headers)-k)*[fpfmt]) + newline
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

        # Allocate arrays
        for i, h in enumerate(headers):
            if i >= k:
                vars(self)[h] = numpy.empty(shape=nlines, dtype=self.dtype)

        with open(file) as fd:
            _ = fd.readline()  # skip headers
            for i, line in enumerate(fd):
                data = line.strip().split(self.Delim)
                for j, h in enumerate(headers):
                    if j >= k:
                        vars(self)[h][i] = self.dtype(data[j])
