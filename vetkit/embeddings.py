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
        * Only supports word2vec vector/vocabulary file formats
        * Datatype of vectors is set to float32 for computing efficiency
        * Vocabulary should not contain class delimiter (default is blank)

    .. code: python

        v.set_processing_count(k=1, frac=1.)
        v.set_processing_count(k=2, frac=1.)
        v.set_processing_count(k=3, frac=0.25)
    """
    FPPrecision = 6
    ColWidth = 10
    Delim = ' '

    def __init__(self, **kwargs):
        # Initialize using reset method, internally object's data members are
        # created with default values.
        self.reset()

        self.dtype = kwargs.get('dtype', self.dtype)
        self.label = kwargs.get('label', self.label)
        self._model = kwargs.get('model', self.model)

        self.load_embedding_model(vectors=kwargs.get('vectors', None),
                                  vocab=kwargs.get('vocabulary', None),
                                  model=self.model,
                                  n=kwargs.get('n', 1.))

    def _clear_vectors(self):
        """Clear vectors current state, set to initial state."""
        # Set data members directly to create them when called from __init__
        self._fileVectors = ""
        self._vectors = numpy.empty(shape=0, dtype=self.dtype)
        self.similarities = numpy.empty(shape=0, dtype=self.dtype)
        self.distances = numpy.empty(shape=0, dtype=self.dtype)
        self.point_distances = numpy.empty(shape=0, dtype=self.dtype)
        self.angle_pairs = numpy.empty(shape=0, dtype=self.dtype)
        self.angle_triplets = numpy.empty(shape=0, dtype=self.dtype)

    def _clear_vocabulary(self):
        """Clear vocabulary current state, set to initial state."""
        # Set data members directly to create them when called from __init__
        self._fileVocabulary = ""
        self._vocabulary = collections.OrderedDict()

    def clear(self):
        """Clear vocabulary and vectors current state, set to initial state."""
        self._clear_vocabulary()
        self._clear_vectors()

    def reset(self):
        """Reset object to initial state."""
        # Set data members directly to create them when called from __init__
        self.dtype = numpy.float32  # set first to make it available
        self.label = ""
        self._model = 'word2vec'
        self.clear()
        self._vectorProcessingCount = 0
        self._pairProcessingCount = 0
        self._tripletProcessingCount = 0

    def __str__(self):
        return "Label: {label}\n" \
               "Model: {model}\n" \
               "Vectors:\n" \
               "  File: {vectors}\n" \
               "  Size: {vector_size}\n" \
               "Vocabulary:\n" \
               "  File: {vocabulary}\n" \
               "  Size: {vocab_size}" \
               .format(label=self.label,
                       model=self.model,
                       vectors=self.fileVectors,
                       vocabulary=self.fileVocabulary,
                       vector_size=self.vectors.shape,
                       vocab_size=len(self.vocabulary))

    def __len__(self):
        return self.vectors.shape[0]

    def __getitem__(self, *keys):
        """Use dictionary syntax to access vector/vocabulary.

        Args:
            keys (str, int, slice, iterable): If string, then consider it as a
                vocabulary word and return vector. If integer or slice, then
                consider it as an index and return vocabulary word. *keys* can
                be a slice or an iterable of all integers or all strings.

        Returns:
            numpy.ndarray: Vectors.
            str or list of str: Words.

        Raises:
            KeyError: If *keys* is invalid type.
        """
        data = []
        for key in keys:
            # slice: returns a list of words [w1,w2,...]
            if isinstance(key, slice):
                data = list(self.vocabulary)[key]
            else:
                # Ensure 'key' is a single level iterable to allow loop processing.
                # When an iterable or multiple keys are passed, the arguments are
                # automatically organized as a tuple of tuple of values
                # ((arg1,arg2),)
                if not hasattr(key, '__iter__') or isinstance(key, str):
                    key = [key]

                # str: returns an array of vectors [v1,v2,...]
                # int: returns a list of words [w1,w2,...]
                if all(isinstance(k, str) for k in key):
                    vocab = list(self.vocabulary)
                    data = numpy.empty(shape=(len(key), self.vectors.shape[1]), dtype=self.dtype)
                    for i, k in enumerate(key):
                        data[i][:] = self.vectors[vocab.index(k)]
                elif all(isinstance(k, int) for k in key):
                    vocab = list(self.vocabulary)
                    data = [vocab[k] for k in key]
                else:
                    raise KeyError('Invalid key \'{}\' for {} '
                                   'object'.format(key, repr(self)))

        # Do not return a list when there is only a single item.
        # This allows direct vector arithmetic.
        return data[0] if len(data) == 1 else data

    @property
    def model(self):
        return self._model

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
    def vectors(self, arg):
        self.load_vectors(arg)

    @property
    def vocabulary(self):
        return self._vocabulary

    @vocabulary.setter
    def vocabulary(self, arg):
        self.load_vocabulary(arg)

    def load_embedding_model(self, vectors, vocab="", model='', n=1.):
        """Load an embedding model and vocabulary.

        Args:
            vectors: See :meth:`load_vectors`.
            vocab: See :meth:`load_vocabulary`.
            n: See :meth:`load_vectors` or :meth:`load_vocabulary`.
        """
        # Vocabulary data has precedence over vector data because words and
        # occurrences are obtained from vocabulary data. If no vocabulary data
        # is provided then only words are obtained from vector data.
        if vocab:
            self.load_vocabulary(vocab, model, n)
        if vectors:
            self.load_vectors(vectors, model, n)

    def _get_range(self, n, *arg):
        """Construct a range from different forms of input.

        Todo:
            * Remove n and merge into args parameter.
            * Raise error instead of printing warnings.

        Args:
            slice, list, tuple, numeric, multiple args: Values representing a
                range. List and tuple should have at least 2 elements, [begin,
                end]. For integer, range is first *n* elements.  Multiple
                arguments represent [begin, end, step]. Default step is 1.

        Returns:
            list: [begin, end, step] or [begin, None, step].
                For invalid cases [0, 0, 1].
        """
        # Extract ranges
        if len(arg) == 0:
            if isinstance(n, slice):
                r = [n.start, n.stop, n.step if n.step else 1]
            elif isinstance(n, (list, tuple)):
                r = [n[0], n[1], n[2] if len(n) > 2 else 1]
            elif isinstance(n , (int, float)) or n is None:
                r = [0, n, 1]
            else:
                print("WARN: Invalid type of range values, {}".format([n, *arg]))
                return [0, 0, 1]
        elif len(arg) == 1:
            r = [n, *arg, 1]
        elif len(arg) == 2:
            r = [n, *arg]
        else:
            print("WARN: Too many range values, {}".format([n, *arg]))
            return [0, 0, 1]

        # Follow slice behavior, support None
        if r[0] is None:
            r[0] = 0
        if r[2] is None:
            r[2] = 1
        if r[1] is None:
            # No support for negative ranges because the length of the data is
            # unknown. Also, step has to move from begin to end.
            if r[0] < 0 or r[2] <= 0:
                print("WARN: Invalid range, {}".format(r))
                return [0, 0, 1]
            else:
                return [int(x) if x else x for x in r]

        # No support for negative ranges because the length of the data is
        # unknown. Also, step has to move from begin to end.
        else:
            if r[0] < 0 or r[1] < 0 or r[2] == 0 or \
                (r[0] > r[1] and r[2] > 0) or (r[0] < r[1] and r[2] < 0):
                print("WARN: Invalid range, {}".format(r))
                return [0, 0, 1]

        # Swap
        if r[0] > r[1]:
            r[:2] = r[1::-1]
            r[2] = abs(r[2])
        return [int(x) for x in r]

    def _load_word2vec_vectors(self, n=1., load_vocab=False):
        """Load word2vec embedding model."""
        # Check if file format is ASCII or binary
        # Get data dimensions
        try:
            with open(self.fileVectors) as fd:
                dims = tuple(int(dim) for dim in fd.readline().strip().split())
            binary = 0
        except UnicodeDecodeError as ex:
            with open(self.fileVectors, 'rb') as fd:
                dims = tuple(int(dim) for dim in fd.readline().strip().split())
            binary = 1

        # Get line range to process
        if isinstance(n, (int, float)):
            n = self._processing_count(dims[0], n)
        r = self._get_range(n)

        if binary == 0:
            # ASCII format
            with open(self.fileVectors) as fd:
                _ = fd.readline()  # discard header, already read
                self._vectors = numpy.empty(shape=(r[1] - r[0], dims[1]), dtype=self.dtype)
                line_curr = r[0]
                j = 0
                for i, line in enumerate(fd):
                    if i < r[0]: continue

                    # For iterables, if None is used as the end value, the
                    # range returns None as well.
                    if r[1] is not None and i >= r[1]: break
                    if i == line_curr:
                        line_curr += r[2]
                        word, vector = line.strip().split(maxsplit=1)
                        if load_vocab:
                            self._vocabulary[word] = 0
                        self._vectors[j][:] = numpy.fromstring(vector, sep=' ', dtype=self.dtype)
                        j += 1
        elif binary == 1:
            # Binary format
            with open(self.fileVectors, 'rb') as fd:
                _ = fd.readline()  # discard header, already read
                self._vectors = numpy.empty(shape=(r[1] - r[0], dims[1]), dtype=self.dtype)
                line_len = dims[1] * numpy.dtype(self.dtype).itemsize
                line_curr = r[0]
                parse_completed = False
                chunk_size = 1024 * 1024
                chunk = b''
                tmp_chunk = b''
                i = 0
                j = 0
                while True:
                    # First part of current line
                    if not chunk:
                        chunk = fd.read(chunk_size)

                        # EOF?
                        if not chunk: break

                    #word, chunk = chunk.split(maxsplit=1)
                    blank_idx = chunk.index(b' ')
                    word = chunk[:blank_idx]
                    chunk = chunk[blank_idx + 1:]  # skip blank space

                    # Read remaining vector bytes
                    while (len(chunk) <= line_len):
                        tmp_chunk = fd.read(chunk_size)

                        # EOF? but we are not done
                        if not tmp_chunk:
                            raise Exception("ERROR: failed to parse vector file")
                        chunk += tmp_chunk

                    # Extract vector
                    vector = chunk[:line_len]

                    # Trim chunk, skip newline
                    chunk = chunk[line_len + 1:]

                    i += 1

                    if i < r[0]: continue
                    if i >= r[1]: break
                    if i == line_curr:
                        print(i, word)
                        self._vectors[j][:] = numpy.frombuffer(vector, dtype=self.dtype)
                        if load_vocab:
                            self._vocabulary[word] = 0
                        line_curr += r[2]
                        j += 1
                    if i == r[1]:
                        parse_completed = True
                        break
                if not parse_completed:
                    raise Exception("ERROR: failed to parse vector file")

    def load_vectors(self, vectors, model='', n=1.):
        """Load an embedding model from word2vec output.

        If vocabulary is empty then the words in vector file will be used to
        populate the vocabulary (occurrences will be set to 0).

        .. code: python

            # Process entire file
            v.load_vectors("...")
            v.load_vectors("...", n=1.)

            # Process first 10 lines
            v.load_vectors("...", n=10)

            # Process lines 10 through 20, [10,20]
            v.load_vectors("...", n=slice(10,21))
            v.load_vectors("...", n=(10,21))
            v.load_vectors("...", n=[10,21])

            # Process every other line from lines 10 through 20, [10,20]
            v.load_vectors("...", n=slice(10,21,2))
            v.load_vectors("...", n=(10,21,2))
            v.load_vectors("...", n=[10,21,2])

        Args:
            vectors (str, numpy.ndarray): If string, then consider it as a
                vector file based on original word2vec format, words and
                vectors. Vectors are ordered by occurrences in decreasing order
                which is the default behavior of original word2vec. If
                numpy.ndarray, then a deep copy is performed.

            n (float, int, slice): If float (0,1], then it is used as a
                fraction corresponding to first n*N words, [0,n*N). If int,
                then it is used to select the first n words, [0,floor(n)). If
                slice, tuple, or list then it is used as a range of words,
                [n1,n2,ns).

        Raises:
            ValueError: If *vectors* is not a valid type.
        """
        if model:
            self._model = model

        self._clear_vectors()
        if isinstance(vectors, str):
            if not os.path.isfile(vectors):
                raise FileNotFoundError(vectors)
            self._fileVectors = os.path.abspath(vectors)

            # Check either fileVocabulary or dictionary because they should be
            # consistent.
            if not self._fileVocabulary or len(self._vocabulary) == 0
                self._fileVocabulary = self._fileVectors
                load_vocab = True
            else:
                load_vocab = False

            if self.model == 'word2vec':
                self._load_word2vec_vectors(n, load_vocab)
            else:
                raise ValueError('ERROR: Invalid vectors model \'{}\''.format(model))
        elif isinstance(vectors, numpy.ndarray):
            # Get line range to process
            if isinstance(n, (int, float)):
                n = self._processing_count(vectors.shape[0], n)
            self._vectors = copy.deepcopy(vectors[slice(*self._get_range(n))])
        else:
            raise ValueError('ERROR: Invalid vectors type \'{}\', allowed values are '
                             'string and numpy.ndarray'.format(type(vectors)))

        print(self.vectors.shape)
        self.set_processing_count(k=1, frac=1.)
        self.set_processing_count(k=2, frac=1.)
        self.set_processing_count(k=3, frac=1.)

        if len(self._vocabulary) > 0 and not self._is_consistent():
            print('WARN: Vector and vocabulary are inconsistent in size')

    def load_vocabulary(self, vocab, model='', n=1.):
        """Load vocabulary from file or given object.

        Todo:
            * Add hooks to select vocabulary data based on model.

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
            v.load_vocabulary("...", n=(10,21,2))
            v.load_vocabulary("...", n=[10,21,2])

        Args:
            vocab (str, collections.OrderedDict): If string, then consider it
                as a vocabulary file with 2 columns, words and occurrences.
                Words are ordered by occurrences in decreasing order which is
                the default behavior of original word2vec. If
                collections.OrderedDict, then a deep copy is performed.

            n (float, int, slice): If float (0,1], then it is used as a
                fraction corresponding to first n*N words, [0,n*N). If int,
                then it is used to select the first n words, [0,floor(n)). If
                slice, tuple, or list then it is used as a range of words,
                [n1,n2,ns).

        Raises:
            ValueError: If *vocab* is not a valid type.
        """
        # Hack to prevent call from __init__ fail
        if vocab is None: return

        if isinstance(vocab, str):
            self._clear_vocabulary()
            if not os.path.isfile(vocab):
                raise FileNotFoundError(vocab)
            self._fileVocabulary= os.path.abspath(vocab)

            # Calculate lines to process
            if isinstance(n, slice):
                line_begin = n.start
                line_end = n.stop
                line_step = n.step if n.step else 1
            # Iterable must have at least 2 elements (start, stop, step)
            elif isinstance(n, (list, tuple)):
                line_begin = n[0]
                line_end = n[1]
                line_step = n[2] if len(n) > 2 else 1
            # Assume it is an integer
            else:
                # Get number of lines from file only if n = (0,1).
                # If n = 1, then set line_end = -1 to process all lines.
                nlines = n if n > 1 else self._get_nlines_from_file(self.fileVocabulary)
                line_begin = 0
                line_end = -1 if n == 1 else self._processing_count(nlines, n)
                line_step = 1

            if line_end != 0:
                with open(self.fileVocabulary) as fd:
                    line_curr = line_begin
                    for i, line in enumerate(fd):
                        # Do not stop if line_end is negative
                        if line_end > 0 and i >= line_end: break
                        if i < line_begin: continue
                        if i == line_curr:
                            word, count = line.strip().split(maxsplit=1)
                            self._vocabulary[word] = int(count)
                            line_curr += line_step
        elif isinstance(vocab, collections.OrderedDict):
            self._clear_vocabulary()
            self._fileVocabulary= vocab

            # Calculate lines to process
            if isinstance(n, slice):
                line_begin = n.start
                line_end = n.stop
                line_step = n.step if n.step else 1
            # Iterable must have at least 2 elements (start, stop, step)
            elif isinstance(n, (list, tuple)):
                line_begin = n[0]
                line_end = n[1]
                line_step = n[2] if len(n) > 2 else 1
            # Assume it is an integer
            else:
                line_begin = 0
                line_end = self._processing_count(len(vocab), n)
                line_step = 1

            if line_end != 0:
                if line_end == len(vocab):
                    self._vocabulary = copy.deepcopy(vocab)
                else:
                    line_curr = line_begin
                    for i, (word, count) in enumerate(vocab.items()):
                        if i >= line_end: break
                        if i < line_begin: continue
                        if i == line_curr:
                            self._vocabulary[word] = count
                            line_curr += line_step
        else:
            self._clear_vocabulary()
            raise ValueError('Invalid vocabulary type \'{}\', allowed values '
                             'are string and collections.OrderedDict'
                             .format(type(vocab)))

        self.set_processing_count(k=1, frac=1)

        if model:
            self._model = model

        if self.vectors.size > 0 and not self._is_consistent():
            print('WARN: Vector and vocabulary are inconsistent in size')

    def _get_nlines_from_file(self, file, sizehint=1024 * 1024):
        """Calculate number of lines in a given text file.

        Notes:
            Does not support binary encoded files.

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

    def write_vectors(self, file="", n=1.):
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
        # Iterable must have at least 2 elements (start, stop, step)
        elif isinstance(n, (list, tuple)):
            line_begin = n[0]
            line_end = n[1]
            line_step = n[2] if len(n) > 2 else 1
        # Assume it is numeric
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

    def write_vocabulary(self, file="", n=1.):
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
        # Iterable must have at least 2 elements (start, stop, step)
        elif isinstance(n, (list, tuple)):
            line_begin = n[0]
            line_end = n[1]
            line_step = n[2] if len(n) > 2 else 1
        # Assume it is numeric
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

    def _processing_count(self, n, frac=1.):
        """
        Calculate number of data to processed based on either:
            * Fraction (float) - n = (0.,1.]
            * Selection (int) - n = [0, ...]

        Floating-point values are considered as percentages.
        Negative values are set to zero and values greater than 1. are set to
        1..

        n is considered as an integer.
        """
        N = 0
        if isinstance(frac, float):
            if frac < 0.:
                frac = 0.
            elif frac > 1.:
                frac = 1.
            N = round(frac * int(n))
        elif isinstance(frac, int):
            N = min(frac, int(n))
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
