"""Interface functions to gloVe embedding models."""


import os
from math import ceil
from collections import OrderedDict
import numpy
from .utils import convert_to_range
import smarttimers


@smarttimers.smarttime
def load_vectors_glove(file, load_vocab=True, filter=None, blacklist=False, dtype=numpy.float32):
    """Load vectors of embedding model from given file in gloVe format.

    Args:
        file (str): Input file.
        load_vocab (bool, optional): If True, vocabulary will be extracted from
            file (occurrences will be set to 1). Otherwise an empty vocabulary
            is returned. Default is True.
        filter (range, slice, list, tuple, float, int, set, dict, None, optional):
            Values representing a filter operation for file processing, see
            *utils.convert_to_range()*. If string, consider it a file with a
            list of words. If None, entire file is processed. Default is None.
        blacklist (bool, optional): If True, consider *filter* as a blacklist.
            If False, consider *filter* as a whitelist. Only applicable when
            *filter* is a set or dict. Default is False.
        dtype (numpy.dtype, optional): Type of vector data. Default is
            numpy.float32.

    Returns:
        numpy.ndarray, OrderedDict: Vectors and vocabulary of embedding model.
    """
    # Get lines to process
    if isinstance(filter, (set, dict)):
        erange = convert_to_range(None, file)
    else:
        blacklist = None  # Disable blacklisting
        erange = convert_to_range(filter, file)

    # Get size of vectors
    with open(file) as fd:
        dims1 = len(fd.readline().split()) - 1

    n_elems = ceil((erange[1] - erange[0]) / erange[2])
    vectors = numpy.empty(shape=(n_elems, dims1), dtype=dtype)
    vocab= OrderedDict()

    with open(file) as fd:
        _ = fd.readline()  # discard header, already read
        next_line = erange[0]
        j = 0
        for i, line in enumerate(fd):
            if i < erange[0]: continue
            if i >= erange[1]: break
            if i == next_line:
                word, vector = line.split(maxsplit=1)
                if blacklist is None or (not blacklist and word in filter) or (blacklist and word not in filter):
                    if load_vocab:
                        vocab[word] = 1
                    vectors[j][:] = numpy.fromstring(vector, dtype, sep=' ')
                    j += 1
                next_line += erange[2]

    # Resize array, only if given a blacklist where final size is unknown
    if j < vectors.shape[0]:
        vectors = vectors[:j,:]
    return vectors, vocab


@smarttimers.smarttime
def load_vocabulary_glove(file, filter=None, blacklist=False):
    """Load vocabulary of embedding model from given file in gloVe format.

    Notes:
        * *file* consists of two columns, words and occurrences.

    Args:
        file (str): Input file.
        filter (range, slice, list, tuple, float, int, set, dict, None, optional):
            Values representing a filter operation for file processing, see
            *utils.convert_to_range()*. If string, consider it a file with a
            list of words. If None, entire file is processed. Default is None.
        blacklist (bool, optional): If True, consider *filter* as a blacklist.
            If False, consider *filter* as a whitelist. Only applicable when
            *filter* is a set or dict. Default is False.
    """
    # Get lines to process
    if isinstance(filter, (set, dict)):
        erange = convert_to_range(None, file)
    else:
        blacklist = None
        erange = convert_to_range(filter, file)
    vocab = OrderedDict()
    with open(file) as fd:
        next_line = erange[0]
        for i, line in enumerate(fd):
            if i < erange[0]: continue
            if erange[1] is not None and i >= erange[1]: break
            if i == next_line:
                word, count = line.split(maxsplit=1)
                if blacklist is None or (not blacklist and word in filter) or (blacklist and word not in filter):
                    vocab[word] = int(count)
                next_line += erange[2]
    return vocab


@smarttimers.smarttime
def dump_vectors_glove(file, vectors, vocab):
    """Write vectors of embedding model to given file in gloVe format.

    Notes:
        * Order of vectors and vocabulary should match.
        * For ASCII format, floating-point precision is 6 decimal places.

    Args:
        file (str): Output file.
        vectors (numpy.ndarray): Vectors of embedding model.
        vocab (dict): Vocabulary of embedding model.
    """
    with open(file, 'w') as fd:
        newline = os.linesep
        fmt = ' '.join(['{}'] + vectors.shape[1] * ['{:6f}']) + newline
        for word, vector in zip(vocab.keys(), vectors):
            fd.write(fmt.format(word, *vector))


@smarttimers.smarttime
def dump_vocabulary_glove(file, vocab):
    """Write vocabulary of embedding model to given file in gloVe format.

    Args:
        file (str): Output file.
        vocab (dict): Vocabulary of embedding model.
    """
    with open(file, 'w') as fd:
        newline = os.linesep
        fmt = "{} {}" + newline
        for word, count in vocab.items():
            fd.write(fmt.format(word, count))
