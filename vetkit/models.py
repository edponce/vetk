"""Interface functions to word2vec embedding models.

Todo:
    * In docstrings, add links for references to module functions.
"""


import os
from math import ceil
from collections import OrderedDict
import numpy
from .utils import convert_to_range


def load_vectors_word2vec(file, load_vocab=True, filter=None, blacklist=False, dtype=numpy.float32):
    """Load vectors of embedding model from given file in word2vec format.

    Notes:
        * *file* encoding (ASCII or binary) is automatically detected during
          processing.

    Args:
        file (str): Input file.
        load_vocab (bool, optional): If True, vocabulary will be extracted from
            file (occurrences will be set to 1). Otherwise an empty vocabulary
            is returned. Default is True.
        filter (range, slice, list, tuple, float, int, dict, None, optional):
            Values representing a filter operation for file processing, see
            *utils.convert_to_range()*. If string, consider it a file with a
            list of words. If None, entire file is processed. Default is None.
        blacklist (bool, optional): If True, consider *filter* as a blacklist.
            If False, consider *filter* as a whitelist. Only applicable when
            *filter* is a dict. Default is False.
        dtype (numpy.dtype, optional): Type of vector data. Default is
            numpy.float32.

    Returns:
        numpy.ndarray, OrderedDict: Vectors and vocabulary of embedding model.

    Raises:
        EOFError: For *file* in binary format, if EOF is reached before all
            possible data requested is extracted.
    """
    # Check file format and get data dimensions
    try:
        with open(file) as fd:
            dims = tuple(int(dim) for dim in fd.readline().split())
        binary = False
    except UnicodeDecodeError as ex:
        with open(file, 'rb') as fd:
            dims = tuple(int(dim) for dim in fd.readline().split())
        binary = True

    # Get lines to process
    if isinstance(filter, dict):
        erange = convert_to_range(None, dims[0])
    else:
        blacklist = None  # Disable blacklisting
        erange = convert_to_range(filter, dims[0])

    n_elems = ceil((erange[1] - erange[0]) / erange[2])
    vectors = numpy.empty(shape=(n_elems, dims[1]), dtype=dtype)
    vocab= OrderedDict()

    if binary:
        with open(file, 'rb') as fd:
            _ = fd.readline()  # discard header, already read
            next_line = erange[0]
            line_length = dims[1] * 4  # float is default in word2vec
            chunk_size = 2**20  # read file in 1MB chunks
            chunk = b''
            i = -1  # begin at -1 because i+=1 is done before comparisons
            j = 0
            while True:
                i += 1
                if i >= erange[1]: break

                # First part of current line
                if not chunk:
                    chunk = fd.read(chunk_size)

                    # EOF?
                    if not chunk: break

                blank_idx = chunk.index(b' ')  # find word/vector separator
                word = chunk[:blank_idx]
                chunk = chunk[blank_idx + 1:]  # skip blank space

                # Read remaining vector bytes
                while (len(chunk) <= line_length):
                    partial_chunk = fd.read(chunk_size)

                    # EOF? We are not done processing file
                    if not partial_chunk: break
                    chunk += partial_chunk

                # Extract vector
                vector = chunk[:line_length]

                # Trim chunk, skip newline
                chunk = chunk[line_length + 1:]

                if i < erange[0]: continue
                if i == next_line:
                    word = word.decode()
                    if blacklist is None or (not blacklist and word in filter) or (blacklist and word not in filter):
                        if load_vocab:
                            vocab[word] = 1
                        vectors[j][:] = numpy.frombuffer(vector, dtype=dtype)
                        j += 1
                    next_line += erange[2]

            # Check if processing stopped before it should, not if blacklisting
            if blacklist is None and j < n_elems and erange[1] - j >= erange[2]:
                raise EOFError("failed to parse vectors file")
    else:
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


def load_vocabulary_word2vec(file, filter=None, blacklist=False):
    """Load vocabulary of embedding model from given file in word2vec format.

    Notes:
        * *file* consists of two columns, words and occurrences.

    Args:
        file (str): Input file.
        filter (range, slice, list, tuple, float, int, dict, None, optional):
            Values representing a filter operation for file processing, see
            *utils.convert_to_range()*. If string, consider it a file with a
            list of words. If None, entire file is processed. Default is None.
        blacklist (bool, optional): If True, consider *filter* as a blacklist.
            If False, consider *filter* as a whitelist. Only applicable when
            *filter* is a dict. Default is False.
    """
    # Get lines to process
    if isinstance(filter, dict):
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


def dump_vectors_word2vec(file, vectors, vocab, binary=False):
    """Write vectors of embedding model to given file in word2vec format.

    Notes:
        * Order of vectors and vocabulary should match.
        * For ASCII format, floating-point precision is 6 decimal places.

    Args:
        file (str): Output file.
        vectors (numpy.ndarray): Vectors of embedding model.
        vocab (dict): Vocabulary of embedding model.
        binary (bool, optional): Select encoding format. Default is False.
    """
    if binary:
        with open(file, 'wb') as fd:
            newline = os.linesep.encode()
            fd.write("{} {}".format(*vectors.shape).encode() + newline)
            fmt = "{} "
            for word, vector in zip(vocab.keys(), vectors):
                fd.write(fmt.format(word).encode())
                vector.tofile(fd)
                fd.write(newline)
    else:
        with open(file, 'w') as fd:
            newline = os.linesep
            fd.write("{} {}".format(*vectors.shape) + newline)
            fmt = ' '.join(['{}'] + vectors.shape[1] * ['{:6f}']) + newline
            for word, vector in zip(vocab.keys(), vectors):
                fd.write(fmt.format(word, *vector))


def dump_vocabulary_word2vec(file, vocab):
    """Write vocabulary of embedding model to given file in word2vec format.

    Args:
        file (str): Output file.
        vocab (dict): Vocabulary of embedding model.
    """
    with open(file, 'w') as fd:
        newline = os.linesep
        fmt = "{} {}" + newline
        for word, count in vocab.items():
            fd.write(fmt.format(word, count))
