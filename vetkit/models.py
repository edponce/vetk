"""Interface functions to word2vec embedding models.

Todo:
    * In docstrings, add links for references to module functions.
"""


from math import ceil
from collections import OrderedDict
import numpy
from .utils import convert_to_range


def load_vectors_word2vec(file, load_vocab=True, lines=None, dtype=numpy.float32):
    """Load vectors of embedding model from given file in word2vec format.

    Notes:
        * *file* encoding (ASCII or binary) is automatically detected during
          processing.

    Args:
        file (str): Input file.
        lines (range, slice, list, tuple, float, int, None, optional): Values
            representing a filter operation for file processing, see
            *utils.convert_to_range()*. If None, entire file is processed.
            Default is None.
        load_vocab (bool, optional): If True, vocabulary will be extracted from
            file (occurrences will be set to 1). Otherwise an empty vocabulary
            is returned. Default is True.
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
            dims = tuple(int(dim) for dim in fd.readline().strip().split())
        binary = False
    except UnicodeDecodeError as ex:
        with open(file, 'rb') as fd:
            dims = tuple(int(dim) for dim in fd.readline().strip().split())
        binary = True

    # Get lines to process
    erange = convert_to_range(lines, dims[0])

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
            i = 0
            j = 0
            while True:
                if i >= erange[1]: break

                # First part of current line
                if not chunk:
                    chunk = fd.read(chunk_size)

                    # EOF?
                    if not chunk: break

                blank_idx = chunk.index(b' ')
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
                    vectors[j][:] = numpy.frombuffer(vector, dtype=dtype)
                    if load_vocab:
                        vocab[word.decode()] = 1
                    next_line += erange[2]
                    j += 1
                i += 1

            # Check if processing stopped before it should
            if j < n_elems and dims[0] - j >= erange[2]:
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
                    word, vector = line.strip().split(maxsplit=1)
                    if load_vocab:
                        vocab[word] = 1
                    vectors[j][:] = numpy.fromstring(vector, dtype, sep=' ')
                    next_line += erange[2]
                    j += 1
    return vectors, vocab


def load_vocabulary_word2vec(file, lines=None):
    """Load vocabulary of embedding model from given file in word2vec format.

    Notes:
        * *file* consists of two columns, words and occurrences.

    Args:
        file (str): Input file.
        lines (range, slice, list, tuple, float, int, None, optional): Values
            representing a filter operation for file processing, see
            *utils.convert_to_range()*. If None, entire file is processed.
            Default is None.
    """
    erange = convert_to_range(lines, file)
    vocab = OrderedDict()
    with open(file) as fd:
        next_line = erange[0]
        for i, line in enumerate(fd):
            if i < erange[0]: continue
            if erange[1] is not None and i >= erange[1]: break
            if i == next_line:
                word, count = line.strip().split(maxsplit=1)
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
        binary (bool, optional): Set write to use binary format. Default is
            False.
    """
    if binary:
        with open(file, 'wb') as fd:
            fd.write("{} {}\n".format(*vectors.shape).encode())
            fmt = "{} "
            newline = b'\n'
            for word, vector in zip(vocab.keys(), vectors):
                fd.write(fmt.format(word).encode())
                vector.tofile(fd)
                fd.write(newline)
    else:
        with open(file, 'w') as fd:
            fd.write("{} {}\n".format(*vectors.shape))
            fmt = ' '.join(['{}'] + vectors.shape[1] * ['{:6f}']) + '\n'
            for word, vector in zip(vocab.keys(), vectors):
                fd.write(fmt.format(word, *vector))


def dump_vocabulary_word2vec(file, vocab):
    """Write vocabulary of embedding model to given file in word2vec format.

    Args:
        file (str): Output file.
        vocab (dict): Vocabulary of embedding model.
    """
    with open(file, 'w') as fd:
        fmt = "{} {}\n"
        for word, count in vocab.items():
            fd.write(fmt.format(word, count))
