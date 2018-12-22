"""Embedding models functions.

Todo:
    * Add vector/vocabulary filters using a list of words or indices.
"""


from math import ceil
from collections import OrderedDict
import numpy
from .utils import convert_to_range


def load_vectors_word2vec(file, lines=None, load_vocab=True, dtype=numpy.float32):
    """Load word2vec embedding model from a given file.

    Args:
        file (str): Input vector file, ASCII or binary.

        lines (range, slice, list, tuple, float, int, None): Values
            representing a range for vectors, see *utils.convert_to_range()*.
            If None, entire file is processed.

        load_vocab (bool): If True, vocabulary will be extracted from file
            (occurrences will be set to 1).

        dtype (numpy.dtype): Type of vector data.

    Returns:
        numpy.ndarray, OrderedDict: Vector array and vocabulary dictionary.

    Raises:
        EOFError: For binary format, if EOF is reached before extracting
            requested data.
    """
    # Check file format, ASCII or binary
    # Get data dimensions
    try:
        with open(file) as fd:
            dims = tuple(int(dim) for dim in fd.readline().strip().split())
        binary = 0
    except UnicodeDecodeError as ex:
        with open(file, 'rb') as fd:
            dims = tuple(int(dim) for dim in fd.readline().strip().split())
        binary = 1

    # Get line range to process
    r = convert_to_range(lines, dims[0])
    n_elems = ceil((r[1] - r[0]) / r[2])

    vectors = numpy.empty(shape=(n_elems, dims[1]), dtype=dtype)
    vocab= OrderedDict()

    if binary == 0:
        # ASCII format
        with open(file) as fd:
            _ = fd.readline()  # discard header, already read
            line_curr = r[0]
            j = 0
            for i, line in enumerate(fd):
                if i < r[0]: continue
                if i >= r[1]: break
                if i == line_curr:
                    word, vector = line.strip().split(maxsplit=1)
                    if load_vocab:
                        vocab[word] = 1
                    vectors[j][:] = numpy.fromstring(vector, dtype, sep=' ')
                    line_curr += r[2]
                    j += 1
    else:
        # Binary format
        with open(file, 'rb') as fd:
            _ = fd.readline()  # discard header, already read
            line_curr = r[0]
            line_len = dims[1] * 4  # float
            chunk_size = 1024 * 1024
            chunk = b''
            i = 0
            j = 0
            while True:
                if i >= r[1]: break

                # First part of current line
                if not chunk:
                    chunk = fd.read(chunk_size)

                    # EOF?
                    if not chunk: break

                blank_idx = chunk.index(b' ')
                word = chunk[:blank_idx]
                chunk = chunk[blank_idx + 1:]  # skip blank space

                # Read remaining vector bytes
                while (len(chunk) <= line_len):
                    tmp_chunk = fd.read(chunk_size)

                    # EOF? We are not done processing file
                    if not tmp_chunk: break
                    chunk += tmp_chunk

                # Extract vector
                vector = chunk[:line_len]

                # Trim chunk, skip newline
                chunk = chunk[line_len + 1:]

                if i < r[0]: continue
                if i == line_curr:
                    vectors[j][:] = numpy.frombuffer(vector, dtype=dtype)
                    if load_vocab:
                        vocab[word.decode()] = 1
                    line_curr += r[2]
                    j += 1
                i += 1

            # Check if processing stopped before it should
            if j < n_elems and dims[0] - j >= r[2]:
                raise EOFError("failed to parse vector file")
    return vectors, vocab


def load_vocabulary_word2vec(file, lines=None):
    """Load vocabulary from a word2vec vocabulary file.

    File consists of two columns, words and occurrences.

    Args:
        file (str): Input vocabulary file.

        lines (range, slice, list, tuple, float, int, None): Values
            representing a range for vocabulary, see
            *utils.convert_to_range()*. If None, entire file is processed.
    """
    r = convert_to_range(lines, file)
    vocab = OrderedDict()
    with open(file) as fd:
        next_line = r[0]
        for i, line in enumerate(fd):
            if i < r[0]: continue
            if r[1] is not None and i >= r[1]: break
            if i == next_line:
                word, count = line.strip().split(maxsplit=1)
                vocab[word] = int(count)
                next_line += r[2]
    return vocab
