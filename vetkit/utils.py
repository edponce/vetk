"""Utility functions.
"""


import os
from math import factorial


def convert_to_range(n, data=None):
    """Construct a range [start,stop,step] from different forms of input.

    Notes:
        * Support None for range values similar to slice behavior, [0,None or
          data size,1].
        * If *data* is provided, then :func:`select_from_data` is invoked.
        * For range, slice, and iterables, step is set to 1 if step is zero,
          None, or not given.

    Todo:
        * Support negative range values.

    Args:
        n (int, float, range, slice, iterable, None): Values representing a
            range. If integer, then it is considered as a selection [0,min(data
            size,n),1]. If float, then it is considered as a ratio [0,n*data
            size,1]. Range and slice values are used directly. Iterables should
            have 1 or more elements, [start,None,1], [start,stop,1], or [start,
            stop,step]. If None, and data size is available, use [0,data size,
            1], otherwise [0,None,1].
        data (str, int, object, None, optional): If string, it is considered as
            a file and size of data corresponds to the number of newlines, see
            :func:`count_lines_in_file`. If integer, it is considered as the
            actual data size. If object, then *len(object)* is used as data
            size. If None, it is ignored. Default is None.

    Returns:
        list: Range [start,stop,step] or [start,None,step].

    Raises:
        TypeError: If *n* or *data* are invalid types.
        ValueError: If *n* or *data* are invalid values.
    """
    # Get data size
    if data is None:
        data_size = None
    elif isinstance(data, str):
        # ASCII file
        data_size = count_lines_in_file(data)
    elif isinstance(data, int):
        # Explicit size
        if data < 0:
            raise ValueError("invalid data value, {}".format(data))
        data_size = data
    elif hasattr(data, '__len__'):
        # list, tuple, numpy.ndarray, etc.
        data_size = len(data)
    else:
        raise TypeError("invalid data type, {}".format(data))

    # Extract range
    if n is None:
        r = [0, data_size, 1]
    elif isinstance(n, int):
        # selection
        r = [0, n, 1]
    elif isinstance(n, float):
        # ratio, size of data is required
        if data_size is None:
            raise ValueError("invalid data value, {}".format(data))
        r = [0, n, 1]
    elif isinstance(n, (range, slice)):
        # range
        r = [n.start, n.stop, n.step]
    elif hasattr(n, '__iter__'):
        # range
        if len(n) == 1:
            r = [*n, None, 1]
        elif len(n) == 2:
            r = [*n, 1]
        elif len(n) > 2:
            r = [*n]
    else:
        raise TypeError("invalid range type, {}".format(n))

    if r[0] is None:  # start cannot be None
        r[0] = 0
    # Data size is available so truncate stop limit
    if data_size is not None:
        if r[1] is None:
            r[1] = data_size
        else:
            r[1] = select_from_data(data_size, r[1])
    if not r[2]:  # step cannot be 0 nor None
        r[2] = 1

    # Negative range values are not allowed
    for x in r:
        if x is not None and x < 0:
            raise ValueError("invalid range values, {}".format(r))

    return [int(x) if x is not None else x for x in r]


def select_from_data(n, select=1.):
    """Calculate number of data based on a fraction or selection value.

    Args:
        n (int): Size of data.
        frac (int, float, optional): Fraction or selection value. A
            floating-point value is used as a ratio with respect to n, [0, n*N]
            (value is bounded by 1.0). An integer value is used as a selection
            value with respect to n, [0, n] (if greater than n, it is set to
            n). Negative values are set to 0. Default is 1.0.

    Returns:
        int: Number of elements.

    Raises:
        ValueError: If *select* is a negative value.
        TypeError: If *select* is non-numeric.
    """
    if isinstance(select, int):
        if select < 0:
            raise ValueError("invalid select value, {}".format(select))
        return min(select, int(n))
    elif isinstance(select, float):
        if select < 0.:
            raise ValueError("invalid select value, {}".format(select))
        return round(min(select, 1.) * int(n))
    else:
        raise TypeError("invalid select type, {}".format(select))


def n_choose_k(n, k):
    """Calculate number of k-combinations given n items.

    C(n,k) = n!/(k!(n-k)!))

    Args:
        n (int): Number of elements.
        k (int): k-distinct elements.

    Returns:
        int: Number of combinations.
    """
    return factorial(n) // (factorial(k) * factorial(n - k))


def read_buf_gen(reader, size_hint=2**20):
    """Read generator with variable-sized buffer.

    Args:
        size_hint (int, optional): Bytes to read at a time. Default is 1MB.

    Returns:
        str: Yield buffer.
    """
    buf = reader(size_hint)
    while buf:
        yield buf
        buf = reader(size_hint)


def count_lines_in_file(file, size_hint=2**20):
    """Calculate number of newlines in a given text file.

    Notes:
        * Assumes last line of file ends with a newline.

    Args:
        file (str): Input file.
        size_hint (int, optional): Bytes to read at a time. Default is 1MB.

    Returns:
        int: Number of newlines.
    """
    with open(file, 'rb') as fd:
        fd_gen = read_buf_gen(fd.raw.read, size_hint)
        newline = os.linesep.encode()
        return sum(buf.count(newline) for buf in fd_gen)


def extract_tokens_from_file(file, delim=None, size_hint=2**20):
    """Extract all delimited tokens from a given text file.

    Notes:
        * Parses on newlines and assumes last line of file ends with a newline.

    Args:
        file (str): Input file.
        delim (str, None, optional): Token delimiter. If None, whitespace is
            used. Default is None.
        size_hint (int, optional): Bytes to read at a time. Default is 1MB.

    Returns:
        dict: Tokens.
    """
#Method 1:
#    tokens = {}
#    with open(file) as fd:
#        for line in fd:
#            for token in line.split(delim):
#                tokens[token] = 1
#    return tokens
    tokens = {}
    with open(file, 'rb') as fd:
        fd_gen = read_buf_gen(fd.raw.read, size_hint)
        newline = os.linesep.encode()
        if delim is not None:
            delim = delim.encode()
        tmp_buf = b''
        for buf in fd_gen:
            buf = tmp_buf + buf  # prepend partial buffer from previous loop
            newline_idx = buf.rindex(newline) + 1  # find last newline
            tmp_buf = buf[newline_idx:]
            for token in buf[:newline_idx].decode().split(delim):
                tokens[token] = tokens.get(token, 0) + 1  # count repeats
    return tokens
