"""Utility functions.

Todo:
    * Improve logic of convert_to_range, if n is None but data is given, then
    use size of data for n.
"""


from math import factorial


def convert_to_range(n, data=None):
    """Construct a range from different forms of input.

    Notes:
        * Setting *arg* to None should be used when *data* is a file and if
          counting lines in file is not desired.

    Args:
        n (range, slice, list, tuple, float, int, None): Values representing a
            range. List and tuple should have at least 2 elements, [start,
            stop]. For integer and floating point, invokes
            *select_from_data(size of data, n*.
        data (str, int, numpy.ndarray, object, None, optional): If str it is
            considered a file and newlines are counted as the total size of
            data. If integer it is considered as the actual data size. If numpy
            array or object, then its length is used as data size. If None,
            *data* has no effect. Default is None.

    Returns:
        list: Range representation [start, stop, step] or [start, None, step].

    Raises:
        TypeError: If *n* or *data* are invalid types.
        ValueError: Invalid range values.
    """
    # Get data size
    if data is None:
        pass
    elif isinstance(data, str):
        # ASCII file
        data_size = count_lines_in_file(data)
    elif isinstance(data, int):
        # Explicit size
        data_size = data
    elif hasattr(data, 'shape'):
        # Numpy array
        data_size = data.shape[0]
    elif hasattr(data, '__len__'):
        # Data structure with len()
        data_size = len(data)
    else:
        raise TypeError("invalid data type, {}".format(data))

    # Extract range
    if isinstance(n, (range, slice)):
        r = [n.start, n.stop, n.step if n.step else 1]
    elif isinstance(n, (list, tuple)):
        r = [n[0], n[1], n[2] if len(n) > 2 else 1]
    elif isinstance(n , (int, float)):
        if data is None:
            r = [0, n, 1]
        else:
            r = [0, select_from_data(data_size, n), 1]
    elif n is None:
        if data is None:
            r = [0, n, 1]
        else:
            r = [0, data_size, 1]
    else:
        raise TypeError("invalid range type, {}".format(n))

    # Follow slice behavior, support None
    if r[0] is None:
        r[0] = 0
    if r[2] is None:
        r[2] = 1
    if r[1] is None:
        # No support for negative ranges because the length of the data is
        # unknown. Also, step has to move from begin to end.
        if r[0] < 0 or r[2] <= 0:
            raise ValueError("invalid range values, {}".format(r))
        else:
            return [int(x) if x else x for x in r]

    # No support for negative ranges because the length of the data is
    # unknown. Also, step has to move from begin to end.
    else:
        # We have a data size so truncate stop limit
        if data is not None:
            r[1] = min(data_size, r[1])

        if r[0] < 0 or r[1] < 0 or r[2] == 0 or \
          (r[0] > r[1] and r[2] > 0) or (r[0] < r[1] and r[2] < 0):
            raise ValueError("invalid range values, {}".format(r))

    # Swap
    if r[0] > r[1]:
        r[:2] = r[1::-1]
        r[2] = abs(r[2])
    return [int(x) for x in r]


def select_from_data(n, frac=1.):
    """Calculate number of data based on a fraction or selection value.

    Args:
        n (int): Size of data.
        frac (int, float, optional): Fraction or selection value. A
            floating-point value is used as a ratio with respect to n, [0, n*N]
            (if greater than 1., it is set to 1.). An integer value is used as
            a selection value with respect to n, [0, n] (if greater than n, it
            is set to n). Negative values are set to 0. Default is 1..

    Returns:
        int: Number of elements.
    """
    N = 0
    if frac > 0.:
        if isinstance(frac, float):
            if frac > 1.:
                frac = 1.
            N = round(frac * int(n))
        elif isinstance(frac, int):
            N = min(frac, int(n))
    return N


def n_choose_k(n, k):
    """Calculate number of k-combinations given n items.

    C(n,k) = n!/(k!(n-k)!))

    Args:
        n (int): Number of elements.
        k (int): k-distinct elements.

    Returns:
        int: Number of combinations.
    """
    N = 0
    if k > 0:
        N = factorial(n) // (factorial(k) * factorial(n - k))
    return N


def count_lines_in_file(file, size_hint=2**20):
    """Calculate number of newlines in a given text file.

    Args:
        file (str): Input file.
        size_hint (int, optional): Bytes to read at a time. Default is 1 MB.

    Returns:
        int: Number of newlines.
    """
    N = 0
    with open(file) as fd:
        buf = fd.read(size_hint)
        while buf:
            N += buf.count('\n')
            buf = fd.read(size_hint)
    return N
