"""
Copyright 2018 - Raad Khraishi
"""
import dateutil


CODE_EXTENSIONS = {'py'}
DATA_EXTENSIONS = {'csv'}


def parse_date(date):
    """
    Parses a string as a date.

    Parameters
    ----------
    date : str

    Returns
    -------
    date : datetime.date
        The parsed date
    """
    return dateutil.parser.parse(str(date)).date()


def sorted_lists_equal(x, y):
    """
    Checks if two lists have the same elements.

    Parameters
    ----------
    x : list
    y : list

    Returns
    -------
    are_equal : bool
        True if both lists have the same elements, False otherwise.
    """
    return sorted(x) == sorted(y)


def allowed_file(filename, extensions={'csv'}):
    """
    Checks if a filename contains an allowable extension.

    Parameters
    ----------
    filename : str
        The filename to check.
    extensions : set
        The set of allowable file extensions.

    Returns
    -------
    allowed : bool
        True if allowable extension, False otherwise.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in extensions
