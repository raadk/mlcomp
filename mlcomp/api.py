import pandas as pd
import requests
import codecs
from urllib.parse import urljoin
from .helpers import allowed_file, CODE_EXTENSIONS, DATA_EXTENSIONS

DEFAULT_URL = "http://127.0.0.1:5000/"


def get_leaderboard(url=DEFAULT_URL):
    """
    Returns a leaderboard with rankings of submissions.

    Parameters
    ----------
    url : str
        The url of the running Flask app.

    Returns
    -------
    leaderboard : pd.DataFrame
    """
    return pd.read_csv(urljoin(url, "/leaderboard/download/"))


def get_data(url=DEFAULT_URL):
    pass



def get_predictions(url=DEFAULT_URL):
    """
    Get all submitted predictions

    Returns
    -------
    predictions : pd.DataFrame
    """
    return pd.read_csv(urljoin(url, "/predictions/download"))


def get_prediction(id_, url=DEFAULT_URL):
    """
    Get the predictions for a specific submission.

    Parameters
    ----------
    id_ : int
        The id of the submission.
    url : str
        The url of the running Flask app.

    Returns
    -------
    prediction : pd.DataFrame
    """
    return pd.read_csv(
        urljoin(url, "/submission/%i/predictions/download/" % id_)
    )


def delete_submission(id_, url=DEFAULT_URL):
    """
    Delete a specific submission.

    Parameters
    ----------
    id_ : int
        The id of the submission to remove.
    url : str
        The url of the running Flask app.

    Returns
    -------
    response : requests.models.Response
    """
    return requests.delete(urljoin(url, "/submission/%i/delete/" % id_))


def get_code(id_, url=DEFAULT_URL):
    """
    Get the code for a specific submission (if submitted).

    Parameters
    ----------
    id_ : int
        The id of the submission.
    url : str
        The url of the running Flask app.

    Returns
    -------
    code : str
    """
    code = requests.get(urljoin(url, "/submission/%i/code/download/" % id_))
    return code.content.decode('utf-8')


def get_sample_submission(url=DEFAULT_URL):
    """
    Returns an example submission as a pd.DataFrame

    Parameters
    ----------
    url : str
        The url of the running Flask app.

    Returns
    -------
    sample_submission : pd.DataFrame
    """
    return pd.read_csv(urljoin(url, "/submission/sample/download/"))


def submit(preds, team, description="", code=None, url=DEFAULT_URL):
    """
    Submit predictions to the competition.

    Parameters
    ----------
    preds : pd.DataFrame/str
        A pd.DataFrame of predictions that contains columns 'id' and 'pred'.

        A path to a csv file.
    team : str
        the name of the team the predictions will be associated with
    description (optional) : str
        An optional description of the submission.
    code (optional) : str/list
        A optional path to a python script to store

        May also be a list of strings containing the python script.
    url : str
        The url of the running Flask app.

    Returns
    -------
    response : requests.models.Response
    """
    if isinstance(preds, str):
        if not allowed_file(preds, DATA_EXTENSIONS):
            raise ValueError('Invalid argument, preds. Only csv files'
                             'and pd.DataFrame are currently supported')
        preds = pd.read_csv(preds)

    files = {
        "pred_file": ('preds.csv', preds.to_csv(index=False), 'text/csv'),
        "team": (None, team),
        "description": (None, description)
    }

    if code is not None:
        if isinstance(code, str):
            if not allowed_file(code, CODE_EXTENSIONS):
                raise ValueError('Invalid argument, code file. '
                                 'Only .py files currently supported')
            files['code_file'] = ('code.py',
                                  codecs.open(code, 'r', encoding='utf-8'),
                                  'text/plain')
        else:
            # Need to clean for next version...
            files['code_file'] = ('code.py', ''.join(code))

    return requests.post(urljoin(url, "/submit/"), files=files)
