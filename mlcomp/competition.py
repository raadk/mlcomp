"""
Copyright 2018 - Raad Khraishi
"""
import os
import uuid
import codecs
import datetime
import dill
import gzip
import re
import shutil
import pandas as pd
import numpy as np
from sklearn import metrics

from .helpers import (
    parse_date, sorted_lists_equal, allowed_file,
    CODE_EXTENSIONS, DATA_EXTENSIONS
)

COLUMN_NAMES = ['id', 'y_true', 'final']
PRED_COLUMN_NAMES = ['id', 'pred']


class CompetitionNotCreatedError(Exception):
    """Error thrown if competition has not been created"""
    pass


class EvalMetric:
    """ Class for evaluation metrics """
    def __init__(self, fun, max_, name=None):
        """
        Parameters
        ----------
        fun : function
            A function that takes two objects, y_true and y_pred,
            and returns a scalar value.
        max_ : bool
            Is a higher value better?
        """
        self.fun = fun
        self.max_ = max_

        if name is None:
            self.name = self.fun.__name__
        else:
            self.name = name

    def __call__(self, *args, **kwargs):
        return self.fun(*args, **kwargs)

    def get_name(self):
        """Returns the name of the metric"""
        return self.name


eval_metrics = {
    "cohen_kappa_score": EvalMetric(metrics.cohen_kappa_score, True),
    "f1_score": EvalMetric(metrics.f1_score, True),
    "log_loss": EvalMetric(metrics.log_loss, False),
    "mean_absolute_error": EvalMetric(metrics.mean_absolute_error, False),
    "mean_squared_error": EvalMetric(metrics.mean_squared_error, False),
    "median_absolute_error": EvalMetric(metrics.median_absolute_error, False),
    "precision_score": EvalMetric(metrics.precision_score, True),
    "recall_score": EvalMetric(metrics.recall_score, True),
    "roc_auc_score": EvalMetric(metrics.roc_auc_score, True),
}


class Competition:
    # @TODO FINISH THE EXAMPLE
    """Class to create and manage a machine learning competition"""

    _competition_not_created_error = CompetitionNotCreatedError(
        "Competition does not exist.Use Competition.create(...)"
    )

    _competition_file = 'competition.pklz'
    _rand = np.random.RandomState(333)

    def __init__(self, path):
        """
        Create an instance of the competition class given an existing
        competition.

        Use the class method, Competition.create, to first create  a
        competition

        Parameters
        ----------
        path : string
            The path to where the competition folder is located.
        """
        self.path = path
        self.created = False

        file_path = os.path.join(path, Competition._competition_file)

        if os.path.isfile(file_path):
            with gzip.open(file_path, 'rb') as f:
                try:
                    self.predictions, self.submission_info, \
                        self.competition_info, self.code = dill.load(f)
                except EOFError:
                    raise Competition._competition_not_created_error
            self.created = True
        else:
            raise Competition._competition_not_created_error

            # Will probably load more often than you create

    @classmethod
    def create(cls, path, title,
               y_test, eval_metric,
               end_date, start_date=datetime.date.today(),
               description="",
               final_frac=0,
               train=None, test=None):
        """
        Creates  a file that stores the competition data at the path provided

        Parameters
        ----------
        path : string
            The path to where the competition folder is located
        title : str
            The name of the competition
        y_test : pd.DataFrame/str/np.ndarray
            A data frame containing columns id, y_true, and optionally final.
            The final column denotes which observations will be
            used to calculate the final score once the competition is closed

            If not contained, will use final_frac to randomly select
            a subset of observations to be used in final scoring.

            Or, a path to a csv file.

            Or, a np.ndarray where the first two columns will be interpretted
            as id and y_true. An optional third column will be used to
            determine which observations will be used for final scoring.
        eval_metric : EvalMetric
            The evaluation metric to use.
        end_date : str/datetime.date
            The date the competition ends.
            If a string, it will parse with dateutil.parser.parse.
        start_date (optional) : str/datetime.date
            The date the competition started.
            If a string,  it will parse with dateutil.parser.parse.
        description (optional) : str
            A description of the competition
        final_frac : float
            A number between 0 and 1 that represents the proportion of the
            data to be used in final scoring if a final column is not contained
            in y_test.

        Returns
        -------
        Competition : An instance of the competition class.
        """

        if os.path.exists(path):
            raise FileExistsError("A competition may already exist at " + path)
        else:
            os.makedirs(path)

        if isinstance(y_test, str):
            if not allowed_file(y_test, DATA_EXTENSIONS):
                raise ValueError('Invalid argument, y_test. Only csv files '
                                 'and pd.DataFrame are currently supported')
            y_test = pd.read_csv(y_test)

        if isinstance(y_test, np.ndarray):
            k = y_test.shape[1]
            if k < 2:
                raise ValueError('Invalid argument, y_test. At least two '
                                 'columns must be contained in the np.array')
            y_test = pd.DataFrame(y_test, columns=COLUMN_NAMES[0:k])

        n_obs = y_test.shape[0]

        if not set(COLUMN_NAMES[:2]).issubset(y_test.columns.values):
            raise ValueError("y_test must contain columns 'id' and 'y_true'")

        if 'final' not in y_test.columns.values:
            if (final_frac is None) or (final_frac < 0) or (final_frac >= 1):
                raise ValueError("final_frac must be between 0 and 1")
            if final_frac == 0:
                y_test['final'] = False
            else:
                y_test['final'] = Competition._rand.rand(n_obs) < final_frac
        else:
            y_test['final'] = y_test['final'].astype(bool)
            if y_test['final'].sum() >= n_obs:
                raise ValueError("final column must indicate "
                                 "sample of observations (< 100%) "
                                 "to be used in final scoring")

        if not isinstance(eval_metric, EvalMetric):
            raise TypeError("eval_metric must be an "
                            "instance of class EvalMetric")

        start_date = parse_date(start_date)
        end_date = parse_date(end_date)

        predictions = y_test.copy()
        predictions.index.name = 'id'

        submission_info = pd.DataFrame(columns=['lb_score',
                                                'final_score',
                                                'team',
                                                'description',
                                                'submitted'])

        if description:
            description = description.split('\n')

        if train is not None:
            train_data_path = os.path.join(path, 'train.csv')
            if isinstance(train, str):
                if not allowed_file(train, DATA_EXTENSIONS):
                    raise ValueError(
                        'Invalid argument, train. Only csv files '
                        'and pd.DataFrame are currently supported'
                    )
                shutil.copy2(train, train_data_path)
            elif isinstance(train, pd.DataFrame):
                train.to_csv(train_data_path, index=False)
            else:
                raise TypeError(
                    'Invalid argument, train. Only csv files '
                    'and pd.DataFrame are currently supported.'
                )

        if test is not None:
            test_data_path = os.path.join(path, 'test.csv')
            if isinstance(test, str):
                if not allowed_file(test, DATA_EXTENSIONS):
                    raise ValueError(
                        'Invalid argument, test. Only csv files '
                        'and pd.DataFrame are currently supported'
                    )
                shutil.copy2(test, test_data_path)
            elif isinstance(test, pd.DataFrame):
                test.to_csv(test_data_path, index=False)
            else:
                raise TypeError(
                    'Invalid argument, test. Only csv files '
                    'and pd.DataFrame are currently supported.'
                )

        competition_info = {
            'title': title,
            'eval_metric': eval_metric,
            'date_created': datetime.datetime.now(),
            'start_date': start_date,
            'end_date': end_date,
            'description': description
        }

        code = dict()

        Competition._save(path, predictions, submission_info,
                          competition_info, code)
        return cls(path)

    @staticmethod
    def _save(path, predictions, submission_info, competition_info, code):
        """
        Save the predictions, submission info, and competition info
        to path

        Parameters
        ----------
        path : string
            The path to the competition folder.
        predictions : pd.DataFrame
            The DataFrame of predictions.
        submission_info : pd.DataFrame
            The DataFrame of information on submissions.
        competition_info : dict
            The dictionary of competition information.
        code : dict
            The dictionary containing any uploaded scripts.
        """
        file_path = os.path.join(path, Competition._competition_file)

        with gzip.open(file_path, 'wb') as f:
            dill.dump([predictions, submission_info,
                       competition_info, code], f)

    def save(self):
        """ Saves the competition to the supplied file path """
        Competition._save(self.path, self.predictions,
                          self.submission_info, self.competition_info,
                          self.code)

    @staticmethod
    def score_predictions(y_true, y_pred, eval_fun):
        """
        Wrapper function to score predictions.

        Parameters
        ----------
        y_true : array
            An array of the observed values.
        y_pred : array
            An array of the predicted values.
        eval_fun : function
            Should take arguments y_true and y_pred and return a scalar.
        """
        return eval_fun(y_true, y_pred)

    def _new_id(self, use='row'):
        """
        Creates a new id for a submission

        Parameters
        ----------
        use : str
            one of 'row', 'datetime', or 'uuid'

        Returns
        -------
        int : an id
        """
        if use == 'uuid':
            id_ = int(re.sub('\-', '', uuid.uuid4()))
        elif use == "datetime":
            id_ = int(re.sub('\-|\:|\.|\s', '', str(datetime.datetime.now())))
        else:
            if self.submission_info.empty:
                id_ = 1
            else:
                id_ = np.max(self.submission_info.index.values) + 1
        return id_

    def is_closed(self):
        """Checks whether a competition is still open based on the end_date"""
        delta = datetime.date.today() - self.competition_info['end_date']
        return delta.days >= 0

    def _get_competition_metric(self, what="fun"):
        """Gets either the fun or max_ attributes of the competition metric"""
        eval_metric = self.competition_info['eval_metric']
        if what == "fun":
            return eval_metric.fun
        else:
            return eval_metric.max_

    def submit_predictions(self, preds, team, description="", code=None):
        """
        Submit predictions to the competition.

        Parameters
        ----------
        preds : pd.DataFrame/str
            A pd.DataFrame of predictions that contains columns 'id' and 'pred'.

            Or, a path to a csv file.

            Or, a np.ndarray where the first two columns will be interpretted
            as 'id' and 'pred'.
        team : str
            the name of the team the predictions will be associated with
        description (optional) : str
            An optional description of the submission.
        code (optional) : str/list
            A optional path to a python script to store in the competition.

            May also be a list of strings containing the python script.
        """

        if isinstance(preds, str):
            if not allowed_file(preds, DATA_EXTENSIONS):
                raise ValueError('Invalid argument, preds. Only csv files'
                                 'and pd.DataFrame are currently supported')
            preds = pd.read_csv(preds)

        if isinstance(preds, np.ndarray):
            k = preds.shape[1]
            if k < 2:
                raise ValueError("preds must have columns id and pred only")
            preds = pd.DataFrame(preds, columns=PRED_COLUMN_NAMES)

        if not set(PRED_COLUMN_NAMES) == set(preds.columns.values):
            raise ValueError("preds must have columns id and pred only")

        if preds.shape[0] != self.predictions.shape[0]:
            raise ValueError("preds must contain %i rows" %
                             self.predictions.shape[0])

        if preds.isnull().values.any():
            raise ValueError("preds may not contain missing values")

        if not sorted_lists_equal(preds['id'].values,
                                  self.predictions['id'].values):
            raise ValueError("preds must contain the correct ids")

        id_ = self._new_id()
        preds = preds.copy()
        preds.rename(columns={'pred': id_}, inplace=True)

        self.predictions = pd.merge(self.predictions, preds, on='id')

        final_mask = self.predictions['final']

        lb_score = Competition.score_predictions(
            self.predictions.loc[~final_mask, 'y_true'],
            self.predictions.loc[~final_mask, id_],
            self._get_competition_metric('fun')
        )

        if final_mask.sum() <= 0:
            final_score = lb_score
        else:
            final_score = Competition.score_predictions(
                self.predictions.loc[final_mask, 'y_true'],
                self.predictions.loc[final_mask, id_],
                self._get_competition_metric('fun')
            )

        self.submission_info.loc[id_] = {
            'lb_score': lb_score,
            'final_score': final_score,
            'team': team,
            'submitted': datetime.datetime.now(),
            'description': description
        }

        if code is not None:
            if isinstance(code, str):
                if not allowed_file(code, CODE_EXTENSIONS):
                    raise ValueError("Invalid argument, code file. "
                                     "Only .py files currently supported")
                with codecs.open(code, 'r', encoding='utf-8') as f:
                    code = f.readlines()
            self.code[id_] = code

    def delete_submission(self, id_):
        """
        Delete a specific submission.

        Parameters
        ----------
        id_ : int
            The id of the submission to remove.
        """
        try:
            self.submission_info.drop(id_, axis=0, inplace=True)
            self.predictions.drop(id_, axis=1, inplace=True)
            if id_ in self.code:
                del self.code[id_]
        except KeyError:
            return

    def get_sample_submission(self):
        """
        Get an example submission

        Returns
        -------
        pd.DataFrame : a sample submission filled with dummy data
        """
        sample_submission = self.predictions[['id']].copy()
        dummy_data = self._rand.choice(
            self.predictions['y_true'].sample(10, replace=True),
            size=self.predictions.shape[0],
            replace=True
        )
        sample_submission['pred'] = dummy_data
        return sample_submission

    def get_predictions(self):
        """
        Get all submitted predictions

        Returns
        -------
        pd.DataFrame : a DataFrame of all predictions
        """
        preds = self.predictions.copy()
        return preds.drop(['final', 'y_true'], axis=1)

    def get_code(self, id_):
        """
        Get the code for a specific submission (if submitted)

        Parameters
        ----------
        id_ : int
            The id of the submission.

        Returns
        -------
        str : the uploaded code
        """
        try:
            return self.code[id_]
        except KeyError:
            return

    def get_data(self):
        """
        Get the competition data

        Returns
        -------
        train, test - tuple
            The training and testing sets if they exist, else None

        """

        train_file = os.path.join(self.path, 'train.csv')
        test_file = os.path.join(self.path, 'test.csv')

        train = None
        test = None

        if os.path.isfile(train_file):
            train = pd.read_csv(train_file)

        if os.path.isfile(test_file):
            test = pd.read_csv(test_file)

        return train, test

    def get_prediction(self, id_):
        """
        Get the predictions for a specific submission

        Parameters
        ----------
        id_ : int
            The id of the submission.

        Returns
        -------
        pd.DataFrame : a DataFrame containing a submission's predictions
        """
        try:
            return self.predictions[['id', id_]].rename(columns={id_: 'pred'})
        except KeyError:
            return

    def get_submission_info(self, id_):
        """
        Get information on a specific submission.

        Parameters
        ----------
        id_ : int
            the id of the submission.

        Returns
        -------
        pd.Series : information on a submission
        """
        sub_info = self.submission_info.copy()

        if not self.is_closed():
            sub_info.drop('final_score', axis=1, inplace=True)

        try:
            return sub_info.loc[id_, :]
        except KeyError:
            return

    def leaderboard(self, show_late=True):
        """
        Get leaderboard rankings of submissions

        Returns
        -------
        pd.DataFrame : a leaderboard of submissions ranked by score
        """
        lb = self.submission_info.copy()
        ascending = not self._get_competition_metric('max')
        if self.is_closed():
            by = 'final_score'
        else:
            by = 'lb_score'
            lb.drop('final_score', axis=1, inplace=True)

        if not show_late:
            lb = lb.loc[lb['submitted'] <= self.competition_info['end_date'], :]

        return lb.sort_values(by, axis=0, ascending=ascending)

    def __str__(self):
        if not self.created:
            return "Competition has not been created or loaded"
        s = "{title} competition from {start_date} to {end_date}"
        return s.format(**self.competition_info)

    def __repr__(self):
        return "Competition(path='%s')" % self.path
