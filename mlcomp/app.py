"""
Copyright 2018 - Raad Khraishi
"""
import os
import uuid
import zipfile
import pandas as pd
import numpy as np
from io import BytesIO
from flask import (
    Flask, redirect, abort, make_response,
    url_for, render_template, request, send_file
)

from .competition import Competition
from .helpers import allowed_file, CODE_EXTENSIONS, DATA_EXTENSIONS


def _make_file_response(rv, filename, content_type='text/csv'):
    """
    Converts the return value from a view function into a real
    response object that is an instance of response_class.

    Sets response headers and filename
    """
    resp = make_response(rv)
    resp.headers["Content-Disposition"] = (
        "attachment; filename=%s" % filename
    )
    resp.headers["Content-Type"] = content_type
    return resp


def create_app(competition, display_digits=4, secret_key=str(uuid.uuid4())):
    """
    Creates a Flask app given an object of class Competition.

    Parameters
    ----------
    competition : Competition
        A competition object to build the app around.
    display_digits : int
        Number of digits to display in the app.
    secret_key : str
        The secret key for the Flask app.

    Returns
    -------
    app : flask.app.Flask
        The Flask web app
    """
    if not isinstance(competition, Competition):
        raise TypeError("competition object must be an "
                        "instance of the Competition class")

    app = Flask(__name__)
    app.secret_key = secret_key
    app.url_map.strict_slashes = False

    template_kwargs = {'digits': display_digits,
                       'title': competition.competition_info['title'],
                       'start_date': competition.competition_info['start_date'],
                       'end_date': competition.competition_info['end_date']}

    train_file = os.path.join(competition.path, 'train.csv')
    test_file = os.path.join(competition.path, 'test.csv')
    data_exists = os.path.isfile(train_file) or os.path.isfile(test_file)

    @app.route("/")
    def leaderboard():
        lb = competition.leaderboard()
        description = competition.competition_info['description']

        eval_metric = competition.competition_info['eval_metric']
        eval_metric_name = eval_metric.get_name().replace('_', ' ')

        final_n = competition.predictions['final'].sum()
        lb_n = competition.predictions.shape[0] - final_n

        return render_template('leaderboard.html',
                               lb=lb,
                               eval_metric_name=eval_metric_name,
                               description=description,
                               lb_n=lb_n,
                               final_n=final_n,
                               data_download=data_exists,
                               **template_kwargs)

    @app.route("/submission/<int:id_>/")
    def submission_info(id_):
        sub_info = competition.get_submission_info(id_)
        if sub_info is None:
            abort(404)

        lb = competition.leaderboard()
        rank = np.where(lb.index.values == id_)[0][0] + 1

        code = competition.get_code(id_)
        if code is not None:
            code = ''.join(code)

        return render_template('submission_details.html',
                               sub_info=sub_info,
                               id_=id_,
                               code=code,
                               rank=rank,
                               **template_kwargs)

    # @TODO Very messy
    @app.route("/submit/", methods=['GET', 'POST'])
    def submit():
        errors = []
        pred_file_error = 'Invalid predictions file'
        code_file_error = 'Invalid script'

        if request.method == 'POST':

            # General
            team = request.values.get('team', '')
            if team == '' or team is None:
                errors += ['Invalid team']
            description = request.values.get('description', '')

            # Pred file
            if 'pred_file' not in request.files:
                errors += [pred_file_error]

            pred_file = request.files['pred_file']

            if pred_file and allowed_file(pred_file.filename, DATA_EXTENSIONS):
                try:
                    pred_df = pd.read_csv(pred_file)
                except Exception:
                    errors += [pred_file_error]
            else:
                errors += [pred_file_error]

            # Code
            code = None
            if 'code_file' in request.files:
                code_file = request.files['code_file']

                # @FIXME
                if code_file:
                    if allowed_file(code_file.filename, CODE_EXTENSIONS):
                        if isinstance(code_file, list):
                            # Quick fix to work with api
                            code = code_file
                        else:
                            try:
                                code = list(code_file.read().decode('utf-8'))
                            except Exception as e:
                                print(e)
                                errors += [code_file_error]
                    else:
                        errors += [code_file_error]

            # Submit
            if not errors:
                try:
                    competition.submit_predictions(preds=pred_df,
                                                   team=team,
                                                   description=description,
                                                   code=code)
                    competition.save()
                    return redirect(url_for('leaderboard'))
                except ValueError:
                    errors += ['Please check your submission and try again']

        return render_template('submit_form.html',
                               **template_kwargs,
                               errors=set(errors)), 400 if errors else 200

    @app.route("/submission/<int:id_>/delete/", methods=['POST', 'DELETE'])
    def delete_submission(id_):
        competition.delete_submission(id_)
        competition.save()
        return redirect(url_for('leaderboard'))

    @app.route("/leaderboard/download/")
    def get_leaderboard():
        lb = competition.leaderboard()
        resp = _make_file_response(lb.to_csv(index=False),
                                   'leaderboard.csv',
                                   'text/csv')
        return resp

    @app.route("/submission/sample/download/")
    def get_sample_submission():
        sample_submission = competition.get_sample_submission()
        resp = _make_file_response(sample_submission.to_csv(index=False),
                                   'sample_submission.csv',
                                   'text/csv')
        return resp

    @app.route("/predictions/download/")
    def get_predictions():
        preds = competition.get_predictions()
        resp = _make_file_response(preds.to_csv(index=False),
                                   'predictions.csv',
                                   'text/csv')
        return resp

    @app.route("/submission/<int:id_>/predictions/download/")
    def get_prediction(id_):
        preds = competition.get_prediction(id_)
        if preds is None:
            abort(404)

        resp = _make_file_response(preds.to_csv(index=False),
                                   'prediction_%s.csv' % str(id_),
                                   'text/csv')
        return resp

    @app.route("/submission/<int:id_>/code/download/")
    def get_code(id_):
        code = competition.get_code(id_)
        if code is None:
            abort(404)

        resp = _make_file_response(''.join(code),
                                   'code_%s.py' % str(id_),
                                   'text/plain')
        return resp

    @app.route("/data/download/")
    def get_data():
        # Should be streaming the data instead
        files = []

        if os.path.isfile(train_file):
            files.append(train_file)

        if os.path.isfile(test_file):
            files.append(test_file)

        if not files:
            abort(400)

        mem_file = BytesIO()

        with zipfile.ZipFile(mem_file, 'w') as zipped_file:
            for f in files:
                zipped_file.write(f)

        mem_file.seek(0)

        return send_file(
            mem_file,
            attachment_filename='data.zip',
            as_attachment=True
        )

    return app
