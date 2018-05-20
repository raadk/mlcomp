import os
import pandas as pd
import numpy as np
import datetime
from threading import Thread
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
)
from sklearn.linear_model import TheilSenRegressor, Lars, Lasso
from mlcomp import Competition, create_app, eval_metrics


if __name__ == "__main__":
    print("Running")

    # Load the data
    boston = load_boston()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df = df.assign(y_true=boston.target, id=df.index.values)

    # Split it into training and testing sets
    train, test = train_test_split(df, random_state=333, test_size=0.2)

    y_test = test[['id', 'y_true']].copy()
    y_test.assign(final=np.random.rand(test.shape[0]) < 0.5, inplace=True)

    test = test.drop(['y_true'], axis=1)

    path = "boston_comp"

    if os.path.isdir(path):
        comp = Competition(path)
    else:
        comp = Competition.create(
            path=path,
            title="Boston Competition",
            y_test=y_test,
            eval_metric=eval_metrics['mean_squared_error'],
            end_date=datetime.date.today() + datetime.timedelta(days=10),
            train=train,
            test=test
        )

    X_train = train.drop(['id', 'y_true'], axis=1)
    y_train = train['y_true']

    X_test = test.drop(['id'], axis=1)

    # A simple linear model
    from sklearn.linear_model import LinearRegression
    regr = LinearRegression()
    regr.fit(X_train, y_train)

    # Submit
    submission = comp.get_sample_submission()  # A template for our submission
    submission['pred'] = regr.predict(X_test)
    comp.submit_predictions(submission,
                            team='Blue Lightning',
                            description='Linear regression')

    for mod in [RandomForestRegressor, AdaBoostRegressor,
                GradientBoostingRegressor, TheilSenRegressor, Lars, Lasso]:
        regr = mod()
        regr.fit(X_train, y_train)
        submission['pred'] = regr.predict(X_test)
        comp.submit_predictions(submission,
                                team='Blue Lightning',
                                description=mod.__name__)
    print(comp.leaderboard())

    app = create_app(comp)
    thread = Thread(target=app.run)
    thread.start()
