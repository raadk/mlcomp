# mlcomp: Machine learning competitions

`mlcomp` is a python package that allows you to create and run machine learning and data science competitions.

## Example

Let's run through a quick example using the Boston housing data from `sklearn` to illustrate how the package works.

First, let's setup the data by splitting it into three datasets:

1) The training set which contains the target variable, ids, and features
2) The test set which contains ids and features, but no target variable
3) The target variable and ids for the test set which will be used to evaluate predictions

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Load the data
boston = load_boston()
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df = df.assign(y_test=boston.target, id=df.index.values)

# Split it into training and testing sets
train, test = train_test_split(df, random_state=333, test_size=0.2)

y_test = test[['id', 'y_test']].copy()
y_test.assign(final=np.random.rand(test.shape[0]) < 0.5, inplace=True)

test = test.drop(['y_test'], axis=1)
```

### Creating a competition
Now that we have our data, we are ready to create a competition.

```python
import datetime
from mlcomp import Competition, eval_metrics

comp = Competition.create(
    path='boston_comp',
    title="Boston Competition",
    y_test=y_test,
    eval_metric=eval_metrics['mean_squared_error'],
    end_date=datetime.date.today() + datetime.timedelta(days=10),
    train=train,
    test=test
)
```

### Loading a competition
In the future, we can just load the competition directly given the path to the competition folder.

```python
comp = Competition(path='boston_comp')
```

### Submitting predictions
With our competition created, we can estimate a model and submit our predictions.

```python
# Setup data for modelling
X_train = train.drop(['id', 'y_test'], axis=1)
y_train = train['y_test']

X_test = test.drop(['id'], axis=1)

# A simple linear model
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(X_train, y_train)

# Submit
submission = comp.get_sample_submission()  # A template for our submission
submission['pred'] = regr.predict(X_test)
comp.submit_predictions(submission, team='Blue Lightning', description='Linear regression')
```

Let's add a few more models for fun.

```python
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import TheilSenRegressor, Lars, Lasso

for mod in [RandomForestRegressor, AdaBoostRegressor,
            GradientBoostingRegressor, TheilSenRegressor, Lars, Lasso]:
    regr = mod()
    regr.fit(X_train, y_train)
    submission['pred'] = regr.predict(X_test)
    comp.submit_predictions(submission, team='Blue Lightning', description=mod.__name__)
```
How well did the predictions perform?

```python
comp.leaderboard()
```

### Saving the competition
By default, when working directly with the competition object, the competition is not saved to disk after each submission. When you are done with your session, run `comp.save()` to store your new predictions.


### Using a custom evaluation metric

Although some evaluation metrics have been included in the package, it is easy to use your own.

Just create an instance of the `EvaluationMetric` class and supply it to `Competition.create(...)`.
The function you supply should take the observed values (e.g. `y_true`) and predicted values (e.g. `y_pred`) as arguments and return a scalar score.

For example, to create a custom metric:

```python
from mlcomp import EvaluationMetric

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

root_mean_squared_error = EvaluationMetric(rmse, max_=False)
```

## Flask app

![Example competition](img/boston.png?raw=true "Boston Competition")



It is easy to run a simple Flask app using the created competition:


```python
from mlcomp import create_app

app = create_app(comp)
app.run(host='127.0.0.1', port=5000)
```

Submissions made directly to the flask app using the website or api will be saved immediately.


### API
There is also an API if you wish to work with the app rather than the Competition object.
This can be useful if you are running the app on your local network and would like others to access it.

```python
from mlcomp.api import get_leaderboard, get_prediction
get_leaderboard()
get_prediction(1)
```


## Contributions are welcome!
