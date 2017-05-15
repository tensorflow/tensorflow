# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""An estimator is a rule for calculating an estimate of a given quantity.

# Estimators

* **Estimators** are used to train and evaluate TensorFlow models.
They support regression and classification problems.
* **Classifiers** are functions that have discrete outcomes.
* **Regressors** are functions that predict continuous values.

## Choosing the correct estimator

* For **Regression** problems use one of the following:
    * `LinearRegressor`: Uses linear model.
    * `DNNRegressor`: Uses DNN.
    * `DNNLinearCombinedRegressor`: Uses Wide & Deep.
    * `TensorForestEstimator`: Uses RandomForest.
      See tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator.
    * `Estimator`: Use when you need a custom model.

* For **Classification** problems use one of the following:
    * `LinearClassifier`: Multiclass classifier using Linear model.
    * `DNNClassifier`: Multiclass classifier using DNN.
    * `DNNLinearCombinedClassifier`: Multiclass classifier using Wide & Deep.
    * `TensorForestEstimator`: Uses RandomForest.
      See tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator.
    * `SVM`: Binary classifier using linear SVMs.
    * `LogisticRegressor`: Use when you need custom model for binary
       classification.
    * `Estimator`: Use when you need custom model for N class classification.

## Pre-canned Estimators

Pre-canned estimators are machine learning estimators premade for general
purpose problems. If you need more customization, you can always write your
own custom estimator as described in the section below.

Pre-canned estimators are tested and optimized for speed and quality.

### Define the feature columns

Here are some possible types of feature columns used as inputs to a pre-canned
estimator.

Feature columns may vary based on the estimator used. So you can see which
feature columns are fed to each estimator in the below section.

```python
sparse_feature_a = sparse_column_with_keys(
    column_name="sparse_feature_a", keys=["AB", "CD", ...])

embedding_feature_a = embedding_column(
    sparse_id_column=sparse_feature_a, dimension=3, combiner="sum")

sparse_feature_b = sparse_column_with_hash_bucket(
    column_name="sparse_feature_b", hash_bucket_size=1000)

embedding_feature_b = embedding_column(
    sparse_id_column=sparse_feature_b, dimension=16, combiner="sum")

crossed_feature_a_x_b = crossed_column(
    columns=[sparse_feature_a, sparse_feature_b], hash_bucket_size=10000)

real_feature = real_valued_column("real_feature")
real_feature_buckets = bucketized_column(
    source_column=real_feature,
    boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
```

### Create the pre-canned estimator

DNNClassifier, DNNRegressor, and DNNLinearCombinedClassifier are all pretty
similar to each other in how you use them. You can easily plug in an
optimizer and/or regularization to those estimators.

#### DNNClassifier

A classifier for TensorFlow DNN models.

```python
my_features = [embedding_feature_a, embedding_feature_b]
estimator = DNNClassifier(
    feature_columns=my_features,
    hidden_units=[1024, 512, 256],
    optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001
    ))
```

#### DNNRegressor

A regressor for TensorFlow DNN models.

```python
my_features = [embedding_feature_a, embedding_feature_b]

estimator = DNNRegressor(
feature_columns=my_features,
hidden_units=[1024, 512, 256])

# Or estimator using the ProximalAdagradOptimizer optimizer with
# regularization.
estimator = DNNRegressor(
    feature_columns=my_features,
    hidden_units=[1024, 512, 256],
    optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001
    ))
```

#### DNNLinearCombinedClassifier

A classifier for TensorFlow Linear and DNN joined training models.

* Wide and deep model
* Multi class (2 by default)

```python
my_linear_features = [crossed_feature_a_x_b]
my_deep_features = [embedding_feature_a, embedding_feature_b]
estimator = DNNLinearCombinedClassifier(
      # Common settings
      n_classes=n_classes,
      weight_column_name=weight_column_name,
      # Wide settings
      linear_feature_columns=my_linear_features,
      linear_optimizer=tf.train.FtrlOptimizer(...),
      # Deep settings
      dnn_feature_columns=my_deep_features,
      dnn_hidden_units=[1000, 500, 100],
      dnn_optimizer=tf.train.AdagradOptimizer(...))
```

#### LinearClassifier

Train a linear model to classify instances into one of multiple possible
classes. When number of possible classes is 2, this is binary classification.

```python
my_features = [sparse_feature_b, crossed_feature_a_x_b]
estimator = LinearClassifier(
    feature_columns=my_features,
    optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001
    ))
```

#### LinearRegressor

Train a linear regression model to predict a label value given observation of
feature values.

```python
my_features = [sparse_feature_b, crossed_feature_a_x_b]
estimator = LinearRegressor(
    feature_columns=my_features)
```

### LogisticRegressor

Logistic regression estimator for binary classification.

```python
# See tf.contrib.learn.Estimator(...) for details on model_fn structure
def my_model_fn(...):
  pass

estimator = LogisticRegressor(model_fn=my_model_fn)

# Input builders
def input_fn_train:
  pass

estimator.fit(input_fn=input_fn_train)
estimator.predict(x=x)
```

#### SVM - Support Vector Machine

Support Vector Machine (SVM) model for binary classification.

Currently only linear SVMs are supported.

```python
my_features = [real_feature, sparse_feature_a]
estimator = SVM(
    example_id_column='example_id',
    feature_columns=my_features,
    l2_regularization=10.0)
```

#### DynamicRnnEstimator

An `Estimator` that uses a recurrent neural network with dynamic unrolling.

```python
problem_type = ProblemType.CLASSIFICATION  # or REGRESSION
prediction_type = PredictionType.SINGLE_VALUE  # or MULTIPLE_VALUE

estimator = DynamicRnnEstimator(problem_type,
                                prediction_type,
                                my_feature_columns)
```

### Use the estimator

There are two main functions for using estimators, one of which is for
training, and one of which is for evaluation.
You can specify different data sources for each one in order to use different
datasets for train and eval.

```python
# Input builders
def input_fn_train: # returns x, Y
  ...
estimator.fit(input_fn=input_fn_train)

def input_fn_eval: # returns x, Y
  ...
estimator.evaluate(input_fn=input_fn_eval)
estimator.predict(x=x)
```

## Creating Custom Estimator

To create a custom `Estimator`, provide a function to `Estimator`'s
constructor that builds your model (`model_fn`, below):


```python
estimator = tf.contrib.learn.Estimator(
      model_fn=model_fn,
      model_dir=model_dir)  # Where the model's data (e.g., checkpoints)
                            # are saved.
```

Here is a skeleton of this function, with descriptions of its arguments and
return values in the accompanying tables:

```python
def model_fn(features, targets, mode, params):
   # Logic to do the following:
   # 1. Configure the model via TensorFlow operations
   # 2. Define the loss function for training/evaluation
   # 3. Define the training operation/optimizer
   # 4. Generate predictions
   return predictions, loss, train_op
```

You may use `mode` and check against
`tf.contrib.learn.ModeKeys.{TRAIN, EVAL, INFER}` to parameterize `model_fn`.

In the Further Reading section below, there is an end-to-end TensorFlow
tutorial for building a custom estimator.

## Additional Estimators

There is an additional estimators under
`tensorflow.contrib.factorization.python.ops`:

*   Gaussian mixture model (GMM) clustering

## Further reading

For further reading, there are several tutorials with relevant topics,
including:

*   [Overview of linear models](../../../tutorials/linear/overview.md)
*   [Linear model tutorial](../../../tutorials/wide/index.md)
*   [Wide and deep learning tutorial](../../../tutorials/wide_and_deep/index.md)
*   [Custom estimator tutorial](../../../tutorials/estimators/index.md)
*   [Building input functions](../../../tutorials/input_fn/index.md)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.estimators._sklearn import NotFittedError
from tensorflow.contrib.learn.python.learn.estimators.constants import ProblemType
from tensorflow.contrib.learn.python.learn.estimators.dnn import DNNClassifier
from tensorflow.contrib.learn.python.learn.estimators.dnn import DNNEstimator
from tensorflow.contrib.learn.python.learn.estimators.dnn import DNNRegressor
from tensorflow.contrib.learn.python.learn.estimators.dnn_linear_combined import DNNLinearCombinedClassifier
from tensorflow.contrib.learn.python.learn.estimators.dnn_linear_combined import DNNLinearCombinedEstimator
from tensorflow.contrib.learn.python.learn.estimators.dnn_linear_combined import DNNLinearCombinedRegressor
from tensorflow.contrib.learn.python.learn.estimators.dynamic_rnn_estimator import DynamicRnnEstimator
from tensorflow.contrib.learn.python.learn.estimators.estimator import BaseEstimator
from tensorflow.contrib.learn.python.learn.estimators.estimator import Estimator
from tensorflow.contrib.learn.python.learn.estimators.estimator import infer_real_valued_columns_from_input
from tensorflow.contrib.learn.python.learn.estimators.estimator import infer_real_valued_columns_from_input_fn
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
from tensorflow.contrib.learn.python.learn.estimators.head import binary_svm_head
from tensorflow.contrib.learn.python.learn.estimators.head import Head
from tensorflow.contrib.learn.python.learn.estimators.head import multi_class_head
from tensorflow.contrib.learn.python.learn.estimators.head import multi_head
from tensorflow.contrib.learn.python.learn.estimators.head import multi_label_head
from tensorflow.contrib.learn.python.learn.estimators.head import no_op_train_fn
from tensorflow.contrib.learn.python.learn.estimators.head import poisson_regression_head
from tensorflow.contrib.learn.python.learn.estimators.head import regression_head
from tensorflow.contrib.learn.python.learn.estimators.kmeans import KMeansClustering
from tensorflow.contrib.learn.python.learn.estimators.linear import LinearClassifier
from tensorflow.contrib.learn.python.learn.estimators.linear import LinearEstimator
from tensorflow.contrib.learn.python.learn.estimators.linear import LinearRegressor
from tensorflow.contrib.learn.python.learn.estimators.logistic_regressor import LogisticRegressor
from tensorflow.contrib.learn.python.learn.estimators.metric_key import MetricKey
from tensorflow.contrib.learn.python.learn.estimators.model_fn import ModeKeys
from tensorflow.contrib.learn.python.learn.estimators.model_fn import ModelFnOps
from tensorflow.contrib.learn.python.learn.estimators.prediction_key import PredictionKey
from tensorflow.contrib.learn.python.learn.estimators.rnn_common import PredictionType
from tensorflow.contrib.learn.python.learn.estimators.run_config import ClusterConfig
from tensorflow.contrib.learn.python.learn.estimators.run_config import Environment
from tensorflow.contrib.learn.python.learn.estimators.run_config import RunConfig
from tensorflow.contrib.learn.python.learn.estimators.run_config import TaskType
from tensorflow.contrib.learn.python.learn.estimators.svm import SVM
