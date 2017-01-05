Linear classifier model.

Train a linear model to classify instances into one of multiple possible
classes. When number of possible classes is 2, this is binary classification.

Example:

```python
sparse_column_a = sparse_column_with_hash_bucket(...)
sparse_column_b = sparse_column_with_hash_bucket(...)

sparse_feature_a_x_sparse_feature_b = crossed_column(...)

# Estimator using the default optimizer.
estimator = LinearClassifier(
    feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b])

# Or estimator using the FTRL optimizer with regularization.
estimator = LinearClassifier(
    feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b],
    optimizer=tf.train.FtrlOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001
    ))

# Or estimator using the SDCAOptimizer.
estimator = LinearClassifier(
   feature_columns=[sparse_column_a, sparse_feature_a_x_sparse_feature_b],
   optimizer=tf.contrib.linear_optimizer.SDCAOptimizer(
     example_id_column='example_id',
     num_loss_partitions=...,
     symmetric_l2_regularization=2.0
   ))

# Input builders
def input_fn_train: # returns x, y (where y represents label's class index).
  ...
def input_fn_eval: # returns x, y (where y represents label's class index).
  ...
estimator.fit(input_fn=input_fn_train)
estimator.evaluate(input_fn=input_fn_eval)
estimator.predict(x=x) # returns predicted labels (i.e. label's class index).
```

Input of `fit` and `evaluate` should have following features,
  otherwise there will be a `KeyError`:

* if `weight_column_name` is not `None`, a feature with
  `key=weight_column_name` whose value is a `Tensor`.
* for each `column` in `feature_columns`:
  - if `column` is a `SparseColumn`, a feature with `key=column.name`
    whose `value` is a `SparseTensor`.
  - if `column` is a `WeightedSparseColumn`, two features: the first with
    `key` the id column name, the second with `key` the weight column name.
    Both features' `value` must be a `SparseTensor`.
  - if `column` is a `RealValuedColumn`, a feature with `key=column.name`
    whose `value` is a `Tensor`.
- - -

#### `tf.contrib.learn.LinearClassifier.__init__(feature_columns, model_dir=None, n_classes=2, weight_column_name=None, optimizer=None, gradient_clip_norm=None, enable_centered_bias=False, _joint_weight=False, config=None, feature_engineering_fn=None)` {#LinearClassifier.__init__}

Construct a `LinearClassifier` estimator object.

##### Args:


*  <b>`feature_columns`</b>: An iterable containing all the feature columns used by
    the model. All items in the set should be instances of classes derived
    from `FeatureColumn`.
*  <b>`model_dir`</b>: Directory to save model parameters, graph and etc. This can
    also be used to load checkpoints from the directory into a estimator
    to continue training a previously saved model.
*  <b>`n_classes`</b>: number of label classes. Default is binary classification.
    Note that class labels are integers representing the class index (i.e.
    values from 0 to n_classes-1). For arbitrary label values (e.g. string
    labels), convert to class indices first.
*  <b>`weight_column_name`</b>: A string defining feature column name representing
    weights. It is used to down weight or boost examples during training. It
    will be multiplied by the loss of the example.
*  <b>`optimizer`</b>: The optimizer used to train the model. If specified, it should
    be either an instance of `tf.Optimizer` or the SDCAOptimizer. If `None`,
    the Ftrl optimizer will be used.
*  <b>`gradient_clip_norm`</b>: A `float` > 0. If provided, gradients are clipped
    to their global norm with this clipping ratio. See
    `tf.clip_by_global_norm` for more details.
*  <b>`enable_centered_bias`</b>: A bool. If True, estimator will learn a centered
    bias variable for each class. Rest of the model structure learns the
    residual after centered bias.
  _joint_weight: If True, the weights for all columns will be stored in a
    single (possibly partitioned) variable. It's more efficient, but it's
    incompatible with SDCAOptimizer, and requires all feature columns are
    sparse and use the 'sum' combiner.

*  <b>`config`</b>: `RunConfig` object to configure the runtime settings.
*  <b>`feature_engineering_fn`</b>: Feature engineering function. Takes features and
                    labels which are the output of `input_fn` and
                    returns features and labels which will be fed
                    into the model.

##### Returns:

  A `LinearClassifier` estimator.

##### Raises:


*  <b>`ValueError`</b>: if n_classes < 2.


- - -

#### `tf.contrib.learn.LinearClassifier.bias_` {#LinearClassifier.bias_}

DEPRECATED FUNCTION

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-30.
Instructions for updating:
This method will be removed after the deprecation date. To inspect variables, use get_variable_names() and get_variable_value().


- - -

#### `tf.contrib.learn.LinearClassifier.config` {#LinearClassifier.config}




- - -

#### `tf.contrib.learn.LinearClassifier.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=None, steps=None, metrics=None, name=None, checkpoint_path=None, hooks=None)` {#LinearClassifier.evaluate}

See evaluable.Evaluable. Note: Labels must be integer class indices.


- - -

#### `tf.contrib.learn.LinearClassifier.export(export_dir, input_fn=None, input_feature_key=None, use_deprecated_input_fn=True, signature_fn=None, default_batch_size=1, exports_to_keep=None)` {#LinearClassifier.export}

See BaseEstimator.export.


- - -

#### `tf.contrib.learn.LinearClassifier.export_savedmodel(*args, **kwargs)` {#LinearClassifier.export_savedmodel}

EXPERIMENTAL FUNCTION

THIS FUNCTION IS EXPERIMENTAL. It may change or be removed at any time, and without warning.


- - -

#### `tf.contrib.learn.LinearClassifier.fit(x=None, y=None, input_fn=None, steps=None, batch_size=None, monitors=None, max_steps=None)` {#LinearClassifier.fit}

See trainable.Trainable. Note: Labels must be integer class indices.


- - -

#### `tf.contrib.learn.LinearClassifier.get_estimator()` {#LinearClassifier.get_estimator}




- - -

#### `tf.contrib.learn.LinearClassifier.get_variable_names()` {#LinearClassifier.get_variable_names}




- - -

#### `tf.contrib.learn.LinearClassifier.get_variable_value(name)` {#LinearClassifier.get_variable_value}




- - -

#### `tf.contrib.learn.LinearClassifier.model_dir` {#LinearClassifier.model_dir}




- - -

#### `tf.contrib.learn.LinearClassifier.predict(*args, **kwargs)` {#LinearClassifier.predict}

Runs inference to determine the predicted class (i.e. class index). (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below.


- - -

#### `tf.contrib.learn.LinearClassifier.predict_proba(*args, **kwargs)` {#LinearClassifier.predict_proba}

Runs inference to determine the class probability predictions. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below.


- - -

#### `tf.contrib.learn.LinearClassifier.weights_` {#LinearClassifier.weights_}

DEPRECATED FUNCTION

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-30.
Instructions for updating:
This method will be removed after the deprecation date. To inspect variables, use get_variable_names() and get_variable_value().


