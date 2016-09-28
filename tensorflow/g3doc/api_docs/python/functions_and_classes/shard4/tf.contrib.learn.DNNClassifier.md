A classifier for TensorFlow DNN models.

Example:

```python
education = sparse_column_with_hash_bucket(column_name="education",
                                           hash_bucket_size=1000)
occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                            hash_bucket_size=1000)

education_emb = embedding_column(sparse_id_column=education, dimension=16,
                                 combiner="sum")
occupation_emb = embedding_column(sparse_id_column=occupation, dimension=16,
                                 combiner="sum")

estimator = DNNClassifier(
    feature_columns=[education_emb, occupation_emb],
    hidden_units=[1024, 512, 256])

# Or estimator using the ProximalAdagradOptimizer optimizer with
# regularization.
estimator = DNNClassifier(
    feature_columns=[education_emb, occupation_emb],
    hidden_units=[1024, 512, 256],
    optimizer=tf.train.ProximalAdagradOptimizer(
      learning_rate=0.1,
      l1_regularization_strength=0.001
    ))

# Input builders
def input_fn_train: # returns x, Y
  pass
estimator.fit(input_fn=input_fn_train)

def input_fn_eval: # returns x, Y
  pass
estimator.evaluate(input_fn=input_fn_eval)
estimator.predict(x=x)
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

#### `tf.contrib.learn.DNNClassifier.__init__(hidden_units, feature_columns, model_dir=None, n_classes=2, weight_column_name=None, optimizer=None, activation_fn=relu, dropout=None, gradient_clip_norm=None, enable_centered_bias=None, config=None)` {#DNNClassifier.__init__}

Initializes a DNNClassifier instance.

##### Args:


*  <b>`hidden_units`</b>: List of hidden units per layer. All layers are fully
    connected. Ex. `[64, 32]` means first layer has 64 nodes and second one
    has 32.
*  <b>`feature_columns`</b>: An iterable containing all the feature columns used by
    the model. All items in the set should be instances of classes derived
    from `FeatureColumn`.
*  <b>`model_dir`</b>: Directory to save model parameters, graph and etc. This can also
    be used to load checkpoints from the directory into a estimator to continue
    training a previously saved model.
*  <b>`n_classes`</b>: number of target classes. Default is binary classification.
    It must be greater than 1.
*  <b>`weight_column_name`</b>: A string defining feature column name representing
    weights. It is used to down weight or boost examples during training. It
    will be multiplied by the loss of the example.
*  <b>`optimizer`</b>: An instance of `tf.Optimizer` used to train the model. If
    `None`, will use an Adagrad optimizer.
*  <b>`activation_fn`</b>: Activation function applied to each layer. If `None`, will
    use `tf.nn.relu`.
*  <b>`dropout`</b>: When not `None`, the probability we will drop out a given
    coordinate.
*  <b>`gradient_clip_norm`</b>: A float > 0. If provided, gradients are
    clipped to their global norm with this clipping ratio. See
    tf.clip_by_global_norm for more details.
*  <b>`enable_centered_bias`</b>: A bool. If True, estimator will learn a centered
    bias variable for each class. Rest of the model structure learns the
    residual after centered bias.
*  <b>`config`</b>: `RunConfig` object to configure the runtime settings.

##### Returns:

  A `DNNClassifier` estimator.

##### Raises:


*  <b>`ValueError`</b>: If `n_classes` < 2.


- - -

#### `tf.contrib.learn.DNNClassifier.bias_` {#DNNClassifier.bias_}

DEPRECATED FUNCTION

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-13.
Instructions for updating:
This method inspects the private state of the object, and should not be used


- - -

#### `tf.contrib.learn.DNNClassifier.config` {#DNNClassifier.config}




- - -

#### `tf.contrib.learn.DNNClassifier.evaluate(x=None, y=None, input_fn=None, feed_fn=None, batch_size=None, steps=None, metrics=None, name=None)` {#DNNClassifier.evaluate}

See evaluable.Evaluable.


- - -

#### `tf.contrib.learn.DNNClassifier.export(export_dir, input_fn=None, input_feature_key=None, use_deprecated_input_fn=True, signature_fn=None, default_batch_size=1, exports_to_keep=None)` {#DNNClassifier.export}

See BaseEstimator.export.


- - -

#### `tf.contrib.learn.DNNClassifier.fit(x=None, y=None, input_fn=None, steps=None, batch_size=None, monitors=None, max_steps=None)` {#DNNClassifier.fit}

See trainable.Trainable.


- - -

#### `tf.contrib.learn.DNNClassifier.get_variable_names()` {#DNNClassifier.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.DNNClassifier.get_variable_value(name)` {#DNNClassifier.get_variable_value}

Returns value of the variable given by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  `Tensor` object.


- - -

#### `tf.contrib.learn.DNNClassifier.model_dir` {#DNNClassifier.model_dir}




- - -

#### `tf.contrib.learn.DNNClassifier.predict(*args, **kwargs)` {#DNNClassifier.predict}

Returns predicted classes for given features. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below.

    Args:
      x: features.
      input_fn: Input function. If set, x must be None.
      batch_size: Override default batch size.
      as_iterable: If True, return an iterable which keeps yielding predictions
        for each example until inputs are exhausted. Note: The inputs must
        terminate if you want the iterable to terminate (e.g. be sure to pass
        num_epochs=1 if you are using something like read_batch_features).

    Returns:
      Numpy array of predicted classes (or an iterable of predicted classes if
      as_iterable is True).


- - -

#### `tf.contrib.learn.DNNClassifier.predict_proba(*args, **kwargs)` {#DNNClassifier.predict_proba}

Returns prediction probabilities for given features. (deprecated arguments)

SOME ARGUMENTS ARE DEPRECATED. They will be removed after 2016-09-15.
Instructions for updating:
The default behavior of predict() is changing. The default value for
as_iterable will change to True, and then the flag will be removed
altogether. The behavior of this flag is described below.

    Args:
      x: features.
      input_fn: Input function. If set, x and y must be None.
      batch_size: Override default batch size.
      as_iterable: If True, return an iterable which keeps yielding predictions
        for each example until inputs are exhausted. Note: The inputs must
        terminate if you want the iterable to terminate (e.g. be sure to pass
        num_epochs=1 if you are using something like read_batch_features).

    Returns:
      Numpy array of predicted probabilities (or an iterable of predicted
      probabilities if as_iterable is True).


- - -

#### `tf.contrib.learn.DNNClassifier.weights_` {#DNNClassifier.weights_}

DEPRECATED FUNCTION

THIS FUNCTION IS DEPRECATED. It will be removed after 2016-10-13.
Instructions for updating:
This method inspects the private state of the object, and should not be used


