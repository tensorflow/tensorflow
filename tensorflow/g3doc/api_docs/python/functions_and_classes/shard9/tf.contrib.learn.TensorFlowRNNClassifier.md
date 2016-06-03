TensorFlow RNN Classifier model.

Parameters:
  rnn_size: The size for rnn cell, e.g. size of your word embeddings.
  cell_type: The type of rnn cell, including rnn, gru, and lstm.
  num_layers: The number of layers of the rnn model.
  input_op_fn: Function that will transform the input tensor, such as
    creating word embeddings, byte list, etc. This takes
    an argument X for input and returns transformed X.
  bidirectional: boolean, Whether this is a bidirectional rnn.
  sequence_length: If sequence_length is provided, dynamic calculation is
    performed. This saves computational time when unrolling past max sequence
    length.
  initial_state: An initial state for the RNN. This must be a tensor of
    appropriate type and shape [batch_size x cell.state_size].
  n_classes: Number of classes in the target.
  batch_size: Mini batch size.
  steps: Number of steps to run over data.
  optimizer: Optimizer name (or class), for example "SGD", "Adam", "Adagrad".
  learning_rate: If this is constant float value, no decay function is
    used. Instead, a customized decay function can be passed that accepts
    global_step as parameter and returns a Tensor.
    e.g. exponential decay function:
    def exp_decay(global_step):
        return tf.train.exponential_decay(
            learning_rate=0.1, global_step,
            decay_steps=2, decay_rate=0.001)
  class_weight: None or list of n_classes floats. Weight associated with
    classes for loss computation. If not given, all classes are
    supposed to have weight one.
  continue_training: when continue_training is True, once initialized
    model will be continuely trained on every call of fit.
  config: RunConfig object that controls the configurations of the session,
    e.g. num_cores, gpu_memory_fraction, etc.
- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.__init__(rnn_size, n_classes, cell_type='gru', num_layers=1, input_op_fn=null_input_op_fn, initial_state=None, bidirectional=False, sequence_length=None, batch_size=32, steps=50, optimizer='Adagrad', learning_rate=0.1, class_weight=None, clip_gradients=5.0, continue_training=False, config=None, verbose=1)` {#TensorFlowRNNClassifier.__init__}




- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.bias_` {#TensorFlowRNNClassifier.bias_}

Returns bias of the rnn layer.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.evaluate(x=None, y=None, input_fn=None, steps=None)` {#TensorFlowRNNClassifier.evaluate}

See base class.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.fit(x, y, steps=None, monitors=None, logdir=None)` {#TensorFlowRNNClassifier.fit}

Neural network model from provided `model_fn` and training data.

Note: called first time constructs the graph and initializers
variables. Consecutives times it will continue training the same model.
This logic follows partial_fit() interface in scikit-learn.
To restart learning, create new estimator.

##### Args:


*  <b>`x`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
  iterator that returns arrays of features. The training input
  samples for fitting the model.

*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
  iterator that returns array of targets. The training target values
  (class labels in classification, real numbers in regression).

*  <b>`steps`</b>: int, number of steps to train.
         If None or 0, train for `self.steps`.
*  <b>`monitors`</b>: List of `BaseMonitor` objects to print training progress and
    invoke early stopping.
*  <b>`logdir`</b>: the directory to save the log file that can be used for
  optional visualization.

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.get_params(deep=True)` {#TensorFlowRNNClassifier.get_params}

Get parameters for this estimator.

##### Args:


*  <b>`deep`</b>: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

##### Returns:

  params : mapping of string to any
  Parameter names mapped to their values.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.get_tensor(name)` {#TensorFlowRNNClassifier.get_tensor}

Returns tensor by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Tensor.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.get_variable_names()` {#TensorFlowRNNClassifier.get_variable_names}

Returns list of all variable names in this model.

##### Returns:

  List of names.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.get_variable_value(name)` {#TensorFlowRNNClassifier.get_variable_value}

Returns value of the variable given by name.

##### Args:


*  <b>`name`</b>: string, name of the tensor.

##### Returns:

  Numpy array - value of the tensor.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.model_dir` {#TensorFlowRNNClassifier.model_dir}




- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.partial_fit(x, y)` {#TensorFlowRNNClassifier.partial_fit}

Incremental fit on a batch of samples.

This method is expected to be called several times consecutively
on different or the same chunks of the dataset. This either can
implement iterative training or out-of-core/online training.
This is especially useful when the whole dataset is too big to
fit in memory at the same time. Or when model is taking long time
to converge, and you want to split up training into subparts.

##### Args:


*  <b>`x`</b>: matrix or tensor of shape [n_samples, n_features...]. Can be
  iterator that returns arrays of features. The training input
  samples for fitting the model.

*  <b>`y`</b>: vector or matrix [n_samples] or [n_samples, n_outputs]. Can be
  iterator that returns array of targets. The training target values
  (class label in classification, real numbers in regression).

##### Returns:

  Returns self.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.predict(x, axis=1, batch_size=None)` {#TensorFlowRNNClassifier.predict}

Predict class or regression for X.

For a classification model, the predicted class for each sample in X is
returned. For a regression model, the predicted value based on X is
returned.

##### Args:


*  <b>`x`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`axis`</b>: Which axis to argmax for classification.
    By default axis 1 (next after batch) is used.
    Use 2 for sequence predictions.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
    it into mini batches. By default the batch_size member
    variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples]. The predicted classes or predicted
  value.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.predict_proba(x, batch_size=None)` {#TensorFlowRNNClassifier.predict_proba}

Predict class probability of the input samples X.

##### Args:


*  <b>`x`</b>: array-like matrix, [n_samples, n_features...] or iterator.
*  <b>`batch_size`</b>: If test set is too big, use batch size to split
    it into mini batches. By default the batch_size member variable is used.

##### Returns:


*  <b>`y`</b>: array of shape [n_samples, n_classes]. The predicted
  probabilities for each class.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.restore(cls, path, config=None)` {#TensorFlowRNNClassifier.restore}

Restores model from give path.

##### Args:


*  <b>`path`</b>: Path to the checkpoints and other model information.
*  <b>`config`</b>: RunConfig object that controls the configurations of the session,
    e.g. num_cores, gpu_memory_fraction, etc. This is allowed to be
      reconfigured.

##### Returns:

  Estimator, object of the subclass of TensorFlowEstimator.

##### Raises:


*  <b>`ValueError`</b>: if `path` does not contain a model definition.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.save(path)` {#TensorFlowRNNClassifier.save}

Saves checkpoints and graph to given path.

##### Args:


*  <b>`path`</b>: Folder to save model to.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.set_params(**params)` {#TensorFlowRNNClassifier.set_params}

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The former have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

##### Args:


*  <b>`**params`</b>: Parameters.

##### Returns:

  self

##### Raises:


*  <b>`ValueError`</b>: If params contain invalid names.


- - -

#### `tf.contrib.learn.TensorFlowRNNClassifier.weights_` {#TensorFlowRNNClassifier.weights_}

Returns weights of the rnn layer.


