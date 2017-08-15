# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ===================================================================

"""TPUEstimator class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import threading
from six.moves import queue as Queue  # pylint: disable=redefined-builtin

from tensorflow.contrib.tpu.python.tpu import tpu
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_feed
from tensorflow.contrib.tpu.python.tpu import tpu_function
from tensorflow.contrib.tpu.python.tpu import training_loop

from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator import util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training


_INITIAL_LOSS = 1e7
_BATCH_SIZE_KEY = 'batch_size'
_RESERVED_PARAMS_KEYS = [_BATCH_SIZE_KEY]


def _tpu_job(run_config):
  # The tpu job is determined by the run_config. Right now, this method is
  # required as tpu_config is not part of the RunConfig.
  return None if run_config.master in ['', 'local'] else 'tpu_worker'


def _per_shard_batch_size(global_batch_size, run_config, use_tpu):
  """Returns the batch size for each shard."""
  if use_tpu:
    return global_batch_size // run_config.tpu_config.num_shards
  else:
    return global_batch_size


class _SIGNAL(object):
  """Signal used to control the input thread of infeed."""
  NEXT_BATCH = 1
  STOP = 2


class InfeedThreadController(object):
  """This wraps the infeed thread and stops when Estimator train finishes.

  For model_fn wrapper, it is not possible to know when the `train` API will
  stop. It could be the cases that the `max_steps` is reached or some hook
  requests the stop in the monitored_session.

  This controller (with coordination with `TPUInfeedSessionHook`) does the
  following:

  1) It pre-infeeds one `batch` data for current TPU iterations.

  2) When `before_run` of `TPUInfeedSessionHook` is called, one more `batch`
  data will be infed.

  3) When `end` of `TPUInfeedSessionHook` is called, the thread will end
  gracefully.

  So, we might need to adjust the algorithrm here if the IO is slower than the
  computation.
  """

  def __init__(self, session, enqueue_ops, iterations):
    self._signal_queue = Queue.Queue()
    self._input_thd = threading.Thread(target=self._input_thread_fn_for_loading,
                                       args=(session, enqueue_ops, iterations))
    self._input_thd.daemon = True
    self._input_thd.start()

  def _input_thread_fn_for_loading(self, session, enqueue_ops, iterations):
    count = 0
    while True:
      signal = self._signal_queue.get()
      if signal == _SIGNAL.STOP:
        logging.info('Stop Infeed input thread.')
        return

      for i in range(iterations):
        logging.debug('InfeedEnqueue data for iteration (%d, %d)', count, i)
        session.run(enqueue_ops)
      count += 1

  def load_next_batch(self):
    self._signal_queue.put(_SIGNAL.NEXT_BATCH)

  def join(self):
    logging.info('Waiting for InputThread to exit.')
    self._signal_queue.put(_SIGNAL.STOP)
    self._input_thd.join()


class TPUInfeedSessionHook(session_run_hook.SessionRunHook):
  """A Session hook setting up the TPU initialization and infeed.

  This hook does two major things:
  1. initialize and shutdown TPU system (maybe a separated hook)
  2. launch and join the input thread for infeed.
  """

  def __init__(self, run_config, enqueue_fn):
    self._iterations = run_config.tpu_config.iterations_per_loop
    self._enqueue_fn = enqueue_fn
    self._tpu_job = _tpu_job(run_config)

  def begin(self):
    self._enqueue_ops = self._enqueue_fn()
    logging.info('TPU job name %s', self._tpu_job)
    self._init_op = [tpu.initialize_system(job=self._tpu_job)]
    self._finalize_op = [tpu.shutdown_system(job=self._tpu_job)]

  def after_create_session(self, session, coord):
    logging.info('Init TPU system')
    session.run(self._init_op)

    logging.info('Start infeed input thread controller')
    self._infeed_thd_controller = InfeedThreadController(
        session, self._enqueue_ops, self._iterations)

  def before_run(self, run_context):
    logging.info('Load next batch of data to infeed.')
    self._infeed_thd_controller.load_next_batch()

  def end(self, session):
    logging.info('Stop infeed input thread controller')
    self._infeed_thd_controller.join()

    logging.info('Shutdown TPU system.')
    session.run(self._finalize_op)


class _PerShardOutput(object):
  """Wraps input_fn's outputs into per-shard outputs.

  Used so that the model_fn can distinguish between sharded input and unsharded
  inputs (e.g., for export_savedmodel()).
  """

  def __init__(self, output):
    self.output = output

  def as_list(self):
    return self.output


class _InputsHolder(object):
  """A inputs holder holds the `features` and `labels' for TPU system.

  Model inputs returned by the `input_fn` can have one of the following forms:
  1. features
  2. (features, labels)

  Internally, form 1 is reformed to `(features, None)` as features and labels
  are passed separatedly to underlying methods. For TPU training, TPUEstimator
  expects multiple `features` and `labels` tuples one for each shard.

  In addition, TPUEstimator allows various different structures for inputs
  (namely `features` and `labels`).  `features` can be `Tensor` or dict of
  string name to `Tensor`, and `labels` could be `None`, `Tensor`, or dict of
  string name to `Tensor`. TPU infeed/outfeed library expects flattened tensor
  list. So, `features` and `labels` need to be flattened, before infeed enqueue,
  and the structure of them needs to be recorded, in order to restore them after
  infeed dequeue.

  `_InputsHolder` could hold the `features` and `labels` tuple for all shards
  (usually multi-host TPU training) or for one host (usually for single-host TPU
  evaluation), records the structure details (including presence, dict or single
  tensor, dict names), validates the structure consistency cross all shards, and
  encapsulates the flatten/unflatten logic.
  """

  def __init__(self, features=None, labels=None, num_shards=None):
    """Constructor.

    Args:
      features: features for one host or a list of features one for each shard
        (must be type `_PerShardOutput`). Once provided, the corresponding
        `labels` should be set also and this `_InputsHolder` is frozen to
        prevent from future modification. If `None`, it is expected to add
        features and labels for each shard by calling `append_tuple` later.
      labels: labels for one host or a list of labels one for each shard
        (must be type `_PerShardOutput`).
      num_shards: Number of shards in the TPU system. Must be provided unless it
        can be deduced from `features`.

    Raises:
      ValueError: If both `sharded_features` and `num_shards` are `None`.
    """
    # Holds the features and labels for all shards.
    self._feature_list = []
    self._label_list = []

    # Holds the structure of inputs
    self._feature_names = []
    self._label_names = []
    self._has_labels = False

    # Internal state.
    self._initialized = False
    self._frozen = False
    self._sharded = False

    if features is None:
      if num_shards is None:
        raise ValueError(
            '`features` and `num_shards` cannot be both None')
      self._num_shards = num_shards
    elif isinstance(features, _PerShardOutput):
      self._from_sharded_inputs(features, labels, num_shards)
    else:
      if num_shards is None:
        raise ValueError(
            '`num_shards` cannot be None for unsharded features.')
      self._from_unsharded_inputs(features, labels, num_shards)

  def _from_unsharded_inputs(self, features, labels, num_shards):
    """Initializes the inputs with unsharded features and labels."""
    self._num_shards = num_shards
    if labels is not None:
      self._has_labels = True
      self.append_tuple((features, labels))
    else:
      self.append_tuple(features)

    self._sharded = False
    self._frozen = True

  def _from_sharded_inputs(self, sharded_features, sharded_labels, num_shards):
    """Initializes the inputs with sharded features and labels."""
    if not isinstance(sharded_features, _PerShardOutput):
      raise ValueError('`sharded_features` must have type `_PerShardOutput`.')
    features = sharded_features.as_list()

    if num_shards is not None and num_shards != len(features):
      raise ValueError(
          '`num_shards` should be same as the length of sharded_features.')

    self._num_shards = len(features)
    if not self._num_shards:
      raise ValueError('`sharded_features` should not be empty.')

    if sharded_labels is not None:
      if not isinstance(sharded_labels, _PerShardOutput):
        raise ValueError('sharded_labels` must have type `_PerShardOutput`.')

      self._has_labels = True
      labels = sharded_labels.as_list()
      if self._num_shards != len(labels):
        raise ValueError(
            'Length of `sharded_features` and `sharded_labels` mismatch.')

    if self._has_labels:
      for (f, l) in zip(features, labels):
        self.append_tuple((f, l))
    else:
      for f in features:
        self.append_tuple(f)

    self._sharded = True
    self._frozen = True

  def _extract_key_names(self, tensor_or_dict):
    if tensor_or_dict is None:
      return []

    return tensor_or_dict.keys() if isinstance(tensor_or_dict, dict) else []

  def _validate(self, features, labels):
    has_labels = labels is not None
    feature_names = self._extract_key_names(features)
    label_names = self._extract_key_names(labels)

    if self._initialized:
      self._sharded = True
      # The following should never happen.
      assert feature_names == self._feature_names, 'feature keys mismatched'
      assert label_names == self._label_names, 'label keys mismatched'
      assert has_labels == self._has_labels, 'label presence mismatched'
    else:
      self._initialized = True
      self._feature_names = feature_names
      self._label_names = label_names
      self._has_labels = has_labels

  @property
  def sharded(self):
    if not self._frozen:
      raise RuntimeError('_InputsHolder has not been frozen yet.')
    return self._sharded

  @property
  def num_shards(self):
    if not self._frozen:
      raise RuntimeError('_InputsHolder has not been frozen yet.')
    return self._num_shards

  def append_tuple(self, inputs):
    """Appends `inputs` for one shard into holder.

    Args:
      inputs: The return from `input_fn`, which could be features or tuple of
        (features, labels). After the first `inputs` appended into
        `_InputsHolder`, the structure of `features` and `labels is recorded.
        Any future invocation should provide the `inputs` with same structure.

    Raises:
      RuntimeError: If the internal data has been frozen already.
    """
    if self._frozen:
      raise RuntimeError('InputsHolder has frozen, which cannot be mutated.')

    # input_fn may return either features or (features, labels)
    if isinstance(inputs, tuple):
      features, labels = inputs
    else:
      features, labels = inputs, None

    self._validate(features, labels)

    self._feature_list.append(features)
    if labels is not None:
      self._label_list.append(labels)

  def as_features_and_labels_tuple(self):
    """Returns features and labels as grouped tuple.

    This is intended to be used to pass features and labels for all shards from
    input_fn to model_fn as the parent class `Estimator` does not have the
    concept of shards. So, grouped tuple is required.

    Once called, the internal data is frozen and `append_tuple` cannot be
    invoked anymore.

    Returns:
      A tuple of features and labels. Both have type `_PerShardOutput`, holding
      the inputs for all shards. `labels` could be `None`.

    Raises:
      RuntimeError: If the internal data has not been initialized.
    """
    self._frozen = True
    if not self._initialized:
      raise RuntimeError('InputsHolder has not been initialized.')

    assert len(self._feature_list) == self._num_shards
    if not self._label_list or all(l is None for l in self._label_list):
      return _PerShardOutput(self._feature_list), None

    assert len(self._label_list) == self._num_shards
    return (_PerShardOutput(self._feature_list),
            _PerShardOutput(self._label_list))

  def as_sharded_flattened_inputs(self):
    """Flatten the features and label as tensor lists for all shards.

    Flattened tensor list contains all tensors in `features` (dict) and `labels`
    (dict). Conceptually, it has the predicated structure like:

    ```python
    flatten_list = []
    for name in features:
      flatten_list.append(features[name])
    for name in labels:
      flatten_list.append(labels[name])
    ```

    This method handles the label is None case and single tensor case nicely.

    Once called, the internal data is frozen and `append_tuple` cannot be
    invokded anymore.

    Returns:
      A list of flattened inputs one for each shard.

    Raises:
      RuntimeError: If the internal data has not been initialized.
      ValueError: If the inputs are sharded.
    """
    self._frozen = True
    if not self._initialized:
      raise RuntimeError('InputsHolder has not been initialized.')
    if not self._sharded:
      raise ValueError('Inputs are not sharded.')

    sharded_inputs = []

    for shard in range(self._num_shards):
      flattened_inputs = self._as_flattened_inputs(
          self._feature_list[shard],
          self._label_list[shard] if self._has_labels else None)
      sharded_inputs.append(flattened_inputs)

    return sharded_inputs

  def as_flattened_inputs(self):
    """Flatten the features and label as a single tensor list for one host."""
    self._frozen = True
    if not self._initialized:
      raise RuntimeError('InputsHolder has not been initialized.')
    if self._sharded:
      raise ValueError('Inputs are sharded.')

    return self._as_flattened_inputs(
        self._feature_list[0],
        self._label_list[0] if self._has_labels else None)

  def _as_flattened_inputs(self, features, labels):
    """Flattens the `features` and `labels` to a single tensor list."""
    flattened_inputs = []
    if self._feature_names:
      # We need a fixed ordering for enqueueing and dequeueing.
      flattened_inputs.extend([features[name] for name in self._feature_names])
    else:
      flattened_inputs.append(features)

    if labels is not None:
      if self._label_names:
        # We need a fixed ordering for enqueueing and dequeueing.
        flattened_inputs.extend([labels[name] for name in self._label_names])
      else:
        flattened_inputs.append(labels)
    return flattened_inputs

  def unflatten_features_and_labels(self, flattened_inputs):
    """Restores the flattened inputs to original features and labels form.

    Once called, the internal data is frozen and `append_tuple` cannot be
    invokded anymore.

    Args:
      flattened_inputs: Flattened inputs for one each, which should be created
      by the `as_sharded_flattened_inputs` API.

    Returns:
      A tuple of (`features`, `labels`), where `labels` could be None.
      Each one, if present, should have identical structure (single tensor vs
      dict) as the one returned by input_fn.

    Raises:
      RuntimeError: If the internal data has not been initialized.
      ValueError: If the number of expected tensors from `flattened_inputs`
        mismatches the recorded structure.
    """
    self._frozen = True
    if not self._initialized:
      raise RuntimeError('InputsHolder has not been initialized.')

    expected_num_features = (len(self._feature_names) if self._feature_names
                             else 1)
    if self._has_labels:
      expected_num_labels = (len(self._label_names) if self._label_names
                             else 1)
    else:
      expected_num_labels = 0

    expected_num_tensors = expected_num_features + expected_num_labels

    if expected_num_tensors != len(flattened_inputs):
      raise ValueError(
          'The number of flattened tensors mismatches expected num. '
          'Expected {}, got {}'.format(expected_num_tensors,
                                       len(flattened_inputs)))
    if self._feature_names:
      unflattened_features = dict(zip(self._feature_names,
                                      flattened_inputs[:expected_num_features]))
    else:
      # Single tensor case
      unflattened_features = flattened_inputs[0]

    if expected_num_labels == 0:
      unflattened_label = None
    elif self._label_names:
      unflattened_label = dict(zip(self._label_names,
                                   flattened_inputs[expected_num_features:]))
    else:
      # Single tensor case.
      unflattened_label = flattened_inputs[expected_num_features]

    return unflattened_features, unflattened_label


class _ModelFnWrapper(object):
  """A `model_fn` wrapper.

  This makes calling model_fn on CPU and TPU easier and more consistent and
  performs necessary check and mutation required by TPU training.

  In addition, this wrapper manages converting the `model_fn` to a single TPU
  train step.
  """

  def __init__(self, model_fn, config, params, mode, train_batch_size):
    self._model_fn = model_fn
    self._config = config
    self._params = params
    self._mode = mode
    self._train_batch_size = train_batch_size

  def call_without_tpu(self, features, labels):
    # Let CrossShardOptimizer be called without TPU in model_fn, since it's
    # common to set the train_op even when running evaluate() or predict().
    with tpu_function.tpu_shard_context(1):
      return self._call_model_fn(features, labels, False)

  def convert_to_single_tpu_train_step(self, dequeue_fn):
    """Converts the `model_fn` as a single train step on TPU."""

    def train_step(loss):
      """Training step function for use inside a while loop."""
      del loss  # unused; required in function signature.
      features, labels = dequeue_fn()

      # Makes deep copy with `config` and params` in case user mutates them.
      estimator_spec = self._verify_estimator_spec(
          self._call_model_fn(features, labels, True))
      loss, train_op = estimator_spec.loss, estimator_spec.train_op
      with ops.control_dependencies([train_op]):
        return array_ops.identity(loss)
    return train_step

  @property
  def config(self):
    return self._config

  def _call_model_fn(self, features, labels, use_tpu):
    """Calls the model_fn with required parameters."""
    model_fn_args = util.fn_args(self._model_fn)
    kwargs = {}

    config = copy.deepcopy(self._config)
    params = copy.deepcopy(self._params)

    if 'labels' in model_fn_args:
      kwargs['labels'] = labels
    else:
      if labels is not None:
        raise ValueError(
            'model_fn does not take labels, but input_fn returns labels.')
    if 'mode' in model_fn_args:
      kwargs['mode'] = self._mode
    if 'config' in model_fn_args:
      kwargs['config'] = config
    if 'params' in model_fn_args:
      kwargs['params'] = params

    if 'params' not in model_fn_args:
      raise ValueError(
          'model_fn ({}) does not include params argument, '
          'required by TPUEstimator to pass batch size as '
          'params[\'batch_size\']'.format(self._model_fn))
    if self._mode == model_fn_lib.ModeKeys.TRAIN:
      # For TPU training. `params` is never `None`.
      params[_BATCH_SIZE_KEY] = _per_shard_batch_size(
          self._train_batch_size, config, use_tpu)

    return self._model_fn(features=features, **kwargs)

  def _verify_estimator_spec(self, estimator_spec):
    """Validates the estimator_spec."""
    err_msg = '{} returned by EstimatorSpec is not supported in TPUEstimator.'
    if estimator_spec.training_chief_hooks:
      raise ValueError(err_msg.format('training_chief_hooks'))
    if estimator_spec.training_hooks:
      raise ValueError(err_msg.format('training_hooks'))
    return estimator_spec


class TPUEstimator(estimator_lib.Estimator):
  """Estimator with TPU support.

  TPUEstimator handles many of the details of running on TPU devices, such as
  replicating inputs and models for each core, and returning to host
  periodically to run hooks.

  Note: For training (evaluate and predict support on TPU are not yet
  implemented), TPUEstimator transforms a global batch size in params to a
  per-shard batch size when calling the `input_fn` and `model_fn`. Users should
  specify `train_batch_size` in constructor, and then get the batch size for
  each shard in `input_fn` and `model_fn` by `params['batch_size']`.
  """

  def __init__(self,
               model_fn=None,
               model_dir=None,
               config=None,
               params=None,
               use_tpu=True,
               train_batch_size=None):
    """Constructs an `TPUEstimator` instance.

    Args:
      model_fn: Model function as required by `Estimator`. For training, the
        returned `EstimatorSpec` cannot have hooks as it is not supported in
        `TPUEstimator`.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model. If `None`, the model_dir in
        `config` will be used if set. If both are set, they must be same. If
        both are `None`, a temporary directory will be used.
      config: An `tpu_config.RunConfig` configuration object. Cannot be `None`.
      params: An optional `dict` of hyper parameters that will be passed into
        `input_fn` and `model_fn`.  Keys are names of parameters, values are
        basic python types. There are reserved keys for `TPUEstimator`,
        including 'batch_size'.
      use_tpu: A bool indicating whether TPU support is enabled. Currently, only
        applied to training. Evaluate and predict still happen on CPU.
      train_batch_size: An int representing the global training batch size.
        TPUEstimator transforms this global batch size to a per-shard batch
        size, as params['batch_size'], when calling `input_fn` and `model_fn`.
        Cannot be `None` if `use_tpu` is `True`. Must be divisible by
        `config.tpu_config.num_shards`.

    Raises:
      ValueError: `params` has reserved keys already.
    """
    if config is None or not isinstance(config, tpu_config.RunConfig):
      raise ValueError(
          '`config` must be provided with type `tpu_config.RunConfig`')

    if params is not None and any(k in params for k in _RESERVED_PARAMS_KEYS):
      raise ValueError(
          '{} are reserved keys but existed in params {}.'.format(
              _RESERVED_PARAMS_KEYS, params))

    if use_tpu:
      if train_batch_size is None:
        raise ValueError('`train_batch_size` cannot be `None`')
      if not isinstance(train_batch_size, int):
        raise ValueError('`train_batch_size` must be an int')
      if train_batch_size < 1:
        raise ValueError('`train_batch_size` must be positive')

      # The specified batch size is the batch size for the entire computation.
      # The input_fn and model_fn are called per-shard, so we want to calculate
      # the per-shard batch size and pass that.
      if train_batch_size % config.tpu_config.num_shards != 0:
        raise ValueError(
            'batch size {} must be divisible by number of shards {}'
            .format(train_batch_size, config.tpu_config.num_shards))

    # Verifies the model_fn signature according to Estimator framework.
    estimator_lib._verify_model_fn_args(model_fn, params)  # pylint: disable=protected-access
    # We cannot store config and params in this constructor as parent
    # constructor might change them, such as assigning a temp dir for
    # config.model_dir.
    model_function = _augment_model_fn(model_fn, train_batch_size, use_tpu)

    super(TPUEstimator, self).__init__(
        model_fn=model_function,
        model_dir=model_dir,
        config=config,
        params=params)
    self._use_tpu = use_tpu
    self._train_batch_size = train_batch_size

  def _create_global_step(self, graph):
    """Creates a global step suitable for TPUs.

    Args:
      graph: The graph in which to create the global step.

    Returns:
      A global step `Tensor`.

    Raises:
      ValueError: if the global step tensor is already defined.
    """
    graph = graph or ops.get_default_graph()
    if training.get_global_step(graph) is not None:
      raise ValueError('"global_step" already exists.')
    # Create in proper graph and base name_scope.
    with graph.as_default() as g, g.name_scope(None):
      return variable_scope.get_variable(
          ops.GraphKeys.GLOBAL_STEP,
          shape=[],
          dtype=dtypes.int32,
          initializer=init_ops.zeros_initializer(),
          trainable=False,
          use_resource=True,
          collections=[ops.GraphKeys.GLOBAL_VARIABLES,
                       ops.GraphKeys.GLOBAL_STEP])

  def _call_input_fn(self, input_fn, mode):
    """Calls the input function.

    Args:
      input_fn: The input function.
      mode: ModeKeys

    Returns:
      Either features or (features, labels) where features and labels are:
        features - `Tensor` or dictionary of string feature name to `Tensor`.
        labels - `Tensor` or dictionary of `Tensor` with labels.

    Raises:
      ValueError: if input_fn takes invalid arguments or does not have `params`.
    """
    input_fn_args = util.fn_args(input_fn)
    config = self.config  # a deep copy.
    kwargs = {}
    if 'params' in input_fn_args:
      kwargs['params'] = self.params  # a deep copy.
    else:
      raise ValueError('input_fn ({}) does not include params argument, '
                       'required by TPUEstimator to pass batch size as '
                       'params["batch_size"]'.format(input_fn))
    if 'config' in input_fn_args:
      kwargs['config'] = config

    # Now for TPU training.
    if mode == model_fn_lib.ModeKeys.TRAIN:
      kwargs['params'][_BATCH_SIZE_KEY] = (
          _per_shard_batch_size(self._train_batch_size, config, self._use_tpu)
          if not config.tpu_config.per_host_input_for_training else
          self._train_batch_size)

    if not self._use_tpu or mode != model_fn_lib.ModeKeys.TRAIN:
      with ops.device('/cpu:0'):
        return input_fn(**kwargs)

    job = _tpu_job(config)
    def placement_function(index):
      if job is None:
        return '/replica:0/task:0/device:CPU:0'
      else:
        return '/job:%s/replica:0/task:%d/device:CPU:0' % (job, index / 8)

    if not config.tpu_config.per_host_input_for_training:
      num_shards = config.tpu_config.num_shards
      inputs = _InputsHolder(num_shards=num_shards)
      for i in range(config.tpu_config.num_shards):
        with ops.device(placement_function(i)):
          inputs.append_tuple(input_fn(**kwargs))

      return inputs.as_features_and_labels_tuple()
    else:
      # TODO(xiejw): Extend this to multi-host support.
      with ops.device(placement_function(0)):
        return input_fn(**kwargs)


def _create_infeed_enqueue_ops_and_dequeue_fn(inputs_holder, run_config):
  """Utility to convert input_fn to enqueue and dequeue fns for TPU.

  Args:
    inputs_holder: An `_InputsHolder` holding features and labels.
    run_config: A `RunConfig` instance.

  Returns:
    A tuple of (dequeue_fn, enqueue_fn)
  """
  if inputs_holder.sharded:
    sharded_inputs = inputs_holder.as_sharded_flattened_inputs()

    infeed_queue = tpu_feed.InfeedQueue(
        number_of_tuple_elements=len(sharded_inputs[0]))
    infeed_queue.set_configuration_from_sharded_input_tensors(sharded_inputs)
  else:
    unsharded_inputs = inputs_holder.as_flattened_inputs()
    infeed_queue = tpu_feed.InfeedQueue(
        tuple_types=[t.dtype for t in unsharded_inputs],
        tuple_shapes=[t.shape for t in unsharded_inputs])
    infeed_queue.set_number_of_shards(inputs_holder.num_shards)

  def dequeue_fn():
    """dequeue_fn is used by the train_step in TPU to retrieve the tensors."""
    values = infeed_queue.generate_dequeue_op()
    return inputs_holder.unflatten_features_and_labels(values)

  def tpu_ordinal_function(index):
    """Return the TPU ordinal associated with a shard.

    Required because the enqueue ops are placed on CPU.

    Args:
      index: the shard index

    Returns:
      The ordinal of the TPU device the shard's infeed should be placed on.
    """
    return index % 8

  def enqueue_fn():
    """enqueue_fn is used to add ops to the graph to send tensors."""
    if inputs_holder.sharded:
      return infeed_queue.generate_enqueue_ops(
          sharded_inputs, tpu_ordinal_function=tpu_ordinal_function)
    else:
      job = _tpu_job(run_config)
      def placement_function(index):
        if job is None:
          return '/replica:0/task:0/device:CPU:0'
        else:
          return '/job:%s/replica:0/task:%d/device:CPU:0' % (job, index / 8)
      return infeed_queue.split_inputs_and_generate_enqueue_ops(
          unsharded_inputs, placement_function=placement_function)

  return (dequeue_fn, enqueue_fn)


def _augment_model_fn(model_fn, train_batch_size, use_tpu):
  """Returns a new model_fn, which wraps the TPU support."""

  def _model_fn(features, labels, mode, config, params):
    """A Estimator `model_fn` for TPUEstimator."""
    model_fn_wrapper = _ModelFnWrapper(model_fn, config, params, mode,
                                       train_batch_size)

    # TODO(jhseu): Move to EVAL and PREDICT to TPU.
    if not use_tpu or mode != model_fn_lib.ModeKeys.TRAIN:
      return model_fn_wrapper.call_without_tpu(features, labels)

    inputs = _InputsHolder(features=features, labels=labels,
                           num_shards=config.tpu_config.num_shards)

    dequeue_fn, enqueue_fn = _create_infeed_enqueue_ops_and_dequeue_fn(
        inputs, config)

    loss = _train_on_tpu_system(model_fn_wrapper, dequeue_fn)

    # Gets the variables back from TPU nodes. This means the variables updated
    # by TPU will now be *synced* to host memory.
    update_ops = [
        array_ops.check_numerics(v.read_value(),
                                 'Gradient for %s is NaN' % v.name).op
        for v in variables.trainable_variables()
    ]

    hooks = [
        TPUInfeedSessionHook(config, enqueue_fn),
        training.LoggingTensorHook(
            {'loss': array_ops.identity(loss),
             'step': training.get_global_step()},
            every_n_secs=30)
    ]

    return model_fn_lib.EstimatorSpec(
        mode,
        loss=array_ops.identity(loss),
        training_hooks=hooks,
        train_op=control_flow_ops.group(*update_ops))
  return _model_fn


def _train_on_tpu_system(model_fn_wrapper, dequeue_fn):
  """Executes `model_fn_wrapper` multiple times on all TPU shards."""
  config = model_fn_wrapper.config.tpu_config
  iterations_per_loop = config.iterations_per_loop
  num_shards = config.num_shards

  single_tpu_train_step = model_fn_wrapper.convert_to_single_tpu_train_step(
      dequeue_fn)

  multi_tpu_train_steps_on_single_shard = (lambda: training_loop.repeat(  # pylint: disable=g-long-lambda
      iterations_per_loop, single_tpu_train_step, [_INITIAL_LOSS], name='loop'))

  (loss,) = tpu.shard(multi_tpu_train_steps_on_single_shard,
                      inputs=[],
                      num_shards=num_shards,
                      outputs_from_all_shards=False)
  return loss
