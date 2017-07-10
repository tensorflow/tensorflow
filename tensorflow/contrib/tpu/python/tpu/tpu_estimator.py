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

"""TpuEstimator class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import threading
from six.moves import queue as Queue  # pylint: disable=redefined-builtin

from tensorflow.contrib.tpu.python.tpu import tpu
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_feed
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


_BATCH_SIZE_KEY = 'batch_size'
_RESERVED_PARAMS_KEYS = [_BATCH_SIZE_KEY]


def _tpu_job(run_config):
  # The tpu job is determined by the run_config. Right now, this method is
  # required as tpu_config is not part of the RunConfig.
  return None if run_config.master in ['', 'local'] else 'tpu_worker'


def _per_shard_batch_size(global_batch_size, run_config):
  """Returns the batch size for each shard."""
  return global_batch_size // run_config.tpu_config.num_shards


class _SIGNAL(object):
  """Signal used to control the input thread of infeed."""
  NEXT_BATCH = 1
  STOP = 2


class InfeedThreadController(object):
  """This wraps the infeed thread and stops when Estimator train finishes.

  For model_fn wrapper, it is not possible to know when the `train` API will
  stop. It could be the cases that the `max_steps` is reached or some hook
  requests the stop in the monitored_session.

  This controller (with coordination with `TpuInfeedSessionHook`) does the
  following:

  1) It pre-infeeds one `batch` data for current TPU iterations.

  2) When `before_run` of `TpuInfeedSessionHook` is called, one more `batch`
  data will be infed.

  3) When `end` of `TpuInfeedSessionHook` is called, the thread will end
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


class TpuInfeedSessionHook(session_run_hook.SessionRunHook):
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

  Used so that the wrapped model_fn can distinguish between sharded input and
  unsharded inputs (e.g., for export_savedmodel()).
  """

  def __init__(self, output):
    self.output = output

  def as_list(self):
    return self.output


class TpuEstimator(estimator_lib.Estimator):
  """Estimator with TPU support.

  TpuEstimator handles many of the details of running on TPU devices, such as
  replicating inputs and models for each core, and returning to host
  periodically to run hooks.

  Note: For training (evaluate and predict support on TPU are not yet
  implemented), TpuEstimator transforms a global batch size in params to a
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
    """Constructs an `TpuEstimator` instance.

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
        TpuEstimator transforms this global batch size to a per-shard batch
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

    if use_tpu:
      # Verifies the model_fn signature according to Estimator framework.
      estimator_lib._verify_model_fn_args(model_fn, params)  # pylint: disable=protected-access
      # We cannot store config and params in this constructor as parent
      # constructor might change them, such as assigning a temp dir for
      # config.model_dir.
      model_function = wrapped_model_fn(model_fn, train_batch_size)
    else:
      model_function = model_fn

    super(TpuEstimator, self).__init__(
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
    if not self._use_tpu or mode != model_fn_lib.ModeKeys.TRAIN:
      return super(TpuEstimator, self)._call_input_fn(input_fn, mode)

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
    per_shard_batch_size = _per_shard_batch_size(self._train_batch_size, config)
    kwargs['params'][_BATCH_SIZE_KEY] = per_shard_batch_size

    job = _tpu_job(config)
    def placement_function(index):
      if job is None:
        return '/replica:0/task:0/device:CPU:0'
      else:
        return '/job:%s/replica:0/task:%d/device:CPU:0' % (job, index / 8)

    features = []
    labels = []
    for i in range(config.tpu_config.num_shards):
      with ops.device(placement_function(i)):
        result = input_fn(**kwargs)
        # input_fn may return either features or (features, labels)
        if isinstance(result, tuple):
          features.append(result[0])
          labels.append(result[1])
        else:
          features.append(result)

    if not labels or all(l is None for l in labels):
      return _PerShardOutput(features), None

    return _PerShardOutput(features), _PerShardOutput(labels)


def _verify_estimator_spec(estimator_spec):
  """Validates the estimator_spec."""
  err_msg = '{} returned by EstimatorSpec is not supported in TPUEstimator.'
  if estimator_spec.training_chief_hooks:
    raise ValueError(err_msg.format('training_chief_hooks'))
  if estimator_spec.training_hooks:
    raise ValueError(err_msg.format('training_hooks'))
  return estimator_spec


def _call_model_fn(model_fn, features, labels, mode, config, params,
                   require_params=False):
  """Calls the model_fn with required parameters."""
  model_fn_args = util.fn_args(model_fn)
  kwargs = {}
  if 'labels' in model_fn_args:
    kwargs['labels'] = labels
  else:
    if labels is not None:
      raise ValueError(
          'model_fn does not take labels, but input_fn returns labels.')
  if 'mode' in model_fn_args:
    kwargs['mode'] = mode
  if 'config' in model_fn_args:
    kwargs['config'] = config
  if 'params' in model_fn_args:
    kwargs['params'] = params
  elif require_params:
    raise ValueError(
        'model_fn ({}) does not include params argument, '
        'required by TPUEstimator to pass batch size as '
        'params[\'batch_size\']'.format(model_fn))
  return model_fn(features=features, **kwargs)


def _call_model_fn_with_tpu(model_fn, features, labels, mode, config, params):
  """Calls user provided `model_fn` and verifies the estimator_spec."""
  # Makes deep copy with `config` and params` in case user mutates them.
  config = copy.deepcopy(config)
  params = copy.deepcopy(params)
  return _verify_estimator_spec(_call_model_fn(
      model_fn, features, labels, mode, config, params, require_params=True))


def _call_model_fn_without_tpu(
    model_fn, features, labels, mode, config, params):
  # Deepcopy of config and params is not required in this branch.
  return _call_model_fn(model_fn, features, labels, mode, config, params)


# TODO(xiejw): Improve the structure of this input_fn to infeed converion.
# The code now looks not like Estimator style. We need to abstract many
# details.
def _create_infeed_enqueue_ops_and_dequeue_fn(run_config, features, labels):
  """Utility to convert input_fn to enqueue and dequeue fns for TPU.

  Mainly, three things need to be done here.
  1. Calls the input_fn many times (`num_shards`) to infeed the data into TPU
  2. Create a dequeue_fn used by the train_step inside TPU execution to
  dequeue the tensors.
  3. Sets up the input thread to infeed.

  Args:
    run_config: run_config
    features: features
    labels: labels

  Returns:
    A tuple of (dequeue_fn, enqueue_fn)
  """
  infeed_names = None
  sharded_inputs = []
  if isinstance(features[0], dict):
    # We need a fixed ordering for enqueueing and dequeueing.
    infeed_names = [name for name in features[0]]

  for shard in range(run_config.tpu_config.num_shards):
    inputs = []
    if infeed_names is None:
      inputs.append(features[shard])
    else:
      for name in infeed_names:
        inputs.append(features[shard][name])
    if labels is not None:
      inputs.append(labels[shard])
    sharded_inputs.append(inputs)

  infeed_queue = tpu_feed.InfeedQueue(
      number_of_tuple_elements=len(sharded_inputs[0]))
  infeed_queue.set_configuration_from_sharded_input_tensors(sharded_inputs)

  def dequeue_fn():
    """dequeue_fn is used by the train_step in TPU to retrieve the tensors."""
    values = infeed_queue.generate_dequeue_op()

    expected_num_tensors = 0
    if labels is not None:
      expected_num_tensors += 1
    if infeed_names is None:
      expected_num_tensors += 1
    else:
      expected_num_tensors += len(infeed_names)
    assert len(values) == expected_num_tensors

    dequeue_label = None
    if labels is not None:
      dequeue_label = values[-1]
    if infeed_names is None:
      return values[0], dequeue_label
    # Restore the feature dictionary and label.
    dequeued_features = {}
    for i in range(len(infeed_names)):
      dequeued_features[infeed_names[i]] = values[i]
    return dequeued_features, dequeue_label

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
    return infeed_queue.generate_enqueue_ops(
        sharded_inputs, tpu_ordinal_function=tpu_ordinal_function)

  return (dequeue_fn, enqueue_fn)


def wrapped_model_fn(model_fn, train_batch_size):
  """Returns a new model_fn, which wraps the TPU support."""

  def _model_fn(features, labels, mode, config, params):
    """model_fn."""

    # TODO(jhseu): Move to EVAL and PREDICT to TPU.
    if mode != model_fn_lib.ModeKeys.TRAIN:
      return _call_model_fn_without_tpu(
          model_fn, features, labels, mode, config, params)

    # Now for TPU training. `params` is never `None`.
    params[_BATCH_SIZE_KEY] = _per_shard_batch_size(train_batch_size, config)

    assert isinstance(features, _PerShardOutput)
    features = features.as_list()
    if labels is not None:
      assert isinstance(labels, _PerShardOutput)
      labels = labels.as_list()

    dequeue_fn, enqueue_fn = (
        _create_infeed_enqueue_ops_and_dequeue_fn(config, features, labels))

    loss = _train_on_tpu_shards(
        config,
        train_step=_convert_model_fn_to_train_step(
            model_fn, dequeue_fn, mode, config, params))

    # Gets the variables back from TPU nodes. This means the variables updated
    # by TPU will now be *synced* to host memory.
    update_ops = [
        array_ops.check_numerics(v.read_value(),
                                 'Gradient for %s is NaN' % v.name).op
        for v in variables.trainable_variables()
    ]

    hooks = [
        TpuInfeedSessionHook(config, enqueue_fn),
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


def _convert_model_fn_to_train_step(model_fn, dequeue_fn, mode, run_config,
                                    params):
  """Generates a train step based on the model_fn."""

  def train_step(loss):
    """Training step function for use inside a while loop."""
    del loss  # unused; required in function signature.
    features, labels = dequeue_fn()

    # TODO(xiejw): how to do we support hook and savers in the original
    # model_fn. Realistically, the original
    # model_fn will be executed on TPU chips in a replica way. The hooks
    # returned by the model_fn cannot be supported at all. If we have to,
    # the graph construction part in the model_fn should be separated from the
    # control part (such as hooks and savers). By that the graph construction
    # could de defered on TPU chip, while the control logic can stay in host.
    estimator_spec = _call_model_fn_with_tpu(
        model_fn, features, labels, mode, run_config, params)
    loss, train_op = estimator_spec.loss, estimator_spec.train_op
    with ops.control_dependencies([train_op]):
      return array_ops.identity(loss)
  return train_step


def _train_on_tpu_shards(run_config, train_step):
  """Executes the `train_step` on all shards."""
  def train_shard():
    return training_loop.repeat(run_config.tpu_config.iterations_per_loop,
                                train_step,
                                [1e7],  # initial_loss
                                name='loop')

  (loss,) = tpu.shard(train_shard,
                      inputs=[],
                      num_shards=run_config.tpu_config.num_shards,
                      outputs_from_all_shards=False)
  return loss
