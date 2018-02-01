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

import collections
from contextlib import contextmanager
import copy
import signal
import threading
import time
import traceback

import six
from six.moves import queue as Queue  # pylint: disable=redefined-builtin
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.summary import summary_ops as contrib_summary
from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import tpu
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_feed
from tensorflow.contrib.tpu.python.tpu import training_loop
from tensorflow.contrib.tpu.python.tpu import util as util_lib

from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.protobuf import config_pb2

from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator import util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import evaluation
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training
from tensorflow.python.training import training_util

_INITIAL_LOSS = 1e7
_ZERO_LOSS = 0.
_TPU_ESTIMATOR = 'tpu_estimator'
_ITERATIONS_PER_LOOP_VAR = 'iterations_per_loop'
_BATCH_SIZE_KEY = 'batch_size'
_CROSS_REPLICA_SUM_OP = 'CrossReplicaSum'
_RESERVED_PARAMS_KEYS = [_BATCH_SIZE_KEY]

# TODO(b/65703635): Flip the value and remove all dead code.
_WRAP_INPUT_FN_INTO_WHILE_LOOP = False


def _create_global_step(graph):
  graph = graph or ops.get_default_graph()
  if training.get_global_step(graph) is not None:
    raise ValueError('"global_step" already exists.')
  # Create in proper graph and base name_scope.
  with graph.as_default() as g, g.name_scope(None):
    return variable_scope.get_variable(
        ops.GraphKeys.GLOBAL_STEP,
        shape=[],
        dtype=dtypes.int64,
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        use_resource=True,
        collections=[ops.GraphKeys.GLOBAL_VARIABLES, ops.GraphKeys.GLOBAL_STEP])


def _create_or_get_iterations_per_loop():
  graph = ops.get_default_graph()
  collection_name = '{}_{}'.format(_TPU_ESTIMATOR, _ITERATIONS_PER_LOOP_VAR)
  iter_vars = graph.get_collection(collection_name)
  if len(iter_vars) == 1:
    return iter_vars[0]
  elif len(iter_vars) > 1:
    raise RuntimeError('Multiple iterations_per_loop_var in collection.')

  with ops.colocate_with(training_util.get_global_step()):
    with variable_scope.variable_scope(
        _TPU_ESTIMATOR, reuse=variable_scope.AUTO_REUSE):
      return variable_scope.get_variable(
          _ITERATIONS_PER_LOOP_VAR,
          initializer=init_ops.zeros_initializer(),
          shape=[],
          dtype=dtypes.int32,
          trainable=False,
          collections=[collection_name, ops.GraphKeys.LOCAL_VARIABLES],
          use_resource=True)


def _sync_variables_ops():
  # Gets the variables back from TPU nodes. This means the variables updated
  # by TPU will now be *synced* to host memory.
  return [
      array_ops.check_numerics(v.read_value(),
                               'Gradient for %s is NaN' % v.name).op
      for v in variables.trainable_variables()
  ]


def _increase_eval_step_op(iterations_per_loop):
  """Returns an op to increase the eval step for TPU evaluation.

  Args:
    iterations_per_loop: Tensor. The number of eval steps runnining in TPU
        system before returning to CPU host for each `Session.run`.

  Returns:
    An operation
  """
  eval_step = evaluation._get_or_create_eval_step()  # pylint: disable=protected-access
  # Estimator evaluate increases 1 by default. So, we increase the difference.
  return state_ops.assign_add(
      eval_step,
      math_ops.cast(iterations_per_loop - 1, dtype=eval_step.dtype),
      use_locking=True)


_DEFAULT_JOB_NAME = 'tpu_worker'
_DEFAULT_COORDINATOR_JOB_NAME = 'coordinator'
_LOCAL_MASTERS = ('', 'local')


class _TPUContext(object):
  """A context holds immutable states of TPU computation.

  This immutable object holds TPUEstimator config, train/eval batch size, and
  `TPUEstimator.use_tpu`, which is expected to be passed around. It also
  provides utility functions, basded on the current state, to determine other
  information commonly required by TPU computation, such as TPU device names,
  TPU hosts, shard batch size, etc.

  N.B. As `mode` is not immutable state in Estimator, but essential to
  distinguish between TPU training and evaluation, a common usage for
  _TPUContext with `mode` is as follows:
  ```
  with _ctx.with_mode(mode) as ctx:
    if ctx.is_running_on_cpu():
       ...
  ```
  """

  def __init__(self, config, train_batch_size, eval_batch_size, use_tpu):
    self._config = config
    self._train_batch_size = train_batch_size
    self._eval_batch_size = eval_batch_size
    self._use_tpu = use_tpu
    self._num_shards_or_none = self._config.tpu_config.num_shards
    self._mode = None

  def _assert_mode(self):
    if self._mode is None:
      raise RuntimeError(
          '`mode` needs to be set via contextmanager `with_mode`.')
    return self._mode

  @property
  def num_of_cores_per_host(self):
    num_cores = self.num_cores
    return min(num_cores, 8)

  @contextmanager
  def with_mode(self, mode):
    new_ctx = copy.copy(self)  # Shallow copy is enough.
    new_ctx._mode = mode  # pylint: disable=protected-access
    yield new_ctx

  @property
  def mode(self):
    return self._assert_mode()

  @property
  def num_cores(self):
    # TODO(xiejw): Adds lazy num_shards initialization.
    return self._num_shards_or_none

  @property
  def num_hosts(self):
    return self.num_cores // self.num_of_cores_per_host

  @property
  def config(self):
    return self._config

  def is_input_sharded_per_core(self):
    """Return true if input_fn is invoked per-core (other than per-host)."""
    self._assert_mode()
    return (self._mode == model_fn_lib.ModeKeys.TRAIN and
            not self._config.tpu_config.per_host_input_for_training)

  def is_running_on_cpu(self):
    """Determines whether the input_fn and model_fn should be invoked on CPU."""
    mode = self._assert_mode()
    return ((not self._use_tpu) or mode == model_fn_lib.ModeKeys.PREDICT or
            (mode == model_fn_lib.ModeKeys.EVAL and
             self._eval_batch_size is None))

  @property
  def global_batch_size(self):
    mode = self._assert_mode()
    if mode == model_fn_lib.ModeKeys.EVAL and self._eval_batch_size is None:
      raise RuntimeError('Internal error, EVAL on TPU is not enabled, but '
                         '`global_batch_size` is called.')
    return (self._train_batch_size
            if mode == model_fn_lib.ModeKeys.TRAIN else self._eval_batch_size)

  @property
  def batch_size_for_input_fn(self):
    """Returns the shard batch size for `input_fn`."""
    mode = self._assert_mode()
    # Special case for eval.
    if mode == model_fn_lib.ModeKeys.EVAL and self._eval_batch_size is None:
      return None
    if self.is_running_on_cpu():
      if mode == model_fn_lib.ModeKeys.TRAIN:
        return self._train_batch_size
      if mode == model_fn_lib.ModeKeys.EVAL:
        return self._eval_batch_size
      return None

    global_batch_size = (
        self._train_batch_size
        if mode == model_fn_lib.ModeKeys.TRAIN else self._eval_batch_size)
    # On TPU
    if self.is_input_sharded_per_core():
      return global_batch_size // self.num_cores
    else:
      return global_batch_size // self.num_hosts

  @property
  def batch_size_for_model_fn(self):
    """Returns the shard batch size for `model_fn`."""
    mode = self._assert_mode()
    # Special case for eval.
    if mode == model_fn_lib.ModeKeys.EVAL and self._eval_batch_size is None:
      return None
    if self.is_running_on_cpu():
      if mode == model_fn_lib.ModeKeys.TRAIN:
        return self._train_batch_size
      if mode == model_fn_lib.ModeKeys.EVAL:
        return self._eval_batch_size
      return None

    # On TPU. always sharded per core.
    if mode == model_fn_lib.ModeKeys.TRAIN:
      return self._train_batch_size // self.num_cores
    else:
      return self._eval_batch_size // self.num_cores

  @property
  def master_job(self):
    """Returns the job name to use to place TPU computations on.

    Returns:
      A string containing the job name, or None if no job should be specified.

    Raises:
      ValueError: If the user needs to specify a tpu_job_name, because we are
        unable to infer the job name automatically, or if the user-specified job
        names are inappropriate.
    """
    run_config = self._config
    # If the user specifies the tpu_job_name, use that.
    if run_config.tpu_config.tpu_job_name:
      return run_config.tpu_config.tpu_job_name

    # The tpu job is determined by the run_config. Right now, this method is
    # required as tpu_config is not part of the RunConfig.
    mode = self._assert_mode()
    master = (
        run_config.evaluation_master
        if mode == model_fn_lib.ModeKeys.EVAL else run_config.master)
    if master in _LOCAL_MASTERS:
      return None

    if (not run_config.session_config or
        not run_config.session_config.cluster_def.job):
      return _DEFAULT_JOB_NAME
    cluster_def = run_config.session_config.cluster_def
    job_names = set([job.name for job in cluster_def.job])
    if _DEFAULT_JOB_NAME in job_names:
      # b/37868888 tracks allowing ClusterSpec propagation to reuse job names.
      raise ValueError('Currently, tpu_worker is not an allowed job name.')
    if len(job_names) == 1:
      return cluster_def.job[0].name
    if len(job_names) == 2:
      if _DEFAULT_COORDINATOR_JOB_NAME in job_names:
        job_names.remove(_DEFAULT_COORDINATOR_JOB_NAME)
        return job_names.pop()
      # TODO(b/67716447): Include more sophisticated heuristics.
    raise ValueError(
        'Could not infer TPU job name. Please specify a tpu_job_name as part '
        'of your TPUConfig.')

  @property
  def tpu_host_placement_function(self):
    """Returns the TPU host place function."""
    master = self.master_job

    def _placement_function(_sentinal=None, core_id=None, host_id=None):  # pylint: disable=invalid-name
      assert _sentinal is None
      if core_id is not None and host_id is not None:
        raise RuntimeError(
            'core_id and host_id can have only one non-None value.')

      if master is None:
        return '/replica:0/task:0/device:CPU:0'
      else:
        # This assumes that if using more than 8 shards,
        # the job configuration varies 'task'.
        if core_id is not None:
          host_id = core_id / 8
        return '/job:%s/task:%d/device:CPU:0' % (master, host_id)

    return _placement_function

  @property
  def tpu_device_placement_function(self):
    master = self.master_job
    job_device = '' if master is None else ('/job:%s' % master)

    def _placement_function(i):
      return '%s/task:%d/device:TPU:%d' % (job_device, i / 8, i % 8)

    return _placement_function

  @property
  def tpu_ordinal_function(self):
    """Returns the TPU ordinal fn."""

    def _tpu_ordinal_function(index):
      """Return the TPU ordinal associated with a shard.

      Required because the enqueue ops are placed on CPU.

      Args:
        index: the shard index

      Returns:
        The ordinal of the TPU device the shard's infeed should be placed on.
      """
      return index % 8

    return _tpu_ordinal_function


class _SIGNAL(object):
  """Signal used to control the thread of infeed/outfeed.

  All preserved signals must be negative numbers. Positive numbers are used to
  indicate the number of iterations for next training/evaluation loop.
  """
  NEXT_BATCH = -1
  STOP = -2


class TPUEstimatorSpec(
    collections.namedtuple('TPUEstimatorSpec', [
        'mode',
        'predictions',
        'loss',
        'train_op',
        'eval_metrics',
        'export_outputs',
        'scaffold_fn',
        'host_call'
    ])):
  """Ops and objects returned from a `model_fn` and passed to `TPUEstimator`.

  See `EstimatorSpec` for `mode`, 'predictions, 'loss', 'train_op', and
  'export_outputs`.

  For evaluation, `eval_metrics `is a tuple of `metric_fn` and `tensors`, where
  `metric_fn` runs on CPU to generate metrics and `tensors` represents the
  `Tensor`s transferred from TPU system to CPU host and passed to `metric_fn`.
  To be precise, TPU evaluation expects a slightly different signature from the
  @{tf.estimator.Estimator}. While `EstimatorSpec.eval_metric_ops` expects a
  dict, `TPUEstimatorSpec.eval_metrics` is a tuple of `metric_fn` and `tensors`.
  The `tensors` could be a list of `Tensor`s or dict of names to `Tensor`s. The
  `tensors` usually specify the model logits, which are transferred back from
  TPU system to CPU host. All tensors must have be batch-major, i.e., the batch
  size is the first dimension. Once all tensors are available at CPU host from
  all shards, they are concatenated (on CPU) and passed as positional arguments
  to the `metric_fn` if `tensors` is list or keyword arguments if `tensors` is
  dict. `metric_fn` takes the `tensors` and returns a dict from metric string
  name to the result of calling a metric function, namely a `(metric_tensor,
  update_op)` tuple. See `TPUEstimator` for MNIST example how to specify the
  `eval_metrics`.

  `scaffold_fn` is a function running on CPU to generate the `Scaffold`. This
  function should not capture any Tensors in `model_fn`.

  `host_call` is a tuple of a `function` and a list or dictionary of `tensors`
  to pass to that function. `host_call` currently works for train() and
  evaluate(). The function's graph is executed on the CPU on every step, so
  there is communication overhead when sending tensors from TPU to CPU. To
  reduce the overhead, try reducing the size of the tensors. The `tensors` are
  concatenated along their major (batch) dimension, and so must be >= rank 1.
  The `host_call` is useful for writing summaries with
  @{tf.contrib.summary.create_file_writer}. Note that `host_call` does not
  currently work if `use_tpu` is set to False.
  """

  def __new__(cls,
              mode,
              predictions=None,
              loss=None,
              train_op=None,
              eval_metrics=None,
              export_outputs=None,
              scaffold_fn=None,
              host_call=None):
    """Creates a validated `TPUEstimatorSpec` instance."""
    host_calls = {}
    if eval_metrics is not None:
      host_calls['eval_metrics'] = eval_metrics
    if host_call is not None:
      host_calls['host_call'] = host_call
    _OutfeedHostCall.validate(host_calls)
    return super(TPUEstimatorSpec, cls).__new__(
        cls,
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metrics=eval_metrics,
        export_outputs=export_outputs,
        scaffold_fn=scaffold_fn,
        host_call=host_call)

  def as_estimator_spec(self):
    """Creates an equivalent `EstimatorSpec` used by CPU train/eval."""
    eval_metric_ops = None
    if self.eval_metrics is not None:
      eval_metric_ops = _OutfeedHostCall.create_cpu_hostcall(
          {'eval_metrics': self.eval_metrics})['eval_metrics']
    scaffold = self.scaffold_fn() if self.scaffold_fn else None
    return model_fn_lib.EstimatorSpec(
        mode=self.mode,
        predictions=self.predictions,
        loss=self.loss,
        train_op=self.train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs=self.export_outputs,
        scaffold=scaffold)


class _OpQueueContext(object):
  """Manages work queue and thread for a infeed/outfeed thread."""

  def __init__(self, name, target, args):
    self._name = name
    self._queue = Queue.Queue()
    args = (self,) + args
    self._thread = threading.Thread(name=name, target=target, args=args)
    self._thread.daemon = True
    self._thread.start()

  def stop(self):
    self._queue.put(_SIGNAL.STOP)

  def send_next_batch_signal(self, iterations):
    self._queue.put(iterations)

  def read_iteration_counts(self):
    while True:
      iterations = self._queue.get(block=True)
      logging.debug('%s read iterations %s', self._name, iterations)
      if iterations == _SIGNAL.STOP:
        logging.info('%s received shutdown signal, stopping.', self._name)
        return
      yield iterations

  def join(self):
    logging.info('Shutting down %s thread.' % self._name)
    self.stop()
    self._thread.join()


class TPUInfeedOutfeedSessionHook(session_run_hook.SessionRunHook):
  """A Session hook setting up the TPU initialization, infeed, and outfeed.

  This hook does two major things:
  1. initialize and shutdown TPU system.
  2. launch and join the threads for infeed enqueue and (optional) outfeed
     dequeue.
  """

  def __init__(self, ctx, enqueue_ops, dequeue_ops):
    self._master_job = ctx.master_job
    self._enqueue_ops = enqueue_ops
    self._dequeue_ops = dequeue_ops
    self._initial_infeed_sleep_secs = (
        ctx.config.tpu_config.initial_infeed_sleep_secs)
    self._session_cancel_timer = None

    self._feed_error = None
    self._finished = False

  def begin(self):
    logging.info('TPU job name %s', self._master_job)
    self._iterations_per_loop_var = _create_or_get_iterations_per_loop()
    self._init_ops = [tpu.initialize_system(job=self._master_job)]
    self._finalize_ops = [tpu.shutdown_system(job=self._master_job)]

    summary_writer_init_ops = contrib_summary.summary_writer_initializer_op()
    self._init_ops.extend(summary_writer_init_ops)
    # Get all the writer resources from the initializer, so we know what to
    # flush.
    for op in summary_writer_init_ops:
      self._finalize_ops.append(contrib_summary.flush(writer=op.inputs[0]))

  def _log_error(self, session, error):
    """Log an infeed or outfeed error.

    This logs a short error message immediately, and schedules a timer to
    emit the full stack trace and error message after a short period of time.
    If the main session has terminated by the time the timer triggers, we
    assume the real source of the error was from the main session and avoid
    emitting a stack trace for the infeed.

    Args:
      session: `tf.Session`, session to be terminated error: exception that
        triggered logging.
      error: the Exception to log.
    """
    logging.warning(
        '\n\n'
        'Error occurred during infeed/outfeed.  This may be due to a compile '
        'error in the main session.  Waiting for a short time for the main '
        'session to come back.\n\n%s', error)

    self._feed_error = traceback.format_exc()

    # If we've already encountered a feed error, don't schedule another
    # cancellation op.
    if self._session_cancel_timer:
      return

    def _cancel_session():
      # Close the session to avoid the main thread from hanging. If input
      # pipeline triggers any error, the infeed thread dies but the main thread
      # for TPU computation waits for the infeed enqueue forever. Close the
      # Session to cancel the main thread Session.run execution.
      #
      # However, sleep for 2 minutes before explicit closing to give some time
      # for the TPU compilation error, if any, propagating, from TPU to CPU
      # host. Compilation errors should be reported by the main thread so that
      # the program can be interrupted and users can take action.  Due to a race
      # condition, the infeed thread might see an error first.  Closing the
      # session here immediately would result in a session cancellation
      # exception in the main thread, instead of the expected compile error.
      # User code that depends on having the proper exception type will
      # therefore be confused.
      time.sleep(5)

      # If the main session is still running, the infeed/outfeed errors are
      # legitimate, and should be logged.
      if not self._finished:
        logging.error('Feed error: %s', self._feed_error)
        logging.error('Closing session.  A RuntimeError should follow.')
        session.close()

    self._session_cancel_timer = threading.Thread(target=_cancel_session)
    self._session_cancel_timer.daemon = True
    self._session_cancel_timer.start()

  def _run_infeed(self, queue_ctx, session):
    logging.info('Starting infeed thread controller.')
    if self._initial_infeed_sleep_secs:
      logging.info('%s thread sleeping for %d seconds.', self._name,
                   self._initial_infeed_sleep_secs)
      time.sleep(self._initial_infeed_sleep_secs)
      logging.info('%s thread starting after sleep', self._name)

    try:
      if _WRAP_INPUT_FN_INTO_WHILE_LOOP:
        for _ in queue_ctx.read_iteration_counts():
          session.run(self._enqueue_ops)
      else:
        for count, steps in enumerate(queue_ctx.read_iteration_counts()):
          for i in xrange(steps):
            logging.debug('Infeed enqueue for iteration (%d, %d)', count, i)
            session.run(self._enqueue_ops)
      logging.debug('Infeed thread finished, shutting down.')
    except Exception as e:  # pylint: disable=broad-except
      self._log_error(session, e)

  def _run_outfeed(self, queue_ctx, session):
    logging.info('Starting outfeed thread controller.')
    try:
      for count, steps in enumerate(queue_ctx.read_iteration_counts()):
        for i in xrange(steps):
          logging.debug('Outfeed dequeue for iteration (%d, %d)', count, i)
          session.run(self._dequeue_ops)
    except Exception as e:  # pylint: disable=broad-except
      self._log_error(session, e)

  def after_create_session(self, session, coord):
    logging.info('Init TPU system')
    session.run(self._init_ops,
                options=config_pb2.RunOptions(timeout_in_ms=5 * 60 * 1000))

    logging.info('Start infeed thread controller')
    self._infeed_controller = _OpQueueContext(
        name='InfeedController', target=self._run_infeed, args=(session,))

    logging.info('Start outfeed thread controller')
    self._outfeed_controller = _OpQueueContext(
        name='OutfeedController', target=self._run_outfeed, args=(session,))

  def before_run(self, run_context):
    if self._feed_error:
      logging.warning('Feed error occurred, terminating session.')
      run_context.request_stop()
      return

    iterations = run_context.session.run(self._iterations_per_loop_var)

    logging.info('Enqueue next (%d) batch(es) of data to infeed.', iterations)
    self._infeed_controller.send_next_batch_signal(iterations)

    # TODO(xiejw): Refactor the outfeed dequeue into tf.while_loop.
    logging.info('Dequeue next (%d) batch(es) of data from outfeed.',
                 iterations)
    self._outfeed_controller.send_next_batch_signal(iterations)

  def end(self, session):
    if self._session_cancel_timer:
      logging.warning('Feed error occurred; waiting for message.')
      self._session_cancel_timer.join()

    self._finished = True
    logging.info('Stop infeed thread controller')
    self._infeed_controller.join()

    logging.info('Stop output thread controller')
    self._outfeed_controller.join()

    logging.info('Shutdown TPU system.')
    session.run(self._finalize_ops)


class _TPUStopAtStepHook(session_run_hook.SessionRunHook):
  """Hook that requests stop at a specified step.

  This hook is similar to the `session_run_hook._StopAfterNEvalsHook` with
  following differences for TPU training:

  1. This hook sets the variable for iterations_per_loop, which is used by
     `TPUInfeedOutfeedSessionHook` to control the iterations for infeed/outfeed.
     As the hook execution order is not guaranteed, the variable update is
     handled in `after_create_session` and `after_run` as
     `TPUInfeedOutfeedSessionHook` reads the variable value in `before_run`.

  2. For each training loop (session.run), the global step could be increased
     multiple times on TPU. The global step tensor value will be explicitly read
     again in `after_run` to ensure the latest value is retrieved to avoid race
     condition.
  """

  def __init__(self, iterations, num_steps=None, last_step=None):
    """Initializes a `StopAtStepHook`.

    Args:
      iterations: The number of iterations to run optimizer per training loop.
      num_steps: Number of steps to execute.
      last_step: Step after which to stop.

    Raises:
      ValueError: If one of the arguments is invalid.
    """
    if num_steps is None and last_step is None:
      raise ValueError('One of num_steps or last_step must be specified.')
    if num_steps is not None and last_step is not None:
      raise ValueError('Only one of num_steps or last_step can be specified.')
    self._num_steps = num_steps
    self._last_step = last_step
    self._iterations = iterations

  def _next_iterations(self, global_step, last_step):
    gap = last_step - global_step
    return min(gap, self._iterations)

  def begin(self):
    self._global_step_tensor = training_util.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError('Global step should be created.')

    self._iterations_per_loop_var = _create_or_get_iterations_per_loop()

  def after_create_session(self, session, coord):
    global_step = session.run(self._global_step_tensor)
    if self._last_step is None:
      self._last_step = global_step + self._num_steps

    iterations = self._next_iterations(global_step, self._last_step)

    self._iterations_per_loop_var.load(iterations, session=session)

  def after_run(self, run_context, run_values):
    # Global step cannot be retrieved via SessionRunArgs and before_run due to
    # race condition.
    global_step = run_context.session.run(self._global_step_tensor)
    if global_step >= self._last_step:
      run_context.request_stop()
    else:
      iterations = self._next_iterations(global_step, self._last_step)
      self._iterations_per_loop_var.load(
          iterations, session=run_context.session)


class _SetEvalIterationsHook(session_run_hook.SessionRunHook):
  """Hook that requests stop at a specified step."""

  def __init__(self, num_steps):
    """Initializes a `_SetEvalIterationsHook`.

    Args:
      num_steps: Number of steps to execute.
    """
    self._num_steps = num_steps

  def begin(self):
    self._iterations_per_loop_var = _create_or_get_iterations_per_loop()

  def after_create_session(self, session, coord):
    self._iterations_per_loop_var.load(self._num_steps, session=session)


def generate_per_core_enqueue_ops_fn_for_host(ctx, input_fn,
                                              inputs_structure_recorder):
  """Generates infeed enqueue ops for per-core input_fn on a single host."""
  captured_infeed_queue = _CapturedObject()

  def enqueue_ops_fn():
    """A fn returns enqueue_ops."""
    num_cores_per_host = ctx.num_of_cores_per_host
    per_host_sharded_inputs = []
    for core_ordinal in range(num_cores_per_host):
      with ops.name_scope('ordinal_%d' % (core_ordinal)):
        inputs = input_fn()
        if isinstance(inputs, tuple):
          features, labels = inputs
        else:
          features, labels = inputs, None

        inputs_structure_recorder.validate_and_record_structure(
            features, labels)
        flattened_inputs = (
            inputs_structure_recorder.flatten_features_and_labels(
                features, labels))
        per_host_sharded_inputs.append(flattened_inputs)

    infeed_queue = tpu_feed.InfeedQueue(
        number_of_tuple_elements=len(per_host_sharded_inputs[0]))
    captured_infeed_queue.capture(infeed_queue)
    infeed_queue.set_configuration_from_sharded_input_tensors(
        per_host_sharded_inputs)

    per_host_enqueue_ops = infeed_queue.generate_enqueue_ops(
        per_host_sharded_inputs, tpu_ordinal_function=ctx.tpu_ordinal_function)
    return per_host_enqueue_ops

  return enqueue_ops_fn, captured_infeed_queue


def generate_per_host_enqueue_ops_fn_for_host(
    ctx, input_fn, inputs_structure_recorder, batch_axis, device):
  """Generates infeed enqueue ops for per-host input_fn on a single host."""
  captured_infeed_queue = _CapturedObject()

  def enqueue_ops_fn():
    with ops.device(device):
      num_cores_per_host = ctx.num_of_cores_per_host
      inputs = input_fn()
      if isinstance(inputs, tuple):
        features, labels = inputs
      else:
        features, labels = inputs, None
      inputs_structure_recorder.validate_and_record_structure(features, labels)
      unsharded_tensor_list = (
          inputs_structure_recorder.flatten_features_and_labels(
              features, labels))

      infeed_queue = tpu_feed.InfeedQueue(
          tuple_types=[t.dtype for t in unsharded_tensor_list],
          tuple_shapes=[t.shape for t in unsharded_tensor_list],
          shard_dimensions=batch_axis)
      captured_infeed_queue.capture(infeed_queue)
      infeed_queue.set_number_of_shards(num_cores_per_host)

      per_host_enqueue_ops = (
          infeed_queue.split_inputs_and_generate_enqueue_ops(
              unsharded_tensor_list, placement_function=lambda x: device))
      return per_host_enqueue_ops

  return enqueue_ops_fn, captured_infeed_queue


class _InputPipeline(object):
  """`_InputPipeline` handles invoking `input_fn` and piping to infeed queue.

  `_InputPipeline` abstracts the per-core/per-host `input_fn` invocation from
  call site.  To be precise, based on the configuration in `_TPUContext`,  it
  invokes `input_fn` for all cores (usually multi-host TPU training) or for one
  host (usually for single-host TPU evaluation), and sends all `features` and
  `labels` returned by `input_fn` to TPU infeed. For per-core invocation,
  `features` and `labels` are piped to infeed directly, one tuple for each
  core. For per-host invocation,  `features` and `labels` are split at host
  (with respect to `batch_axis`) and piped to all cores accordingly.

  In addition, flatten/unflatten are handled by `_InputPipeline` also.  Model
  inputs returned by the `input_fn` can have one of the following forms:
  1. features
  2. (features, labels)

  Internally, form 1 is reformed to `(features, None)` as features and labels
  are passed separatedly to underlying methods. For TPU training, TPUEstimator
  may expect multiple `features` and `labels` tuples one for each core.

  TPUEstimator allows various different structures for inputs (namely `features`
  and `labels`).  `features` can be `Tensor` or dict of string name to `Tensor`,
  and `labels` could be `None`, `Tensor`, or dict of string name to `Tensor`.
  TPU infeed/outfeed library expects flattened tensor list. So, `features` and
  `labels` need to be flattened, before infeed enqueue, and the structure of
  them needs to be recorded, in order to restore them after infeed dequeue.
  """

  class InputsStructureRecorder(object):
    """The recorder to record inputs structure."""

    def __init__(self):
      # Holds the structure of inputs
      self._feature_names = []
      self._label_names = []
      self._has_labels = False

      # Internal state.
      self._initialized = False

    def has_labels(self):
      return self._has_labels

    def validate_and_record_structure(self, features, labels):
      """Validates and records the structure of features` and `labels`."""

      def _extract_key_names(tensor_or_dict):
        if tensor_or_dict is None:
          return []
        return tensor_or_dict.keys() if isinstance(tensor_or_dict, dict) else []

      # Extract structure.
      has_labels = labels is not None
      feature_names = _extract_key_names(features)
      label_names = _extract_key_names(labels)

      if self._initialized:
        # Verify the structure is same. The following should never happen.
        assert feature_names == self._feature_names, 'feature keys mismatched'
        assert label_names == self._label_names, 'label keys mismatched'
        assert has_labels == self._has_labels, 'label presence mismatched'
      else:
        # Record structure.
        self._initialized = True
        self._feature_names = feature_names
        self._label_names = label_names
        self._has_labels = has_labels

    def flatten_features_and_labels(self, features, labels):
      """Flattens the `features` and `labels` to a single tensor list."""
      flattened_inputs = []
      if self._feature_names:
        # We need a fixed ordering for enqueueing and dequeueing.
        flattened_inputs.extend(
            [features[name] for name in self._feature_names])
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

      Args:
        flattened_inputs: Flattened inputs for each shard.

      Returns:
        A tuple of (`features`, `labels`), where `labels` could be None.
        Each one, if present, should have identical structure (single tensor vs
        dict) as the one returned by input_fn.

      Raises:
        ValueError: If the number of expected tensors from `flattened_inputs`
          mismatches the recorded structure.
      """
      expected_num_features = (
          len(self._feature_names) if self._feature_names else 1)
      if self._has_labels:
        expected_num_labels = (
            len(self._label_names) if self._label_names else 1)
      else:
        expected_num_labels = 0

      expected_num_tensors = expected_num_features + expected_num_labels

      if expected_num_tensors != len(flattened_inputs):
        raise ValueError(
            'The number of flattened tensors mismatches expected num. '
            'Expected {}, got {}'.format(expected_num_tensors,
                                         len(flattened_inputs)))
      if self._feature_names:
        unflattened_features = dict(
            zip(self._feature_names, flattened_inputs[:expected_num_features]))
      else:
        # Single tensor case
        unflattened_features = flattened_inputs[0]

      if expected_num_labels == 0:
        unflattened_label = None
      elif self._label_names:
        unflattened_label = dict(
            zip(self._label_names, flattened_inputs[expected_num_features:]))
      else:
        # Single tensor case.
        unflattened_label = flattened_inputs[expected_num_features]

      return unflattened_features, unflattened_label

  def __init__(self, input_fn, batch_axis, ctx):
    """Constructor.

    Args:
      input_fn: input fn for train or eval.
      batch_axis: A python tuple of int values describing how each tensor
        produced by the Estimator `input_fn` should be split across the TPU
        compute shards.
      ctx: A `_TPUContext` instance with mode.

    Raises:
      ValueError: If both `sharded_features` and `num_cores` are `None`.
    """
    self._inputs_structure_recorder = _InputPipeline.InputsStructureRecorder()

    self._sharded_per_core = ctx.is_input_sharded_per_core()
    self._input_fn = input_fn
    self._infeed_queue = None
    self._ctx = ctx
    self._batch_axis = batch_axis

  def generate_infeed_enqueue_ops_and_dequeue_fn(self):
    """Generates infeed enqueue ops and dequeue_fn."""
    # While tf.while_loop is called, the body function, which invokes
    # `enqueue_fn` passed in, is called to construct the graph. So, input_fn
    # structure is recorded.
    enqueue_ops = self._invoke_input_fn_and_record_structure()

    self._validate_input_pipeline()

    def dequeue_fn():
      """dequeue_fn is used by TPU to retrieve the tensors."""
      values = self._infeed_queue.generate_dequeue_op()
      # The unflatten process uses the structure information recorded above.
      return self._inputs_structure_recorder.unflatten_features_and_labels(
          values)

    return (enqueue_ops, dequeue_fn)

  def _invoke_input_fn_and_record_structure(self):
    """Deploys the input pipeline and record input structure."""
    enqueue_ops = []
    infeed_queues = []
    num_hosts = self._ctx.num_hosts
    tpu_host_placement_fn = self._ctx.tpu_host_placement_function
    if self._sharded_per_core:
      # Per-Core input pipeline deployment.
      # Invoke input pipeline for each core and placed on the corresponding
      # host.
      for host_id in range(num_hosts):
        host_device = tpu_host_placement_fn(host_id=host_id)
        with ops.device(host_device):
          with ops.name_scope('input_pipeline_task%d' % (host_id)):
            enqueue_ops_fn, captured_infeed_queue = (
                generate_per_core_enqueue_ops_fn_for_host(
                    self._ctx, self._input_fn, self._inputs_structure_recorder))

            if _WRAP_INPUT_FN_INTO_WHILE_LOOP:
              enqueue_ops.append(
                  _wrap_computation_in_while_loop(
                      device=host_device, op_fn=enqueue_ops_fn))
            else:
              enqueue_ops.append(enqueue_ops_fn())
            # Infeed_queue_getter must be called after enqueue_ops_fn is called.
            infeed_queues.append(captured_infeed_queue.get())

    else:
      for host_id in range(num_hosts):
        host_device = tpu_host_placement_fn(host_id=host_id)
        with ops.device(host_device):
          with ops.name_scope('input_pipeline_task%d' % (host_id)):
            enqueue_ops_fn, captured_infeed_queue = (
                generate_per_host_enqueue_ops_fn_for_host(
                    self._ctx, self._input_fn, self._inputs_structure_recorder,
                    self._batch_axis, host_device))

            if _WRAP_INPUT_FN_INTO_WHILE_LOOP:
              enqueue_ops.append(
                  _wrap_computation_in_while_loop(
                      device=host_device, op_fn=enqueue_ops_fn))
            else:
              enqueue_ops.append(enqueue_ops_fn())
            infeed_queues.append(captured_infeed_queue.get())
    # infeed_queue is used to generate dequeue ops. The only thing it uses for
    # dequeue is dtypes and types. So, any one can be used. Here, grab the
    # first one.
    self._infeed_queue = infeed_queues[0]
    return enqueue_ops

  def _validate_input_pipeline(self):
    # Perform some sanity checks to log user friendly information. We should
    # error out to give users better error message. But, if
    # _WRAP_INPUT_FN_INTO_WHILE_LOOP is False (legacy behavior), we cannot break
    # user code, so, log a warning.
    if ops.get_default_graph().get_collection(ops.GraphKeys.QUEUE_RUNNERS):
      err_msg = ('Input pipeline contains one or more QueueRunners. '
                 'It could be slow and not scalable. Please consider '
                 'converting your input pipeline to use `tf.data` instead (see '
                 'https://www.tensorflow.org/programmers_guide/datasets for '
                 'instructions.')
      if _WRAP_INPUT_FN_INTO_WHILE_LOOP:
        raise RuntimeError(err_msg)
      else:
        logging.warn(err_msg)


class _ModelFnWrapper(object):
  """A `model_fn` wrapper.

  This makes calling model_fn on CPU and TPU easier and more consistent and
  performs necessary check and mutation required by TPU training and evaluation.

  In addition, this wrapper manages converting the `model_fn` to a single TPU
  train and eval step.
  """

  def __init__(self, model_fn, config, params, ctx):
    self._model_fn = model_fn
    self._config = config
    self._params = params
    self._ctx = ctx

  def call_without_tpu(self, features, labels):
    return self._call_model_fn(features, labels)

  def convert_to_single_tpu_train_step(self, dequeue_fn):
    """Converts user provided model_fn` as a single train step on TPU.

    The user provided `model_fn` takes input tuple
    (features, labels) and produces the EstimatorSpec with train_op and loss for
    train `mode`. This usually represents a single train computation on CPU.

    For TPU training, a train (computation) step is first wrapped in a
    tf.while_loop control flow to repeat for many times and then replicated to
    all TPU shards. Besides the input should be taken from TPU infeed rather
    than input pipeline (input_fn) directly. To fit TPU loop and replicate
    pattern, the original train computation should be reformed, which is the
    returned `train_step`.

    Args:
      dequeue_fn: The function to retrieve inputs, features and labels, from TPU
        infeed dequeue channel.

    Returns:
      A Fn representing the train step for TPU.
    """

    host_call = _OutfeedHostCall(self._ctx)
    captured_scaffold_fn = _CapturedObject()

    def train_step(loss):
      """Training step function for use inside a while loop."""
      del loss  # unused; required in function signature.
      features, labels = dequeue_fn()

      estimator_spec = self._verify_estimator_spec(
          self._call_model_fn(features, labels))
      loss, train_op = estimator_spec.loss, estimator_spec.train_op

      host_call_outfeed_ops = []
      if isinstance(estimator_spec, TPUEstimatorSpec):
        captured_scaffold_fn.capture(estimator_spec.scaffold_fn)
        if estimator_spec.host_call is not None:
          host_call.record({'host_call': estimator_spec.host_call})
          host_call_outfeed_ops = host_call.create_enqueue_op()
      else:
        captured_scaffold_fn.capture(None)

      with ops.control_dependencies([train_op] + host_call_outfeed_ops):
        return array_ops.identity(loss)

    return train_step, host_call, captured_scaffold_fn

  def convert_to_single_tpu_eval_step(self, dequeue_fn):
    """Converts user provided model_fn` as a single eval step on TPU.

    Similar to training, the user provided `model_fn` takes input tuple
    (features, labels) and produces the TPUEstimatorSpec with eval_metrics for
    eval `mode`. This usually represents a single evaluation computation on CPU.

    For TPU evaluation, a eval (computation) step is first wrapped in a
    tf.while_loop control flow to repeat for many times and then replicated to
    all TPU shards. Besides the input and output are slightly different. Input,
    features and labels, should be taken from TPU infeed rather than input
    pipeline (input_fn) directly. Output is managed in two stages.  First, the
    model outputs as the result of evaluation computation, usually model logits,
    should be transferred from TPU system to CPU. Then, all model outputs are
    concatenated first on CPU and sent to the metric_fn for metrics computation.
    To fit TPU evaluation pattern, the original eval computation should be
    reformed, which is the returned `eval_step`.

    Args:
      dequeue_fn: The function to retrieve inputs, features and labels, from TPU
        infeed dequeue channel.

    Returns:
      A tuple of eval_fn and eval_metrics. The eval_fn representing the eval
      step for TPU. and eval_metrics is an `_OutfeedHostCall` instance.
    """
    host_calls = _OutfeedHostCall(self._ctx)
    captured_scaffold_fn = _CapturedObject()

    def eval_step(total_loss):
      """Evaluation step function for use inside a while loop."""
      features, labels = dequeue_fn()

      tpu_estimator_spec = self._call_model_fn(features, labels)
      if not isinstance(tpu_estimator_spec, TPUEstimatorSpec):
        raise RuntimeError(
            'estimator_spec used by TPU evaluation must have type'
            '`TPUEstimatorSpec`. Got {}'.format(type(tpu_estimator_spec)))

      loss = tpu_estimator_spec.loss
      captured_scaffold_fn.capture(tpu_estimator_spec.scaffold_fn)
      to_record = {}
      to_record['eval_metrics'] = tpu_estimator_spec.eval_metrics
      if tpu_estimator_spec.host_call is not None:
        to_record['host_call'] = tpu_estimator_spec.host_call
      host_calls.record(to_record)

      with ops.control_dependencies(host_calls.create_enqueue_op()):
        return math_ops.add(total_loss, loss)

    return eval_step, host_calls, captured_scaffold_fn

  def _call_model_fn(self, features, labels):
    """Calls the model_fn with required parameters."""
    model_fn_args = util.fn_args(self._model_fn)
    kwargs = {}

    # Makes deep copy with `config` and params` in case user mutates them.
    config = copy.deepcopy(self._config)
    params = copy.deepcopy(self._params)

    if 'labels' in model_fn_args:
      kwargs['labels'] = labels
    elif labels is not None:
      raise ValueError(
          'model_fn does not take labels, but input_fn returns labels.')
    if 'mode' in model_fn_args:
      kwargs['mode'] = self._ctx.mode
    if 'config' in model_fn_args:
      kwargs['config'] = config
    if 'params' in model_fn_args:
      kwargs['params'] = params

    if 'params' not in model_fn_args:
      raise ValueError('model_fn ({}) does not include params argument, '
                       'required by TPUEstimator to pass batch size as '
                       'params[\'batch_size\']'.format(self._model_fn))

    batch_size_for_model_fn = self._ctx.batch_size_for_model_fn
    if batch_size_for_model_fn is not None:
      params[_BATCH_SIZE_KEY] = batch_size_for_model_fn

    estimator_spec = self._model_fn(features=features, **kwargs)
    if (self._ctx.is_running_on_cpu() and
        isinstance(estimator_spec, TPUEstimatorSpec)):
      # The estimator_spec will be passed to `Estimator` directly, which expects
      # type `EstimatorSpec`.
      return estimator_spec.as_estimator_spec()
    else:
      return estimator_spec

  def _verify_estimator_spec(self, estimator_spec):
    """Validates the estimator_spec."""
    if isinstance(estimator_spec, TPUEstimatorSpec):
      return estimator_spec

    err_msg = '{} returned by EstimatorSpec is not supported in TPUEstimator.'
    if estimator_spec.training_chief_hooks:
      raise ValueError(err_msg.format('training_chief_hooks'))
    if estimator_spec.training_hooks:
      raise ValueError(err_msg.format('training_hooks'))
    if estimator_spec.evaluation_hooks:
      raise ValueError(err_msg.format('evaluation_hooks'))

    if estimator_spec.scaffold:
      logging.warning('EstimatorSpec.Scaffold is ignored by TPU train/eval. '
                      'Please use TPUEstimatorSpec.')
    return estimator_spec


class _OutfeedHostCall(object):
  """Support for `eval_metrics` and `host_call` in TPUEstimatorSpec."""

  def __init__(self, ctx):
    self._ctx = ctx
    self._names = []
    # All of these are dictionaries of lists keyed on the name.
    self._host_fns = {}
    self._tensor_keys = collections.defaultdict(list)
    self._tensors = collections.defaultdict(list)
    self._tensor_dtypes = collections.defaultdict(list)
    self._tensor_shapes = collections.defaultdict(list)

  @staticmethod
  def validate(host_calls):
    """Validates the `eval_metrics` and `host_call` in `TPUEstimatorSpec`."""

    for name, host_call in host_calls.items():
      if not isinstance(host_call, (tuple, list)):
        raise ValueError('{} should be tuple or list'.format(name))
      if len(host_call) != 2:
        raise ValueError('{} should have two elements.'.format(name))
      if not callable(host_call[0]):
        raise TypeError('{}[0] should be callable.'.format(name))
      if not isinstance(host_call[1], (tuple, list, dict)):
        raise ValueError('{}[1] should be tuple or list, or dict.'.format(name))

      if isinstance(host_call[1], (tuple, list)):
        fn_args = util.fn_args(host_call[0])
        if len(host_call[1]) != len(fn_args):
          raise RuntimeError(
              'In TPUEstimatorSpec.{}, length of tensors does not '
              'match method args of metric_fn.'.format(name))

  @staticmethod
  def create_cpu_hostcall(host_calls):
    """Runs on the host_call on CPU instead of TPU when use_tpu=False."""

    _OutfeedHostCall.validate(host_calls)
    ret = {}
    for name, host_call in host_calls.items():
      host_fn, tensors = host_call
      if isinstance(tensors, (tuple, list)):
        ret[name] = host_fn(*tensors)
      else:
        # Must be dict.
        try:
          ret[name] = host_fn(**tensors)
        except TypeError as e:
          logging.warning(
              'Exception while calling %s: %s. It is likely the tensors '
              '(%s[1]) do not match the '
              'function\'s arguments', name, e, name)
          raise e
    return ret

  def record(self, host_calls):
    """Records the host_call structure."""

    for name, host_call in host_calls.items():
      host_fn, tensor_list_or_dict = host_call
      self._names.append(name)
      self._host_fns[name] = host_fn

      if isinstance(tensor_list_or_dict, dict):
        for (key, tensor) in six.iteritems(tensor_list_or_dict):
          self._tensor_keys[name].append(key)
          self._tensors[name].append(tensor)
          self._tensor_dtypes[name].append(tensor.dtype)
          self._tensor_shapes[name].append(tensor.shape)
      else:
        # List or tuple.
        self._tensor_keys[name] = None
        for tensor in tensor_list_or_dict:
          self._tensors[name].append(tensor)
          self._tensor_dtypes[name].append(tensor.dtype)
          self._tensor_shapes[name].append(tensor.shape)

  def create_enqueue_op(self):
    """Create the op to enqueue the recorded host_calls.

    Returns:
      A list of enqueue ops, which is empty if there are no host calls.
    """
    if not self._names:
      return []

    tensors = []
    # TODO(jhseu): Consider deduping tensors.
    for name in self._names:
      tensors.extend(self._tensors[name])
    return [tpu_ops.outfeed_enqueue_tuple(tensors)]

  def create_tpu_hostcall(self):
    """Sends the tensors through outfeed and runs the host_fn on CPU.

    The tensors are concatenated along dimension 0 to form a global tensor
    across all shards. The concatenated function is passed to the host_fn and
    executed on the first host.

    Returns:
      A dictionary mapping name to the return type of the host_call by that
      name.

    Raises:
      RuntimeError: If outfeed tensor is scalar.
    """
    if not self._names:
      return []

    ret = {}
    # For each i, dequeue_ops[i] is a list containing the tensors from all
    # shards. This list is concatenated later.
    dequeue_ops = []
    tensor_dtypes = []
    tensor_shapes = []
    for name in self._names:
      for _ in self._tensors[name]:
        dequeue_ops.append([])
      for dtype in self._tensor_dtypes[name]:
        tensor_dtypes.append(dtype)
      for shape in self._tensor_shapes[name]:
        tensor_shapes.append(shape)

    # Outfeed ops execute on each JF node. Note: we must constraint it such that
    # we have at most one outfeed dequeue and enqueue.
    tpu_device_placement_fn = self._ctx.tpu_device_placement_function
    for i in xrange(self._ctx.num_cores):
      with ops.device(tpu_device_placement_fn(i)):
        outfeed_tensors = tpu_ops.outfeed_dequeue_tuple(
            dtypes=tensor_dtypes, shapes=tensor_shapes)
        for j, item in enumerate(outfeed_tensors):
          dequeue_ops[j].append(item)

    # Deconstruct dequeue ops.
    dequeue_ops_by_name = {}
    pos = 0
    for name in self._names:
      dequeue_ops_by_name[name] = dequeue_ops[pos:pos+len(self._tensors[name])]
      pos += len(self._tensors[name])

    # It is assumed evaluation always happens on single host TPU system. So,
    # place all ops on tpu host if possible.
    #
    # TODO(jhseu): Evaluate whether this is right for summaries.
    with ops.device(self._ctx.tpu_host_placement_function(core_id=0)):
      for name in self._names:
        dequeue_ops = dequeue_ops_by_name[name]
        for i, item in enumerate(dequeue_ops):
          if dequeue_ops[i][0].shape.ndims == 0:
            raise RuntimeError(
                'All tensors outfed from TPU should preserve batch size '
                'dimension, but got scalar {}'.format(dequeue_ops[i][0]))
          # TODO(xiejw): Allow users to specify the axis for batch size
          # dimension.
          dequeue_ops[i] = array_ops.concat(dequeue_ops[i], axis=0)

        if self._tensor_keys[name] is not None:
          # The user-provided eval_metrics[1] is a dict.
          dequeue_ops = dict(zip(self._tensor_keys[name], dequeue_ops))
          try:
            ret[name] = self._host_fns[name](**dequeue_ops)
          except TypeError as e:
            logging.warning(
                'Exception while calling %s: %s. It is likely the tensors '
                '(%s[1]) do not match the '
                'function\'s arguments', name, e, name)
            raise e
        else:
          ret[name] = self._host_fns[name](*dequeue_ops)

    return ret


class ExamplesPerSecondHook(basic_session_run_hooks.StepCounterHook):
  """Count examples during runtime."""

  def __init__(self,
               batch_size,
               every_n_steps=100,
               every_n_secs=None,
               output_dir=None,
               summary_writer=None):
    self._batch_size = batch_size
    super(ExamplesPerSecondHook, self).__init__(
        every_n_steps=every_n_steps,
        every_n_secs=every_n_secs,
        output_dir=output_dir,
        summary_writer=summary_writer)

  def _log_and_record(self, elapsed_steps, elapsed_time, global_step):
    examples_per_sec = self._batch_size * elapsed_steps / elapsed_time
    if self._summary_writer is not None:
      example_summary = Summary(value=[
          Summary.Value(tag='examples_sec', simple_value=examples_per_sec)
      ])
      self._summary_writer.add_summary(example_summary, global_step)
    logging.info('examples/sec: %g', examples_per_sec)


class InstallSignalHandlerHook(session_run_hook.SessionRunHook):
  """Change SIGINT (CTRL^C) handler to force quit the process.

  The default behavior often results in hanging processes.
  The original handler is restored after training/evaluation.
  """

  def __init__(self):
    self._signal_fn = signal.getsignal(signal.SIGINT)

  def before_run(self, run_context):
    signal.signal(signal.SIGINT, signal.SIG_DFL)

  def end(self, session):
    signal.signal(signal.SIGINT, self._signal_fn)


class TPUEstimator(estimator_lib.Estimator):
  """Estimator with TPU support.

  TPUEstimator handles many of the details of running on TPU devices, such as
  replicating inputs and models for each core, and returning to host
  periodically to run hooks.

  If `use_tpu` is false, all training, evaluation, and predict are executed on
  CPU.

  For training, TPUEstimator transforms a global batch size in params to a
  per-shard batch size when calling the `input_fn` and `model_fn`. Users should
  specify `train_batch_size` in constructor, and then get the batch size for
  each shard in `input_fn` and `model_fn` by `params['batch_size']`. If
  `TPUConfig.per_host_input_for_training` is `True`, `input_fn` is invoked per
  host rather than per core. In this case, a global batch size is transformed a
  per-host batch size in params for `input_fn`, but `model_fn` still gets
  per-core batch size.

  For evaluation, if `eval_batch_size` is None, it is executed on CPU, even if
  `use_tpu` is `True`. If `eval_batch_size` is not `None`, it is executed on
  TPU, which is an experimental feature. In this case, `model_fn` should return
  `TPUEstimatorSpec` instead of `EstimatorSpec`, which expects the
  `eval_metrics` for TPU evaluation.

  `TPUEstimatorSpec.eval_metrics` is a tuple of `metric_fn` and `tensors`, where
  `tensors` could be a list of `Tensor`s or dict of names to `Tensor`s. (See
  `TPUEstimatorSpec` for details).  `metric_fn` takes the `tensors` and returns
  a dict from metric string name to the result of calling a metric function,
  namely a `(metric_tensor, update_op)` tuple.

  Current limitations:

  1. TPU evaluation only works on single host.
  2. `input_fn` for evaluation should not throw OutOfRange error for all
  evaluation steps and all batches should have the same size.

  Example (MNIST):
  ```
  # The metric Fn which runs on CPU.
  def metric_fn(labels, logits):
    predictions = tf.argmax(logits, 1)
    return {
      'accuracy': tf.metrics.precision(
          labels=labels, predictions=predictions),
    }

  # Your model Fn which runs on TPU (eval_metrics is list in this example)
  def model_fn(features, labels, mode, config, params):
    ...
    logits = ...

    if mode = tf.estimator.ModeKeys.EVAL:
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metrics=(metric_fn, [labels, logits]))

  # or specify the eval_metrics tensors as dict.
  def model_fn(features, labels, mode, config, params):
    ...
    final_layer_output = ...

    if mode = tf.estimator.ModeKeys.EVAL:
      return tpu_estimator.TPUEstimatorSpec(
          mode=mode,
          loss=loss,
          eval_metrics=(metric_fn, {
              'labels': labels,
              'logits': final_layer_output,
          }))
  ```

  Predict support on TPU is not yet implemented. So, `predict` and
  `export_savedmodel` are executed on CPU, even if `use_tpu` is true.
  """

  def __init__(self,
               model_fn=None,
               model_dir=None,
               config=None,
               params=None,
               use_tpu=True,
               train_batch_size=None,
               eval_batch_size=None,
               batch_axis=None):
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
      use_tpu: A bool indicating whether TPU support is enabled. Currently,
        - TPU training respects this bit.
        - If true, see `eval_batch_size` for evaluate support.
        - Predict still happens on CPU.
      train_batch_size: An int representing the global training batch size.
        TPUEstimator transforms this global batch size to a per-shard batch
        size, as params['batch_size'], when calling `input_fn` and `model_fn`.
        Cannot be `None` if `use_tpu` is `True`. Must be divisible by
        `config.tpu_config.num_shards`.
      eval_batch_size: An int representing the global training batch size.
        Currently, if `None`, evaluation is still executed on CPU (even when
        `use_tpu` is True). In near future, `use_tpu` will be the only option to
        switch between TPU/CPU evaluation.
      batch_axis: A python tuple of int values describing how each tensor
        produced by the Estimator `input_fn` should be split across the TPU
        compute shards. For example, if your input_fn produced (images, labels)
        where the images tensor is in `HWCN` format, your shard dimensions would
        be [3, 0], where 3 corresponds to the `N` dimension of your images
        Tensor, and 0 corresponds to the dimension along which to split the
        labels to match up with the corresponding images. If None is supplied,
        and per_host_input_for_training is True, batches will be sharded based
        on the major dimension. If tpu_config.per_host_input_for_training is
        False, batch_axis is ignored.

    Raises:
      ValueError: `params` has reserved keys already.
    """
    if config is None or not isinstance(config, tpu_config.RunConfig):
      raise ValueError(
          '`config` must be provided with type `tpu_config.RunConfig`')

    if params is not None and any(k in params for k in _RESERVED_PARAMS_KEYS):
      raise ValueError('{} are reserved keys but existed in params {}.'.format(
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
            'train batch size {} must be divisible by number of shards {}'
            .format(train_batch_size, config.tpu_config.num_shards))

      if eval_batch_size is not None:
        if config.tpu_config.num_shards > 8:
          raise NotImplementedError(
              'TPU evaluation is only supported with one host.')

        if eval_batch_size % config.tpu_config.num_shards != 0:
          raise ValueError(
              'eval batch size {} must be divisible by number of shards {}'
              .format(eval_batch_size, config.tpu_config.num_shards))

    # Verifies the model_fn signature according to Estimator framework.
    estimator_lib._verify_model_fn_args(model_fn, params)  # pylint: disable=protected-access
    # We cannot store config and params in this constructor as parent
    # constructor might change them, such as assigning a temp dir for
    # config.model_dir.
    model_function = self._augment_model_fn(model_fn, batch_axis)

    # Passing non-None params as wrapped model_fn has it.
    params = params or {}
    super(TPUEstimator, self).__init__(
        model_fn=model_function,
        model_dir=model_dir,
        config=config,
        params=params)
    self._iterations_per_training_loop = (
        self._config.tpu_config.iterations_per_loop)

    # All properties passed to _TPUContext are immutable.
    self._ctx = _TPUContext(self._config, train_batch_size, eval_batch_size,
                            use_tpu)

  def _create_global_step(self, graph):
    """Creates a global step suitable for TPUs.

    Args:
      graph: The graph in which to create the global step.

    Returns:
      A global step `Tensor`.

    Raises:
      ValueError: if the global step tensor is already defined.
    """
    return _create_global_step(graph)

  def _convert_train_steps_to_hooks(self, steps, max_steps):
    with self._ctx.with_mode(model_fn_lib.ModeKeys.TRAIN) as ctx:
      if ctx.is_running_on_cpu():
        return super(TPUEstimator, self)._convert_train_steps_to_hooks(
            steps, max_steps)

    # On TPU.
    if steps is None and max_steps is None:
      raise ValueError(
          'For TPU training, one of `steps` or `max_steps` must be set. '
          'Cannot be both `None`.')

    # Estimator.train has explicit positiveness check.
    if steps is not None:
      util_lib.check_positive_integer(steps, 'Train steps')
    if max_steps is not None:
      util_lib.check_positive_integer(max_steps, 'Train max_steps')

    return [
        _TPUStopAtStepHook(self._iterations_per_training_loop, steps, max_steps)
    ]

  def _convert_eval_steps_to_hooks(self, steps):
    with self._ctx.with_mode(model_fn_lib.ModeKeys.EVAL) as ctx:
      if ctx.is_running_on_cpu():
        return super(TPUEstimator, self)._convert_eval_steps_to_hooks(steps)

    if steps is None:
      raise ValueError('Evaluate `steps` must be set on TPU. Cannot be `None`.')

    util_lib.check_positive_integer(steps, 'Eval steps')

    return [
        evaluation._StopAfterNEvalsHook(  # pylint: disable=protected-access
            num_evals=steps),
        _SetEvalIterationsHook(steps)
    ]

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

    with self._ctx.with_mode(mode) as ctx:
      # Setting the batch size in params first. This helps user to have same
      # input_fn for use_tpu=True/False.
      batch_size_for_input_fn = ctx.batch_size_for_input_fn
      if batch_size_for_input_fn is not None:
        kwargs['params'][_BATCH_SIZE_KEY] = batch_size_for_input_fn

      if ctx.is_running_on_cpu():
        with ops.device('/device:CPU:0'):
          return input_fn(**kwargs)

      # For TPU computation, input_fn should be invoked in a tf.while_loop for
      # performance. While constructing the tf.while_loop, the structure of
      # inputs returned by the `input_fn` needs to be recorded. The structure
      # includes whether features or labels is dict or single Tensor, dict keys,
      # tensor shapes, and dtypes. The recorded structure is used to create the
      # infeed dequeue ops, which must be wrapped and passed as a Fn, called
      # inside the TPU computation, as the TPU computation is wrapped inside a
      # tf.while_loop also. So, we either pass input_fn to model_fn or pass
      # dequeue_fn to model_fn. Here, `input_fn` is passed directly as
      # `features` in `model_fn` signature.
      def _input_fn():
        return input_fn(**kwargs)

      return _input_fn

  def _augment_model_fn(self, model_fn, batch_axis):
    """Returns a new model_fn, which wraps the TPU support."""

    def _model_fn(features, labels, mode, config, params):
      """A Estimator `model_fn` for TPUEstimator."""
      with self._ctx.with_mode(mode) as ctx:
        model_fn_wrapper = _ModelFnWrapper(model_fn, config, params, ctx)

        # TODO(jhseu): Move to PREDICT to TPU.
        if ctx.is_running_on_cpu():
          logging.info('Running %s on CPU', mode)
          return model_fn_wrapper.call_without_tpu(features, labels)

        assert labels is None, '`labels` passed to `model_fn` must be `None`.'
        # TPUEstimator._call_input_fn passes `input_fn` as features to here.
        assert callable(features), '`input_fn` is not callable.'
        input_fn = features

        input_holders = _InputPipeline(input_fn, batch_axis, ctx)
        enqueue_ops, dequeue_fn = (
            input_holders.generate_infeed_enqueue_ops_and_dequeue_fn())

        if mode == model_fn_lib.ModeKeys.TRAIN:
          loss, host_call, scaffold = (
              _train_on_tpu_system(ctx, model_fn_wrapper, dequeue_fn))
          host_ops = host_call.create_tpu_hostcall()
          if host_ops is None:
            host_ops = []
          hooks = [
              TPUInfeedOutfeedSessionHook(ctx, enqueue_ops, host_ops),
              ExamplesPerSecondHook(ctx.global_batch_size),
              InstallSignalHandlerHook(),
              training.LoggingTensorHook(
                  {
                      'loss': array_ops.identity(loss),
                      'step': training.get_global_step()
                  },
                  every_n_secs=30)
          ]
          summary.scalar(model_fn_lib.LOSS_METRIC_KEY, loss)
          with ops.control_dependencies([loss]):
            update_ops = _sync_variables_ops()

          # Validate the TPU training graph to catch basic errors
          _validate_tpu_training_graph()

          return model_fn_lib.EstimatorSpec(
              mode,
              loss=loss,
              training_hooks=hooks,
              train_op=control_flow_ops.group(*update_ops),
              scaffold=scaffold)

        # Now eval.
        total_loss, host_calls, scaffold = _eval_on_tpu_system(
            ctx, model_fn_wrapper, dequeue_fn)
        iterations_per_loop_var = _create_or_get_iterations_per_loop()
        mean_loss = math_ops.div(total_loss,
                                 math_ops.cast(
                                     iterations_per_loop_var,
                                     dtype=total_loss.dtype))

        # Creates a dummy metric update_op for all metrics. Estimator expects
        # all metrics in eval_metric_ops have update_op and calls them one by
        # one. The real metric update_ops are invoked in a separated thread. So,
        # here give Estimator the dummy op for all metrics.
        with ops.control_dependencies([mean_loss]):
          # After TPU evaluation computation is done (the mean_loss tensor),
          # reads all variables back from TPU and updates the eval step counter
          # properly
          internal_ops_to_run = _sync_variables_ops()
          internal_ops_to_run.append(
              _increase_eval_step_op(iterations_per_loop_var))
          with ops.control_dependencies(internal_ops_to_run):
            dummy_update_op = control_flow_ops.no_op()

        host_call_ret = host_calls.create_tpu_hostcall()
        eval_metric_ops = {}
        eval_update_ops = []
        for k, v in host_call_ret['eval_metrics'].items():
          eval_metric_ops[k] = (v[0], dummy_update_op)
          eval_update_ops.append(v[1])

        if 'host_call' not in host_call_ret:
          host_ops = []
        else:
          host_ops = host_call_ret['host_call']
        hooks = [
            TPUInfeedOutfeedSessionHook(ctx, enqueue_ops,
                                        eval_update_ops + host_ops),
        ]

        return model_fn_lib.EstimatorSpec(
            mode,
            loss=mean_loss,
            evaluation_hooks=hooks,
            eval_metric_ops=eval_metric_ops,
            scaffold=scaffold)

    return _model_fn


def _eval_on_tpu_system(ctx, model_fn_wrapper, dequeue_fn):
  """Executes `model_fn_wrapper` multiple times on all TPU shards."""
  num_cores = ctx.num_cores
  iterations_per_loop_var = _create_or_get_iterations_per_loop()

  single_tpu_eval_step, host_calls, captured_scaffold_fn = (
      model_fn_wrapper.convert_to_single_tpu_eval_step(dequeue_fn))

  def multi_tpu_eval_steps_on_single_shard():
    return training_loop.repeat(
        iterations_per_loop_var,
        single_tpu_eval_step, [_ZERO_LOSS],
        name='loop')

  (loss,) = tpu.shard(
      multi_tpu_eval_steps_on_single_shard,
      inputs=[],
      num_shards=num_cores,
      outputs_from_all_shards=False)

  scaffold = _get_scaffold(captured_scaffold_fn)
  return loss, host_calls, scaffold


def _train_on_tpu_system(ctx, model_fn_wrapper, dequeue_fn):
  """Executes `model_fn_wrapper` multiple times on all TPU shards."""
  num_cores = ctx.num_cores
  iterations_per_loop_var = _create_or_get_iterations_per_loop()

  single_tpu_train_step, host_call, captured_scaffold_fn = (
      model_fn_wrapper.convert_to_single_tpu_train_step(dequeue_fn))

  def multi_tpu_train_steps_on_single_shard():
    return training_loop.repeat(
        iterations_per_loop_var,
        single_tpu_train_step, [_INITIAL_LOSS],
        name=b'loop')

  (loss,) = tpu.shard(
      multi_tpu_train_steps_on_single_shard,
      inputs=[],
      num_shards=num_cores,
      outputs_from_all_shards=False)

  scaffold = _get_scaffold(captured_scaffold_fn)
  return loss, host_call, scaffold


def _wrap_computation_in_while_loop(device, op_fn):
  """Wraps the ops generated by `op_fn` in tf.while_loop."""

  def computation(i):
    with ops.control_dependencies(op_fn()):
      return i + 1

  iterations_per_loop_var = _create_or_get_iterations_per_loop()
  # By setting parallel_iterations=1, the parallel execution in while_loop is
  # basically turned off.
  with ops.device(device):
    iterations = array_ops.identity(iterations_per_loop_var)
    return control_flow_ops.while_loop(
        lambda i: i < iterations,
        computation, [constant_op.constant(0)],
        parallel_iterations=1)


def _validate_tpu_training_graph():
  """Validate graph before running distributed training.

  Raises:
    ValueError: If the graph seems invalid for running on device
  """
  operations = ops.get_default_graph().get_operations()

  # Check if there is atleast one CrossReplicaSum operation in the graph
  # This should be introduced by using the CrossShardOptimizer wrapper
  cross_replica_sum_ops = [
      o for o in operations if o.type == _CROSS_REPLICA_SUM_OP
  ]
  if not cross_replica_sum_ops:
    raise ValueError(
        'CrossShardOptimizer must be used for model training on TPUs.')


class _CapturedObject(object):
  """A placeholder to capture an object.

  This is useful when we need to capture a Python object in the Tensorflow
  control flow body function and use it outside the control flow.
  """

  def __init__(self):
    self._object = None
    self._captured = False

  def capture(self, o):
    if self._captured:
      raise RuntimeError(
          'InternalError: Object can be captured only. Please file bug .')

    self._captured = True
    self._object = o

  def get(self):
    if not self._captured:
      raise RuntimeError(
          'InternalError: Object is not captured properly before `get`. '
          'Please file bug .')
    return self._object


def _get_scaffold(captured_scaffold_fn):
  """Retrieves the Scaffold from `captured_scaffold_fn`."""
  with _CapturingContext(message='Inside scaffold_fn'):
    scaffold_fn = captured_scaffold_fn.get()
    if scaffold_fn:
      scaffold = scaffold_fn()
      if scaffold is None:
        raise ValueError(
            'TPUEstimatorSpec.scaffold_fn returns None, which is not allowed')
    else:
      scaffold = None

  if scaffold:
    wrapped_finalize = scaffold.finalize

    def _finalize():
      with _CapturingContext('Inside Scaffold.finalize'):
        wrapped_finalize()

    scaffold.finalize = _finalize
  return scaffold


class _CapturingContext(control_flow_ops.ControlFlowContext):
  """Tracks references to Tensors defined in TPU replication."""

  def __init__(self, message):
    control_flow_ops.ControlFlowContext.__init__(self)
    self._message = message

  def AddOp(self, op):  # pylint: disable=invalid-name
    for c in op.inputs:
      if tpu._TPU_REPLICATE_ATTR in c.op.node_def.attr:  # pylint: disable=protected-access
        raise ValueError('{}: Op {} depends on TPU computation {}, '
                         'which is not allowed.'.format(self._message, op, c))

  def __enter__(self):
    # pylint: disable=protected-access
    self._g = ops.get_default_graph()
    self._old = self._g._get_control_flow_context()
    self._g._set_control_flow_context(self)
    # pylint: enable=protected-access

  def __exit__(self, _, __, ___):  # pylint: disable=invalid-name
    self._g._set_control_flow_context(self._old)  # pylint: disable=protected-access
