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
"""A Context that captures profile and performs profiling/dumping.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import random
import sys
import threading

from tensorflow.core.protobuf import config_pb2
from tensorflow.python import pywrap_tensorflow as print_mdl
from tensorflow.python.client import session
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.util import compat

WARMUP_STEPS = 10
MAX_TRACED_STEPS = 100


def _profiled_init(self, target='', graph=None, config=None):
  """Overwrites the session.__init__."""
  self._profiler_init_internal(target, graph, config)  # pylint: disable=protected-access


def _profiled_run(self,
                  fetches,
                  feed_dict=None,
                  options=None,
                  run_metadata=None):
  """Overwrites the session.run()."""
  # pylint: disable=protected-access
  # Count the session steps.
  with self.profile_context._new_step() as state:
    step, locked = state
    # Fast path if no need for profiling.
    if locked and not self.profile_context._is_fast_path(step):
      # Maybe trace this step.
      if self.profile_context._should_trace(step, self.graph, fetches):
        if self.profile_context._debug:
          sys.stderr.write('debug: tracing step: %d\n' % step)
        # Enable tracing, perform auto profiling or auto dump.
        if not run_metadata:
          run_metadata = config_pb2.RunMetadata()

        if not options:
          options = config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE)
          old_trace_level = options.trace_level
        else:
          old_trace_level = options.trace_level
          options.trace_level = config_pb2.RunOptions.FULL_TRACE

        ret = self._profiler_run_internal(
            fetches, feed_dict, options, run_metadata)
        if self.profile_context._debug:
          self.profile_context._dump_file(run_metadata, 'run_meta_%d' % step)

        self.profile_context.profiler._graph = self.graph
        self.profile_context.profiler.add_step(step, run_metadata)
        options.trace_level = old_trace_level
      else:
        ret = self._profiler_run_internal(fetches, feed_dict, options)

      # Maybe dump profile.
      self.profile_context._maybe_dump(step)

      # Maybe profile:
      to_profiles = self.profile_context._profile_candidates()
      for to_prof in to_profiles:
        cmd, opts, _ = to_prof
        if self.profile_context._debug:
          sys.stderr.write('debug: profiling %s step: %d\n' % (cmd, step))
        if cmd == 'graph':
          self.profile_context.profiler.profile_graph(opts)
        elif cmd == 'scope':
          self.profile_context.profiler.profile_name_scope(opts)
        elif cmd == 'op':
          self.profile_context.profiler.profile_operations(opts)
        elif cmd == 'code':
          self.profile_context.profiler.profile_python(opts)
        else:
          raise ValueError('Unknown cmd: %s\n' % cmd)
      return ret
  # Fast no lock path.
  return self._profiler_run_internal(
      fetches, feed_dict, options, run_metadata)
  # pylint: enable=protected-access


class ProfileContext(object):
  """A Context that captures RunMetadata and performs profiling.

  ```python
    # Trace steps 100~200, profile at [150, 200] and dump profile at 200.
    with tf.contrib.tfprof.ProfileContext('/tmp/train_dir',
                                          trace_steps=range(100, 200, 3),
                                          dump_steps=[200]) as pctx:
      opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
      pctx.add_auto_profiling('op', opts, [150, 200])
      train_loop().

    # Tracing only.
    with tf.contrib.tfprof.ProfileContext('/tmp/train_dir') as pctx:
      # Run train/eval loop for at least few hundred steps. Profiles will be
      # dumped to train_dir. Use web UI or command line to do profiling.
      train_loop().

    # When session object is available, do explicit trace, profile and dump.
    with tf.contrib.tfprof.ProfileContext('/tmp/train_dir',
                                          trace_steps=[],
                                          dump_steps=[]) as pctx:
      opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
      pctx.trace_next_step()
      _ = session.run(train_op)
      pctx.profiler.profile_operations(options=opts)
  ```

  Args:
    profile_dir: Directory to store profiles.
    trace_steps: A list of session run steps to trace. If None, use
        pre-defined steps.
    dump_steps: A list of steps to dump the profile to `profile_dir`. If None,
        use pre-defined steps.
    enabled: If false, everything is disabled with minimal overhead. It allows
        user to only enable profiling when needed.
    debug: If true, also dumps the raw trace RunMetadata text file to
        profile_dir. And print debugging message. Useful for bug report.
  """

  def __init__(self,
               profile_dir,
               trace_steps=None,
               dump_steps=None,
               enabled=True,
               debug=False):
    self._enabled = enabled
    if not self._enabled:
      return

    self._debug = debug
    if not profile_dir:
      raise ValueError('Must have a directory for profile.\n')
    self._profiler_dir = profile_dir

    if trace_steps is None:
      self._trace_steps = set()
      self._auto_tracing = True
    else:
      if len(trace_steps) > MAX_TRACED_STEPS:
        raise ValueError('Only support tracing up to 100 steps.\n')
      self._trace_steps = set(trace_steps[:])
      self._auto_tracing = False

    if dump_steps is None:
      self._dump_steps = set([MAX_TRACED_STEPS])
    else:
      self._dump_steps = set(dump_steps[:])

    self._rng = random.Random(111)
    self._fetched = set()
    self._slow_path_steps = self._dump_steps | self._trace_steps
    self._trace_next_step = False
    self._dump_next_step = False
    self._step = 0
    self._traced_steps = 0
    self._auto_profiles = []
    self._profiler = None
    self._lock = threading.Lock()

  def add_auto_profiling(self, cmd, options, profile_steps):
    """Traces and profiles at some session run steps.

    Args:
      cmd: The profiling commands. (i.e. scope, op, python, graph)
      options: The profiling options.
      profile_steps: A list/set of integers. The profiling command and options
          will be run automatically at these integer steps. Each step is
          a session.run.
    """
    if not self._enabled:
      return
    self._auto_profiles.append((cmd, options, profile_steps[:]))
    self._slow_path_steps |= set(profile_steps)
    self._trace_steps |= set(profile_steps)

  @property
  def profiler(self):
    """Returns the current profiler object."""
    if not self._enabled:
      return None
    if not self._profiler:
      self._profiler = model_analyzer.Profiler(ops.get_default_graph())
    return self._profiler

  def trace_next_step(self):
    """Enables tracing and adds traces to profiler at next step."""
    if not self._enabled:
      return
    self._trace_next_step = True
    self._slow_path_steps.add(self._step)

  def dump_next_step(self):
    """Enable tracing and dump profiles at next step."""
    if not self._enabled:
      return
    self._dump_next_step = True
    self._slow_path_steps.add(self._step)

  def _is_fast_path(self, step):
    if step in self._slow_path_steps:
      return False
    # When user doesn't set the tracing steps explicitly, auto decide it.
    if (self._auto_tracing and step > WARMUP_STEPS and
        self._traced_steps <= MAX_TRACED_STEPS):
      return False
    return True

  def _should_trace(self, step, graph, fetches):
    """Whether should do tracing at current step."""
    if self._traced_steps > MAX_TRACED_STEPS:
      return False
    # Check user-set tracing steps.
    if step in self._trace_steps or self._trace_next_step:
      self._traced_steps += 1
      return True

    # If no user-set tracing steps set and passes warm up steps, auto trace.
    if self._auto_tracing and step > WARMUP_STEPS:
      # If the fetches have not been seen before, trace it.
      with graph.as_default():
        fetch_names = [f.name for f in
                       session._FetchMapper.for_fetch(fetches).unique_fetches()]  # pylint: disable=protected-access
      fetch_name = '-'.join(sorted(fetch_names))
      if self._debug:
        sys.stderr.write('debug: trace fetches: %s\n' % fetch_name)
      if fetch_name not in self._fetched:
        self._fetched.add(fetch_name)
        self._traced_steps += 1
        return True
      # If the trace coverage is low, does some random tracing.
      if (self.profiler._coverage < 0.5 and step < MAX_TRACED_STEPS and  # pylint: disable=protected-access
          self._rng.randint(0, 10) < 2):
        self._traced_steps += 1
        return True
    return False

  def _maybe_dump(self, step):
    """Maybe dump the profile file."""
    if not (step in self._dump_steps or self._dump_next_step):
      return
    if self._debug:
      sys.stderr.write('debug: dumping file at step: %d\n' % step)
    if not gfile.Exists(self._profiler_dir):
      gfile.MakeDirs(self._profiler_dir)

    filename = os.path.join(compat.as_bytes(self._profiler_dir),
                            compat.as_bytes('profile_%d' % step))
    self.profiler._write_profile(filename)  # pylint: disable=protected-access

  def _dump_file(self, pb, basename):
    if not gfile.Exists(self._profiler_dir):
      gfile.MakeDirs(self._profiler_dir)
    with gfile.Open(os.path.join(self._profiler_dir, basename), 'w') as f:
      f.write('%s' % pb)

  @contextlib.contextmanager
  def _new_step(self):
    acquired = self._lock.acquire(False)
    yield (self._step, acquired)
    self._step += 1
    self._trace_next_step = False
    self._dump_next_step = False
    if acquired:
      self._lock.release()

  def _profile_candidates(self):
    to_profile = []
    for auto_prof in self._auto_profiles:
      _, _, prof_steps = auto_prof
      if self._step in prof_steps:
        to_profile.append(auto_prof)
    return to_profile

  def __enter__(self):
    if self._enabled:
      self.old_run = getattr(session.BaseSession, 'run', None)
      self.old_init = getattr(session.BaseSession, '__init__', None)
      if not self.old_run:
        raise errors.InternalError(None, None, 'BaseSession misses run method.')
      elif not self.old_init:
        raise errors.InternalError(None, None,
                                   'BaseSession misses __init__ method.')
      elif getattr(session.BaseSession, '_profiler_run_internal', None):
        raise errors.InternalError(None, None,
                                   'Already in context or context not cleaned.')
      elif getattr(session.BaseSession, '_profiler_init_internal', None):
        raise errors.InternalError(None, None,
                                   'Already in context or context not cleaned.')
      else:
        setattr(session.BaseSession, 'run', _profiled_run)
        setattr(session.BaseSession, '__init__', _profiled_init)
        setattr(session.BaseSession, '_profiler_run_internal', self.old_run)
        setattr(session.BaseSession, '_profiler_init_internal', self.old_init)
        setattr(session.BaseSession, 'profile_context', self)
        return self
    else:
      return self

  def __exit__(self, exec_type, exec_value, exec_tb):
    if not self._enabled:
      return
    print_mdl.DeleteProfiler()
    setattr(session.BaseSession, 'run', self.old_run)
    setattr(session.BaseSession, '__init__', self.old_init)
    setattr(session.BaseSession, '_profiler_run_internal', None)
    setattr(session.BaseSession, '_profiler_init_internal', None)
    setattr(session.BaseSession, 'profile_context', None)
