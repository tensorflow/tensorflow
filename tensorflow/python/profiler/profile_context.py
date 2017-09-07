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

import os
import threading

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import errors
from tensorflow.python.platform import gfile
from tensorflow.python.profiler import model_analyzer
from tensorflow.python.profiler import tfprof_logger


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
  self.profile_context._new_step()
  # Fast path if no need for profiling.
  to_profiles = self.profile_context._profile_candidates()
  to_dumps = self.profile_context._dump_candidates()
  if (not to_profiles and not to_dumps and
      not self.profile_context._is_capture_enforced()):
    return self._profiler_run_internal(
        fetches, feed_dict, options, run_metadata)

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

  ret = self._profiler_run_internal(fetches, feed_dict, options, run_metadata)

  if self.profile_context._capture_next_step:
    self.profile_context._add_run_meta(run_metadata)

  for to_dump in to_dumps:
    outdir, _ = to_dump
    if not gfile.Exists(outdir):
      gfile.MakeDirs(outdir)
    with gfile.Open(os.path.join(outdir, 'graph.pbtxt'), 'w') as f:
      f.write('%s' % self.graph.as_graph_def(add_shapes=True))
    with gfile.Open(os.path.join(outdir, 'run_metadata'), 'w') as f:
      f.write(run_metadata.SerializeToString())
    tfprof_logger.write_op_log(
        self.graph, outdir, run_meta=run_metadata, add_trace=True)

  for to_prof in to_profiles:
    cmd, opts, _ = to_prof
    model_analyzer.profile(
        self.graph, run_meta=run_metadata, cmd=cmd, options=opts)

  # Restore to default.
  options.trace_level = old_trace_level
  return ret
  # pylint: enable=protected-access


class ProfileContext(object):
  """A Context that captures RunMetadata and performs profiling.

  ```python
    # Auto profiling at step 1, 100 and 1000.:
    with tf.contrib.tfprof.ProfileContext() as pctx:
      # Create the profiling options.
      opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
      # Run profiling at certain steps. Multiple ones can be added.
      pctx.add_auto_profiling('op', opts, [1, 100, 1000])
      # Or dump the profile files at certain steps.
      pctx.add_auto_profile_dump('/tmp/profiles', [1000])
      # Run train/eval loop.
      train_loop().

    # Alternatively, enable and capture RunMetadata of next step.
    with tf.contrib.tfprof.ProfileContext() as pctx:
      pctx.capture_next_run_meta()
      opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
      _ = session.run(train_op)
      tf.profiler.profile(session.graph,
                          run_meta=pctx.run_meta(),
                          cmd='op',
                          options=opts)
  ```
  """

  def __init__(self):
    self._lock = threading.Lock()
    self._capture_next_step = False
    self._step = 0
    self._auto_profiles = []
    self._auto_dumps = []
    self._run_meta = None

  def add_auto_profiling(self, cmd, profile_options, profile_steps):
    """Runs profiling at some steps with provided command and options.

    Args:
      cmd: The profiling commands.
      profile_options: The profiling options.
      profile_steps: A list/set of integers. The profiling command and options
          will be run automatically at these integer steps. Each step is
          a session.run.
    """
    with self._lock:
      self._auto_profiles.append((cmd, profile_options, profile_steps))

  def add_auto_profile_dump(self, outdir, dump_steps):
    """Dumps profiles at some steps to the directory.

    Args:
      outdir: The directory to dump the profile files.
      dump_steps: A list/set of integers. The profile files will be dump at
          these integer steps. Each step is a session.run.
    """
    with self._lock:
      self._auto_dumps.append((outdir, dump_steps))

  def capture_next_run_meta(self):
    """Enables tracing and captures RunMetadata at next session.run.

      The captured RunMetadata can be retrieved via run_meta(). It
      will be cleared one step later.
    """
    with self._lock:
      self._capture_next_step = True

  def run_meta(self):
    """Returns the RunMetadata captured at previous session.run.

      Needs to call capture_next_run_meta() before session.run to enable
      capturing.
    """
    with self._lock:
      assert self._run_meta, 'Need to call capture_next_run_meta()'
      return self._run_meta

  def _is_capture_enforced(self):
    with self._lock:
      return self._capture_next_step

  def _add_run_meta(self, run_meta):
    with self._lock:
      self._run_meta = run_meta
      self._capture_next_step = False

  def _new_step(self):
    with self._lock:
      self._run_meta = None
      self._step += 1

  def _profile_candidates(self):
    to_profile = []
    with self._lock:
      for auto_prof in self._auto_profiles:
        _, _, prof_steps = auto_prof
        if self._step - 1 in prof_steps:
          to_profile.append(auto_prof)
    return to_profile

  def _dump_candidates(self):
    to_dump = []
    with self._lock:
      for auto_dump in self._auto_dumps:
        _, dump_steps = auto_dump
        if self._step - 1 in dump_steps:
          to_dump.append(auto_dump)
    return to_dump

  def __enter__(self):
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

  def __exit__(self, exec_type, exec_value, exec_tb):
    setattr(session.BaseSession, 'run', self.old_run)
    setattr(session.BaseSession, '__init__', self.old_init)
    setattr(session.BaseSession, '_profiler_run_internal', None)
    setattr(session.BaseSession, '_profiler_init_internal', None)
    setattr(session.BaseSession, 'profile_context', None)
