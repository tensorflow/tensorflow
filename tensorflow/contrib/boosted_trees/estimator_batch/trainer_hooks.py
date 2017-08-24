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
# ==============================================================================
"""Hooks for use with GTFlow Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.contrib.learn.python.learn import session_run_hook
from tensorflow.contrib.learn.python.learn.session_run_hook import SessionRunArgs
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import training_util
from tensorflow.python.training.summary_io import SummaryWriterCache


class FeatureImportanceSummarySaver(session_run_hook.SessionRunHook):
  """Hook to save feature importance summaries."""

  def __init__(self, model_dir, every_n_steps=1):
    """Create a FeatureImportanceSummarySaver Hook.

    This hook creates scalar summaries representing feature importance
    for each feature column during training.

    Args:
      model_dir: model base output directory.
      every_n_steps: frequency, in number of steps, for logging summaries.

    Raises:
      ValueError: If one of the arguments is invalid.
    """
    if model_dir is None:
      raise ValueError("model dir must be specified.")
    self._model_dir = model_dir
    self._every_n_steps = every_n_steps
    self._last_triggered_step = None

  def begin(self):
    self._global_step_tensor = training_util.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use FeatureImportanceSummarySaver.")
    graph = ops.get_default_graph()
    self._feature_names_tensor = graph.get_tensor_by_name(
        "gbdt/feature_names:0")
    self._feature_usage_counts_tensor = graph.get_tensor_by_name(
        "gbdt/feature_usage_counts:0")
    self._feature_gains_tensor = graph.get_tensor_by_name(
        "gbdt/feature_gains:0")

  def before_run(self, run_context):
    del run_context  # Unused by feature importance summary saver hook.
    requests = {
        "global_step": self._global_step_tensor,
        "feature_names": self._feature_names_tensor,
        "feature_usage_counts": self._feature_usage_counts_tensor,
        "feature_gains": self._feature_gains_tensor
    }
    return SessionRunArgs(requests)

  def after_run(self, run_context, run_values):
    del run_context  # Unused by feature importance summary saver hook.

    # Read result tensors.
    global_step = run_values.results["global_step"]
    feature_names = run_values.results["feature_names"]
    feature_usage_counts = run_values.results["feature_usage_counts"]
    feature_gains = run_values.results["feature_gains"]

    # Ensure summaries are logged at desired frequency
    if (self._last_triggered_step is not None and
        global_step < self._last_triggered_step + self._every_n_steps):
      return

    # Validate tensors.
    if (len(feature_names) != len(feature_usage_counts) or
        len(feature_names) != len(feature_gains)):
      raise RuntimeError(
          "Feature names and importance measures have inconsistent lengths.")

    # Compute total usage.
    total_usage_count = 0.0
    for usage_count in feature_usage_counts:
      total_usage_count += usage_count
    usage_count_norm = 1.0 / total_usage_count if total_usage_count else 1.0

    # Compute total gain.
    total_gain = 0.0
    for gain in feature_gains:
      total_gain += gain
    gain_norm = 1.0 / total_gain if total_gain else 1.0

    # Output summary for each feature.
    self._last_triggered_step = global_step
    for (name, usage_count, gain) in zip(feature_names, feature_usage_counts,
                                         feature_gains):
      output_dir = os.path.join(self._model_dir, name.decode("utf-8"))
      summary_writer = SummaryWriterCache.get(output_dir)
      usage_count_summary = Summary(value=[
          Summary.Value(
              tag="feature_importance/usage_counts",
              simple_value=usage_count)
      ])
      usage_fraction_summary = Summary(value=[
          Summary.Value(
              tag="feature_importance/usage_fraction",
              simple_value=usage_count * usage_count_norm)
      ])
      summary_writer.add_summary(usage_count_summary, global_step)
      summary_writer.add_summary(usage_fraction_summary, global_step)
      gains_summary = Summary(
          value=[Summary.Value(
              tag="feature_importance/gains",
              simple_value=gain)])
      gains_fraction_summary = Summary(
          value=[Summary.Value(
              tag="feature_importance/gains_fraction",
              simple_value=gain * gain_norm)])
      summary_writer.add_summary(gains_summary, global_step)
      summary_writer.add_summary(gains_fraction_summary, global_step)


class FeedFnHook(session_run_hook.SessionRunHook):
  """Runs feed_fn and sets the feed_dict accordingly."""

  def __init__(self, feed_fn):
    self.feed_fn = feed_fn

  def before_run(self, run_context):
    del run_context  # unused by FeedFnHook.
    return session_run_hook.SessionRunArgs(
        fetches=None, feed_dict=self.feed_fn)


class StopAfterNTrees(session_run_hook.SessionRunHook):
  """Stop training after building N full trees."""

  def __init__(self, n, num_attempted_trees_tensor, num_finalized_trees_tensor):
    self._num_trees = n
    # num_attempted_trees_tensor and num_finalized_trees_tensor are both
    # tensors.
    self._num_attempted_trees_tensor = num_attempted_trees_tensor
    self._num_finalized_trees_tensor = num_finalized_trees_tensor

  def before_run(self, run_context):
    del run_context  # unused by StopTrainingAfterNTrees.
    return session_run_hook.SessionRunArgs({
        "num_attempted_trees": self._num_attempted_trees_tensor,
        "num_finalized_trees": self._num_finalized_trees_tensor,
    })

  def after_run(self, run_context, run_values):
    num_attempted_trees = run_values.results["num_attempted_trees"]
    num_finalized_trees = run_values.results["num_finalized_trees"]
    assert num_attempted_trees is not None
    assert num_finalized_trees is not None
    if (num_finalized_trees >= self._num_trees or
        num_attempted_trees > self._num_trees):
      logging.info("Requesting stop since we have reached %d trees.",
                   num_finalized_trees)
      run_context.request_stop()
