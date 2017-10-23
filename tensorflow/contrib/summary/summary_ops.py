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

"""Operations to emit summaries."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.summary import gen_summary_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.layers import utils
from tensorflow.python.ops import summary_op_util
from tensorflow.python.training import training_util

# Name for a collection which is expected to have at most a single boolean
# Tensor. If this tensor is True the summary ops will record summaries.
_SHOULD_RECORD_SUMMARIES_NAME = "ShouldRecordSummaries"


def should_record_summaries():
  """Returns boolean Tensor which is true if summaries should be recorded."""
  should_record_collection = ops.get_collection(_SHOULD_RECORD_SUMMARIES_NAME)
  if not should_record_collection:
    return False
  if len(should_record_collection) != 1:
    raise ValueError(
        "More than one tensor specified for whether summaries "
        "should be recorded: %s" % should_record_collection)
  return should_record_collection[0]


# TODO(apassos) consider how to handle local step here.
def record_summaries_every_n_global_steps(n):
  """Sets the should_record_summaries Tensor to true if global_step % n == 0."""
  collection_ref = ops.get_collection_ref(_SHOULD_RECORD_SUMMARIES_NAME)
  collection_ref[:] = [training_util.get_global_step() % n == 0]


def always_record_summaries():
  """Sets the should_record_summaries Tensor to always true."""
  collection_ref = ops.get_collection_ref(_SHOULD_RECORD_SUMMARIES_NAME)
  collection_ref[:] = [True]


def never_record_summaries():
  """Sets the should_record_summaries Tensor to always false."""
  collection_ref = ops.get_collection_ref(_SHOULD_RECORD_SUMMARIES_NAME)
  collection_ref[:] = [False]


def create_summary_file_writer(logdir,
                               max_queue=None,
                               flush_secs=None,
                               filename_suffix=None,
                               name=None):
  """Creates a summary file writer in the current context."""
  if max_queue is None:
    max_queue = constant_op.constant(10)
  if flush_secs is None:
    flush_secs = constant_op.constant(120)
  if filename_suffix is None:
    filename_suffix = constant_op.constant("")
  resource = gen_summary_ops.summary_writer(shared_name=name)
  gen_summary_ops.create_summary_file_writer(resource, logdir, max_queue,
                                             flush_secs, filename_suffix)
  context.context().summary_writer_resource = resource


def _nothing():
  """Convenient else branch for when summaries do not record."""
  return False


def summary_writer_function(name, tensor, function, family=None):
  """Helper function to write summaries.

  Args:
    name: name of the summary
    tensor: main tensor to form the summary
    function: function taking a tag and a scope which writes the summary
    family: optional, the summary's family

  Returns:
    The result of writing the summary.
  """
  def record():
    with summary_op_util.summary_scope(
        name, family, values=[tensor]) as (tag, scope):
      function(tag, scope)
      return True

  return utils.smart_cond(
      should_record_summaries(), record, _nothing, name="")


def generic(name, tensor, metadata, family=None):
  """Writes a tensor summary if possible."""

  def function(tag, scope):
    gen_summary_ops.write_summary(context.context().summary_writer_resource,
                                  training_util.get_global_step(), tensor,
                                  tag, metadata, name=scope)
  return summary_writer_function(name, tensor, function, family=family)


def scalar(name, tensor, family=None):
  """Writes a scalar summary if possible."""

  def function(tag, scope):
    gen_summary_ops.write_scalar_summary(
        context.context().summary_writer_resource,
        training_util.get_global_step(), tag, tensor, name=scope)

  return summary_writer_function(name, tensor, function, family=family)


def histogram(name, tensor, family=None):
  """Writes a histogram summary if possible."""

  def function(tag, scope):
    gen_summary_ops.write_histogram_summary(
        context.context().summary_writer_resource,
        training_util.get_global_step(), tag, tensor, name=scope)

  return summary_writer_function(name, tensor, function, family=family)


def image(name, tensor, bad_color=None, max_images=3, family=None):
  """Writes an image summary if possible."""

  def function(tag, scope):
    if bad_color is None:
      bad_color_ = constant_op.constant([255, 0, 0, 255], dtype=dtypes.uint8)
    gen_summary_ops.write_image_summary(
        context.context().summary_writer_resource,
        training_util.get_global_step(), tag, tensor, bad_color_, max_images,
        name=scope)

  return summary_writer_function(name, tensor, function, family=family)


def audio(name, tensor, sample_rate, max_outputs, family=None):
  """Writes an audio summary if possible."""

  def function(tag, scope):
    gen_summary_ops.write_audio_summary(
        context.context().summary_writer_resource,
        training_util.get_global_step(),
        tag,
        tensor,
        sample_rate=sample_rate,
        max_outputs=max_outputs,
        name=scope)

  return summary_writer_function(name, tensor, function, family=family)
