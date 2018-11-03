# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Upgrader for Python scripts from 1.* TensorFlow to 2.0 TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import renames_v2


class TFAPIChangeSpec(ast_edits.APIChangeSpec):
  """List of maps that describe what changed in the API."""

  def __init__(self):
    # Maps from a function name to a dictionary that describes how to
    # map from an old argument keyword to the new argument keyword.
    self.function_keyword_renames = {}

    # Mapping from function to the new name of the function
    self.function_renames = renames_v2.renames
    # pylint: disable=line-too-long
    self.function_renames.update({
        "tf.FixedLengthRecordReader": "tf.compat.v1.FixedLengthRecordReader",
        "tf.IdentityReader": "tf.compat.v1.IdentityReader",
        "tf.LMDBReader": "tf.compat.v1.LMDBReader",
        "tf.ReaderBase": "tf.compat.v1.ReaderBase",
        "tf.TFRecordReader": "tf.compat.v1.TFRecordReader",
        "tf.TextLineReader": "tf.compat.v1.TextLineReader",
        "tf.WholeFileReader": "tf.compat.v1.WholeFileReader",
        "tf.saved_model.builder.SavedModelBuilder": "tf.compat.v1.saved_model.Builder",
        "tf.saved_model.loader.load": "tf.compat.v1.saved_model.load",
        "tf.saved_model.main_op.main_op": "tf.compat.v1.saved_model.main_op",
        "tf.saved_model.main_op.main_op_with_restore": "tf.compat.v1.saved_model.main_op_with_restore",
        "tf.saved_model.simple_save": "tf.compat.v1.saved_model.simple_save",
        "tf.saved_model.utils.build_tensor_info": "tf.compat.v1.saved_model.build_tensor_info",
        "tf.saved_model.utils.get_tensor_from_tensor_info": "tf.compat.v1.saved_model.get_tensor_from_tensor_info",
        "tf.train.QueueRunner": "tf.compat.v1.QueueRunner",
        "tf.train.add_queue_runner": "tf.compat.v1.add_queue_runner",
        "tf.train.batch": "tf.compat.v1.train.batch",
        "tf.train.batch_join": "tf.compat.v1.train.batch_join",
        "tf.train.input_producer": "tf.compat.v1.train.input_producer",
        "tf.train.limit_epochs": "tf.compat.v1.train.limit_epochs",
        "tf.train.maybe_batch": "tf.compat.v1.train.maybe_batch",
        "tf.train.maybe_batch_join": "tf.compat.v1.train.maybe_batch_join",
        "tf.train.maybe_shuffle_batch": "tf.compat.v1.train.maybe_shuffle_batch",
        "tf.train.maybe_shuffle_batch_join": "tf.compat.v1.train.maybe_shuffle_batch_join",
        "tf.train.queue_runner.QueueRunner": "tf.compat.v1.queue_runner.QueueRunner",
        "tf.train.queue_runner.add_queue_runner": "tf.compat.v1.queue_runner.add_queue_runner",
        "tf.train.queue_runner.start_queue_runners": "tf.compat.v1.queue_runner.start_queue_runners",
        "tf.train.range_input_producer": "tf.compat.v1.train.range_input_producer",
        "tf.train.shuffle_batch": "tf.compat.v1.train.shuffle_batch",
        "tf.train.shuffle_batch_join": "tf.compat.v1.train.shuffle_batch_join",
        "tf.train.slice_input_producer": "tf.compat.v1.train.slice_input_producer",
        "tf.train.string_input_producer": "tf.compat.v1.train.string_input_producer",
        "tf.train.start_queue_runners": "tf.compat.v1.start_queue_runners",
    })
    # pylint: enable=line-too-long

    # TODO(amitpatankar): Fix the function rename script
    # to handle constants without hardcoding.
    self.function_renames["QUANTIZED_DTYPES"] = "dtypes.QUANTIZED_DTYPES"

    # Variables that should be changed to functions.
    self.change_to_function = {}

    # Functions that were reordered should be changed to the new keyword args
    # for safety, if positional arguments are used. If you have reversed the
    # positional arguments yourself, this could do the wrong thing.
    self.function_reorders = {}

    # Specially handled functions.
    self.function_handle = {}

    decay_function_comment = (
        "ERROR: <function name> has been changed to return a callable instead "
        "of a tensor when graph building, but its functionality remains "
        "unchanged during eager execution (returns a callable like "
        "before). The converter cannot detect and fix this reliably, so "
        "you need to inspect this usage manually.\n"
    )

    # Function warnings. <function name> placeholder inside warnings will be
    # replaced by function name.
    self.function_warnings = {
        "tf.train.exponential_decay": decay_function_comment,
        "tf.train.piecewise_constant": decay_function_comment,
        "tf.train.polynomial_decay": decay_function_comment,
        "tf.train.natural_exp_decay": decay_function_comment,
        "tf.train.inverse_time_decay": decay_function_comment,
        "tf.train.cosine_decay": decay_function_comment,
        "tf.train.cosine_decay_restarts": decay_function_comment,
        "tf.train.linear_cosine_decay": decay_function_comment,
        "tf.train.noisy_linear_cosine_decay": decay_function_comment,
    }


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      formatter_class=argparse.RawDescriptionHelpFormatter,
      description="""Convert a TensorFlow Python file to 2.0

Simple usage:
  tf_convert_v2.py --infile foo.py --outfile bar.py
  tf_convert_v2.py --intree ~/code/old --outtree ~/code/new
""")
  parser.add_argument(
      "--infile",
      dest="input_file",
      help="If converting a single file, the name of the file "
      "to convert")
  parser.add_argument(
      "--outfile",
      dest="output_file",
      help="If converting a single file, the output filename.")
  parser.add_argument(
      "--intree",
      dest="input_tree",
      help="If converting a whole tree of files, the directory "
      "to read from (relative or absolute).")
  parser.add_argument(
      "--outtree",
      dest="output_tree",
      help="If converting a whole tree of files, the output "
      "directory (relative or absolute).")
  parser.add_argument(
      "--copyotherfiles",
      dest="copy_other_files",
      help=("If converting a whole tree of files, whether to "
            "copy the other files."),
      type=bool,
      default=False)
  parser.add_argument(
      "--reportfile",
      dest="report_filename",
      help=("The name of the file where the report log is "
            "stored."
            "(default: %(default)s)"),
      default="report.txt")
  args = parser.parse_args()

  upgrade = ast_edits.ASTCodeUpgrader(TFAPIChangeSpec())
  report_text = None
  report_filename = args.report_filename
  files_processed = 0
  if args.input_file:
    if not args.output_file:
      raise ValueError(
          "--outfile=<output file> argument is required when converting a "
          "single file.")
    files_processed, report_text, errors = upgrade.process_file(
        args.input_file, args.output_file)
    files_processed = 1
  elif args.input_tree:
    if not args.output_tree:
      raise ValueError(
          "--outtree=<output directory> argument is required when converting a "
          "file tree.")
    files_processed, report_text, errors = upgrade.process_tree(
        args.input_tree, args.output_tree, args.copy_other_files)
  else:
    parser.print_help()
  if report_text:
    open(report_filename, "w").write(report_text)
    print("TensorFlow 2.0 Upgrade Script")
    print("-----------------------------")
    print("Converted %d files\n" % files_processed)
    print("Detected %d errors that require attention" % len(errors))
    print("-" * 80)
    print("\n".join(errors))
    print("\nMake sure to read the detailed log %r\n" % report_filename)
