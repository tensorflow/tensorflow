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
    self.symbol_renames = renames_v2.renames
    # pylint: disable=line-too-long
    # Add additional renames not in renames_v2.py here.
    self.symbol_renames.update({
    })
    # pylint: enable=line-too-long

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

    # TODO(b/118888586): add default value change to update script.
    default_loss_reduction_changed = (
        "WARNING: default value of loss_reduction has been changed to "
        "SUM_OVER_BATCH_SIZE.\n"
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
        "tf.estimator.LinearClassifier": default_loss_reduction_changed,
    }
    # Right now we can't have both a rename and a warning.
    self.symbol_renames = {
        name: new_name for name, new_name in self.symbol_renames.items()
        if name not in self.function_warnings
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
