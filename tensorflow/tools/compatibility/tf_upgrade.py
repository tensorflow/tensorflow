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
"""Upgrader for Python scripts from pre-1.0 TensorFlow to 1.0 TensorFlow."""

import argparse

import six

from tensorflow.tools.compatibility import ast_edits


class TFAPIChangeSpec(ast_edits.APIChangeSpec):
  """List of maps that describe what changed in the API."""

  def __init__(self):
    # Maps from a function name to a dictionary that describes how to
    # map from an old argument keyword to the new argument keyword.
    self.function_keyword_renames = {
        "tf.batch_matmul": {
            "adj_x": "adjoint_a",
            "adj_y": "adjoint_b",
        },
        "tf.count_nonzero": {
            "reduction_indices": "axis"
        },
        "tf.reduce_all": {
            "reduction_indices": "axis"
        },
        "tf.reduce_any": {
            "reduction_indices": "axis"
        },
        "tf.reduce_max": {
            "reduction_indices": "axis"
        },
        "tf.reduce_mean": {
            "reduction_indices": "axis"
        },
        "tf.reduce_min": {
            "reduction_indices": "axis"
        },
        "tf.reduce_prod": {
            "reduction_indices": "axis"
        },
        "tf.reduce_sum": {
            "reduction_indices": "axis"
        },
        "tf.reduce_logsumexp": {
            "reduction_indices": "axis"
        },
        "tf.expand_dims": {
            "dim": "axis"
        },
        "tf.argmax": {
            "dimension": "axis"
        },
        "tf.argmin": {
            "dimension": "axis"
        },
        "tf.reduce_join": {
            "reduction_indices": "axis"
        },
        "tf.sparse_concat": {
            "concat_dim": "axis"
        },
        "tf.sparse_split": {
            "split_dim": "axis"
        },
        "tf.sparse_reduce_sum": {
            "reduction_axes": "axis"
        },
        "tf.reverse_sequence": {
            "seq_dim": "seq_axis",
            "batch_dim": "batch_axis"
        },
        "tf.sparse_reduce_sum_sparse": {
            "reduction_axes": "axis"
        },
        "tf.squeeze": {
            "squeeze_dims": "axis"
        },
        "tf.split": {
            "split_dim": "axis",
            "num_split": "num_or_size_splits"
        },
        "tf.concat": {
            "concat_dim": "axis"
        },
    }

    # Mapping from function to the new name of the function
    self.symbol_renames = {
        "tf.inv": "tf.reciprocal",
        "tf.contrib.deprecated.scalar_summary": "tf.summary.scalar",
        "tf.contrib.deprecated.histogram_summary": "tf.summary.histogram",
        "tf.listdiff": "tf.setdiff1d",
        "tf.list_diff": "tf.setdiff1d",
        "tf.mul": "tf.multiply",
        "tf.neg": "tf.negative",
        "tf.sub": "tf.subtract",
        "tf.train.SummaryWriter": "tf.summary.FileWriter",
        "tf.scalar_summary": "tf.summary.scalar",
        "tf.histogram_summary": "tf.summary.histogram",
        "tf.audio_summary": "tf.summary.audio",
        "tf.image_summary": "tf.summary.image",
        "tf.merge_summary": "tf.summary.merge",
        "tf.merge_all_summaries": "tf.summary.merge_all",
        "tf.image.per_image_whitening": "tf.image.per_image_standardization",
        "tf.all_variables": "tf.global_variables",
        "tf.VARIABLES": "tf.GLOBAL_VARIABLES",
        "tf.initialize_all_variables": "tf.global_variables_initializer",
        "tf.initialize_variables": "tf.variables_initializer",
        "tf.initialize_local_variables": "tf.local_variables_initializer",
        "tf.batch_matrix_diag": "tf.matrix_diag",
        "tf.batch_band_part": "tf.band_part",
        "tf.batch_set_diag": "tf.set_diag",
        "tf.batch_matrix_transpose": "tf.matrix_transpose",
        "tf.batch_matrix_determinant": "tf.matrix_determinant",
        "tf.batch_matrix_inverse": "tf.matrix_inverse",
        "tf.batch_cholesky": "tf.cholesky",
        "tf.batch_cholesky_solve": "tf.cholesky_solve",
        "tf.batch_matrix_solve": "tf.matrix_solve",
        "tf.batch_matrix_triangular_solve": "tf.matrix_triangular_solve",
        "tf.batch_matrix_solve_ls": "tf.matrix_solve_ls",
        "tf.batch_self_adjoint_eig": "tf.self_adjoint_eig",
        "tf.batch_self_adjoint_eigvals": "tf.self_adjoint_eigvals",
        "tf.batch_svd": "tf.svd",
        "tf.batch_fft": "tf.fft",
        "tf.batch_ifft": "tf.ifft",
        "tf.batch_fft2d": "tf.fft2d",
        "tf.batch_ifft2d": "tf.ifft2d",
        "tf.batch_fft3d": "tf.fft3d",
        "tf.batch_ifft3d": "tf.ifft3d",
        "tf.select": "tf.where",
        "tf.complex_abs": "tf.abs",
        "tf.batch_matmul": "tf.matmul",
        "tf.pack": "tf.stack",
        "tf.unpack": "tf.unstack",
        "tf.op_scope": "tf.name_scope",
    }

    self.change_to_function = {
        "tf.ones_initializer",
        "tf.zeros_initializer",
    }

    # Functions that were reordered should be changed to the new keyword args
    # for safety, if positional arguments are used. If you have reversed the
    # positional arguments yourself, this could do the wrong thing.
    self.function_reorders = {
        "tf.split": ["axis", "num_or_size_splits", "value", "name"],
        "tf.sparse_split": ["axis", "num_or_size_splits", "value", "name"],
        "tf.concat": ["concat_dim", "values", "name"],
        "tf.svd": ["tensor", "compute_uv", "full_matrices", "name"],
        "tf.nn.softmax_cross_entropy_with_logits": [
            "logits", "labels", "dim", "name"
        ],
        "tf.nn.sparse_softmax_cross_entropy_with_logits": [
            "logits", "labels", "name"
        ],
        "tf.nn.sigmoid_cross_entropy_with_logits": ["logits", "labels", "name"],
        "tf.op_scope": ["values", "name", "default_name"],
    }

    # Warnings that should be printed if corresponding functions are used.
    self.function_warnings = {
        "tf.reverse": (
            ast_edits.ERROR,
            "tf.reverse has had its argument semantics changed "
            "significantly. The converter cannot detect this reliably, so "
            "you need to inspect this usage manually.\n"),
    }

    self.module_deprecations = {}


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      formatter_class=argparse.RawDescriptionHelpFormatter,
      description="""Convert a TensorFlow Python file to 1.0

Simple usage:
  tf_convert.py --infile foo.py --outfile bar.py
  tf_convert.py --intree ~/code/old --outtree ~/code/new
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
    files_processed, report_text, errors = upgrade.process_file(
        args.input_file, args.output_file)
    files_processed = 1
  elif args.input_tree:
    files_processed, report_text, errors = upgrade.process_tree(
        args.input_tree, args.output_tree, args.copy_other_files)
  else:
    parser.print_help()
  if report_text:
    open(report_filename, "w").write(six.ensure_str(report_text))
    print("TensorFlow 1.0 Upgrade Script")
    print("-----------------------------")
    print("Converted %d files\n" % files_processed)
    print("Detected %d errors that require attention" % len(errors))
    print("-" * 80)
    print("\n".join(errors))
    print("\nMake sure to read the detailed log %r\n" % report_filename)
