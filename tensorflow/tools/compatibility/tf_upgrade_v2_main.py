# Lint as: python2, python3
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
"""Upgrader for Python scripts from 1.x TensorFlow to 2.0 TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import six

from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import ipynb
from tensorflow.tools.compatibility import tf_upgrade_v2
from tensorflow.tools.compatibility import tf_upgrade_v2_safety

# Make straightforward changes to convert to 2.0. In harder cases,
# use compat.v1.
_DEFAULT_MODE = "DEFAULT"

# Convert to use compat.v1.
_SAFETY_MODE = "SAFETY"

# Whether to rename to compat.v2
_IMPORT_RENAME_DEFAULT = False


def process_file(in_filename, out_filename, upgrader):
  """Process a file of type `.py` or `.ipynb`."""

  if six.ensure_str(in_filename).endswith(".py"):
    files_processed, report_text, errors = \
      upgrader.process_file(in_filename, out_filename)
  elif six.ensure_str(in_filename).endswith(".ipynb"):
    files_processed, report_text, errors = \
      ipynb.process_file(in_filename, out_filename, upgrader)
  else:
    raise NotImplementedError(
        "Currently converter only supports python or ipynb")

  return files_processed, report_text, errors


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.RawDescriptionHelpFormatter,
      description="""Convert a TensorFlow Python file from 1.x to 2.0

Simple usage:
  tf_upgrade_v2.py --infile foo.py --outfile bar.py
  tf_upgrade_v2.py --infile foo.ipynb --outfile bar.ipynb
  tf_upgrade_v2.py --intree ~/code/old --outtree ~/code/new
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
      default=True)
  parser.add_argument(
      "--inplace",
      dest="in_place",
      help=("If converting a set of files, whether to "
            "allow the conversion to be performed on the "
            "input files."),
      action="store_true")
  parser.add_argument(
      "--no_import_rename",
      dest="no_import_rename",
      help=("Not to rename import to compat.v2 explicitly."),
      action="store_true")
  parser.add_argument(
      "--no_upgrade_compat_v1_import",
      dest="no_upgrade_compat_v1_import",
      help=("If specified, don't upgrade explicit imports of "
            "`tensorflow.compat.v1 as tf` to the v2 apis. Otherwise, "
            "explicit imports of  the form `tensorflow.compat.v1 as tf` will "
            "be upgraded."),
      action="store_true")
  parser.add_argument(
      "--reportfile",
      dest="report_filename",
      help=("The name of the file where the report log is "
            "stored."
            "(default: %(default)s)"),
      default="report.txt")
  parser.add_argument(
      "--mode",
      dest="mode",
      choices=[_DEFAULT_MODE, _SAFETY_MODE],
      help=("Upgrade script mode. Supported modes:\n"
            "%s: Perform only straightforward conversions to upgrade to "
            "2.0. In more difficult cases, switch to use compat.v1.\n"
            "%s: Keep 1.* code intact and import compat.v1 "
            "module." %
            (_DEFAULT_MODE, _SAFETY_MODE)),
      default=_DEFAULT_MODE)
  parser.add_argument(
      "--print_all",
      dest="print_all",
      help="Print full log to stdout instead of just printing errors",
      action="store_true")
  args = parser.parse_args()

  if args.mode == _SAFETY_MODE:
    change_spec = tf_upgrade_v2_safety.TFAPIChangeSpec()
  else:
    if args.no_import_rename:
      change_spec = tf_upgrade_v2.TFAPIChangeSpec(
          import_rename=False,
          upgrade_compat_v1_import=not args.no_upgrade_compat_v1_import)
    else:
      change_spec = tf_upgrade_v2.TFAPIChangeSpec(
          import_rename=_IMPORT_RENAME_DEFAULT,
          upgrade_compat_v1_import=not args.no_upgrade_compat_v1_import)
  upgrade = ast_edits.ASTCodeUpgrader(change_spec)

  report_text = None
  report_filename = args.report_filename
  files_processed = 0
  if args.input_file:
    if not args.in_place and not args.output_file:
      raise ValueError(
          "--outfile=<output file> argument is required when converting a "
          "single file.")
    if args.in_place and args.output_file:
      raise ValueError(
          "--outfile argument is invalid when when converting in place")
    output_file = args.input_file if args.in_place else args.output_file
    files_processed, report_text, errors = process_file(
        args.input_file, output_file, upgrade)
    errors = {args.input_file: errors}
    files_processed = 1
  elif args.input_tree:
    if not args.in_place and not args.output_tree:
      raise ValueError(
          "--outtree=<output directory> argument is required when converting a "
          "file tree.")
    if args.in_place and args.output_tree:
      raise ValueError(
          "--outtree argument is invalid when when converting in place")
    output_tree = args.input_tree if args.in_place else args.output_tree
    files_processed, report_text, errors = upgrade.process_tree(
        args.input_tree, output_tree, args.copy_other_files)
  else:
    parser.print_help()
  if report_text:
    num_errors = 0
    report = []
    for f in errors:
      if errors[f]:
        num_errors += len(errors[f])
        report.append(six.ensure_str("-" * 80) + "\n")
        report.append("File: %s\n" % f)
        report.append(six.ensure_str("-" * 80) + "\n")
        report.append("\n".join(errors[f]) + "\n")

    report = ("TensorFlow 2.0 Upgrade Script\n"
              "-----------------------------\n"
              "Converted %d files\n" % files_processed +
              "Detected %d issues that require attention" % num_errors + "\n" +
              six.ensure_str("-" * 80) + "\n") + "".join(report)
    detailed_report_header = six.ensure_str("=" * 80) + "\n"
    detailed_report_header += "Detailed log follows:\n\n"
    detailed_report_header += six.ensure_str("=" * 80) + "\n"

    with open(report_filename, "w") as report_file:
      report_file.write(report)
      report_file.write(detailed_report_header)
      report_file.write(six.ensure_str(report_text))

    if args.print_all:
      print(report)
      print(detailed_report_header)
      print(report_text)
    else:
      print(report)
    print("\nMake sure to read the detailed log %r\n" % report_filename)

if __name__ == "__main__":
  main()
