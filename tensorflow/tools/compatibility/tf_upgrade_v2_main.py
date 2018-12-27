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
from tensorflow.tools.compatibility import tf_upgrade_v2


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.RawDescriptionHelpFormatter,
      description="""Convert a TensorFlow Python file to 2.0

Simple usage:
  tf_upgrade_v2.py --infile foo.py --outfile bar.py
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
      "--reportfile",
      dest="report_filename",
      help=("The name of the file where the report log is "
            "stored."
            "(default: %(default)s)"),
      default="report.txt")
  args = parser.parse_args()

  upgrade = ast_edits.ASTCodeUpgrader(tf_upgrade_v2.TFAPIChangeSpec())
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


if __name__ == "__main__":
  main()
