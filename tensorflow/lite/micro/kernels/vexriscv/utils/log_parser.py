# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
r"""Analyze function call stack from GDB or Renode

See README for detail usage

Example usage:

python log_parser.py profile.txt --regex=gdb_regex.json --visualize --top=7

* To add a title in the graph, use the optional argument --title to set it

Example usage:

python log_parser.py profile.txt --regex=gdb_regex.json \
--visualize --top=7 --title=magic_wand

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import os
import re
import matplotlib.pyplot as plt


def readlines(filename):
  """
  Arg:
    filename(str):

  Return:
    (list of str):
  """
  with open(filename, "r") as f:
    content = f.read().splitlines()

  return content


def writelines(data, filename):
  # Write parsed log to file
  with open(filename, "w") as f:
    for line in data:
      f.write(line + "\n")


def load_regex_parser(filename):
  """
  Arg:
    filename: string for the input json file containing regex
  """
  assert filename is not None

  with open(filename, "r") as f:
    content = json.load(f)

  regex_parser = {}
  for key, val in content.items():
    if isinstance(val, list):
      regexs = []
      for pattern in val:
        regexs.append(re.compile(pattern))

      regex_parser[key] = regexs
    else:
      regex_parser[key] = re.compile(val)

  return regex_parser


def gdb_log_parser(data, output, re_file, ignore_list=None, full_trace=False):
  """
  Args:
    data: list of strings of logs from GDB
    output: string of output filename
    re_file: path to the regex *.json file
    ignore_list: list of string (functions) to ignore
    full_trace: bool to generate full stack trace of the log
  """
  regex_parser = load_regex_parser(re_file)

  trace = collections.defaultdict(list)
  stack = []
  processed = []
  for line in data:
    # Skip invalid lines
    if not line.startswith("#"):
      continue

    # Skip redundant lines
    if not full_trace and not line.startswith("#0"):
      continue

    # Remove ANSI color symbols
    # line = ANSI_CLEANER.sub("", line)
    line = regex_parser["base"].sub("", line)

    # Extract function names with regex
    find = None
    for r in regex_parser["custom"]:
      find = r.findall(line)

      if len(find) != 0:
        break

    if find is None or len(find) == 0:
      continue

    # Extract content from `re.findall` results
    target = find[0][0] if isinstance(find[0], tuple) else find[0]

    # Extract function name from `$ADDR in $NAME`, e.g.
    # `0x40002998 in __addsf3` -> `__addsf3`
    if " in " in target:
      target = target.split()[-1]

    # Remove leading/trailing spaces
    target = target.strip()

    if full_trace:
      if line.startswith("#0") and stack:
        # Encode the trace to string
        temp = "/".join(stack)
        trace[stack[0]].append(temp)

        # Clear up previous stack
        stack.clear()

      stack.append(target)

    if not line.startswith("#0"):
      continue

    if ignore_list and target in ignore_list:
      continue

    # Strip the string before adding into parsed list
    processed.append(target)

  print("Extracted {} lines".format(len(processed)))

  # Write parsed log to file
  writelines(processed, output)

  if full_trace:
    content = {}
    for top, paths in trace.items():
      content[top] = []
      counter = collections.Counter(paths)

      for path, counts in counter.items():
        info = {"counts": counts, "path": path.split("/")}
        content[top].append(info)

    name = os.path.splitext(output)[0]
    with open(name + ".json", "w") as f:
      json.dump(content, f, sort_keys=True, indent=4)

  print("Parsed the log to `{}`".format(output))


def renode_log_parser(data, output, ignore_list=None):
  """
  Args:
    data: list of strings of logs from Renode
    output: string of output filename
    ignore_list: list of string (functions) to ignore
  """
  message = "Entering function"
  extractor = re.compile(r"{} (.*) at".format(message))

  ignore_count = 0
  processed = []
  for idx, line in enumerate(data):
    print("Processing {:.2f}%".format((idx + 1) / len(data) * 100.), end="\r")

    if message not in line:
      continue

    find = extractor.findall(line)

    # Skip invalid find or unnamed functions
    if len(find) == 0 or len(find[0].split()) == 0:
      continue

    entry = find[0].split()[0]

    if ignore_list and entry in ignore_list:
      ignore_count += 1
      continue

    processed.append(entry)

  print("Extracted {} lines ({:.2f}%); {} lines are ignored ({:.2f}%)".format(
      len(processed),
      len(processed) / len(data) * 100., ignore_count,
      ignore_count / len(data) * 100.))

  # Write parsed log to file
  writelines(processed, output)

  print("Parsed the log to `{}`".format(output))


def parse_log(filename,
              output=None,
              re_file=None,
              source="gdb",
              ignore=None,
              full_trace=False):
  """
  Args:
    filename(str)
    output(str)
  """
  data = readlines(filename)
  print("Raw log: {} lines".format(len(data)))

  ignore_list = None
  if ignore is not None:
    ignore_list = set(readlines(ignore))
    print("* {} patterns in the ignore list".format(len(ignore_list)))

  name, ext = None, None
  if output is None:
    name, ext = os.path.splitext(filename)
    output = "{}-parsed{}".format(name, ext)

  if source == "gdb":
    gdb_log_parser(data, output, re_file, ignore_list, full_trace)
  elif source == "renode":
    renode_log_parser(data, output, ignore_list=ignore_list)
  else:
    raise NotImplementedError


def visualize_log(filename, top=None, title=None, show=False, save=True):
  """
  Arg:
    filename(str)
  """
  data = readlines(filename)
  print("Parsed log: {} lines".format(len(data)))

  x, y = get_frequency(data)

  if top is not None:
    top *= -1
    x, y = x[top:], y[top:]

  plt.figure(figsize=(3, 5))
  plt.barh(x, y)
  plt.xlabel("Frequency")

  if title:
    plt.title(title)

  if show:
    plt.show()

  if save:
    fig_name = "{}.png".format(os.path.splitext(filename)[0])
    plt.savefig(fname=fig_name, bbox_inches="tight", dpi=300)
    print("Figure saved in {}".format(fig_name))


def get_frequency(data):
  """
  Arg:
    data(list of str):

  Return:
    keys(list of str):
    vals(list of str):
  """
  counter = collections.Counter(data)

  keys = [pair[0] for pair in sorted(counter.items(), key=lambda x: x[1])]
  vals = sorted(counter.values())

  return keys, vals


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("input", type=str, help="Input raw log file.")
  parser.add_argument("--output",
                      type=str,
                      help="Parsed log file. Default: [NAME]-parsed.[EXT]")
  parser.add_argument("--regex",
                      type=str,
                      help="Path to the regex files for parsing GDB log.")
  parser.add_argument("--visualize",
                      action="store_true",
                      help="Parse and visualize")
  parser.add_argument("--top", type=int, help="Top # to visualize")
  parser.add_argument("--source",
                      type=str,
                      default="gdb",
                      choices=["gdb", "renode"],
                      help="Source of where the log is captured")
  parser.add_argument(
      "--ignore",
      type=str,
      help="List of functions (one for each line in the file) to \
                  ignore after parsing.")
  parser.add_argument("--full-trace", action="store_true", help="")
  parser.add_argument("--title",
                      type=str,
                      help="Set title for the visualized image")

  args = parser.parse_args()

  if args.output is None:
    fname, extension = os.path.splitext(args.input)
    args.output = "{}-parsed{}".format(fname, extension)

  parse_log(args.input, args.output, args.regex, args.source, args.ignore,
            args.full_trace)

  if args.visualize:
    visualize_log(args.output, top=args.top, title=args.title)
