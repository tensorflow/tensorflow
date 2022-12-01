#!/usr/bin/env python3
# pylint:disable=protected-access
#
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Merge all JUnit test.xml files in one directory into one.

Usage: squash_testlogs.py START_DIRECTORY OUTPUT_FILE

Example: squash_testlogs.py /tf/pkg/testlogs /tf/pkg/merged.xml

Recursively find all the JUnit test.xml files in one directory, and merge any
of them that contain failures into one file. The TensorFlow DevInfra team
uses this to generate a simple overview of an entire pip and nonpip test
invocation, since the normal logs that Bazel creates are too large for the
internal invocation viewer.
"""
import collections
import os
import re
import subprocess
import sys
from junitparser import JUnitXml

result = JUnitXml()
try:
  files = subprocess.check_output(
      ["grep", "-rlE", '(failures|errors)="[1-9]', sys.argv[1]])
except subprocess.CalledProcessError as e:
  print("No failures found to log!")
  exit(0)

# For test cases, only show the ones that failed that have text (a log)
seen = collections.Counter()
runfiles_matcher = re.compile(r"(/.*\.runfiles/)")


for f in files.strip().splitlines():
  # Just ignore any failures, they're probably not important
  try:
    r = JUnitXml.fromfile(f)
  except Exception as e:  # pylint: disable=broad-except
    print("Ignoring this XML parse failure in {}: ".format(f), str(e))

  source_file = re.search(r"/(bazel_pip|tensorflow)/.*",
                          f.decode("utf-8")).group(0)
  for testsuite in r:
    testsuite._elem.set("source_file", source_file)
    # Remove empty testcases
    for p in testsuite._elem.xpath(".//testcase"):
      if not len(p):  # pylint: disable=g-explicit-length-test
        testsuite._elem.remove(p)
    # Change "testsuite > testcase,system-out" to "testsuite > testcase > error"
    for p in testsuite._elem.xpath(".//system-out"):
      for c in p.getparent().xpath(".//error | .//failure"):
        c.text = p.text
      p.getparent().remove(p)
    # Remove duplicate results of the same exact test (e.g. due to retry
    # attempts)
    for p in testsuite._elem.xpath(".//error | .//failure"):
      # Sharded tests have target names like this:
      # WindowOpsTest.test_tflite_convert0 (<function hann_window at
      #     0x7fc61728dd40>, 10, False, tf.float32)
      # Where 0x... is a thread ID (or something) that is not important for
      # debugging, but breaks this "number of failures" counter because it's
      # different for repetitions of the same test. We use re.sub(r"0x\w+")
      # to remove it.
      key = re.sub(r"0x\w+", "", p.getparent().get("name", "")) + p.text
      if key in seen:
        testsuite._elem.remove(p.getparent())
      seen[key] += 1
    # Remove this testsuite if it doesn't have anything in it any more
    if len(testsuite) == 0:  # pylint: disable=g-explicit-length-test
      r._elem.remove(testsuite._elem)
  if len(r) > 0:  # pylint: disable=g-explicit-length-test
    result += r

# Insert the number of failures for each test to help identify flakes
# need to clarify for shard
for p in result._elem.xpath(".//error | .//failure"):
  key = re.sub(r"0x\w+", "", p.getparent().get("name", "")) + p.text
  p.text = runfiles_matcher.sub("[testroot]/", p.text)
  source_file = p.getparent().getparent().get("source_file", "")
  p.text += f"\nNOTE: From {source_file}"
  if "bazel_pip" in source_file:
    p.text += ("\nNOTE: This is a --config=pip test. Remove 'bazel_pip' to find"
               " the file.")
  n_failures = seen[key]
  p.text += f"\nNOTE: Number of failures for this test: {seen[key]}."
  p.text += "\n      Most TF jobs run tests three times to root out flakes."
  if seen[key] == 3:
    p.text += ("\n      Since there were three failures, this is not flaky, and"
               " it")
    p.text += "\n      probably caused the Kokoro invocation to fail."
  else:
    p.text += ("\n      Since there were not three failures, this is probably a"
               " flake.")
    p.text += ("\n      Flakes make this pkg/pip_and_nonpip_tests target show "
               "as failing,")
    p.text += "\n      but do not make the Kokoro invocation fail."

os.makedirs(os.path.dirname(sys.argv[2]), exist_ok=True)
result.update_statistics()
result.write(sys.argv[2])
