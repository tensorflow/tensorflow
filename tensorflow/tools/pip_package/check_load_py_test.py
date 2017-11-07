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
"""Tests to check that py_test are properly loaded in BUILD files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess


def check_output_despite_error(args):
  """Get output of args from command line, even if there are errors.

  Args:
    args: a list of command line args.

  Returns:
    output as string.
  """
  try:
    output = subprocess.check_output(args, stderr=subprocess.STDOUT)
  except subprocess.CalledProcessError as e:
    output = e.output
  return output.strip()


def main():
  # Get all py_test target, note bazel query result will also include
  # cuda_py_test etc.
  try:
    targets = subprocess.check_output([
        'bazel', 'query',
        'kind(py_test, //tensorflow/contrib/... + '
        '//tensorflow/python/... - '
        '//tensorflow/contrib/tensorboard/...)']).strip()
  except subprocess.CalledProcessError as e:
    targets = e.output

  # Only keep py_test targets, and filter out targets with 'no_pip' tag.
  valid_targets = []
  for target in targets.split('\n'):
    kind = check_output_despite_error(['buildozer', 'print kind', target])
    if kind == 'py_test':
      tags = check_output_despite_error(['buildozer', 'print tags', target])
      if 'no_pip' not in tags:
        valid_targets.append(target)

  # Get all BUILD files for all valid targets.
  build_files = set()
  for target in valid_targets:
    build_files.add(os.path.join(target[2:].split(':')[0], 'BUILD'))

  # Check if BUILD files load py_test.
  files_missing_load = []
  for build_file in build_files:
    updated_build_file = subprocess.check_output(
        ['buildozer', '-stdout', 'new_load //tensorflow:tensorflow.bzl py_test',
         build_file])
    with open(build_file, 'r') as f:
      if f.read() != updated_build_file:
        files_missing_load.append(build_file)

  if files_missing_load:
    raise RuntimeError('The following files are missing %s:\n %s' % (
        'load("//tensorflow:tensorflow.bzl", "py_test").\nThis load statement'
        ' is needed because otherwise pip tests will try to use their '
        'dependencies, which are not visible to them.',
        '\n'.join(files_missing_load)))
  else:
    print('TEST PASSED.')


if __name__ == '__main__':
  main()
