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
"""Utility for running programs in a symlinked execroot."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import shutil
import subprocess
import sys
import tempfile


def run(inputs, program, outputs):
  """Creates temp symlink tree, runs program, and copies back outputs.

  Args:
    inputs: List of fake paths to real paths, which are used for symlink tree.
    program: List containing real path of program and its arguments. The
        execroot directory will be appended as the last argument.
    outputs: List of fake outputted paths to copy back to real paths.
  Returns:
    0 if succeeded or nonzero if failed.
  """
  root = tempfile.mkdtemp()
  try:
    cwd = os.getcwd()
    for fake, real in inputs:
      parent = os.path.join(root, os.path.dirname(fake))
      if not os.path.exists(parent):
        os.makedirs(parent)
      os.symlink(os.path.join(cwd, real), os.path.join(root, fake))
    if subprocess.call(program + [root]) != 0:
      return 1
    for fake, real in outputs:
      shutil.copyfile(os.path.join(root, fake), real)
    return 0
  finally:
    shutil.rmtree(root)


def main(args):
  """Invokes run function using a JSON file config.

  Args:
    args: CLI args, which can be a JSON file containing an object whose
        attributes are the parameters to the run function. If multiple JSON
        files are passed, their contents are concatenated.
  Returns:
    0 if succeeded or nonzero if failed.
  Raises:
    Exception: If input data is missing.
  """
  if not args:
    raise Exception('Please specify at least one JSON config path')
  inputs = []
  program = []
  outputs = []
  for arg in args:
    with open(arg) as fd:
      config = json.load(fd)
    inputs.extend(config.get('inputs', []))
    program.extend(config.get('program', []))
    outputs.extend(config.get('outputs', []))
  if not program:
    raise Exception('Please specify a program')
  return run(inputs, program, outputs)


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
