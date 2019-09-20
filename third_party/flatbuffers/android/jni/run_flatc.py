#!/usr/bin/python
# Copyright 2015 Google Inc. All rights reserved.
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

import os
import platform
import subprocess
import sys

EXECUTABLE_EXTENSION = '.exe' if platform.system() == 'Windows' else ''
# Paths to search for flatc relative to the current working directory.
FLATC_SEARCH_PATHS = [os.path.curdir, 'Release', 'Debug']

def main():
  """Script that finds and runs flatc built from source."""
  if len(sys.argv) < 2:
    sys.stderr.write('Usage: run_flatc.py flatbuffers_dir [flatc_args]\n')
    return 1
  cwd = os.getcwd()
  flatc = ''
  flatbuffers_dir = sys.argv[1]
  for path in FLATC_SEARCH_PATHS:
    current = os.path.join(flatbuffers_dir, path,
                           'flatc' + EXECUTABLE_EXTENSION)
    if os.path.exists(current):
      flatc = current
      break
  if not flatc:
    sys.stderr.write('flatc not found\n')
    return 1
  command = [flatc] + sys.argv[2:]
  return subprocess.call(command)

if __name__ == '__main__':
  sys.exit(main())
