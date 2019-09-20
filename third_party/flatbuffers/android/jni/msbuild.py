#!/usr/bin/python
# Copyright 2014 Google Inc. All rights reserved.
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

"""Simple script that locates the newest MSBuild in one of several locations.

This script will find the highest version number of MSBuild and run it,
passing its arguments through to MSBuild.
"""

import glob
import os
import re
import string
import subprocess
import sys

SYSTEMROOT = os.getenv("SYSTEMROOT", "c:\\windows")
PROGRAM_FILES = os.getenv("ProgramFiles", "c:\\Program Files")
PROGRAM_FILES_X86 = os.getenv("ProgramFiles(x86)", "c:\\Program Files (x86)")

SEARCH_FOLDERS = [ PROGRAM_FILES + "\\MSBuild\\*\\Bin\\MSBuild.exe",
                   PROGRAM_FILES_X86 + "\\MSBuild\\*\\Bin\\MSBuild.exe",
                   SYSTEMROOT + "\\Microsoft.NET\Framework\\*\\MSBuild.exe" ]

def compare_version(a, b):
  """Compare two version number strings of the form W.X.Y.Z.

  The numbers are compared most-significant to least-significant.
  For example, 12.345.67.89 > 2.987.88.99.

  Args:
    a: First version number string to compare
    b: Second version number string to compare

  Returns:
    0 if the numbers are identical, a positive number if 'a' is larger, and
    a negative number if 'b' is larger.
  """
  aa = string.split(a, ".")
  bb = string.split(b, ".")
  for i in range(0, 4):
    if aa[i] != bb[i]:
      return cmp(int(aa[i]), int(bb[i]))
  return 0

def main():
  msbuilds = []

  for folder in SEARCH_FOLDERS:
    for file in glob.glob(folder):
      p = subprocess.Popen([file, "/version"], stdout=subprocess.PIPE)
      out, err = p.communicate()
      match = re.search("^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+$", out, re.M)
      if match:
        msbuilds.append({ 'ver':match.group(), 'exe':file })
  msbuilds.sort(lambda x, y: compare_version(x['ver'], y['ver']), reverse=True)
  if len(msbuilds) == 0:
    print "Unable to find MSBuild.\n"
    return -1;
  cmd = [msbuilds[0]['exe']]
  cmd.extend(sys.argv[1:])
  return subprocess.call(cmd)

if __name__ == '__main__':
  sys.exit(main())
