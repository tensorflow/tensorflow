#!/usr/bin/env python
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
#
# Checks that the options mentioned in syslibs_configure.bzl are consistent with
# the valid options in workspace.bzl
# Expects the tensorflow source folder as the first argument

import glob
import os
import sys

tf_source_path = sys.argv[1]

syslibs_configure_path = os.path.join(tf_source_path, 'third_party',
                                      'systemlibs', 'syslibs_configure.bzl')
workspace_path = os.path.join(tf_source_path, 'tensorflow', 'workspace.bzl')
third_party_path = os.path.join(tf_source_path, 'third_party')
third_party_glob = os.path.join(third_party_path, '*', 'workspace.bzl')

if not (os.path.isdir(tf_source_path) and os.path.isfile(syslibs_configure_path)
        and os.path.isfile(workspace_path)):
  raise ValueError('The path to the TensorFlow source must be passed as'
                   ' the first argument')


def extract_valid_libs(filepath):
  """Evaluate syslibs_configure.bzl, return the VALID_LIBS set from that file."""

  # Stub only
  def repository_rule(**kwargs):  # pylint: disable=unused-variable
    del kwargs

  # Populates VALID_LIBS
  with open(filepath, 'r') as f:
    f_globals = {'repository_rule': repository_rule}
    f_locals = {}
    exec(f.read(), f_globals, f_locals)  # pylint: disable=exec-used

  return set(f_locals['VALID_LIBS'])


def extract_system_builds(filepath):
  """Extract the 'name' argument of all rules with a system_build_file argument."""
  lib_names = []
  system_build_files = []
  current_name = None
  with open(filepath, 'r') as f:
    for line in f:
      line = line.strip()
      if line.startswith('name = '):
        current_name = line[7:-1].strip('"')
      elif line.startswith('system_build_file = '):
        lib_names.append(current_name)
        # Split at '=' to extract rhs, then extract value between quotes
        system_build_spec = line.split('=')[-1].split('"')[1]
        assert system_build_spec.startswith('//')
        system_build_files.append(system_build_spec[2:].replace(':', os.sep))
  return lib_names, system_build_files


syslibs = extract_valid_libs(syslibs_configure_path)

syslibs_from_workspace = set()
system_build_files_from_workspace = []
for current_path in [workspace_path] + glob.glob(third_party_glob):
  cur_lib_names, build_files = extract_system_builds(current_path)
  syslibs_from_workspace.update(cur_lib_names)
  system_build_files_from_workspace.extend(build_files)

missing_build_files = [
    file for file in system_build_files_from_workspace
    if not os.path.isfile(os.path.join(tf_source_path, file))
]

has_error = False

if missing_build_files:
  has_error = True
  print('Missing system build files: ' + ', '.join(missing_build_files))

if syslibs != syslibs_from_workspace:
  has_error = True
  # Libs present in workspace files but not in the allowlist
  missing_syslibs = syslibs_from_workspace - syslibs
  if missing_syslibs:
    libs = ', '.join(sorted(missing_syslibs))
    print('Libs missing from syslibs_configure: ' + libs)
  # Libs present in the allow list but not in workspace files
  additional_syslibs = syslibs - syslibs_from_workspace
  if additional_syslibs:
    libs = ', '.join(sorted(additional_syslibs))
    print('Libs missing in workspace (or superfluous in syslibs_configure): ' +
          libs)

sys.exit(1 if has_error else 0)
