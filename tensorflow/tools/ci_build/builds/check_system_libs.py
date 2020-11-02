#!/usr/bin/env python

# Checks that the options mentioned in syslibs_configure.bzl are consistent with the valid options in workspace.bzl
# Expects the tensorflow source folder as the first argument

import sys
import os
from glob import glob

tf_source_path = sys.argv[1]

if not os.path.isdir(tf_source_path):
  raise ValueError('The path to the TensorFlow source must be passed as'
                   ' the first argument')

syslibs_configure_path = os.path.join(tf_source_path, 'third_party',
                                      'systemlibs', 'syslibs_configure.bzl')
workspace_path = os.path.join(tf_source_path, 'tensorflow', 'workspace.bzl')
third_party_path = os.path.join(tf_source_path, 'third_party')
third_party_glob = os.path.join(third_party_path, '*', 'workspace.bzl')

# Stub only
def repository_rule(**kwargs):
  del kwargs

# Populates VALID_LIBS
with open(syslibs_configure_path, 'r') as f:
  exec(f.read())
syslibs = set(VALID_LIBS)

syslibs_from_workspace = set()

def extract_system_builds(filepath):
  current_name = None
  with open(filepath, 'r') as f:
    for line in f:
      line = line.strip()
      if line.startswith('name = '):
        current_name = line[7:-1].strip('"')
      elif line.startswith('system_build_file = '):
        syslibs_from_workspace.add(current_name)

for current_path in [workspace_path] + glob(third_party_glob):
  extract_system_builds(current_path)

if syslibs != syslibs_from_workspace:
  missing_syslibs = syslibs_from_workspace - syslibs
  if missing_syslibs:
    libs = ', '.join(sorted(missing_syslibs))
    print('Libs missing from syslibs_configure: ' + libs)
  additional_syslibs = syslibs - syslibs_from_workspace
  if additional_syslibs:
    libs = ', '.join(sorted(additional_syslibs))
    print('Libs missing in workspace (or superfluous in syslibs_configure): '
          + libs)
  sys.exit(1)
