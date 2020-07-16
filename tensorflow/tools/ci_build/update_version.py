#!/usr/bin/python
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
#
# Automatically update TensorFlow version in source files
#
# Usage:
#           ./tensorflow/tools/ci_build/update_version.py --version 1.4.0-rc1
#           ./tensorflow/tools/ci_build/update_version.py --nightly
#
"""Update version of TensorFlow script."""

# pylint: disable=superfluous-parens

import argparse
import os
import re
import subprocess
import time

# File parameters.
TF_SRC_DIR = "tensorflow"
VERSION_H = "%s/core/public/version.h" % TF_SRC_DIR
SETUP_PY = "%s/tools/pip_package/setup.py" % TF_SRC_DIR
README_MD = "./README.md"
TENSORFLOW_BZL = "%s/tensorflow.bzl" % TF_SRC_DIR
RELEVANT_FILES = [TF_SRC_DIR, VERSION_H, SETUP_PY, README_MD]

# Version type parameters.
NIGHTLY_VERSION = 1
REGULAR_VERSION = 0


def check_existence(filename):
  """Check the existence of file or dir."""
  if not os.path.exists(filename):
    raise RuntimeError("%s not found. Are you under the TensorFlow source root"
                       " directory?")


def check_all_files():
  """Check all relevant files necessary for upgrade."""
  for file_name in RELEVANT_FILES:
    check_existence(file_name)


def replace_string_in_line(search, replace, filename):
  """Replace with sed when regex is required."""
  with open(filename, "r") as source:
    content = source.read()
  with open(filename, "w") as source:
    source.write(re.sub(search, replace, content))


class Version(object):
  """Version class object that stores SemVer version information."""

  def __init__(self, major, minor, patch, identifier_string, version_type):
    """Constructor.

    Args:
      major: major string eg. (1)
      minor: minor string eg. (3)
      patch: patch string eg. (1)
      identifier_string: extension string eg. (-rc0)
      version_type: version parameter ((REGULAR|NIGHTLY)_VERSION)
    """
    self.major = major
    self.minor = minor
    self.patch = patch
    self.identifier_string = identifier_string
    self.version_type = version_type
    self._update_string()

  def _update_string(self):
    self.string = "%s.%s.%s%s" % (self.major,
                                  self.minor,
                                  self.patch,
                                  self.identifier_string)

  def __str__(self):
    return self.string

  def set_identifier_string(self, identifier_string):
    self.identifier_string = identifier_string
    self._update_string()

  @property
  def pep_440_str(self):
    if self.version_type == REGULAR_VERSION:
      return_string = "%s.%s.%s%s" % (self.major,
                                      self.minor,
                                      self.patch,
                                      self.identifier_string)
      return return_string.replace("-", "")
    else:
      return_string = "%s.%s.%s" % (self.major,
                                    self.minor,
                                    self.identifier_string)
      return return_string.replace("-", "")

  @staticmethod
  def parse_from_string(string, version_type):
    """Returns version object from Semver string.

    Args:
      string: version string
      version_type: version parameter

    Raises:
      RuntimeError: If the version string is not valid.
    """
    # Check validity of new version string.
    if not re.search(r"[0-9]+\.[0-9]+\.[a-zA-Z0-9]+", string):
      raise RuntimeError("Invalid version string: %s" % string)

    major, minor, extension = string.split(".", 2)

    # Isolate patch and identifier string if identifier string exists.
    extension_split = extension.split("-", 1)
    patch = extension_split[0]
    if len(extension_split) == 2:
      identifier_string = "-" + extension_split[1]
    else:
      identifier_string = ""

    return Version(major,
                   minor,
                   patch,
                   identifier_string,
                   version_type)


def get_current_semver_version():
  """Returns a Version object of current version.

  Returns:
    version: Version object of current SemVer string based on information from
    core/public/version.h
  """

  # Get current version information.
  version_file = open(VERSION_H, "r")
  for line in version_file:
    major_match = re.search("^#define TF_MAJOR_VERSION ([0-9]+)", line)
    minor_match = re.search("^#define TF_MINOR_VERSION ([0-9]+)", line)
    patch_match = re.search("^#define TF_PATCH_VERSION ([0-9]+)", line)
    extension_match = re.search("^#define TF_VERSION_SUFFIX \"(.*)\"", line)
    if major_match:
      old_major = major_match.group(1)
    if minor_match:
      old_minor = minor_match.group(1)
    if patch_match:
      old_patch_num = patch_match.group(1)
    if extension_match:
      old_extension = extension_match.group(1)
      break

  if "dev" in old_extension:
    version_type = NIGHTLY_VERSION
  else:
    version_type = REGULAR_VERSION

  return Version(old_major,
                 old_minor,
                 old_patch_num,
                 old_extension,
                 version_type)


def update_version_h(old_version, new_version):
  """Update tensorflow/core/public/version.h."""
  replace_string_in_line("#define TF_MAJOR_VERSION %s" % old_version.major,
                         "#define TF_MAJOR_VERSION %s" % new_version.major,
                         VERSION_H)
  replace_string_in_line("#define TF_MINOR_VERSION %s" % old_version.minor,
                         "#define TF_MINOR_VERSION %s" % new_version.minor,
                         VERSION_H)
  replace_string_in_line("#define TF_PATCH_VERSION %s" % old_version.patch,
                         "#define TF_PATCH_VERSION %s" % new_version.patch,
                         VERSION_H)
  replace_string_in_line(
      "#define TF_VERSION_SUFFIX \"%s\"" % old_version.identifier_string,
      "#define TF_VERSION_SUFFIX \"%s\"" % new_version.identifier_string,
      VERSION_H)


def update_setup_dot_py(old_version, new_version):
  """Update setup.py."""
  replace_string_in_line("_VERSION = '%s'" % old_version.string,
                         "_VERSION = '%s'" % new_version.string, SETUP_PY)


def update_readme(old_version, new_version):
  """Update README."""
  pep_440_str = new_version.pep_440_str
  replace_string_in_line(r"%s\.%s\.([[:alnum:]]+)-" % (old_version.major,
                                                       old_version.minor),
                         "%s-" % pep_440_str, README_MD)


def update_tensorflow_bzl(old_version, new_version):
  """Update tensorflow.bzl."""
  old_mmp = "%s.%s.%s" % (old_version.major, old_version.minor,
                          old_version.patch)
  new_mmp = "%s.%s.%s" % (new_version.major, new_version.minor,
                          new_version.patch)
  replace_string_in_line('VERSION = "%s"' % old_mmp,
                         'VERSION = "%s"' % new_mmp, TENSORFLOW_BZL)


def major_minor_change(old_version, new_version):
  """Check if a major or minor change occurred."""
  major_mismatch = old_version.major != new_version.major
  minor_mismatch = old_version.minor != new_version.minor
  if major_mismatch or minor_mismatch:
    return True
  return False


def check_for_lingering_string(lingering_string):
  """Check for given lingering strings."""
  formatted_string = lingering_string.replace(".", r"\.")
  try:
    linger_str_output = subprocess.check_output(
        ["grep", "-rnoH", formatted_string, TF_SRC_DIR])
    linger_strs = linger_str_output.decode("utf8").split("\n")
  except subprocess.CalledProcessError:
    linger_strs = []

  if linger_strs:
    print("WARNING: Below are potentially instances of lingering old version "
          "string \"%s\" in source directory \"%s/\" that are not "
          "updated by this script. Please check them manually!"
          % (lingering_string, TF_SRC_DIR))
    for linger_str in linger_strs:
      print(linger_str)
  else:
    print("No lingering old version strings \"%s\" found in source directory"
          " \"%s/\". Good." % (lingering_string, TF_SRC_DIR))


def check_for_old_version(old_version, new_version):
  """Check for old version references."""
  for old_ver in [old_version.string, old_version.pep_440_str]:
    check_for_lingering_string(old_ver)

  if major_minor_change(old_version, new_version):
    old_r_major_minor = "r%s.%s" % (old_version.major, old_version.minor)
    check_for_lingering_string(old_r_major_minor)


def main():
  """This script updates all instances of version in the tensorflow directory.

  Requirements:
    version: The version tag
    OR
    nightly: Create a nightly tag with current date

  Raises:
    RuntimeError: If the script is not being run from tf source dir
  """

  parser = argparse.ArgumentParser(description="Cherry picking automation.")

  # Arg information
  parser.add_argument("--version",
                      help="<new_major_ver>.<new_minor_ver>.<new_patch_ver>",
                      default="")
  parser.add_argument("--nightly",
                      help="disable the service provisioning step",
                      action="store_true")

  args = parser.parse_args()

  check_all_files()
  old_version = get_current_semver_version()

  if args.nightly:
    if args.version:
      new_version = Version.parse_from_string(args.version, NIGHTLY_VERSION)
      new_version.set_identifier_string("-dev" + time.strftime("%Y%m%d"))
    else:
      new_version = Version(old_version.major,
                            str(old_version.minor),
                            old_version.patch,
                            "-dev" + time.strftime("%Y%m%d"),
                            NIGHTLY_VERSION)
  else:
    new_version = Version.parse_from_string(args.version, REGULAR_VERSION)

  update_version_h(old_version, new_version)
  update_setup_dot_py(old_version, new_version)
  update_readme(old_version, new_version)
  update_tensorflow_bzl(old_version, new_version)

  # Print transition details.
  print("Major: %s -> %s" % (old_version.major, new_version.major))
  print("Minor: %s -> %s" % (old_version.minor, new_version.minor))
  print("Patch: %s -> %s\n" % (old_version.patch, new_version.patch))

  check_for_old_version(old_version, new_version)


if __name__ == "__main__":
  main()
