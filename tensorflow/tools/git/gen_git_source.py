#!/usr/bin/env python
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Help include git hash in tensorflow bazel build.

This creates symlinks from the internal git repository directory so
that the build system can see changes in the version state. We also
remember what branch git was on so when the branch changes we can
detect that the ref file is no longer correct (so we can suggest users
run ./configure again).

NOTE: this script is only used in opensource.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import json
import os
import subprocess
import shutil


def parse_branch_ref(filename):
  """Given a filename of a .git/HEAD file return ref path.

  In particular, if git is in detached head state, this will
  return None. If git is in attached head, it will return
  the branch reference. E.g. if on 'master', the HEAD will
  contain 'ref: refs/heads/master' so 'refs/heads/master'
  will be returned.

  Example: parse_branch_ref(".git/HEAD")
  Args:
    filename: file to treat as a git HEAD file
  Returns:
    None if detached head, otherwise ref subpath
  Raises:
    RuntimeError: if the HEAD file is unparseable.
  """

  data = open(filename).read().strip()
  items = data.split(" ")
  if len(items) == 1:
    return None
  elif len(items) == 2 and items[0] == "ref:":
    return items[1].strip()
  else:
    raise RuntimeError("Git directory has unparseable HEAD")


def configure(src_base_path, debug=False):
  """Configure `src_base_path` to embed git hashes if available."""

  # TODO(aselle): No files generated or symlinked here are deleted by
  # the build system. I don't know of a way to do it in bazel. It
  # should only be a problem if somebody moves a sandbox directory
  # without running ./configure again.

  git_path = os.path.join(src_base_path, ".git")
  gen_path = os.path.join(src_base_path, "tensorflow", "tools", "git", "gen")

  # Remove and recreate the path
  if os.path.exists(gen_path):
    if os.path.isdir(gen_path):
      shutil.rmtree(gen_path)
    else:
      raise RuntimeError("Cannot delete non-directory %s, inspect ",
                         "and remove manually" % gen_path)
  os.makedirs(gen_path)

  if not os.path.isdir(gen_path):
    raise RuntimeError("gen_git_source.py: Failed to create dir")

  # file that specifies what the state of the git repo is
  spec = {}

  # value file names will be mapped to the keys
  link_map = {"head": None, "branch_ref": None}

  if not os.path.isdir(git_path):
    # No git directory
    spec["git"] = False
    open(os.path.join(gen_path, "head"), "w").write("")
    open(os.path.join(gen_path, "branch_ref"), "w").write("")
  else:
    # Git directory, possibly detached or attached
    spec["git"] = True
    spec["path"] = src_base_path
    git_head_path = os.path.join(git_path, "HEAD")
    spec["branch"] = parse_branch_ref(git_head_path)
    link_map["head"] = git_head_path
    if spec["branch"] is not None:
      # attached method
      link_map["branch_ref"] = os.path.join(git_path, *
                                            os.path.split(spec["branch"]))
  # Create symlinks or dummy files
  for target, src in link_map.items():
    if src is None:
      open(os.path.join(gen_path, target), "w").write("")
    else:
      if hasattr(os, 'symlink'):
        os.symlink(src, os.path.join(gen_path, target))
      else:
        shutil.copy2(src, os.path.join(gen_path, target))

  json.dump(spec, open(os.path.join(gen_path, "spec.json"), "w"), indent=2)
  if debug:
    print("gen_git_source.py: list %s" % gen_path)
    print("gen_git_source.py: %s" + repr(os.listdir(gen_path)))
    print("gen_git_source.py: spec is %r" % spec)


def get_git_version(git_base_path):
  """Get the git version from the repository.

  This function runs `git describe ...` in the path given as `git_base_path`.
  This will return a string of the form:
  <base-tag>-<number of commits since tag>-<shortened sha hash>

  For example, 'v0.10.0-1585-gbb717a6' means v0.10.0 was the last tag when
  compiled. 1585 commits are after that commit tag, and we can get back to this
  version by running `git checkout gbb717a6`.

  Args:
    git_base_path: where the .git directory is located
  Returns:
    A string representing the git version
  """

  unknown_label = "unknown"
  try:
    val = subprocess.check_output(["git", "-C", git_base_path, "describe",
                                   "--long", "--dirty", "--tags"]).strip()
    return val if val else unknown_label
  except subprocess.CalledProcessError:
    return unknown_label


def write_version_info(filename, git_version):
  """Write a c file that defines the version functions.

  Args:
    filename: filename to write to.
    git_version: the result of a git describe.
  """
  if b"\"" in git_version or b"\\" in git_version:
    git_version = "git_version_is_invalid"  # do not cause build to fail!
  contents = """/*  Generated by gen_git_source.py  */
const char* tf_git_version() {return "%s";}
const char* tf_compiler_version() {return __VERSION__;}
""" % git_version
  open(filename, "w").write(contents)


def generate(arglist):
  """Generate version_info.cc as given `destination_file`.

  Args:
    arglist: should be a sequence that contains
             spec, head_symlink, ref_symlink, destination_file.

  `destination_file` is the filename where version_info.cc will be written

  `spec` is a filename where the file contains a JSON dictionary
    'git' bool that is true if the source is in a git repo
    'path' base path of the source code
    'branch' the name of the ref specification of the current branch/tag

  `head_symlink` is a filename to HEAD that is cross-referenced against
    what is contained in the json branch designation.

  `ref_symlink` is unused in this script but passed, because the build
    system uses that file to detect when commits happen.

  Raises:
    RuntimeError: If ./configure needs to be run, RuntimeError will be raised.
  """

  # unused ref_symlink arg
  spec, head_symlink, _, dest_file = arglist
  data = json.load(open(spec))
  git_version = None
  if not data["git"]:
    git_version = "unknown"
  else:
    old_branch = data["branch"]
    new_branch = parse_branch_ref(head_symlink)
    if new_branch != old_branch:
      raise RuntimeError(
          "Run ./configure again, branch was '%s' but is now '%s'" %
          (old_branch, new_branch))
    git_version = get_git_version(data["path"])
  write_version_info(dest_file, git_version)


def raw_generate(output_file):
  """Simple generator used for cmake/make build systems.

  This does not create any symlinks. It requires the build system
  to build unconditionally.

  Args:
    output_file: Output filename for the version info cc
  """

  git_version = get_git_version(".")
  write_version_info(output_file, git_version)


parser = argparse.ArgumentParser(description="""Git hash injection into bazel.
If used with --configure <path> will search for git directory and put symlinks
into source so that a bazel genrule can call --generate""")

parser.add_argument(
    "--debug",
    type=bool,
    help="print debugging information about paths",
    default=False)

parser.add_argument(
    "--configure", type=str,
    help="Path to configure as a git repo dependency tracking sentinel")

parser.add_argument(
    "--generate",
    type=str,
    help="Generate given spec-file, HEAD-symlink-file, ref-symlink-file",
    nargs="+")

parser.add_argument(
    "--raw_generate",
    type=str,
    help="Generate version_info.cc (simpler version used for cmake/make)")

args = parser.parse_args()

if args.configure is not None:
  configure(args.configure, debug=args.debug)
elif args.generate is not None:
  generate(args.generate)
elif args.raw_generate is not None:
  raw_generate(args.raw_generate)
else:
  raise RuntimeError("--configure or --generate or --raw_generate "
                     "must be used")
