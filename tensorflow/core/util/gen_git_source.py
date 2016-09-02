#!/usr/bin/python

import json
import os
import shutil
import sys
import argparse

def parse_branch_ref(x):
  items = x.split(" ")
  if len(items) == 1:
    return None
  elif len(items) == 2 and items[0]=="ref:":
    return items[1].strip()
  else:
    raise RuntimeError("""Git directory has unparseable HEAD""")

def configure(src_base_path):
  "Configure `src_base_path` to embed git hashes if available"

  git_path = os.path.join(src_base_path, ".git")
  gen_path = os.path.join(src_base_path, "tensorflow", "core", "util", "git")
  util_path = os.path.join(src_base_path, "tensorflow", "core", "util")

  # Remove and recreate the path
  if os.path.exists(gen_path):
    if os.path.isdir(gen_path):
      shutil.rmtree(gen_path)
    else:
      raise RuntimeError("""Cannot delete non-directory %s, inspect
  and remove manually""" % gen_path)
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
    open(os.path.join(gen_path, "head"),"w").write("")
    open(os.path.join(gen_path, "branch_ref"),"w").write("")
  else:
    # Git directory, possibly detached or attached
    spec["git"] = True
    spec["path"] = src_base_path
    git_head_path = os.path.join(git_path,"HEAD")
    head = open(git_head_path).read().strip()
    spec["branch"] = parse_branch_ref(head)
    link_map["head"] = git_head_path
    if spec["branch"] is not None:
      # attached method
      link_map["branch_ref"] = os.path.join(git_path, *os.path.split(spec["branch"]))
  # Create symlinks or dummy files
  for target, src in link_map.items():
    if target==None:
      open(os.path.join(gen_path, target),"w").write("")
    else:
      os.symlink(src, os.path.join(gen_path, target))

  json.dump(spec,open(os.path.join(gen_path,"spec.json"),"w"), indent=2)
  print("gen_git_source.py: list %s"%gen_path)
  print("gen_git_source.py: %s"+repr(os.listdir(gen_path)))
  print("gen_git_source.py: spec is %r"%spec)

def generate(args):
  "Generate version_info.h"
  spec, head_symlink, ref_symlink, dest_file = args
  data = json.load(open(spec))
  strs = {"tf_compiler_version": "__VERSION__"}
  if data["git"] == False:
    strs["tf_git_version"] = "internal"
  else:
    old_branch = data["branch"]
    new_branch = parse_branch_ref(open(head_symlink).read().strip())
    if new_branch != old_branch:
      raise RuntimeError(
        "Run ./configure again, branch was '%s' but is now '%s'" % (
            old_branch, new_branch))
    strs["tf_git_version"] = os.popen(
      "git -C \"%s\" describe --long --dirty --tags" % (
        data["path"],)).read().strip()
  # TODO(aselle): Check for escaping
  file = "\n".join("const char* %s = \"%s\";"%(x,y) for x,y in strs.items())
  open(dest_file,"w").write(file+"\n")


parser = argparse.ArgumentParser(description="""Git hash injection into bazel.
If used with --configure <path> will search for git directory and put symlinks
into source so that a bazel genrule can call --generate""")

parser.add_argument("--configure", type=str,
                    help='Configure in a given source tree')

parser.add_argument("--generate", type=str,
                    help='Generate given spec-file, HEAD-symlink-file, ref-symlink-file',
                    nargs='+')
args = parser.parse_args()

if args.configure != None:
  configure(args.configure)
elif args.generate != None:
  generate(args.generate)
else:
  raise RuntimeError("--configure or --generate must be used")


