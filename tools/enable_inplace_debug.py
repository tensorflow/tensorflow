"""Enable in-place code change and debug without TF packaging and installation

It is a common practice to iteratively change the code, build, packing,
install, and debug to understand/fix/enable certain features.
The packing and installation phases are annoying because they are not strictly
necessary and slowing down the whole cycle.
This tool creates symbolic links from the source code directory to the binary
directory to `merge' these two so that we can change the code in the source
tree, build (not necessary for scripting languages such as Python), and debug
immediately without packing and installation, enabling a more efficient dev
cycle.

Usage:
  --help for help
  --link for enabling in-place debugging
  --unlink for disabling in-place debugging
  --dryrun for side effect inspection
  --check for inspecting special files changed by `bazel build'
"""

import argparse
import sys
import os
import shutil

#
# setup running FLAGS
#
parser = argparse.ArgumentParser()
parser.add_argument('--link',
  action = 'store_true',
  help='enable in-place debugging by creating symbolic links.')
parser.add_argument('--unlink',
  action = 'store_true',
  help='disable in-place debugging by removing symbolic links.')
parser.add_argument('--dryrun',
  action = 'store_true',
  help='print affected files without any side effect.')
parser.add_argument('--check',
  action = 'store_true',
  help='some files (e.g., tensorflow/__init__.py) are updated '
  'by build. In this case, symlink does not work, and we have '
  'to overwrite the files in the source tree (and recover them '
  'with --unlink. This command helps show who they are and '
  '--dryrun is enabled automatically.')
FLAGS, unparsed = parser.parse_known_args()
if FLAGS.check:
  FLAGS.dryrun = True # enable dryrun to avoid any side effect
if FLAGS.link and FLAGS.unlink:
  sys.exit ("--link and --unlink cannot be set simultaneously")

#
# setup statistics
#
scan_count = 0
link_count = 0
unlink_count = 0

#
# setup related directories
#
root_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + "/..")
target_dir = root_dir + "/tensorflow"
input_dir = root_dir + "/bazel-bin/tensorflow"

if not os.path.exists(target_dir):
  sys.exit ("directory %s not found, please make sure it exists and the "
    " current script is under tools dir" % target_dir)

if not os.path.exists(input_dir):
  sys.exit ("directory %s not found, please make sure it exists "
    "(`bazel build' not completed?)" % input_dir)

#
# scan input dirs and do the job
#
for root, dirs, files in os.walk(input_dir):
  for name in files:
    source_file = os.path.abspath(os.path.join(root, name))
    dest_file = source_file.replace(input_dir, target_dir)
    scan_count += 1

    # remove symlinks
    if FLAGS.unlink:
      if os.path.islink(dest_file):
        orig_file = os.readlink(dest_file)
        if source_file == orig_file:
          if FLAGS.dryrun:
            print("unlink %s -> %s" 
              % (dest_file, orig_file))
          else:
            os.unlink(dest_file)
          unlink_count += 1
    
    # create symlinks
    if FLAGS.link:
      if not os.path.exists(dest_file):
        if FLAGS.dryrun:
          print("link.create symlink %s -> %s"
            % (dest_file, source_file))
        else:
          try:
            os.makedirs(os.path.dirname(dest_file))
          except Exception as ex:
            pass
          os.symlink(source_file, dest_file)
        link_count += 1
        
    if FLAGS.check \
      and os.path.exists(dest_file) \
      and not os.path.islink(dest_file):
      print("check: %s %s both exists" % (dest_file, source_file))

#
# deal with source files updated by bazel build,
# run --check for all these files
#
updated_files_by_bazel = [
  "/__init__.py"
  # more files here
  ]
for updated_file in updated_files_by_bazel:
  source_file = target_dir + updated_file
  build_file = input_dir + updated_file

  if FLAGS.unlink:
    print("unlink.recover %s" % source_file)
    if not FLAGS.dryrun:
      if os.path.exists(source_file + ".bak"):
        os.remove(source_file)
        os.rename(source_file + ".bak", source_file)
      else:
        print("backup file %s is missing, skip recover!"
          % (source_file + ".bak"))
  
  if FLAGS.link:
    print("link.backup %s and overwrite it with %s"
      % (source_file, build_file))
    if not FLAGS.dryrun and os.path.exists(source_file) \
       and os.path.exists(build_file):
      os.rename (source_file, source_file + ".bak")
      shutil.copy2(build_file, source_file)

#
# display statistic
#
print("scanned %d files, create %d symlinks, unlink %d symlinks"
  % (scan_count, link_count, unlink_count))
