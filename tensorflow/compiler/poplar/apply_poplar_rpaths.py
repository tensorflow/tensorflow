#!/usr/bin/python

import sys
import subprocess
import os
import distutils.spawn

# This will copy the '.so' file to the final resting place, and apply
# the full path of the poplar and popnn directories to the rpaths

print "Copying " + sys.argv[1] + " to " + sys.argv[2]
subprocess.call(["cp", sys.argv[1], sys.argv[2]])
subprocess.call(["chmod", "755", sys.argv[2]])

install_name_path = distutils.spawn.find_executable('install_name_tool')
if (install_name_path):

  fullpath = os.path.dirname(os.path.abspath(sys.argv[3]))
  print "Adding poplar rpath"
  subprocess.call([install_name_path, "-add_rpath", fullpath, sys.argv[2]])

  fullpath = os.path.dirname(os.path.abspath(sys.argv[4]))
  print "Adding popnn rpath"
  subprocess.call([install_name_path, "-add_rpath", fullpath, sys.argv[2]])
