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

"""create_def_file.py - tool to create a windows def file.

The def file can be used to export symbols from the tensorflow dll to enable
tf.load_library().

Because the linker allows only 64K symbols to be exported per dll
we filter the symbols down to the essentials. The regular expressions
we use for this are specific to tensorflow.

TODO: this works fine but there is an issue with exporting
'const char * const' and importing it from a user_ops. The problem is
on the importing end and using __declspec(dllimport) works around it.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import codecs
import os
import re
import subprocess
import sys
import tempfile

# External tools we use that come with visual studio sdk and
# we assume that the caller has the correct PATH to the sdk
UNDNAME = "undname.exe"
DUMPBIN = "dumpbin.exe"

# Exclude if matched
EXCLUDE_RE = re.compile(r"RTTI|deleting destructor|::internal::")

# Include if matched before exclude
INCLUDEPRE_RE = re.compile(r"google::protobuf::internal::ExplicitlyConstructed|"
                           r"tensorflow::internal::LogMessage|"
                           r"tensorflow::internal::LogString|"
                           r"tensorflow::internal::CheckOpMessageBuilder|"
                           r"tensorflow::internal::PickUnusedPortOrDie|"
                           r"tensorflow::internal::ValidateDevice|"
                           r"tensorflow::ops::internal::Enter|"
                           r"tensorflow::strings::internal::AppendPieces|"
                           r"tensorflow::strings::internal::CatPieces|"
                           r"tensorflow::io::internal::JoinPathImpl")

# Include if matched after exclude
INCLUDE_RE = re.compile(r"^(TF_\w*)$|"
                        r"^(TFE_\w*)$|"
                        r"tensorflow::|"
                        r"functor::|"
                        r"nsync_|"
                        r"perftools::gputools")

# We want to identify data members explicitly in the DEF file, so that no one
# can implicitly link against the DLL if they use one of the variables exported
# from the DLL and the header they use does not decorate the symbol with
# __declspec(dllimport). It is easier to detect what a data symbol does
# NOT look like, so doing it with the below regex.
DATA_EXCLUDE_RE = re.compile(r"[)(]|"
                             r"vftable|"
                             r"vbtable|"
                             r"vcall|"
                             r"RTTI|"
                             r"protobuf::internal::ExplicitlyConstructed")

def get_args():
  """Parse command line."""
  filename_list = lambda x: x.split(";")
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", type=filename_list,
                      help="paths to input libraries separated by semicolons",
                      required=True)
  parser.add_argument("--output", help="output deffile", required=True)
  parser.add_argument("--target", help="name of the target", required=True)
  args = parser.parse_args()
  return args


def main():
  """main."""
  args = get_args()

  # Pipe dumpbin to extract all linkable symbols from libs.
  # Good symbols are collected in candidates and also written to
  # a temp file.
  candidates = []
  tmpfile = tempfile.NamedTemporaryFile(mode="w", delete=False)
  for lib_path in args.input:
    proc = subprocess.Popen([DUMPBIN, "/nologo", "/linkermember:1", lib_path],
                            stdout=subprocess.PIPE)
    for line in codecs.getreader("utf-8")(proc.stdout):
      cols = line.split()
      if len(cols) < 2:
        continue
      sym = cols[1]
      tmpfile.file.write(sym + "\n")
      candidates.append(sym)
    exit_code = proc.wait()
    if exit_code != 0:
      print("{} failed, exit={}".format(DUMPBIN, exit_code))
      return exit_code
  tmpfile.file.close()

  # Run the symbols through undname to get their undecorated name
  # so we can filter on something readable.
  with open(args.output, "w") as def_fp:
    # track dupes
    taken = set()

    # Header for the def file.
    def_fp.write("LIBRARY " + args.target + "\n")
    def_fp.write("EXPORTS\n")
    def_fp.write("\t ??1OpDef@tensorflow@@UEAA@XZ\n")

    # Each symbols returned by undname matches the same position in candidates.
    # We compare on undname but use the decorated name from candidates.
    dupes = 0
    proc = subprocess.Popen([UNDNAME, tmpfile.name], stdout=subprocess.PIPE)
    for idx, line in enumerate(codecs.getreader("utf-8")(proc.stdout)):
      decorated = candidates[idx]
      if decorated in taken:
        # Symbol is already in output, done.
        dupes += 1
        continue

      if not INCLUDEPRE_RE.search(line):
        if EXCLUDE_RE.search(line):
          continue
        if not INCLUDE_RE.search(line):
          continue

      if "deleting destructor" in line:
        # Some of the symbols convered by INCLUDEPRE_RE export deleting
        # destructor symbols, which is a bad idea.
        # So we filter out such symbols here.
        continue

      if DATA_EXCLUDE_RE.search(line):
        def_fp.write("\t" + decorated + "\n")
      else:
        def_fp.write("\t" + decorated + " DATA\n")
      taken.add(decorated)
  exit_code = proc.wait()
  if exit_code != 0:
    print("{} failed, exit={}".format(UNDNAME, exit_code))
    return exit_code

  os.unlink(tmpfile.name)

  print("symbols={}, taken={}, dupes={}"
        .format(len(candidates), len(taken), dupes))
  return 0


if __name__ == "__main__":
  sys.exit(main())
