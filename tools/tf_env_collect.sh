#!/usr/bin/env bash
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

set -u  # Check for undefined variables

die() {
  # Print a message and exit with code 1.
  #
  # Usage: die <error_message>
  #   e.g., die "Something bad happened."

  echo $@
  exit 1
}

echo "Collecting system information..."

OUTPUT_FILE=tf_env.txt
PYTHON_BIN_PATH="$(which python || which python3 || die "Cannot find Python binary")"

HEADER_WIDTH=68
# Create a string of HEADER_WIDTH "=" characters
HEADER=$(printf "%*s" "$HEADER_WIDTH" "" | sed 's/ /=/g')

print_header () {
  # This function simply prints the header with even spacing, 
  # and also prints it to STDERR so that the human running
  # the script sees progress.
  local TITLE="$1"
  echo
  # This line is a bit cryptic, but it essentially
  # just pads the title with "=" to be the desired length.
  local PADDED_TITLE="== $TITLE ${HEADER:${#TITLE}+4}"
  # Echo to STDOUT
  echo "$PADDED_TITLE"
  # Echo to STDERR (to show progress to the user as it runs)
  echo "$PADDED_TITLE" 1>&2
}

# Clear the output file
echo > "$OUTPUT_FILE"

{
  print_header "check python"

  "${PYTHON_BIN_PATH}" <<EOF
import platform

print("""python version: %s
python branch: %s
python build version: %s
python compiler version: %s
python implementation: %s
""" % (
platform.python_version(),
platform.python_branch(),
platform.python_build(),
platform.python_compiler(),
platform.python_implementation(),
))
EOF

  print_header "check os platform"

  "${PYTHON_BIN_PATH}" <<EOF
import platform

PLATFORM_ENTRIES = [
    ("os", "system"),
    ("os kernel version", "version"),
    ("os release version", "release"),
    ("os platform", "platform"),
    ("freedesktop os release", "freedesktop_os_release"),
    ("mac version", "mac_ver"),
    ("uname", "uname"),
    ("architecture", "architecture"),
    ("machine", "machine"),
]

for label, function_name in PLATFORM_ENTRIES:
    if hasattr(platform, function_name):
        function = getattr(platform, function_name)
        result = function()  # Call the function
        print(f"{label}: {result}")
    else:
        print(f"{label}: N/A")
EOF


  print_header 'are we in docker'
  if grep -q docker /proc/1/cgroup; then
    echo "Yes"
  else
    echo "No"
  fi
  
  print_header 'c++ compiler'
  if which c++; then
    c++ --version 2>&1
  else
    echo "Not found"
  fi
  
  print_header 'check pips'
  pip list 2>&1 | grep -E 'proto|numpy|tensorflow|tf_nightly'
  
  
  print_header 'check for virtualenv'
  "${PYTHON_BIN_PATH}" <<EOF
import sys

if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print("Running inside a virtual environment.")
else:
    print("Not running inside a virtual environment.")
EOF

  print_header 'tensorflow import'

  "${PYTHON_BIN_PATH}" <<EOF 2>&1
import tensorflow as tf;
print(f"""
tf.version.VERSION = {tf.version.VERSION}
tf.version.GIT_VERSION = {tf.version.GIT_VERSION}
tf.version.COMPILER_VERSION = {tf.version.COMPILER_VERSION}
""")
print("Sanity check: %r" % tf.constant([1,2,3])[:1])
EOF

  # Record libraries loaded by tensorflow
  LD_DEBUG=libs "${PYTHON_BIN_PATH}" -c "import tensorflow" 2> /tmp/loadedlibs >/dev/null
  if grep libcudnn /tmp/loadedlibs; then
    echo "libcudnn found"
  else
    echo "libcudnn not found"
  fi

  print_header env

  # Note: the usage of "set -u" above would cause these to error if the
  #   basic form [[ -z $LD_LIBRARY_PATH ]] was used.
  if [ -z ${LD_LIBRARY_PATH+x} ]; then
    echo "LD_LIBRARY_PATH is unset";
  else
    echo LD_LIBRARY_PATH ${LD_LIBRARY_PATH} ;
  fi
  if [ -z ${DYLD_LIBRARY_PATH+x} ]; then
    echo "DYLD_LIBRARY_PATH is unset";
  else
    echo DYLD_LIBRARY_PATH ${DYLD_LIBRARY_PATH} ;
  fi
  
  
  print_header nvidia-smi
  nvidia-smi 2>&1
  
  print_header 'cuda libs'

  # Find cudart/cudnn files
  find /usr -type f -name 'libcud*'  2>/dev/null | grep -E 'cuda.*(cudart|cudnn)' | grep -v -F '.cache'

  print_header 'tensorflow installation'
  if ! pip show tensorflow; then
    echo "tensorflow not found"
  fi

  print_header 'tf_nightly installation'
  if ! pip show tf_nightly; then
    echo "tf_nightly not found"
  fi

  print_header 'python version'
  echo '(major, minor, micro, releaselevel, serial)'
  "${PYTHON_BIN_PATH}" -c 'import sys; print(sys.version_info[:])'
  
  print_header 'bazel version'
  bazel version

# Remove any lines with google.
} | grep -v -i google >> "$OUTPUT_FILE"

echo "Wrote environment to ${OUTPUT_FILE}. You can review the contents of that file."
echo "and use it to populate the fields in the github issue template."
echo
echo "cat ${OUTPUT_FILE}"
echo

