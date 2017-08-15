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
python_bin_path=$(which python || which python3 || die "Cannot find Python binary")

{
  echo
  echo "== cat /etc/issue ==============================================="
  uname -a
  uname=`uname -s`
  if [ "$(uname)" == "Darwin" ]; then
    echo Mac OS X `sw_vers -productVersion`
  elif [ "$(uname)" == "Linux" ]; then
    cat /etc/*release | grep VERSION
  fi
  
  echo
  echo '== are we in docker ============================================='
  num=`cat /proc/1/cgroup | grep docker | wc -l`;
  if [ $num -ge 1 ]; then
    echo "Yes"
  else
    echo "No"
  fi
  
  echo
  echo '== compiler ====================================================='
  c++ --version 2>&1
  
  echo
  echo '== uname -a ====================================================='
  uname -a
  
  echo
  echo '== check pips ==================================================='
  pip list 2>&1 | grep "proto\|numpy\|tensorflow"
  
  
  echo
  echo '== check for virtualenv ========================================='
  ${python_bin_path} -c "import sys;print(hasattr(sys, \"real_prefix\"))"
  
  echo
  echo '== tensorflow import ============================================'
} >> ${OUTPUT_FILE}

cat <<EOF > /tmp/check_tf.py
import tensorflow as tf;
print("tf.VERSION = %s" % tf.VERSION)
print("tf.GIT_VERSION = %s" % tf.GIT_VERSION)
print("tf.COMPILER_VERSION = %s" % tf.GIT_VERSION)
with tf.Session() as sess:
  print("Sanity check: %r" % sess.run(tf.constant([1,2,3])[:1]))
EOF
${python_bin_path} /tmp/check_tf.py 2>&1  >> ${OUTPUT_FILE}

DEBUG_LD=libs ${python_bin_path} -c "import tensorflow"  2>>${OUTPUT_FILE} > /tmp/loadedlibs

{
  grep libcudnn.so /tmp/loadedlibs
  echo
  echo '== env =========================================================='
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
  
  
  echo
  echo '== nvidia-smi ==================================================='
  nvidia-smi 2>&1
  
  echo
  echo '== cuda libs  ==================================================='
} >> ${OUTPUT_FILE}

find /usr/local -type f -name 'libcudart*'  2>/dev/null | grep cuda |  grep -v "\\.cache" >> ${OUTPUT_FILE}
find /usr/local -type f -name 'libudnn*'  2>/dev/null | grep cuda |  grep -v "\\.cache" >> ${OUTPUT_FILE}

# Remove any words with google.
mv $OUTPUT_FILE old-$OUTPUT_FILE
grep -v -i google old-${OUTPUT_FILE} > $OUTPUT_FILE

echo "Wrote environment to ${OUTPUT_FILE}. You can review the contents of that file."
echo "and use it to populate the fields in the github issue template."
echo
echo "cat ${OUTPUT_FILE}"
echo

