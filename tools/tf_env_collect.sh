#!/usr/bin/env bash

set -u  # Check for undefined variables

echo "Collecting system information..."

OUTPUT_FILE=tf_env.txt

echo >> $OUTPUT_FILE
echo "== cat /etc/issue ===============================================" >> $OUTPUT_FILE
uname -a >> $OUTPUT_FILE
uname=`uname -s`
if [ "$(uname)" == "Darwin" ]; then
  echo Mac OS X `sw_vers -productVersion` >> $OUTPUT_FILE
elif [ "$(uname)" == "Linux" ]; then
  cat /etc/*release | grep VERSION >> $OUTPUT_FILE
fi


echo >> $OUTPUT_FILE
echo '== are we in docker =============================================' >> $OUTPUT_FILE
num=`cat /proc/1/cgroup | grep docker | wc -l`;
if [ $num -ge 1 ]; then
  echo "Yes" >> $OUTPUT_FILE
else
  echo "No" >> $OUTPUT_FILE
fi

echo >> $OUTPUT_FILE
echo '== compiler =====================================================' >> $OUTPUT_FILE
c++ --version &>> $OUTPUT_FILE

echo >> $OUTPUT_FILE
echo '== uname -a =====================================================' >> $OUTPUT_FILE
uname -a >> $OUTPUT_FILE

echo >> $OUTPUT_FILE
echo '== check pips ===================================================' >> $OUTPUT_FILE
pip list 2>&1 | grep "proto\|numpy\|tensorflow" &>> $OUTPUT_FILE


echo >> $OUTPUT_FILE
echo '== check for virtualenv =========================================' >> $OUTPUT_FILE
python -c "import sys;print(hasattr(sys, \"real_prefix\"))" >> $OUTPUT_FILE

echo >> $OUTPUT_FILE
echo '== tensorflow import ============================================' >> $OUTPUT_FILE
cat <<EOF > /tmp/check_tf.py
import tensorflow as tf;
print("tf.VERSION = %s" % tf.VERSION)
print("tf.GIT_VERSION = %s" % tf.GIT_VERSION)
print("tf.COMPILER_VERSION = %s" % tf.GIT_VERSION)
with tf.Session() as sess:
  print("Sanity check: %r" % sess.run(tf.constant([1,2,3])[:1]))
EOF
python /tmp/check_tf.py &>> ${OUTPUT_FILE}

DEBUG_LD=libs python -c "import tensorflow"  2>>${OUTPUT_FILE} > /tmp/loadedlibs
grep libcudnn.so /tmp/loadedlibs >> $OUTPUT_FILE

echo >> $OUTPUT_FILE
echo '== env ==========================================================' >> $OUTPUT_FILE
if [ -z ${LD_LIBRARY_PATH+x} ]; then
  echo "LD_LIBRARY_PATH is unset" >> $OUTPUT_FILE;
else
  echo LD_LIBRARY_PATH ${LD_LIBRARY_PATH}  >> $OUTPUT_FILE;
fi
if [ -z ${DYLD_LIBRARY_PATH+x} ]; then
  echo "DYLD_LIBRARY_PATH is unset" >> $OUTPUT_FILE;
else
  echo DYLD_LIBRARY_PATH ${DYLD_LIBRARY_PATH}  >> $OUTPUT_FILE;
fi


echo >> $OUTPUT_FILE >> $OUTPUT_FILE
echo '== nvidia-smi ===================================================' >> $OUTPUT_FILE
nvidia-smi &>> $OUTPUT_FILE

echo >> $OUTPUT_FILE

echo '== cuda libs  ===================================================' >> $OUTPUT_FILE
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

