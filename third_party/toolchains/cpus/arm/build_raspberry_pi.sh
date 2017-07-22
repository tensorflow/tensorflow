#!/usr/bin/env bash
set -e

# To be able to build any of the python packages, it is assumed that
# you have a python directory in /usr/include/arm-linux-gnueabihf/.

# To install the architecture includes for python on ubuntu trusty;
# run:
#  sudo dpkg --add-architecture armhf
#  echo "deb [arch=armhf] http://ports.ubuntu.com/ trusty main universe" | \
#    sudo tee -a /etc/apt/sources.list.d/armhf.list
#  # Ignore errors about missing armhf packages in other repos.
#  sudo aptitude update
#  # Use aptitude; apt-get sometimes runs into errors on this command.
#  sudo aptitude install libpython-all-dev:armhf python-numpy
#

yes '' | ./configure

bazel build -c opt --copt=-march=armv6 --copt=-mfpu=vfp \
  --copt=-funsafe-math-optimizations --copt=-ftree-vectorize \
  --copt=-fomit-frame-pointer --cpu=armeabi \
  --crosstool_top=@local_config_arm_compiler//:toolchain \
  --verbose_failures \
  //tensorflow/tools/benchmark:benchmark_model \
  //tensorflow/tools/pip_package:build_pip_package

TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)
echo "Final outputs will go to ${TMPDIR}"

# Build a universal wheel.
BDIST_OPTS="--universal" \
  bazel-bin/tensorflow/tools/pip_package/build_pip_package "${TMPDIR}"

OLD_FN=$(ls "${TMPDIR}" | grep \.whl)
SUB='s/tensorflow-([^-]+)-([^-]+)-.*/tensorflow-\1-\2-none-any.whl/; print'
NEW_FN=$(echo "${OLD_FN}" | perl -ne "${SUB}")
mv "${TMPDIR}/${OLD_FN}" "${TMPDIR}/${NEW_FN}"
cp bazel-bin/tensorflow/tools/benchmark/benchmark_model "${TMPDIR}"

echo "Output can be found here:"
find "${TMPDIR}"
