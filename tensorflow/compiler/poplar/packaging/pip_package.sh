#!/usr/bin/env bash

if [ $# -lt 1 ] ; then
  echo "No destination dir provided"
  exit 1
fi

DEST=$1
TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)

echo $(date) : "=== Using tmpdir: ${TMPDIR}"

if [ ! -d bazel-bin/tensorflow ]; then
  echo "Could not find bazel-bin.  Did you run from the root of the build tree?"
  exit 1
fi

mkdir ${TMPDIR}/poplar_plugin

cp tensorflow/compiler/poplar/packaging/README.rst ${TMPDIR}
cp tensorflow/compiler/poplar/packaging/setup.py ${TMPDIR}
cp tensorflow/compiler/poplar/__init__.py ${TMPDIR}/poplar_plugin
cp tensorflow/compiler/poplar/poplar_plugin.py ${TMPDIR}/poplar_plugin
cp bazel-genfiles/tensorflow/compiler/poplar/* ${TMPDIR}/poplar_plugin


source tools/python_bin_path.sh

pushd ${TMPDIR}

ls -lR

echo $(date) : "=== Building wheel"
"${PYTHON_BIN_PATH:-python}" setup.py bdist_wheel #>/dev/null
mkdir -p ${DEST}
cp dist/* ${DEST}
popd
rm -rf ${TMPDIR}
echo $(date) : "=== Output wheel file is in: ${DEST}"
