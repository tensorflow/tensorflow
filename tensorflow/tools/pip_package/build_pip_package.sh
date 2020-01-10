#!/bin/bash

set -e

function main() {
  if [ $# -lt 1 ] ; then
    echo "No destination dir provided"
    exit 1
  fi

  DEST=$1
  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)

  echo `date` : "=== Using tmpdir: ${TMPDIR}"

  if [ ! -d bazel-bin/tensorflow ]; then
    echo "Could not find bazel-bin.  Did you run from the root of the build tree?"
    exit 1
  fi
  cp -R \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package.runfiles/* \
    ${TMPDIR}

  cp tensorflow/tools/pip_package/MANIFEST.in ${TMPDIR}
  cp tensorflow/tools/pip_package/README ${TMPDIR}
  cp tensorflow/tools/pip_package/setup.py ${TMPDIR}
  pushd ${TMPDIR}
  rm -f MANIFEST
  echo `date` : "=== Building wheel"
  python setup.py sdist bdist_wheel >/dev/null
  mkdir -p ${DEST}
  cp dist/* ${DEST}
  popd
  rm -rf ${TMPDIR}
  echo `date` : "=== Output wheel file is in: ${DEST}"
}

main "$@"
