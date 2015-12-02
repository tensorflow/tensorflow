#!/bin/bash

set -e -o errexit

EXPECTED_PATHS="util/python/python_include util/python/python_lib third_party/py/numpy/numpy_include"

function main {
  argument="$1"
  shift
  case $argument in
    --check)
      check_python
      exit 0
      ;;
    --setup)
      setup_python "$1"
      exit 0
      ;;
  esac
}

function setup_python {
  PYTHON_BIN_PATH="$1";

  if [ -z "$PYTHON_BIN_PATH" ]; then
    echo "PYTHON_BIN_PATH was not provided.  Did you run configure?"
    exit 1
  fi
  if [ ! -x "$PYTHON_BIN_PATH" ]  || [ -d "$PYTHON_BIN_PATH" ]; then
    echo "PYTHON_BIN_PATH is not executable.  Is it the python binary?"
    exit 1
  fi

  local python_include=$("${PYTHON_BIN_PATH}" -c 'from __future__ import print_function; from distutils import sysconfig; print(sysconfig.get_python_inc());')
  if [ "$python_include" == "" ]; then
    echo -e "\n\nERROR: Problem getting python include path.  Is distutils installed?"
    exit 1
  fi
  local python_lib=$("${PYTHON_BIN_PATH}" -c 'from __future__ import print_function; from distutils import sysconfig; print(sysconfig.get_python_lib());')
  if [ "$python_lib" == "" ]; then
    echo -e "\n\nERROR: Problem getting python lib path.  Is distutils installed?"
    exit 1
  fi
  local numpy_include=$("${PYTHON_BIN_PATH}" -c 'from __future__ import print_function; import numpy; print(numpy.get_include());')
  if [ "$numpy_include" == "" ]; then
    echo -e "\n\nERROR: Problem getting numpy include path.  Is numpy installed?"
    exit 1
  fi

  for x in $EXPECTED_PATHS; do
    if [ -e "$x" ]; then
      rm "$x"
    fi
  done

  ln -s "${python_include}" util/python/python_include
  ln -s "${python_lib}" util/python/python_lib
  ln -s "${numpy_include}" third_party/py/numpy/numpy_include
}

function check_python {
  for x in $EXPECTED_PATHS; do
    if [ ! -e "$x" ]; then
      echo -e "\n\nERROR: Cannot find '${x}'.  Did you run configure?\n\n" 1>&2
      exit 1
    fi
    if [ ! -L "${x}" ]; then
      echo -e "\n\nERROR: '${x}' is not a symbolic link.  Internal error.\n\n" 1>&2
      exit 1
    fi
    true_path=$(readlink "${x}")
    if [ ! -d "${true_path}" ]; then
      echo -e "\n\nERROR: '${x}' does not refer to an existing directory: ${true_path}.  Do you need to rerun configure?\n\n" 1>&2
      exit 1
    fi
  done
}

main "$@"
