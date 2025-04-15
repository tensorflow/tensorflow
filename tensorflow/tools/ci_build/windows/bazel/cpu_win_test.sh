#!/bin/bash
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
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

# This script is a CI script maintained by Intel and is used to launch the nightly CI test 
# build on the Windows platform.
# It assumes the standard setup on tensorflow Jenkins Windows machines.
# Update the flags/variables below to make it work on your local system.

# REQUIREMENTS:
# * All installed in standard locations:
#   - JDK8, and JAVA_HOME set.
#   - Microsoft Visual Studio 2015 Community Edition
#   - Msys2
#   - Python 3.x (with pip, setuptools, venv)
# * Bazel Windows executable copied as "bazel.exe" and included in PATH.


# All commands should be visible (-x).
set -x

POSITIONAL_ARGS=()
XBF_ARGS=""
XTF_ARGS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --extra_build_flags)
      XBF_ARGS="$2"
      shift # past argument
      shift # past value
      ;;
    --extra_test_flags)
      XTF_ARGS="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

# Bazelisk (renamed as bazel) is kept in C:\Tools
export PATH=/c/ProgramData/chocolatey/bin:/c/Tools/bazel:/c/Program\ Files/Git:/c/Program\ \
Files/Git/cmd:/c/msys64:/c/msys64/usr/bin:/c/Windows/system32:/c/Windows:/c/Windows/System32/Wbem

# Environment variables to be set by Jenkins before calling this script

export PYTHON_VERSION=${PYTHON_VERSION:-"310"}
export TF_PYTHON_VERSION=${PYTHON_VERSION:0:1}.${PYTHON_VERSION:1}
# keep the tensorflow git repo clone under here as tensorflow subdir
MYTFWS_ROOT=${WORKSPACE:-"C:/Users/mlp_admin"} 
MYTFWS_ROOT=`cygpath -m $MYTFWS_ROOT`
export MYTFWS_ROOT="$MYTFWS_ROOT"
export MYTFWS_NAME="tensorflow"
export MYTFWS="${MYTFWS_ROOT}/${MYTFWS_NAME}"
export MYTFWS_ARTIFACT="${MYTFWS_ROOT}/artifact"


# Import General Test Target
source tensorflow/tools/ci_build/build_scripts/DEFAULT_TEST_TARGETS.sh

# Environment variables specific to the system where this job is running, are to
# be set by a script for the specific system. This needs to be set here by sourcing a file.

export TMP=${TMP:-"${MYTFWS_ROOT}/tmp"}
export TEMP="$TMP"
export TMPDIR=${TMPDIR:-"${MYTFWS}-build"} # used internally by TF build
export TEST_TARGET=${TEST_TARGET:-"${DEFAULT_BAZEL_TARGETS}"}
export MSYS_LOCATION='C:/msys64'
export GIT_LOCATION='C:/Program Files/Git'
export JAVA_LOCATION='C:/Program Files/Eclipse Adoptium/jdk-11.0.14.101-hotspot'
export VS_LOCATION='C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools'
export NATIVE_PYTHON_LOCATION="C:/Python${PYTHON_VERSION}"
export PORTSERVER_LOCATION='C:/Program Files/python_portpicker/src/portserver.py'


echo "*** *** hostname is $(hostname) *** ***"
which bazel
which git
[[ -e "$NATIVE_PYTHON_LOCATION/python.exe" ]] || \
{ echo "Specified Python path is incorrect: $NATIVE_PYTHON_LOCATION"; exit 1;}
[[ -e "$NATIVE_PYTHON_LOCATION/Scripts/pip.exe" ]] || \
{ echo "Specified Python path has no pip: $NATIVE_PYTHON_LOCATION"; exit 1;}
[[ -e "$NATIVE_PYTHON_LOCATION/Lib/venv" ]] || \
{ echo "Specified Python path has no venv: $NATIVE_PYTHON_LOCATION"; exit 1;}

$NATIVE_PYTHON_LOCATION/python.exe -m pip list

# =========================== Start of actual script =========================
# This script sets necessary environment variables and runs TF-Windows build & unit tests
# We also assume a few Software components are also installed in the machine: MS VC++,
# MINGW SYS64, Python 3.x, JAVA, Git, Bazelisk etc.

# Asuumptions
# 1) TF repo cloned into to %WORKSPACE%\tensorflow (aka %TF_LOCATION%)
# 2) Bazelisk is installed in "C:\Tools\Bazel"
# 3) The following jobs-specific env vars will be exported  by the caller
#       WORKSPACE (ex. C:\Jenkins\workspace\tensorflow-eigen-test-win)
#       PYTHON_VERSION  (ex. 38)
#       PIP_MODULES (if set will contain any additional pip packages)
# 4) System-specific env variables for the location of different software
#    components needed for building.

# Create Python virtual env
cd ${MYTFWS_ROOT}
export PYTHON_DIRECTORY="${MYTFWS_ROOT}"/venv_py${PYTHON_VERSION}
"${NATIVE_PYTHON_LOCATION}"/python.exe -mvenv --clear  "${PYTHON_DIRECTORY}"

#activate virtual env
source "${PYTHON_DIRECTORY}"/Scripts/activate

which python
python --version

# Install pip modules specs from tensorflow/tools/ci_build/release/requirements_common.txt
python -m pip install -r $MYTFWS/tensorflow/tools/ci_build/release/requirements_common.txt

# set up other Variables required by Bazel.
export PYTHON_BIN_PATH="${PYTHON_DIRECTORY}"/Scripts/python.exe
export PYTHON_LIB_PATH="${PYTHON_DIRECTORY}"/Lib/site-packages
export BAZEL_VS=${VS_LOCATION}
export BAZEL_VC=${VS_LOCATION}/VC
export JAVA_HOME=${JAVA_LOCATION}
export BAZEL_SH="${MSYS_LOCATION}"/usr/bin/bash.exe

cd ${MYTFWS_ROOT}
mkdir -p "$TMP"
mv summary.log summary.log.bak
mv test_failures.log test_failures.log.bak
mv test_run.log test_run.log.bak
rm -rf ${MYTFWS_ARTIFACT}
mkdir -p ${MYTFWS_ARTIFACT}

cd $MYTFWS

# All commands shall pass
set -e

# Setting up the environment variables Bazel and ./configure needs
source "tensorflow/tools/ci_build/windows/bazel/common_env.sh" \
  || { echo "Failed to source common_env.sh" >&2; exit 1; }

# load bazel_test_lib.sh
source "tensorflow/tools/ci_build/windows/bazel/bazel_test_lib.sh" \
  || { echo "Failed to source bazel_test_lib.sh" >&2; exit 1; }

# Recreate an empty bazelrc file under source root
export TMP_BAZELRC=.tmp.bazelrc
rm -f "${TMP_BAZELRC}"
touch "${TMP_BAZELRC}"

function cleanup {
  # Remove all options in .tmp.bazelrc
  echo "" > "${TMP_BAZELRC}"
}
trap cleanup EXIT

# Enable short object file path to avoid long path issues on Windows.
echo "startup --output_user_root=${TMPDIR}" >> "${TMP_BAZELRC}"

if ! grep -q "import %workspace%/${TMP_BAZELRC}" .bazelrc; then
  echo "import %workspace%/${TMP_BAZELRC}" >> .bazelrc
fi

run_configure_for_cpu_build

# Unset so the script continues even if commands fail, needed to correctly process the logs
set +e   

# start the port server before testing so that each invocation of 
# portpicker will defer to the single instance of portserver
# Define the batch script content
BATCH_SCRIPT_START="
@echo off
set SCRIPT_PATH="${PORTSERVER_LOCATION}"
echo Starting the server...
start \"PORTSERVER\" \"%PYTHON_BIN_PATH%\" \"%SCRIPT_PATH%\"
echo Server started.
"
# Save the batch script content to a temporary batch file
BATCH_SCRIPT_FILE="temp_script.bat"
echo "$BATCH_SCRIPT_START" > "$BATCH_SCRIPT_FILE"

# Run the batch script
cmd.exe /C "$BATCH_SCRIPT_FILE"

# NUMBER_OF_PROCESSORS is predefined on Windows
N_JOBS="${NUMBER_OF_PROCESSORS}"
bazel --windows_enable_symlinks test \
  --action_env=TEMP=${TMP} --action_env=TMP=${TMP} ${XTF_ARGS} \
  --experimental_cc_shared_library --enable_runfiles --nodistinct_host_configuration \
  --build_tag_filters=-no_pip,-no_windows,-no_oss,-gpu,-tpu \
  --test_tag_filters=-no_windows,-no_oss,-gpu,-tpu \
  --build_tests_only --config=monolithic \
  --dynamic_mode=off --config=xla --config=opt \
  --build_tests_only -k \
  --test_env=PORTSERVER_ADDRESS=@unittest-portserver \
  --repo_env=TF_PYTHON_VERSION=${TF_PYTHON_VERSION} \
  --test_size_filters=small,medium --jobs="${N_JOBS}" --test_timeout=300,450,1200,3600 \
  --flaky_test_attempts=3 --verbose_failures \
  ${POSITIONAL_ARGS[@]} \
  -- ${TEST_TARGET} \
  > run.log 2>&1

build_ret_val=$?   # Store the ret value

BATCH_SCRIPT_STOP="
echo Killing the server...
taskkill /FI \"WindowTitle eq PORTSERVER*\" /F /T
echo Server killed.
"
BATCH_SCRIPT_FILEl="temp_script.bat"
echo "$BATCH_SCRIPT_STOP" > "$BATCH_SCRIPT_FILEl"
cmd.exe /C "$BATCH_SCRIPT_FILEl"

# Removing the temporary batch script
rm -f "$BATCH_SCRIPT_FILE"
rm -f "$BATCH_SCRIPT_FILEl"

# process results
cd $MYTFWS_ROOT

# Check to make sure the log was created
[ ! -f "${MYTFWS}"/run.log  ] && exit 1

# handle logs for unit test
cd ${MYTFWS_ARTIFACT}
cp "${MYTFWS}"/run.log ./test_run.log

fgrep "FAILED: Build did NOT complete" test_run.log > summary.log
fgrep "Executed" test_run.log >> summary.log

[ $build_ret_val -eq 0 ] && exit 0

echo "FAILED TESTS:" > test_failures.log
fgrep "FAILED" test_run.log | grep " ms)" | sed -e 's/^.*\] //' -e 's/ .*$//' | sort | \
uniq >> test_failures.log
echo >> test_failures.log
echo "SKIPPED TESTS:" >> test_failures.log
fgrep "SKIPPED" test_run.log | grep -v "listed below:" | sed -e 's/^.*\] //' | sort | \
uniq >> test_failures.log

exit 1
