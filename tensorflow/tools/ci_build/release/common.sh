#!/usr/bin/env bash
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# External `common.sh`

# Keep in sync with tensorflow_estimator and configure.py.
# LINT.IfChange
LATEST_BAZEL_VERSION=3.7.2
# LINT.ThenChange(
#   //tensorflow/opensource_only/configure.py,
#   //tensorflow_estimator/google/kokoro/common.sh,
#   //tensorflow/tools/ci_build/install/install_bazel.sh,
#   //tensorflow/tools/ci_build/install/install_bazel_from_source.sh)

# Run flaky functions with retries.
# run_with_retry cmd
function run_with_retry {
  eval "$1"
  # If the command fails retry again in 60 seconds.
  if [[ $? -ne 0 ]]; then
    sleep 60
    eval "$1"
  fi
}

function die() {
  echo "$@" 1>&2 ; exit 1;
}

# A small utility to run the command and only print logs if the command fails.
# On success, all logs are hidden.
function readable_run {
  # Disable debug mode to avoid printing of variables here.
  set +x
  result=$("$@" 2>&1) || die "$result"
  echo "$@"
  echo "Command completed successfully at $(date)"
  set -x
}

# LINT.IfChange
# Redirect bazel output dir b/73748835
function set_bazel_outdir {
  mkdir -p /tmpfs/bazel_output
  export TEST_TMPDIR=/tmpfs/bazel_output
}

# Downloads bazelisk to ~/bin as `bazel`.
function install_bazelisk {
  date
  case "$(uname -s)" in
    Darwin) local name=bazelisk-darwin-amd64 ;;
    Linux)  local name=bazelisk-linux-amd64  ;;
    *) die "Unknown OS: $(uname -s)" ;;
  esac
  mkdir -p "$HOME/bin"
  wget --no-verbose -O "$HOME/bin/bazel" \
      "https://github.com/bazelbuild/bazelisk/releases/download/v1.3.0/$name"
  chmod u+x "$HOME/bin/bazel"
  if [[ ! ":$PATH:" =~ :"$HOME"/bin/?: ]]; then
    PATH="$HOME/bin:$PATH"
  fi
  set_bazel_outdir
  which bazel
  bazel version
  date
}

# Install the given bazel version on linux
function update_bazel_linux {
  if [[ -z "$1" ]]; then
    BAZEL_VERSION=${LATEST_BAZEL_VERSION}
  else
    BAZEL_VERSION=$1
  fi
  rm -rf ~/bazel
  mkdir ~/bazel

  pushd ~/bazel
  readable_run wget https://github.com/bazelbuild/bazel/releases/download/"${BAZEL_VERSION}"/bazel-"${BAZEL_VERSION}"-installer-linux-x86_64.sh
  chmod +x bazel-*.sh
  ./bazel-"${BAZEL_VERSION}"-installer-linux-x86_64.sh --user
  rm bazel-"${BAZEL_VERSION}"-installer-linux-x86_64.sh
  popd

  PATH="/home/kbuilder/bin:$PATH"
  set_bazel_outdir
  which bazel
  bazel version
}
# LINT.ThenChange(
#   //tensorflow_estimator/google/kokoro/common.sh)

function install_ubuntu_16_pip_deps {
  PIP_CMD="pip"

  while true; do
    if [[ -z "${1}" ]]; then
      break
    fi
    if [[ "$1" == "pip"* ]]; then
      PIP_CMD="$1"
    fi
    shift
  done

  # First, upgrade pypi wheels
  "${PIP_CMD}" install --user --upgrade 'setuptools<53' pip wheel

  # LINT.IfChange(linux_pip_installations_orig)
  # Remove any historical keras package if they are installed.
  "${PIP_CMD}" list
  "${PIP_CMD}" uninstall -y keras
  "${PIP_CMD}" install --user -r tensorflow/tools/ci_build/release/requirements_ubuntu.txt
  # LINT.ThenChange(:mac_pip_installations)
}

# Gradually replace function install_ubuntu_16_pip_deps.
# TODO(lpak): delete install_ubuntu_16_pip_deps when completely replaced.
function install_ubuntu_16_python_pip_deps {
  PIP_CMD="pip"

  while true; do
    if [[ -z "${1}" ]]; then
      break
    fi
    if [[ "$1" == "pip"* ]]; then
      PIP_CMD="$1"
    fi
    if [[ "$1" == "python"* ]]; then
      PIP_CMD="${1} -m pip"
    fi
    shift
  done

  # First, upgrade pypi wheels
  ${PIP_CMD} install --user --upgrade 'setuptools<53' pip wheel

  # LINT.IfChange(linux_pip_installations)
  # Remove any historical keras package if they are installed.
  ${PIP_CMD} list
  ${PIP_CMD} uninstall -y keras
  ${PIP_CMD} install --user -r tensorflow/tools/ci_build/release/requirements_ubuntu.txt
  # LINT.ThenChange(:mac_pip_installations)
}

function install_macos_pip_deps {

  PIP_CMD="python -m pip"

  # First, upgrade pypi wheels
  ${PIP_CMD} install --upgrade 'setuptools<53' pip wheel

  # LINT.IfChange(mac_pip_installations)
  # Remove any historical keras package if they are installed.
  ${PIP_CMD} list
  ${PIP_CMD} uninstall -y keras
  ${PIP_CMD} install -r tensorflow/tools/ci_build/release/requirements_mac.txt
  # LINT.ThenChange(:linux_pip_installations_orig)
  # LINT.ThenChange(:install_macos_pip_deps_no_venv)
  # LINT.ThenChange(:linux_pip_installations)
}

# This hack is unfortunately necessary for MacOS builds that use pip_new.sh
# You cannot deactivate a virtualenv from a subshell.
function install_macos_pip_deps_no_venv {

  PIP_CMD="${1} -m pip"

  # First, upgrade pypi wheels
  ${PIP_CMD} install --user --upgrade 'setuptools<53' pip wheel

  # LINT.IfChange(mac_pip_installations)
  # Remove any historical keras package if they are installed.
  ${PIP_CMD} list
  ${PIP_CMD} uninstall -y keras
  ${PIP_CMD} install --user -r tensorflow/tools/ci_build/release/requirements_mac.txt
  # LINT.ThenChange(:install_macos_pip_deps)
}

function setup_venv_macos () {
  # First argument needs to be the python executable.
  ${1} -m pip install virtualenv
  ${1} -m virtualenv tf_build_env
  source tf_build_env/bin/activate
  install_macos_pip_deps
}

function activate_venv_macos () {
  source tf_build_env/bin/activate
}

function setup_python_from_pyenv_macos {
  if [[ -z "${1}" ]]; then
    PY_VERSION=3.9.1
  else
    PY_VERSION=$1
  fi

  git clone --branch 1.2.27 https://github.com/pyenv/pyenv.git

  PYENV_ROOT="$(pwd)/pyenv"
  export PYENV_ROOT
  export PATH="$PYENV_ROOT/bin:$PATH"

  eval "$(pyenv init -)"

  pyenv install -s "${PY_VERSION}"
  pyenv local "${PY_VERSION}"
  python --version
}

function maybe_skip_v1 {
  # If we are building with v2 by default, skip tests with v1only tag.
  if grep -q "build --config=v2" ".bazelrc"; then
    echo ",-v1only"
  else
    echo ""
  fi
}

# Copy and rename a wheel to a new project name.
# Usage: copy_to_new_project_name <whl_path> <new_project_name>, for example
# copy_to_new_project_name test_dir/tf_nightly-1.15.0.dev20190813-cp35-cp35m-manylinux2010_x86_64.whl tf_nightly_cpu
# will create a wheel with the same tags, but new project name under the same
# directory at
# test_dir/tf_nightly_cpu-1.15.0.dev20190813-cp35-cp35m-manylinux2010_x86_64.whl
function copy_to_new_project_name {
  WHL_PATH="$1"
  NEW_PROJECT_NAME="$2"
  PYTHON_CMD="$3"

  ORIGINAL_WHL_NAME=$(basename "${WHL_PATH}")
  ORIGINAL_WHL_DIR=$(realpath "$(dirname "${WHL_PATH}")")
  ORIGINAL_PROJECT_NAME="$(echo "${ORIGINAL_WHL_NAME}" | cut -d '-' -f 1)"
  FULL_TAG="$(echo "${ORIGINAL_WHL_NAME}" | cut -d '-' -f 2-)"
  NEW_WHL_NAME="${NEW_PROJECT_NAME}-${FULL_TAG}"
  VERSION="$(echo "${FULL_TAG}" | cut -d '-' -f 1)"

  ORIGINAL_WHL_DIR_PREFIX="${ORIGINAL_PROJECT_NAME}-${VERSION}"
  NEW_WHL_DIR_PREFIX="${NEW_PROJECT_NAME}-${VERSION}"

 TMP_DIR="$(mktemp -d)"
 ${PYTHON_CMD} -m wheel unpack "${WHL_PATH}"
 mv "${ORIGINAL_WHL_DIR_PREFIX}" "${TMP_DIR}"
 pushd "${TMP_DIR}/${ORIGINAL_WHL_DIR_PREFIX}"

  mv "${ORIGINAL_WHL_DIR_PREFIX}.dist-info" "${NEW_WHL_DIR_PREFIX}.dist-info"
  if [[ -d "${ORIGINAL_WHL_DIR_PREFIX}.data" ]]; then
    mv "${ORIGINAL_WHL_DIR_PREFIX}.data" "${NEW_WHL_DIR_PREFIX}.data"
  fi

  ORIGINAL_PROJECT_NAME_DASH="${ORIGINAL_PROJECT_NAME//_/-}"
  NEW_PROJECT_NAME_DASH="${NEW_PROJECT_NAME//_/-}"

  # We need to change the name in the METADATA file, but we need to ensure that
  # all other occurences of the name stay the same, otherwise things such as
  # URLs and depedencies might be broken (for example, replacing without care
  # might transform a `tensorflow_estimator` dependency into
  # `tensorflow_gpu_estimator`, which of course does not exist -- except by
  # manual upload of a manually altered `tensorflow_estimator` package)
  sed -i.bak "s/Name: ${ORIGINAL_PROJECT_NAME_DASH}/Name: ${NEW_PROJECT_NAME_DASH}/g" "${NEW_WHL_DIR_PREFIX}.dist-info/METADATA"

  ${PYTHON_CMD} -m wheel pack .
  mv *.whl "${ORIGINAL_WHL_DIR}"

  popd
  rm -rf "${TMP_DIR}"
}

# Create minimalist test XML for web view. It includes the pass/fail status
# of each target, without including errors or stacktraces.
# Remember to "set +e" before calling bazel or we'll only generate the XML for
# passing runs.
function test_xml_summary {
  set +x
  set +e
  mkdir -p "${KOKORO_ARTIFACTS_DIR}/${KOKORO_JOB_NAME}/summary"
  # First build the repeated inner XML blocks, since the header block needs to
  # report the number of test cases / failures / errors.
  # TODO(rsopher): handle build breakages
  # TODO(rsopher): extract per-test times as well
  TESTCASE_XML="$(sed -n '/INFO:\ Build\ completed/,/INFO:\ Build\ completed/p' \
    /tmpfs/kokoro_build.log \
    | grep -E '(PASSED|FAILED|TIMEOUT)\ in' \
    | while read -r line; \
      do echo '<testcase name="'"$(echo "${line}" | tr -s ' ' | cut -d ' ' -f 1)"\
          '" status="run" classname="" time="0">'"$( \
        case "$(echo "${line}" | tr -s ' ' | cut -d ' ' -f 2)" in \
          FAILED) echo '<failure message="" type=""/>' ;; \
          TIMEOUT) echo '<failure message="timeout" type=""/>' ;; \
        esac; \
      )"'</testcase>'; done; \
  )"
  NUMBER_OF_TESTS="$(echo "${TESTCASE_XML}" | wc -l)"
  NUMBER_OF_FAILURES="$(echo "${TESTCASE_XML}" | grep -c '<failure')"
  echo '<?xml version="1.0" encoding="UTF-8"?>'\
  '<testsuites name="1"  tests="1" failures="0" errors="0" time="0">'\
  '<testsuite name="Kokoro Summary" tests="'"${NUMBER_OF_TESTS}"\
  '" failures="'"${NUMBER_OF_FAILURES}"'" errors="0" time="0">'\
  "${TESTCASE_XML}"'</testsuite></testsuites>'\
  > "${KOKORO_ARTIFACTS_DIR}/${KOKORO_JOB_NAME}/summary/sponge_log.xml"
}

# Create minimalist test XML for web view, then exit.
# Ends script with value of previous command, meant to be called immediately
# after bazel as the last call in the build script.
function test_xml_summary_exit {
  RETVAL=$?
  test_xml_summary
  exit "${RETVAL}"
}

# CPU size
MAC_CPU_MAX_WHL_SIZE=200M
LINUX_CPU_MAX_WHL_SIZE=170M
WIN_CPU_MAX_WHL_SIZE=170M
# GPU size
LINUX_GPU_MAX_WHL_SIZE=450M
WIN_GPU_MAX_WHL_SIZE=345M

function test_tf_whl_size() {
  WHL_PATH=${1}
  # First, list all wheels with their sizes:
  echo "Found these wheels: "
  find $WHL_PATH -type f -exec ls -lh {} \;
  echo "===================="
  # Check CPU whl size.
  if [[ "$WHL_PATH" == *"_cpu"* ]]; then
    # Check MAC CPU whl size.
    if [[ "$WHL_PATH" == *"-macos"* ]] && [[ $(find $WHL_PATH -type f -size +${MAC_CPU_MAX_WHL_SIZE}) ]]; then
        echo "Mac CPU whl size has exceeded ${MAC_CPU_MAX_WHL_SIZE}. To keep
within pypi's CDN distribution limit, we must not exceed that threshold."
      return 1
    fi
    # Check Linux CPU whl size.
    if [[ "$WHL_PATH" == *"-manylinux"* ]] && [[ $(find $WHL_PATH -type f -size +${LINUX_CPU_MAX_WHL_SIZE}) ]]; then
        echo "Linux CPU whl size has exceeded ${LINUX_CPU_MAX_WHL_SIZE}. To keep
within pypi's CDN distribution limit, we must not exceed that threshold."
      return 1
    fi
    # Check Windows CPU whl size.
    if [[ "$WHL_PATH" == *"-win"* ]] && [[ $(find $WHL_PATH -type f -size +${WIN_CPU_MAX_WHL_SIZE}) ]]; then
        echo "Windows CPU whl size has exceeded ${WIN_CPU_MAX_WHL_SIZE}. To keep
within pypi's CDN distribution limit, we must not exceed that threshold."
      return 1
    fi
  # Check GPU whl size
  elif [[ "$WHL_PATH" == *"_gpu"* ]]; then
    # Check Linux GPU whl size.
    if [[ "$WHL_PATH" == *"-manylinux"* ]] && [[ $(find $WHL_PATH -type f -size +${LINUX_GPU_MAX_WHL_SIZE}) ]]; then
        echo "Linux GPU whl size has exceeded ${LINUX_GPU_MAX_WHL_SIZE}. To keep
within pypi's CDN distribution limit, we must not exceed that threshold."
      return 1
    fi
    # Check Windows GPU whl size.
    if [[ "$WHL_PATH" == *"-win"* ]] && [[ $(find $WHL_PATH -type f -size +${WIN_GPU_MAX_WHL_SIZE}) ]]; then
        echo "Windows GPU whl size has exceeded ${WIN_GPU_MAX_WHL_SIZE}. To keep
within pypi's CDN distribution limit, we must not exceed that threshold."
      return 1
    fi
  fi
}

