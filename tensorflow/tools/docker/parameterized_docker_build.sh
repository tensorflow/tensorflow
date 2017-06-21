#!/usr/bin/env bash
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# Paramterized build and test for TensorFlow Docker images.
#
# Usage:
#   parameterized_docker_build.sh
#
# The script obeys the following environment variables:
#   TF_DOCKER_BUILD_TYPE: (CPU | GPU)
#     CPU or GPU image
#
#   TF_DOCKER_BUILD_IS_DEVEL: (NO | YES)
#     Is this developer image
#
#   TF_DOCKER_BUILD_DEVEL_BRANCH
#     (Required if TF_DOCKER_BUILD_IS_DEVEL is YES)
#     Specifies the branch to checkout for devel docker images
#
#   TF_DOCKER_BUILD_CENTRAL_PIP
#     (Optional)
#     If set to a non-empty string, will use it as the URL from which the
#     pip wheel file will be downloaded (instead of building the pip locally).
#
#   TF_DOCKER_BUILD_IMAGE_NAME:
#     (Optional)
#     If set to any non-empty value, will use it as the image of the
#     newly-built image. If not set, the tag prefix tensorflow/tensorflow
#     will be used.
#
#   TF_DOCKER_BUILD_VERSION:
#     (Optinal)
#     If set to any non-empty value, will use the version (e.g., 0.8.0) as the
#     tag prefix of the image. Additional strings, e.g., "-devel-gpu", will be
#     appended to the tag. If not set, the default tag prefix "latest" will be
#     used.
#
#   TF_DOCKER_BUILD_PORT
#     (Optional)
#     If set to any non-empty and valid port number, will use that port number
#     during basic checks on the newly-built docker image.
#
#   TF_DOCKER_BUILD_PUSH_CMD
#     (Optional)
#     If set to a valid binary/script path, will call the script with the final
#     tagged image name with an argument, to push the image to a central repo
#     such as gcr.io or Docker Hub.
#
#   TF_DOCKER_BUILD_PYTHON_VERSION
#     (Optional)
#     Specifies the desired Python version. Defaults to PYTHON2.
#
#   TF_DOCKER_BUILD_OPTIONS
#     (Optional)
#     Specifies the desired build options. Defaults to OPT.

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../ci_build/builds/builds_common.sh"

# Help functions
CHECK_FAILED=0
mark_check_failed() {
  # Usage: mark_check_failed <FAILURE_MESSAGE>
  echo $1
  CHECK_FAILED=1
}

TF_DOCKER_BUILD_TYPE=$(to_lower ${TF_DOCKER_BUILD_TYPE})
TF_DOCKER_BUILD_IS_DEVEL=$(to_lower ${TF_DOCKER_BUILD_IS_DEVEL})
TF_DOCKER_BUILD_PYTHON_VERSION=$(to_lower ${TF_DOCKER_BUILD_PYTHON_VERSION:-PYTHON2})
TF_DOCKER_BUILD_OPTIONS=$(to_lower ${TF_DOCKER_BUILD_OPTIONS:-OPT})

echo "Required build parameters:"
echo "  TF_DOCKER_BUILD_TYPE=${TF_DOCKER_BUILD_TYPE}"
echo "  TF_DOCKER_BUILD_IS_DEVEL=${TF_DOCKER_BUILD_IS_DEVEL}"
echo "  TF_DOCKER_BUILD_DEVEL_BRANCH=${TF_DOCKER_BUILD_DEVEL_BRANCH}"
echo ""
echo "Optional build parameters:"
echo "  TF_DOCKER_BUILD_CENTRAL_PIP=${TF_DOCKER_BUILD_CENTRAL_PIP}"
echo "  TF_DOCKER_BUILD_IMAGE_NAME=${TF_DOCKER_BUILD_IMAGE_NAME}"
echo "  TF_DOCKER_BUILD_VERSION=${TF_DOCKER_BUILD_VERSION}"
echo "  TF_DOCKER_BUILD_PORT=${TF_DOCKER_BUILD_PORT}"
echo "  TF_DOCKER_BUILD_PUSH_CMD=${TF_DOCKER_BUILD_PUSH_CMD}"


CONTAINER_PORT=${TF_DOCKER_BUILD_PORT:-8888}

# Make sure that docker is available on path
if [[ -z $(which docker) ]]; then
  die "ERROR: docker is not available on path"
fi

# Validate the environment-variable options and construct the final image name
# Final image name with tag
FINAL_IMAGE_NAME=${TF_DOCKER_BUILD_IMAGE_NAME:-tensorflow/tensorflow}
FINAL_TAG=${TF_DOCKER_BUILD_VERSION:-latest}

# Original (unmodified) Dockerfile
ORIG_DOCKERFILE="Dockerfile"

if [[ ${TF_DOCKER_BUILD_IS_DEVEL} == "yes" ]]; then
  FINAL_TAG="${FINAL_TAG}-devel"
  ORIG_DOCKERFILE="${ORIG_DOCKERFILE}.devel"

  if [[ -z "${TF_DOCKER_BUILD_DEVEL_BRANCH}" ]]; then
    die "ERROR: TF_DOCKER_BUILD_DEVEL_BRANCH is missing for devel docker build"
  fi
elif [[ ${TF_DOCKER_BUILD_IS_DEVEL} == "no" ]]; then
  :
else
  die "ERROR: Unrecognized value in TF_DOCKER_BUILD_IS_DEVEL: "\
"${TF_DOCKER_BUILD_IS_DEVEL}"
fi

if [[ ${TF_DOCKER_BUILD_TYPE} == "cpu" ]]; then
  DOCKER_BINARY="docker"
elif   [[ ${TF_DOCKER_BUILD_TYPE} == "gpu" ]]; then
  DOCKER_BINARY="nvidia-docker"

  FINAL_TAG="${FINAL_TAG}-gpu"
  if [[ ${ORIG_DOCKERFILE} == *"."* ]]; then
    # There is already a dot in the tag, use "-"
    ORIG_DOCKERFILE="${ORIG_DOCKERFILE}-gpu"
  else
    ORIG_DOCKERFILE="${ORIG_DOCKERFILE}.gpu"
  fi
else
  die "ERROR: Unrecognized value in TF_DOCKER_BUILD_TYPE: "\
"${TF_DOCKER_BUILD_TYPE}"
fi

if [[ "${TF_DOCKER_BUILD_PYTHON_VERSION}" == "python2" ]]; then
  :
elif [[ "${TF_DOCKER_BUILD_PYTHON_VERSION}" == "python3" ]]; then
  FINAL_TAG="${FINAL_TAG}-py3"
else
  die "Unrecognized value in TF_DOCKER_BUILD_PYTHON_VERSION: "\
"${TF_DOCKER_BUILD_PYTHON_VERSION}"
fi

# Verify that the original Dockerfile exists
ORIG_DOCKERFILE="${SCRIPT_DIR}/${ORIG_DOCKERFILE}"
if [[ ! -f "${ORIG_DOCKERFILE}" ]]; then
  die "ERROR: Cannot find Dockerilfe at: ${ORIG_DOCKERFILE}"
fi

echo ""
echo "FINAL_IMAGE_NAME: ${FINAL_IMAGE_NAME}"
echo "FINAL_TAG: ${FINAL_TAG}"
echo "Original Dockerfile: ${ORIG_DOCKERFILE}"
echo ""

# Create tmp directory for Docker build
TMP_DIR=$(mktemp -d)
echo ""
echo "Docker build will occur in temporary directory: ${TMP_DIR}"

# Copy all files to tmp directory for Docker build
cp -r ${SCRIPT_DIR}/* "${TMP_DIR}/"

if [[ "${TF_DOCKER_BUILD_IS_DEVEL}" == "no" ]]; then
  DOCKERFILE="${TMP_DIR}/Dockerfile"

  if [[ -z "${TF_DOCKER_BUILD_CENTRAL_PIP}" ]]; then
    # Perform local build of the required PIP whl file
    export TF_BUILD_CONTAINER_TYPE=${TF_DOCKER_BUILD_TYPE}
    export TF_BUILD_PYTHON_VERSION=${TF_DOCKER_BUILD_PYTHON_VERSION}
    export TF_BUILD_OPTIONS=${TF_DOCKER_BUILD_OPTIONS}
    export TF_BUILD_IS_PIP="PIP"

    if [[ "${TF_DOCKER_BUILD_TYPE}" == "gpu" ]]; then
      export TF_BUILD_APPEND_CI_DOCKER_EXTRA_PARAMS=\
  "${TF_BUILD_APPEND_CI_DOCKER_EXTRA_PARAMS} -e TF_CUDA_COMPUTE_CAPABILITIES=3.0,3.5,5.2"
    fi

    pushd "${SCRIPT_DIR}/../../../"
    rm -rf pip_test/whl &&
    tensorflow/tools/ci_build/ci_parameterized_build.sh
    PIP_BUILD_EXIT_CODE=$?
    popd

    # Was the pip build successful?
    if [[ ${PIP_BUILD_EXIT_CODE} != "0" ]]; then
      die "FAIL: Failed to build pip file locally"
    fi

    PIP_WHL=$(ls pip_test/whl/*.whl | head -1)
    if [[ -z "${PIP_WHL}" ]]; then
      die "ERROR: Cannot locate the locally-built pip whl file"
    fi
    echo "Locally-built PIP whl file is at: ${PIP_WHL}"

    # Copy the pip file to tmp directory
    cp "${PIP_WHL}" "${TMP_DIR}/" || \
        die "ERROR: Failed to copy wheel file: ${PIP_WHL}"

    # Use string replacement to put the correct file name into the Dockerfile
    PIP_WHL=$(basename "${PIP_WHL}")

    # Modify the non-devel Dockerfile to point to the correct pip whl file
    # location
    sed -e "/# --- DO NOT EDIT OR DELETE BETWEEN THE LINES --- #/,"\
"/# --- ~ DO NOT EDIT OR DELETE BETWEEN THE LINES --- #/c"\
"COPY ${PIP_WHL} /\n"\
"RUN pip --no-cache-dir install /${PIP_WHL}" "${ORIG_DOCKERFILE}" \
    > "${DOCKERFILE}"
  else
    echo "Downloading pip wheel from: ${TF_DOCKER_BUILD_CENTRAL_PIP}"
    echo

    # Modify the non-devel Dockerfile to point to the correct pip whl URL.
    sed -e "/# --- DO NOT EDIT OR DELETE BETWEEN THE LINES --- #/,"\
"/# --- ~ DO NOT EDIT OR DELETE BETWEEN THE LINES --- #/c"\
"RUN pip --no-cache-dir install ${TF_DOCKER_BUILD_CENTRAL_PIP}" "${ORIG_DOCKERFILE}" \
    > "${DOCKERFILE}"
  fi

  echo "Modified Dockerfile at: ${DOCKERFILE}"
  echo

  # Modify python/pip version if necessary.
  if [[ "${TF_DOCKER_BUILD_PYTHON_VERSION}" == "python3" ]]; then
    if sed -i -e 's/python /python3 /g' "${DOCKERFILE}" && \
        sed -i -e 's/python-dev/python3-dev/g' "${DOCKERFILE}" && \
        sed -i -e 's/pip /pip3 /g' "${DOCKERFILE}" && \
        sed -i -e 's^# RUN ln -s /usr/bin/python3 /usr/bin/python#^RUN ln -s /usr/bin/python3 /usr/bin/python^' "${DOCKERFILE}"
    then
      echo "Modified Dockerfile for python version "\
"${TF_DOCKER_BUILD_PYTHON_VERSION} at: ${DOCKERFILE}"
    else
      die "FAILED to modify ${DOCKERFILE} for python3"
    fi
  fi
else
  DOCKERFILE="${TMP_DIR}/Dockerfile"

  # Modify the devel Dockerfile to specify the git branch
  sed -r "s/([\s]*git checkout )(.*)/\1${TF_DOCKER_BUILD_DEVEL_BRANCH}/g" \
      "${ORIG_DOCKERFILE}" > "${DOCKERFILE}"

  # Modify python/pip version if necessary.
  if [[ "${TF_DOCKER_BUILD_PYTHON_VERSION}" == "python3" ]]; then
    if sed -i -e 's/python-dev/python-dev python3-dev/g' "${DOCKERFILE}" && \
        sed -i -e 's/python /python3 /g' "${DOCKERFILE}" && \
        sed -i -e 's^/tmp/pip^/tmp/pip3^g' "${DOCKERFILE}" && \
        sed -i -e 's/pip /pip3 /g' "${DOCKERFILE}" && \
        sed -i -e 's/ENV CI_BUILD_PYTHON python/ENV CI_BUILD_PYTHON python3/g' "${DOCKERFILE}" && \
        sed -i -e 's^# RUN ln -s /usr/bin/python3 /usr/bin/python#^RUN ln -s /usr/bin/python3 /usr/bin/python^' "${DOCKERFILE}"
    then
      echo "Modified Dockerfile further for python version ${TF_DOCKER_BUILD_PYTHON_VERSION} at: ${DOCKERFILE}"
    else
      die "FAILED to modify ${DOCKERFILE} for python3"
    fi
  fi
fi

# Perform docker build
# Intermediate image name with tag
IMG="${USER}/tensorflow:${FINAL_TAG}"
echo "Building docker image with image name and tag: ${IMG}"

"${DOCKER_BINARY}" build --no-cache --pull -t "${IMG}" -f "${DOCKERFILE}" "${TMP_DIR}"
if [[ $? == "0" ]]; then
  echo "${DOCKER_BINARY} build of ${IMG} succeeded"
else
  die "FAIL: ${DOCKER_BINARY} build of ${IMG} with Dockerfile ${DOCKERFILE} "\
"failed"
fi


# Make sure that there is no other containers of the same image running
# TODO(cais): Move to an earlier place.
if "${DOCKER_BINARY}" ps | grep -q "${IMG}"; then
  die "ERROR: It appears that there are docker containers of the image "\
"${IMG} running. Please stop them before proceeding"
fi

# Start a docker container from the newly-built docker image
DOCKER_RUN_LOG="${TMP_DIR}/docker_run.log"
echo ""
echo "Running docker container from image ${IMG}..."
echo "  (Log file is at: ${DOCKER_RUN_LOG}"
echo ""

if [[ "${TF_DOCKER_BUILD_IS_DEVEL}" == "no" ]]; then
  "${DOCKER_BINARY}" run --rm -p ${CONTAINER_PORT}:${CONTAINER_PORT} \
      -v ${TMP_DIR}/notebooks:/root/notebooks "${IMG}" \
      2>&1 > "${DOCKER_RUN_LOG}" &

  # Get the container ID
  CONTAINER_ID=""
  while [[ -z ${CONTAINER_ID} ]]; do
    sleep 1
    echo "Polling for container ID..."
    CONTAINER_ID=$("${DOCKER_BINARY}" ps | grep "${IMG}" | awk '{print $1}')
  done

  echo "ID of the running docker container: ${CONTAINER_ID}"
  echo ""

  if [[ ${TF_DOCKER_BUILD_IS_DEVEL} == "no" ]]; then
    # Non-devel docker build: Do some basic sanity checks on jupyter notebook
    # on the running docker container
    echo ""
    echo "Performing basic sanity checks on the running container..."
    if wget -qO- "http://127.0.0.1:${CONTAINER_PORT}/tree" &> /dev/null
    then
      echo "  PASS: wget tree"
    else
      mark_check_failed "  FAIL: wget tree"
    fi

    for NB in ${TMP_DIR}/notebooks/*.ipynb; do
      NB_BASENAME=$(basename "${NB}")
      NB_URL="http://127.0.0.1:${CONTAINER_PORT}/notebooks/${NB_BASENAME}"
      if wget -qO- "${NB_URL}" -o "${TMP_DIR}/${NB_BASENAME}" &> /dev/null
      then
        echo "  PASS: wget ${NB_URL}"
      else
        mark_check_failed  "  FAIL: wget ${NB_URL}"
      fi
    done
  fi

  # Stop the running docker container
  sleep 1
  "${DOCKER_BINARY}" stop --time=0 ${CONTAINER_ID}

fi


# Clean up
echo "Cleaning up temporary directory: ${TMP_DIR} ..."
rm -rf "${TMP_DIR}" || echo "ERROR: Failed to remove directory ${TMP_DIR}"


# Summarize result
echo ""
if [[ ${CHECK_FAILED} == "0" ]]; then
  echo "PASS: basic checks on newly-built image \"${IMG}\" succeeded"
else
  die "FAIL: basic checks on newly-built image \"${IMG}\" failed"
fi


# Apply the final image name and tag
FINAL_IMG="${FINAL_IMAGE_NAME}:${FINAL_TAG}"

DOCKER_VER=$("${DOCKER_BINARY}" version | grep Version | head -1 | awk '{print $NF}')
if [[ -z "${DOCKER_VER}" ]]; then
  die "ERROR: Failed to determine ${DOCKER_BINARY} version"
fi
DOCKER_MAJOR_VER=$(echo "${DOCKER_VER}" | cut -d. -f 1)
DOCKER_MINOR_VER=$(echo "${DOCKER_VER}" | cut -d. -f 2)

FORCE_TAG=""
if [[ "${DOCKER_MAJOR_VER}" -le 1 ]] && \
   [[ "${DOCKER_MINOR_VER}" -le 9 ]]; then
  FORCE_TAG="--force"
fi

"${DOCKER_BINARY}" tag ${FORCE_TAG} "${IMG}" "${FINAL_IMG}" || \
    die "Failed to tag intermediate docker image ${IMG} as ${FINAL_IMG}"

echo ""
echo "Successfully tagged docker image: ${FINAL_IMG}"


# Optional: call command specified by TF_DOCKER_BUILD_PUSH_CMD to push image
if [[ ! -z "${TF_DOCKER_BUILD_PUSH_CMD}" ]]; then
  ${TF_DOCKER_BUILD_PUSH_CMD} ${FINAL_IMG}
  if [[ $? == "0" ]]; then
    echo "Successfully pushed Docker image ${FINAL_IMG}"
  else
    die "FAIL: Failed to push Docker image ${FINAL_IMG}"
  fi
fi
