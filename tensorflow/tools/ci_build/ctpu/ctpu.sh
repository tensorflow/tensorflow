#!/bin/bash
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
# Reusable functions for using CTPU in a Kokoro build.
# These functions are unit tested in ctpu_test.sh.

# Installs the Cloud TPU CLI to the current directory.
# Pass pip command as first arg, ex: install_ctpu pip3.7
function install_ctpu {
  PIP_CMD="${1:-pip}"

  # TPUClusterResolver has a runtime dependency on these Python libraries when
  # resolving a Cloud TPU. It's very likely we want these installed if we're
  # using CTPU.
  "${PIP_CMD}" install --user --upgrade google-api-python-client oauth2client

  wget -nv "https://dl.google.com/cloud_tpu/ctpu/latest/linux/ctpu"
  chmod a+x ctpu
}

# Starts a Cloud TPU, storing metadata into artifacts dir for export.
#
# This function supports overriding the default parameters, using optional
# single-letter flags.
#
# Usage:
#   ctpu_up -n [tpu name] -z [zone] -s [tpu size] -v [tf-version] \
#     -p [cloud project] -g [gcp-network]
function ctpu_up {
  local OPTIND o  # Used for flag parsing
  # Generate a unique random name for TPU, as we might be running multiple builds in parallel.
  local name="kokoro-tpu-${RANDOM}"
  local zone="us-central1-c"
  local size="v2-8"
  local version="nightly-2.x"
  local project  # Project automatically detected from environment.
  local gcp_network  # Network needed only if project default is Legacy.

  # Override any of the above params from flags.
  while getopts ":n:z:p:s:v:g:" o; do
    case "${o}" in
      n)
        name="${OPTARG}"
        ;;
      z)
        zone="${OPTARG}"
        ;;
      p)
        project="${OPTARG}"
        ;;
      s)
        size="${OPTARG}"
        ;;
      v)
        version="${OPTARG}"
        ;;
      g)
        gcp_network="${OPTARG}"
        ;;
      *)
        echo "Unexpected parameter for ctpu_up: ${o}"
        exit 1
    esac
  done
  shift $((OPTIND-1))

  export TPU_NAME="${name}"
  export TPU_ZONE="${zone}"

  # Store name and zone into artifacts dir so cleanup job has access.
  echo "${TPU_NAME}" > "${TF_ARTIFACTS_DIR}/tpu_name"
  echo "${TPU_ZONE}" > "${TF_ARTIFACTS_DIR}/tpu_zone"

  local args=(
    "--zone=${zone}"
    "--tf-version=${version}"
    "--name=${name}"
    "--tpu-size=${size}"
    "--tpu-only"
    "-noconf"
  )

  # "-v" is a bash 4.2 builtin for checking that a variable is set.
  if [[ -v gcp_network ]]; then
    args+=("--gcp-network=${gcp_network}")
  fi

  if [[ -v project ]]; then
    args+=("--project=${project}")
    echo "${project}" > "${TF_ARTIFACTS_DIR}/tpu_project"
  fi

  ./ctpu up "${args[@]}"
}

# Delete the Cloud TPU specified by the metadata in the gfile directory.
function ctpu_delete {
  export TPU_NAME="$(cat "${TF_GFILE_DIR}/tpu_name")"
  export TPU_ZONE="$(cat "${TF_GFILE_DIR}/tpu_zone")"
  TPU_PROJECT_FILE="${TF_GFILE_DIR}/tpu_project"
  if [ -f "${TPU_PROJECT_FILE}" ]; then
    export TPU_PROJECT="$(cat ${TPU_PROJECT_FILE})"
  else
    export TPU_PROJECT="tensorflow-testing"
  fi

  # Retry due to rare race condition where TPU creation hasn't propagated by
  # the time we try to delete it.
  for i in 1 2 3; do
    ./ctpu delete \
      --project=${TPU_PROJECT} \
      --zone="${TPU_ZONE}" \
      --name="${TPU_NAME}" \
      --tpu-only \
      -noconf && break || sleep 60
  done
}
