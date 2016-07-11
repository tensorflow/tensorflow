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
#
# Driver script for TensorFlow-GCS smoke test.
#
# Usage:
#   gcs_smoke.sh <GCLOUD_JSON_KEY_PATH> <GCS_BUCKET_URL>
#
# Input arguments:
#   GCLOUD_KEY_JSON_PATH: Path to the Google Cloud JSON key file.
#     See https://cloud.google.com/storage/docs/authentication for details.
#
#   GCS_BUCKET_URL: URL to the GCS bucket for testing.
#     E.g., gs://my-gcs-bucket/test-directory

# Configurations
DOCKER_IMG="tensorflow-gcs-test"

print_usage() {
  echo "Usage: gcs_smoke.sh <GCLOUD_JSON_KEY_PATH> <GCS_BUCKET_URL>"
  echo ""
}


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../ci_build/builds/builds_common.sh"

# Check input arguments
GCLOUD_JSON_KEY_PATH=$1
GCS_BUCKET_URL=$2
if [[ -z "${GCLOUD_JSON_KEY_PATH}" ]]; then
  print_usage
  die "ERROR: Command-line argument GCLOUD_JSON_KEY_PATH is not supplied"
fi
if [[ -z "${GCS_BUCKET_URL}" ]]; then
  print_usage
  die "ERROR: Command-line argument GCS_BUCKET_URL is not supplied"
fi

if [[ ! -f "${GCLOUD_JSON_KEY_PATH}" ]]; then
  die "ERROR: Path to Google Cloud JSON key file is invalid: \""\
"${GCLOUD_JSON_KEY_PATH}\""
fi

DOCKERFILE="${SCRIPT_DIR}/Dockerfile"
if [[ ! -f "${DOCKERFILE}" ]]; then
  die "ERROR: Cannot find Dockerfile at expected path ${DOCKERFILE}"
fi

# Build the docker image for testing
docker build --no-cache \
    -f "${DOCKERFILE}" -t "${DOCKER_IMG}" "${SCRIPT_DIR}" || \
    die "FAIL: Failed to build docker image for testing"

# Run the docker image with the GCS key file mapped and the gcloud-required
# environment variables set.
LOG_FILE="/tmp/tf-gcs-test.log"
rm -rf ${LOG_FILE}

docker run --rm \
    -v ${GCLOUD_JSON_KEY_PATH}:/gcloud-key.json \
    -e "GOOGLE_APPLICATION_CREDENTIALS=/gcloud-key.json" \
    "${DOCKER_IMG}" \
    python /gcs_smoke.py --gcs_bucket_url="${GCS_BUCKET_URL}" \
    2>&1 > "${LOG_FILE}"

if [[ $? != "0" ]]; then
  cat ${LOG_FILE}
  die "FAIL: End-to-end test of GCS access from TensorFlow failed."
fi

cat ${LOG_FILE}
echo ""

# Clean up the newly created tfrecord file in GCS bucket
NEW_TFREC_URL=$(grep "Using input path" "${LOG_FILE}" | \
                awk '{print $NF}')
if [[ -z ${NEW_TFREC_URL} ]]; then
  die "FAIL: Unable to determine the URL to the new tfrecord file in GCS"
fi
gsutil rm "${NEW_TFREC_URL}" && \
    echo "Cleaned up new tfrecord file in GCS: ${NEW_TFREC_URL}" || \
    die "FAIL: Unable to clean up new tfrecord file in GCS: ${NEW_TFREC_URL}"