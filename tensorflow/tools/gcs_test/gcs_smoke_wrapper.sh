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
# In-container wrapper for GCS smoke test.
#
# This script invokes gcs_smoke.py and performs tear down afterwards.
#
# Usage:
#   gcs_smoke_wrapper.sh <GCS_BUCKET_URL>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Helper function: Exit on failure.
die () {
  echo $@
  exit 1
}

print_usage() {
  echo "Usage: gcs_smoke_wrapper.sh <GCS_BUCKET_URL>"
  echo ""
}

# Sanity check on command-line arguments.
GCS_BUCKET_URL=$1
if [[ -z "${GCS_BUCKET_URL}" ]]; then
  print_usage
  die "ERROR: Command-line argument GCS_BUCKET_URL is not supplied"
fi

# Check that gcloud and gsutil binaries are available.
GCLOUD_BIN="/var/gcloud/google-cloud-sdk/bin/gcloud"
if [[ ! -f "${GCLOUD_BIN}" ]]; then
  die "ERROR: Unable to find gcloud at path ${GCLOUD_BIN}"
fi

GSUTIL_BIN="/var/gcloud/google-cloud-sdk/bin/gsutil"
if [[ ! -f "${GSUTIL_BIN}" ]]; then
  die "ERROR: Unable to find gsutil at path ${GSUTIL_BIN}"
fi

# Check environment variable for gcloud credentials
if [[ -z "${GOOGLE_APPLICATION_CREDENTIALS}" ]]; then
  die "ERROR: Required gcloud environment variable "\
"${GOOGLE_APPLICATION_CREDENTIALS} is not set."
fi

# Locate main Python file
GCS_SMOKE_PY="${SCRIPT_DIR}/python/gcs_smoke.py"
if [[ ! -f "${GCS_SMOKE_PY}" ]]; then
  die "ERROR: Unable to find Python file at ${GCS_SMOKE_PY}"
fi


LOG_FILE="/tmp/tf-gcs-test.log"
rm -rf ${LOG_FILE} || \
    die "ERROR: Failed to remove existing log file ${LOG_FILE}"

# Since https://github.com/tensorflow/tensorflow/pull/47247 we need to
# enable legacy filesystem for GCS (or switch to the modular one)
export TF_ENABLE_LEGACY_FILESYSTEM=1

# Invoke main Python file
python "${GCS_SMOKE_PY}" --gcs_bucket_url="${GCS_BUCKET_URL}" \
    > "${LOG_FILE}" 2>&1

if [[ $? != "0" ]]; then
  cat ${LOG_FILE}
  die "FAIL: End-to-end test of GCS access from TensorFlow failed."
fi

cat ${LOG_FILE}
echo ""

# Clean up the newly created tfrecord file in GCS bucket.
# First, activate gcloud service account
"${GCLOUD_BIN}" auth activate-service-account \
    --key-file "${GOOGLE_APPLICATION_CREDENTIALS}" || \
    die "ERROR: Failed to activate gcloud service account with JSON key file"

NEW_TFREC_URL=$(grep "Using input path" "${LOG_FILE}" | \
                awk '{print $NF}')
if [[ -z ${NEW_TFREC_URL} ]]; then
  die "FAIL: Unable to determine the URL to the new tfrecord file in GCS"
fi
if "${GSUTIL_BIN}" rm "${NEW_TFREC_URL}"
then
  echo "Cleaned up new tfrecord file in GCS: ${NEW_TFREC_URL}"
else
  die "FAIL: Unable to clean up new tfrecord file in GCS: ${NEW_TFREC_URL}"
fi
