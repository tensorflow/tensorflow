#!/bin/bash
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
# This script uploads all staged artifacts from all previous builds in the same
# job chain to GCS, GAR, and PyPI.
source "${BASH_SOURCE%/*}/utilities/setup.sh"

# Calculate the version number for choosing the final directory name. This adds
# "-devYYYYMMDD" to the end of the version string for nightly builds.
if [[ "$TFCI_NIGHTLY_UPDATE_VERSION_ENABLE" == 1 ]]; then
  export TF_VER_FULL="$(tfrun bazel run //tensorflow/tools/ci_build:calculate_full_version -- --wheel-type nightly)"
else
  export TF_VER_FULL="$(tfrun bazel run //tensorflow/tools/ci_build:calculate_full_version -- --wheel-type release)"
fi

# Note on gcloud storage commands:
# "gcloud storage cp" always "copies into". It cannot act on the contents of a directory
# and it does not seem possible to e.g. copy "gs://foo/bar" as anything other than
# "/path/bar". This script uses "gcloud storage rsync" instead, which acts on directory
# contents. About arguments to gcloud storage:
# "gcloud storage rsync" runs in parallel.
# "gcloud storage rsync --recursive" is recursive and makes directories work.
# "gcloud storage rsync --delete-unmatched-destination-objects" is "sync and delete files from destination if not present in source"

DOWNLOADS="$(mktemp -d)"
mkdir -p "$DOWNLOADS"
gcloud storage rsync --recursive "$TFCI_ARTIFACT_STAGING_GCS_URI" "$DOWNLOADS"
ls "$DOWNLOADS"

# Upload all build artifacts to e.g. gs://tensorflow/versions/2.16.0-rc1 (releases) or
# gs://tensorflow/nightly/2.16.0-dev20240105 (nightly), overwriting previous values.
if [[ "$TFCI_ARTIFACT_FINAL_GCS_ENABLE" == 1 ]]; then
  gcloud auth activate-service-account --key-file="$TFCI_ARTIFACT_FINAL_GCS_SA_PATH"

  # $TF_VER_FULL will resolve to e.g. "2.15.0-rc2". Since $TF_VER_FULL comes
  # from get_versions.sh, which must be run *after* update_version.py, FINAL_URI
  # can't be set inside the rest of the _upload envs.
  FINAL_URI="$TFCI_ARTIFACT_FINAL_GCS_URI/$TF_VER_FULL"
  gcloud storage rsync --delete-unmatched-destination-objects --recursive "$DOWNLOADS" "$FINAL_URI"

  # Also mirror the latest-uploaded folder to the "latest" directory.
  # GCS does not support symlinks.
  gcloud storage rsync --delete-unmatched-destination-objects --recursive "$FINAL_URI" "$TFCI_ARTIFACT_LATEST_GCS_URI"
fi

if [[ "$TFCI_ARTIFACT_FINAL_PYPI_ENABLE" == 1 ]]; then
  pip install --upgrade twine keyring keyrings.google-artifactregistry-auth
  twine upload $TFCI_ARTIFACT_FINAL_PYPI_ARGS "$DOWNLOADS"/*.whl
  twine upload $TFCI_ARTIFACT_FINAL_GAR_ARGS "$DOWNLOADS"/*.whl
fi
