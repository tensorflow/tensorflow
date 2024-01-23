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
# job chain to GCS and PyPI.
source "${BASH_SOURCE%/*}/utilities/setup.sh"

# Update the version numbers for Nightly only, then fetch the version numbers
# for choosing the final directory name. This adds "-devYYYYMMDD" to the end of
# the version string (.devYYYYMMDD for Python; see pypi.org/project/tf-nightly)
if [[ "$TFCI_NIGHTLY_UPDATE_VERSION_ENABLE" == 1 ]]; then
  tfrun python3 tensorflow/tools/ci_build/update_version.py --nightly
fi
source ci/official/utilities/get_versions.sh

# $TF_VER_FULL will resolve to e.g. "2.15.0-rc2". When given a directory path,
# gsutil copies the basename of the directory into the provided path. Uploading
# /tmp/$TF_VER_FULL to gs://bucket/ will create gs://bucket/$TF_VER_FULL/files.
# Since $TF_VER_FULL comes from get_versions.sh, which must be run *after*
# update_version.py, this can't be set inside the rest of the _upload envs.
DOWNLOADS="$(mktemp -d)/$TF_VER_FULL"
mkdir -p "$DOWNLOADS"
# -r is needed to copy a whole folder.
gsutil -m cp -r "$TFCI_ARTIFACT_STAGING_GCS_URI" "$DOWNLOADS"
ls "$DOWNLOADS"

# Upload all build artifacts to e.g. gs://tensorflow/versions/2.16.0-rc1 (releases) or
# gs://tensorflow/nightly/2.16.0-dev20240105 (nightly), overwriting previous values.
if [[ "$TFCI_ARTIFACT_FINAL_GCS_ENABLE" == 1 ]]; then
  gcloud auth activate-service-account --key-file="$TFCI_ARTIFACT_FINAL_GCS_SA_PATH"
  gsutil -m cp -r "$DOWNLOADS" "$TFCI_ARTIFACT_FINAL_GCS_URI"
  # Also mirror the latest-uploaded folder to the "latest" directory.
  # GCS does not support symlinks. -p preserves ACLs. -d deletes
  # no-longer-present files (it's what makes this act as a mirror).
  gsutil rsync -d -p -r "$TFCI_ARTIFACT_FINAL_GCS_URI" "$TFCI_ARTIFACT_LATEST_GCS_URI"
fi

if [[ "$TFCI_ARTIFACT_FINAL_PYPI_ENABLE" == 1 ]]; then
  twine upload $TFCI_ARTIFACT_FINAL_PYPI_ARGS "$DOWNLOADS"/*.whl
fi
