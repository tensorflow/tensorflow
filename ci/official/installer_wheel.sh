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
source "${BASH_SOURCE%/*}/utilities/setup.sh"

# Update the version numbers for Nightly only
if [[ "$TFCI_NIGHTLY_UPDATE_VERSION_ENABLE" == 1 ]]; then
  python3 tensorflow/tools/ci_build/update_version.py --nightly
fi

# This generates a pure python wheel of the format "*-py3-none-any.whl"
bazel run --HERMETIC_PYTHON_VERSION=3.13 //tensorflow/tools/pip_package:setup_py_binary -- bdist_wheel --dist-dir "$TFCI_OUTPUT_DIR"

# Get the name of the pure python wheel that was built. This should
# resolve to either
# 1. tensorflow-a.b.c-py3-none-any.whl  or, for nightly builds,
# 2. tf_nightly-a.b.c.devYYYYMMDD-py3-none-any.whl
pure_python_whl=$(ls "$TFCI_OUTPUT_DIR"/*py3-none-any*)
pure_python_whl=$(basename "${pure_python_whl}")

# Extract the package name from the wheel name. That is, extract every character
# before the pattern "-py3-" in the wheel name.
pkg_name=$(echo "${pure_python_whl}" | awk -F'-py3-' '{print $1}')

# Save the current working directory and then switch to the output directory.
pushd "${TFCI_OUTPUT_DIR}"

# Unpack the wheel to get all the file contents. The pure python wheel we built
# above is tagged with "py3-none-any". We cannot change the tags by simply
# renaming the wheels as uploading to PyPI would fail with "File already exists"
# error. In order to upload to PyPI, we unpack the wheel and change the tag
# inside a metadata file to the one we want (e.g cp38-cp38-win_amd) and then
# re-pack it to generate it as a platform specific wheel with this new wheel
#tag.
python3 -m wheel unpack "${pure_python_whl}"

# Remove the pure python wheel.
rm -rf "${pure_python_whl}"

# Generate a PyPI upload compatible wheel for each tag in
# $TFCI_INSTALLER_WHL_TAGS.
for whl_tag in $TFCI_INSTALLER_WHL_TAGS; do
  echo "Generating a PyPI upload compatible wheel for ${whl_tag}"
  echo ""
  # Unpacking a wheel creates a directory named
  # {distribution}-{version}.dist-info which contains the metadata files. We
  # replace the old tag in the WHEEL file in this directory with the new tag we
  # have in ${whl_tag}. Replace the line in WHEEL that starts with "Tag:" with
  # "Tag: <new whl tag>"
  sed -i "s/^Tag:.*/Tag: ${whl_tag}/g" "${pkg_name}"/"${pkg_name}".dist-info/WHEEL

  # Repack the wheel. When repacking, the wheel would be automatically tagged
  # with the new tag we provided in ${whl_tag}. Repacking also regnerates the
  # RECORD file which contains hashes of all included files.
  python3 -m wheel pack "${pkg_name}"
done

# Switch back to the original working directory. This is needed to ensure that
# cleanup steps at the end of the script works as expected.
popd

echo "Following installer wheels were generated: "
ls "${TFCI_OUTPUT_DIR}"/*.whl

if [[ "$TFCI_ARTIFACT_STAGING_GCS_ENABLE" == 1 ]]; then
  # Note: -n disables overwriting previously created files.
  gsutil cp -n "$TFCI_OUTPUT_DIR"/*.whl "$TFCI_ARTIFACT_STAGING_GCS_URI"
fi