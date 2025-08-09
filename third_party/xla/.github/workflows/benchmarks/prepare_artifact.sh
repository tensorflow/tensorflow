# Copyright 2025 The OpenXLA Authors. All Rights Reserved.
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
# ============================================================================
# .github/workflows/benchmarks/prepare_artifact.sh
# TODO(juliagmt): convert this to a python script.
#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.

echo "--- prepare_artifact.sh (Self-creating directory version) ---"
echo "SCRIPT: Current PWD: $(pwd)"
echo "SCRIPT: GITHUB_WORKSPACE is: $GITHUB_WORKSPACE"
echo "SCRIPT: Intended OUTPUT_DIR is: $OUTPUT_DIR"

# Create the directory HERE, inside this script, right before using it.
echo "SCRIPT: Ensuring directory '$OUTPUT_DIR' exists by creating it with mkdir -p."
mkdir -p "$OUTPUT_DIR"

# Verify creation immediately
echo "SCRIPT: Verifying directory '$OUTPUT_DIR' after mkdir with 'ls -ld':"
ls -ld "$OUTPUT_DIR" || echo "SCRIPT: 'ls -ld ""$OUTPUT_DIR""' FAILED even after mkdir in script!"

# Now, check with [ -d ... ]
if [ ! -d "$OUTPUT_DIR" ]; then
  echo "::error::SCRIPT: Output directory '$OUTPUT_DIR' STILL NOT found with [ -d ... ] even after mkdir in this script."
  echo "SCRIPT: Listing parent directory '$(dirname "$OUTPUT_DIR")' using 'ls -la':"
  ls -la "$(dirname "$OUTPUT_DIR")" || echo "SCRIPT: Failed to list parent directory."
  exit 1
else
  echo "SCRIPT: Output directory '$OUTPUT_DIR' IS now found with [ -d ... ]."
fi

# --- Original script logic from here ---
echo "--- Preparing Artifact (main logic) ---"

ARTIFACT_FILE_NAME=$(basename "$ARTIFACT_LOCATION")
LOCAL_ARTIFACT_PATH="$OUTPUT_DIR/$ARTIFACT_FILE_NAME"

echo "Target local path: ${LOCAL_ARTIFACT_PATH}"

if [ "$IS_GCS_ARTIFACT" == "true" ]; then
  echo "Downloading GCS artifact from: $ARTIFACT_LOCATION"
   if ! command -v wget &> /dev/null; then
     echo "::error::wget command not found in container. Cannot download GCS artifact."
     exit 1
   fi

   wget -q -nv -O "$LOCAL_ARTIFACT_PATH" "$ARTIFACT_LOCATION"
   WGET_EXIT_CODE=$?
    if [ $WGET_EXIT_CODE -ne 0 ]; then
      echo "::error::wget failed to download GCS artifact from $ARTIFACT_LOCATION (Exit code: $WGET_EXIT_CODE)"
      rm -f "$LOCAL_ARTIFACT_PATH" # Clean up partial file
      exit $WGET_EXIT_CODE
    fi
   echo "GCS artifact downloaded."
else
   REPO_ARTIFACT_PATH="$GITHUB_WORKSPACE/$ARTIFACT_LOCATION" # ARTIFACT_LOCATION is the relative repo path here
   echo "Copying local artifact from workspace path: $REPO_ARTIFACT_PATH (IS_GCS_ARTIFACT was false)"
    if [ ! -f "$REPO_ARTIFACT_PATH" ]; then
       echo "::error::Local artifact not found at repository path: $REPO_ARTIFACT_PATH"
       exit 1
    fi
    cp -v "$REPO_ARTIFACT_PATH" "$LOCAL_ARTIFACT_PATH" || exit 1 # Exit if copy fails
   echo "Local artifact copied successfully."
fi

# Verify the final destination file exists
if [ ! -f "$LOCAL_ARTIFACT_PATH" ]; then
   echo "::error::Final artifact file not found at destination: $LOCAL_ARTIFACT_PATH"
   exit 1
fi
echo "Artifact successfully prepared at $LOCAL_ARTIFACT_PATH."

echo "artifact_local_path=$LOCAL_ARTIFACT_PATH" >> "$GITHUB_OUTPUT"
echo "--- Artifact Prep Finished ---"
