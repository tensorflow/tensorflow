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
# .github/workflows/benchmarks/run_comparison.sh
# TODO(juliagmt): convert this to a python script.
#!/bin/bash

# This script encapsulates the logic to compare benchmark results against a baseline.

set -e  # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.

echo "--- Running Comparison Script ---"
echo "CONFIG_ID for baseline lookup: $CONFIG_ID" # This is now the comprehensive ID
echo "Results Directory: $RESOLVED_OUTPUT_DIR"
echo "Baseline File: $RESOLVED_BASELINE_YAML"
echo "Comparison Python Script: $RESOLVED_COMPARISON_SCRIPT"

ACTUAL_RESULTS_JSON_PATH="${RESOLVED_OUTPUT_DIR}/results.json"

if [ ! -f "$ACTUAL_RESULTS_JSON_PATH" ]; then
  echo "::warning::Primary results file '$ACTUAL_RESULTS_JSON_PATH' not found. Cannot perform baseline comparison."
  # Check for fallback .txt file for more info if primary is missing
  if [ -f "${ACTUAL_RESULTS_JSON_PATH}.txt" ]; then
    echo "Fallback results.json.txt found:"
    cat "${ACTUAL_RESULTS_JSON_PATH}.txt"
  fi
  exit 0 # Exiting cleanly to not block if results are missing; comparison script handles this too
fi
SCRIPT_CONFIG_ID_FOR_BASELINE_LOOKUP="${CONFIG_ID}"

echo "Using Config ID for baseline lookup: $SCRIPT_CONFIG_ID_FOR_BASELINE_LOOKUP"

echo "Constructed Config ID for baseline lookup: $SCRIPT_CONFIG_ID_FOR_BASELINE_LOOKUP"

# Run the python comparison script and capture its exit code.
set +e
python3 "$RESOLVED_COMPARISON_SCRIPT" \
  --results-json-file="$ACTUAL_RESULTS_JSON_PATH" \
  --baseline-yaml-file="$RESOLVED_BASELINE_YAML" \
  --config-id="$SCRIPT_CONFIG_ID_FOR_BASELINE_LOOKUP"
COMPARISON_EXIT_CODE=$?
set -e

# --- Handle Exit Code and Set GitHub Actions Output ---

# Default to no regression detected. This will be the output unless overridden.
echo "regression_detected=false" >> "$GITHUB_OUTPUT"

# Assumed Python script exit codes:
# 0: Success, no regressions.
# 1: Script execution error.
# 2: Success, but regressions were detected.

if [ $COMPARISON_EXIT_CODE -eq 0 ]; then
  echo "Baseline comparison successful: No regressions detected or no applicable baselines found."
elif [ $COMPARISON_EXIT_CODE -eq 2 ]; then
  echo "::error::Performance regression detected when comparing with baseline."
  echo "Setting 'regression_detected=true' for GCS upload."
  # Set the output for the workflow to use in conditional steps.
  echo "regression_detected=true" >> "$GITHUB_OUTPUT"
  exit 0
else
  echo "::error::Baseline comparison script failed with an unexpected error (Exit Code: $COMPARISON_EXIT_CODE)."
  # Exit with the original error code to mark this workflow step as failed.
  exit $COMPARISON_EXIT_CODE
fi

echo "--- Comparison Script Finished ---"