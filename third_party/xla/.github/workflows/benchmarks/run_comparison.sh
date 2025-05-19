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
set -eu # Exit on unset variables and errors

# Input variables from the GitHub Actions workflow
readonly RESULTS_JSON_FILE="$1"
readonly BASELINE_YAML_FILE="$2"
readonly CONFIG_ID_FOR_BASELINE_LOOKUP="$3"
readonly TARGET_METRICS_JSON="$4"
readonly PYTHON_COMPARISON_SCRIPT="$5"

echo "--- Starting Baseline Comparison Script ---"
echo "Results JSON File: $RESULTS_JSON_FILE"
echo "Baseline YAML File: $BASELINE_YAML_FILE"
echo "Config ID for Baseline Lookup: $CONFIG_ID_FOR_BASELINE_LOOKUP"
echo "Target Metrics JSON: $TARGET_METRICS_JSON"
echo "Python Comparison Script: $PYTHON_COMPARISON_SCRIPT" # New line

# Validate inputs
if [ -z "$RESULTS_JSON_FILE" ] || [ ! -f "$RESULTS_JSON_FILE" ]; then
    echo "::error::Results JSON file not found or invalid: '$RESULTS_JSON_FILE'"
    exit 1
fi
if [ -z "$BASELINE_YAML_FILE" ] || [ ! -f "$BASELINE_YAML_FILE" ]; then
    echo "::warning::Baseline YAML file not found or invalid: '$BASELINE_YAML_FILE'. Skipping comparison."
    exit 0 # Exit cleanly if baseline file is missing, as comparison cannot proceed.
fi
if [ -z "$CONFIG_ID_FOR_BASELINE_LOOKUP" ]; then
    echo "::error::Config ID for baseline lookup is empty."
    exit 1
fi
if [ -z "$PYTHON_COMPARISON_SCRIPT" ] || [ ! -f "$PYTHON_COMPARISON_SCRIPT" ]; then # New validation
    echo "::error::Python comparison script not found or invalid: '$PYTHON_COMPARISON_SCRIPT'"
    exit 1
fi

# Ensure Python and jq are available
if ! command -v python3 &> /dev/null; then echo "::error::python3 command not found."; exit 1; fi
if ! command -v jq &> /dev/null; then echo "::error::jq command not found."; exit 1; fi


echo "Executing Python comparison script..."
# Use the passed Python script path
python3 "$PYTHON_COMPARISON_SCRIPT" \
  --results-json-file="$RESULTS_JSON_FILE" \
  --baseline-yaml-file="$BASELINE_YAML_FILE" \
  --config-id="$CONFIG_ID_FOR_BASELINE_LOOKUP" \
  --target-metrics-json="$TARGET_METRICS_JSON"

COMPARISON_EXIT_CODE=$?

if [ $COMPARISON_EXIT_CODE -ne 0 ]; then
  echo "::error::Baseline comparison script failed or regressions detected (Exit Code: $COMPARISON_EXIT_CODE)."
  exit $COMPARISON_EXIT_CODE # This will fail the shell script and the GitHub Actions step
else
  echo "Baseline comparison successful: No regressions detected or no applicable baselines found."
fi

echo "--- Baseline Comparison Script Finished ---"