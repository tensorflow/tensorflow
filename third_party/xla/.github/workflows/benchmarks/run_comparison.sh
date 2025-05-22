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

# Path to the results.json created by run_benchmark.sh
ACTUAL_RESULTS_JSON_PATH="${RESOLVED_OUTPUT_DIR}/results.json"

echo "Results JSON path for comparison script: $ACTUAL_RESULTS_JSON_PATH"
echo "Baseline YAML path for comparison script: $RESOLVED_BASELINE_YAML"
echo "Comparison script path: $RESOLVED_COMPARISON_SCRIPT"

if [ ! -f "$ACTUAL_RESULTS_JSON_PATH" ]; then
  echo "::warning::Primary results file '$ACTUAL_RESULTS_JSON_PATH' not found. Cannot perform baseline comparison."
  # If results.json.txt exists, the comparison script won't use it.
  # This indicates an issue in run_benchmark.sh not creating the primary results.json
  # Depending on policy, this could be an error: exit 1
  exit 0 # Exiting cleanly to not block if results are missing
fi

# Construct the config_id for baseline lookup (must match keys in baseline YAML)
NUM_HOSTS=$(echo "$TOPOLOGY_JSON" | jq -r '.num_hosts // "1"')
DEVICES_PER_HOST=$(echo "$TOPOLOGY_JSON" | jq -r '.num_devices_per_host // "1"')
HW_CAT_LOWER=$(echo "$HARDWARE_CATEGORY" | tr '[:upper:]' '[:lower:]')
# Use CONFIG_ID (which is matrix.benchmark_entry.config_id || matrix.benchmark_entry.benchmark_name)
SCRIPT_CONFIG_ID_FOR_BASELINE_LOOKUP="${CONFIG_ID}_${HW_CAT_LOWER}_${NUM_HOSTS}_host_${DEVICES_PER_HOST}_device"

echo "Constructed Config ID for baseline lookup: $SCRIPT_CONFIG_ID_FOR_BASELINE_LOOKUP"

python3 "$RESOLVED_COMPARISON_SCRIPT" \
  --results-json-file="$ACTUAL_RESULTS_JSON_PATH" \
  --baseline-yaml-file="$RESOLVED_BASELINE_YAML" \
  --config-id="$SCRIPT_CONFIG_ID_FOR_BASELINE_LOOKUP"

COMPARISON_EXIT_CODE=$?
if [ $COMPARISON_EXIT_CODE -ne 0 ]; then
  echo "::error::Baseline comparison script failed or regressions detected (Exit Code: $COMPARISON_EXIT_CODE)."
  exit $COMPARISON_EXIT_CODE # This will fail the GitHub Actions step and block the PR
fi
echo "Baseline comparison successful: No regressions detected or no applicable baselines found."