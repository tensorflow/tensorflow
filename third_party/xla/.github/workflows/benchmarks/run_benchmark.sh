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
set -u # Treat unset variables as an error when substituting.
# IMPORTANT: pipefail is handled specifically around the runner command.
set -e # Exit on errors, EXCEPT where explicitly handled.

echo "--- Running Benchmark ---"

# Reads ENVs from the Step:
#   RUNNER_BINARY, STATS_BINARY, DEVICE_TYPE_FLAG, LOCAL_ARTIFACT_PATH
# Reads ENVs from the Job:
#   BENCHMARK_NAME, CONFIG_ID, HARDWARE_CATEGORY, OUTPUT_DIR,
#   XLA_FLAGS_JSON, RUNTIME_FLAGS_JSON, 
#   COMMIT_SHA, WORKFLOW_RUN_ID

# --- Validate Inputs ---
if [ -z "$LOCAL_ARTIFACT_PATH" ] || [ ! -f "$LOCAL_ARTIFACT_PATH" ]; then echo "::error::LOCAL_ARTIFACT_PATH path is invalid or file not found: '$LOCAL_ARTIFACT_PATH'"; exit 1; fi
if [ -z "$RUNNER_BINARY" ] || [ ! -x "$RUNNER_BINARY" ]; then echo "::error::RUNNER_BINARY path is invalid or file not executable: '$RUNNER_BINARY'"; exit 1; fi
if [ -z "$DEVICE_TYPE_FLAG" ]; then echo "::error::DEVICE_TYPE_FLAG is empty"; exit 1; fi
if [ -z "$STATS_BINARY" ] || [ ! -x "$STATS_BINARY" ]; then echo "::error::STATS_BINARY path is invalid or file not executable: '$STATS_BINARY'"; exit 1; fi
if ! command -v jq &> /dev/null; then echo "::error::jq command not found."; exit 1; fi

RUNNER_STDOUT_FILE="$OUTPUT_DIR/runner_stdout.txt"
XSPACE_FILE_PATH="$OUTPUT_DIR/xspace.pb"
RESULTS_JSON_FILE="$OUTPUT_DIR/results.json"

# --- Prepare flags ---
declare -a xla_flags_array=()
declare -a runtime_flags_array=()

# Use JQ to safely parse JSON and populate bash arrays
if echo "$XLA_FLAGS_JSON" | jq -e '. | arrays and length > 0' > /dev/null; then
    mapfile -t xla_flags_array < <(echo "$XLA_FLAGS_JSON" | jq -r '.[]')
fi
if echo "$RUNTIME_FLAGS_JSON" | jq -e '. | arrays and length > 0' > /dev/null; then
   mapfile -t runtime_flags_array < <(echo "$RUNTIME_FLAGS_JSON" | jq -r '.[]')
fi

# Conditionally add profile flag if needed for stats
needs_profile_flag=true
for flag in "${runtime_flags_array[@]}"; do
    if [[ "$flag" == "--profile_execution"* ]]; then
        needs_profile_flag=false; break
    fi
done
needs_xspace_dump_flag=true # Assume we always want stats if possible
if $needs_profile_flag && $needs_xspace_dump_flag; then
    runtime_flags_array+=("--profile_execution=True")
     echo "INFO: Added --profile_execution=True for stats generation."
fi

# --- Build Runner Command ---
declare -a runner_command_array=("$RUNNER_BINARY" "--device_type=$DEVICE_TYPE_FLAG")
if [ ${#runtime_flags_array[@]} -gt 0 ]; then runner_command_array+=("${runtime_flags_array[@]}"); fi
if [ ${#xla_flags_array[@]} -gt 0 ]; then runner_command_array+=("${xla_flags_array[@]}"); fi
if $needs_xspace_dump_flag; then
   runner_command_array+=("--xla_gpu_dump_xspace_to=$XSPACE_FILE_PATH")
fi
runner_command_array+=("$LOCAL_ARTIFACT_PATH")

# --- Execute Runner ---
echo "Executing HLO Runner command:" 
printf "%q " "${runner_command_array[@]}"; echo # Print quoted command

set +e # Disable exit-on-error temporarily to capture exit code
set -o pipefail # Ensure tee doesn't mask the runner's exit code
"${runner_command_array[@]}" 2>&1 | tee "$RUNNER_STDOUT_FILE"
RUNNER_EXIT_CODE=${PIPESTATUS[0]}
set +o pipefail
set -e # Re-enable exit-on-error

echo "Runner stdout/stderr saved to $RUNNER_STDOUT_FILE"
echo "Runner exited with code: $RUNNER_EXIT_CODE"

# --- Execute Stats or Generate Fallback JSON ---
STATS_EXIT_CODE=0
if [ -f "$XSPACE_FILE_PATH" ] && [ $RUNNER_EXIT_CODE -eq 0 ]; then
  echo "Running compute_xspace_stats_main..."
  STATS_PLATFORM_TYPE=$([[ "$HARDWARE_CATEGORY" == GPU* ]] && echo "GPU" || echo "CPU")
  declare -a stats_command_array=("$STATS_BINARY" "--input=$XSPACE_FILE_PATH" "--device_type=$STATS_PLATFORM_TYPE" "--output_json=$RESULTS_JSON_FILE")

  echo "Executing Stats command:"; printf "%q " "${stats_command_array[@]}"; echo

  set +e # Disable exit-on-error temporarily
  "${stats_command_array[@]}" >> "$RUNNER_STDOUT_FILE" # Append stats stdout to runner log
  STATS_EXIT_CODE=$?
  set -e # Re-enable

  if [ $STATS_EXIT_CODE -ne 0 ]; then
     echo "::warning::compute_xspace_stats_main failed with code $STATS_EXIT_CODE."
      # Fallback to creating JSON with run status and error message for stats failure
      jq -n \
        --arg bn "$BENCHMARK_NAME" --arg cid "$CONFIG_ID" --arg hc "$HARDWARE_CATEGORY" \
        --arg rs "STATS_FAILURE" \
        --arg em "compute_xspace_stats_main failed with code $STATS_EXIT_CODE. Runner was successful." \
        --arg cs "$COMMIT_SHA" --arg wrid "$WORKFLOW_RUN_ID" \
        '{ benchmark_name: $bn, config_id: $cid, hardware_category: $hc, run_status: $rs, error_message: $em, commit_sha: $cs, workflow_run_id: $wrid }' \
        > "$RESULTS_JSON_FILE"
     echo "Fallback results JSON created at $RESULTS_JSON_FILE due to stats failure."
  else
      echo "Stats computed and saved to $RESULTS_JSON_FILE"
  fi
else
   # Create fallback JSON if Runner failed OR if Runner succeeded but produced no XSpace file
   if [ $RUNNER_EXIT_CODE -ne 0 ]; then 
      echo "::warning::Runner failed (Exit Code: $RUNNER_EXIT_CODE), skipping stats."
   else 
     echo "::warning::XSpace file missing at $XSPACE_FILE_PATH, skipping stats."
   fi

   RUN_STATUS=$([ $RUNNER_EXIT_CODE -eq 0 ] && echo "SUCCESS_NO_PROFILE" || echo "FAILURE")
   ERROR_MSG=$([ $RUNNER_EXIT_CODE -ne 0 ] && echo "Runner failed with code $RUNNER_EXIT_CODE" || echo "XSpace file not generated by successful run.")

    jq -n \
      --arg bn "$BENCHMARK_NAME" --arg cid "$CONFIG_ID" --arg hc "$HARDWARE_CATEGORY" \
      --arg rs "$RUN_STATUS" --arg em "$ERROR_MSG" \
       --arg cs "$COMMIT_SHA" --arg wrid "$WORKFLOW_RUN_ID" \
      '{ benchmark_name: $bn, config_id: $cid, hardware_category: $hc, run_status: $rs, error_message: $em, commit_sha: $cs, workflow_run_id: $wrid }' \
       > "$RESULTS_JSON_FILE"

     if [ $? -eq 0 ]; then
        echo "Basic results JSON created at $RESULTS_JSON_FILE."
     else
        # Should not happen if jq is present, but a safety-net
        echo "::error::FATAL: Failed to create basic results JSON using jq."
        echo "Fallback error: Benchmark Name: $BENCHMARK_NAME, Run Status: $RUN_STATUS, Error: $ERROR_MSG" > "$RESULTS_JSON_FILE.txt"
        exit 1 # Make sure this failure is noted
     fi
fi

# --- Final Exit Status ---
if [ $RUNNER_EXIT_CODE -ne 0 ]; then 
  echo "::error::Benchmark run failed (Runner Exit Code: $RUNNER_EXIT_CODE)."
  exit $RUNNER_EXIT_CODE # Propagate the runner's failure code
fi

echo "--- Run Benchmark Script Finished Successfully ---"