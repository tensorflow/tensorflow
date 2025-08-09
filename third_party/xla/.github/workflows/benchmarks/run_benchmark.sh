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
# .github/workflows/benchmarks/run_benchmark.sh
# TODO(juliagmt): convert this to a python script.
#!/bin/bash

# This script encapsulates the logic to run a benchmark and generate results.json.

set -u # Treat unset variables as an error when substituting.
# IMPORTANT: pipefail is handled specifically around the runner command.
set -e # Exit on errors, EXCEPT where explicitly handled.

echo "--- Running Benchmark Script ---"
echo "Benchmark Name: $BENCHMARK_NAME"
echo "Config ID: $CONFIG_ID"
echo "Hardware Category: $HARDWARE_CATEGORY"
echo "Output Directory: $OUTPUT_DIR"
echo "Runner Binary: $RUNNER_BINARY"
echo "Stats Binary: $STATS_BINARY"
echo "Device Type Flag: $DEVICE_TYPE_FLAG"
echo "Local Artifact Path: $LOCAL_ARTIFACT_PATH"
echo "Input Format: $INPUT_FORMAT"
echo "XLA Flags JSON: $XLA_FLAGS_JSON"
echo "Runtime Flags JSON: $RUNTIME_FLAGS_JSON"
echo "Commit SHA: $COMMIT_SHA"
echo "Workflow Run ID: $WORKFLOW_RUN_ID"


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

needs_xspace_dump_flag=true # By default, we enable profiler and xspace dump.

# --- Build Runner Command ---
declare -a runner_command_array=(
    "$RUNNER_BINARY"
    "--device_type=$DEVICE_TYPE_FLAG"
)
if [ ${#runtime_flags_array[@]} -gt 0 ]; then runner_command_array+=("${runtime_flags_array[@]}"); fi
if [ ${#xla_flags_array[@]} -gt 0 ]; then runner_command_array+=("${xla_flags_array[@]}"); fi
if $needs_xspace_dump_flag; then
   runner_command_array+=("--xla_gpu_dump_xspace_to=$XSPACE_FILE_PATH")
fi
runner_command_array+=("$LOCAL_ARTIFACT_PATH")

# --- Execute Runner ---
echo "Executing HLO Runner command:"
printf "%q " "${runner_command_array[@]}"; echo

set +e # Disable exit-on-error temporarily to capture exit code
set -o pipefail # Ensure tee doesn't mask the runner's exit code
"${runner_command_array[@]}" 2>&1 | tee "$RUNNER_STDOUT_FILE"
RUNNER_EXIT_CODE=${PIPESTATUS[0]}
set +o pipefail
set -e # Re-enable exit-on-error

echo "Runner stdout/stderr saved to $RUNNER_STDOUT_FILE"
echo "Runner exited with code: $RUNNER_EXIT_CODE"


# --- Process Stats and Generate results.json ---
STATS_RUN_SUCCESSFUL=false
METRICS_JSON_CONTENT="{}" # Initialize as an empty JSON object

if [ -f "$XSPACE_FILE_PATH" ] && [ $RUNNER_EXIT_CODE -eq 0 ]; then
  echo "XSpace file found. Running compute_xspace_stats_main..."
  STATS_PLATFORM_TYPE=$([[ "$HARDWARE_CATEGORY" == GPU* ]] && echo "GPU" || echo "CPU")

  # Capture the output of compute_xspace_stats_main to parse it
  # Do not write its output directly to results.json yet
  echo "Executing Stats command and capturing its output:"

  set +e # Temporarily disable exit-on-error for stats command
  STATS_OUTPUT=$("$STATS_BINARY" --input="$XSPACE_FILE_PATH" --device_type="$STATS_PLATFORM_TYPE" 2>&1)
  STATS_EXIT_CODE=$?
  set -e

  echo "compute_xspace_stats_main output:"
  echo "$STATS_OUTPUT"
  echo "compute_xspace_stats_main exited with code: $STATS_EXIT_CODE"

  # Append stats tool's raw output to the main runner log for complete record
  echo -e "\n--- compute_xspace_stats_main Raw Output ---" >> "$RUNNER_STDOUT_FILE"
  echo "$STATS_OUTPUT" >> "$RUNNER_STDOUT_FILE"
  echo "--- End compute_xspace_stats_main Raw Output ---" >> "$RUNNER_STDOUT_FILE"


  if [ $STATS_EXIT_CODE -eq 0 ]; then
    STATS_RUN_SUCCESSFUL=true
    metrics_obj_str="{"
    first_metric=true

    # Process each line of STATS_OUTPUT
    while IFS=':' read -r key value; do
        # Trim leading/trailing whitespace from key and value
        key=$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
        value=$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

        # Expecting lines like "Metric Name: 123.45 us"
        if [[ "$value" == *us ]]; then
            num_value=$(echo "$value" | sed 's/ us$//')
            # Convert microseconds to milliseconds using awk
            ms_value=$(LC_ALL=C awk -v num="$num_value" 'BEGIN { printf "%.3f", num / 1000 }')

            # Sanitize base metric key (e.g., "Device Time" -> "DEVICE_TIME")
            base_metric_key=$(echo "$key" | tr ' ' '_' | tr '[:lower:]' '[:upper:]')
            final_metric_key=""

            # Determine the final metric key based on HARDWARE_CATEGORY for baseline matching
            if [[ "$HARDWARE_CATEGORY" == GPU* ]]; then
                if [[ "$base_metric_key" == "DEVICE_TIME" ]]; then
                    final_metric_key="GPU_DEVICE_TIME"
                elif [[ "$base_metric_key" == "DEVICE_MEMCPY_TIME" ]]; then
                    final_metric_key="GPU_DEVICE_MEMCPY_TIME"
                # Add other specific GPU mappings here if needed
                # else
                #    final_metric_key="GPU_${base_metric_key}" # Generic prefix
                else
                    final_metric_key="$base_metric_key" # If no specific GPU mapping, use base
                fi
            elif [[ "$HARDWARE_CATEGORY" == CPU* ]]; then
                if [[ "$base_metric_key" == "CPU_TIME" ]] || [[ "$base_metric_key" == "TIME" ]]; then # Handle "CPU Time" or just "Time"
                    final_metric_key="CPU_TIME"
                elif [[ "$base_metric_key" == "WALL_TIME" ]]; then
                    final_metric_key="WALL_TIME" # Wall time is generic
                # Add other specific CPU mappings here if needed
                # else
                #    final_metric_key="CPU_${base_metric_key}" # Generic prefix
                else
                    final_metric_key="$base_metric_key" # If no specific CPU mapping, use base
                fi
            else
                final_metric_key="$base_metric_key" # For unknown/other categories
            fi

            echo "INFO: Parsed metric: OriginalKey='$key', BaseKey='$base_metric_key', FinalKey='$final_metric_key', ValueMs='$ms_value'"

            if ! $first_metric; then metrics_obj_str+=","; fi
            metrics_obj_str+="\"$final_metric_key\": {\"value_ms\": $ms_value, \"unit\": \"ms\"}"
            first_metric=false
        fi
    done <<< "$STATS_OUTPUT"
    metrics_obj_str+="}"

    if echo "$metrics_obj_str" | jq -e . > /dev/null 2>&1; then
        METRICS_JSON_CONTENT=$(echo "$metrics_obj_str" | jq '.')
        echo "Successfully parsed metrics from stats output."
    else
        echo "::warning::Could not construct valid JSON from stats output. Metrics object will be empty."
        echo "Problematic metrics string constructed: $metrics_obj_str"
        METRICS_JSON_CONTENT="{}"
        STATS_RUN_SUCCESSFUL=false 
    fi
  else
    echo "::warning::compute_xspace_stats_main failed with code $STATS_EXIT_CODE. No metrics will be parsed from its output."
  fi
else
   if [ $RUNNER_EXIT_CODE -ne 0 ]; then 
      echo "::warning::Runner failed (Exit Code: $RUNNER_EXIT_CODE), skipping stats processing."
   else 
     echo "::warning::XSpace file missing at $XSPACE_FILE_PATH, skipping stats processing."
   fi
fi

# --- Construct Final results.json ---
RUN_STATUS_MSG=""
ERROR_MSG_CONTENT=""

if [ $RUNNER_EXIT_CODE -ne 0 ]; then
    RUN_STATUS_MSG="FAILURE"
    ERROR_MSG_CONTENT="Runner failed with code $RUNNER_EXIT_CODE"
elif [ ! -f "$XSPACE_FILE_PATH" ]; then
    RUN_STATUS_MSG="SUCCESS_NO_PROFILE"
    ERROR_MSG_CONTENT="XSpace file not generated by successful run."
elif [ $STATS_EXIT_CODE -ne 0 ] || [ "$STATS_RUN_SUCCESSFUL" = false ] ; then
    RUN_STATUS_MSG="STATS_FAILURE"
    ERROR_MSG_CONTENT="compute_xspace_stats_main failed (code $STATS_EXIT_CODE) or metrics parsing failed. Runner was successful."
else
    RUN_STATUS_MSG="SUCCESS"
    ERROR_MSG_CONTENT=""
fi

# Use jq to build the final JSON, incorporating the parsed metrics
jq -n \
  --arg bn "$BENCHMARK_NAME" \
  --arg cid "$CONFIG_ID" \
  --arg hc "$HARDWARE_CATEGORY" \
  --arg rs "$RUN_STATUS_MSG" \
  --arg em "$ERROR_MSG_CONTENT" \
  --arg cs "$COMMIT_SHA" \
  --arg wrid "$WORKFLOW_RUN_ID" \
  --argjson metrics "$METRICS_JSON_CONTENT" \
  '{
     benchmark_name: $bn,
     config_id: $cid,
     hardware_category: $hc,
     run_status: $rs,
     error_message: $em,
     commit_sha: $cs,
     workflow_run_id: $wrid,
     metrics: $metrics
   }' > "$RESULTS_JSON_FILE"

if [ $? -eq 0 ]; then
    echo "Final results.json created at $RESULTS_JSON_FILE."
else
    echo "::error::FATAL: Failed to create final results.json using jq."
    echo "FATAL JQ ERROR. Benchmark Name: $BENCHMARK_NAME, Run Status: $RUN_STATUS_MSG, Error: $ERROR_MSG_CONTENT" > "$RESULTS_JSON_FILE.txt"
    exit 1 
fi

# --- Debug: Verify file creation ---
echo "DEBUG: Listing contents of OUTPUT_DIR ($OUTPUT_DIR):"
ls -la "$OUTPUT_DIR"
echo "DEBUG: Checking for RESULTS_JSON_FILE ($RESULTS_JSON_FILE):"
if [ -f "$RESULTS_JSON_FILE" ]; then
  echo "DEBUG: RESULTS_JSON_FILE exists. Content (first 20 lines):"
  head -n 20 "$RESULTS_JSON_FILE"
else
  echo "DEBUG: RESULTS_JSON_FILE does NOT exist."
  if [ -f "${RESULTS_JSON_FILE}.txt" ]; then
    echo "DEBUG: RESULTS_JSON_FILE.txt exists. Content:"
    cat "${RESULTS_JSON_FILE}.txt"
  else
    echo "DEBUG: RESULTS_JSON_FILE.txt also does NOT exist."
  fi
fi
echo "DEBUG: End of file check."

# --- Final Exit Status of the script ---
if [ $RUNNER_EXIT_CODE -ne 0 ]; then
  echo "::error::Benchmark run failed (HLO Runner Exit Code: $RUNNER_EXIT_CODE)."
  exit $RUNNER_EXIT_CODE
fi

echo "--- Run Benchmark Script Finished ---"