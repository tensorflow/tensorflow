#!/usr/bin/env bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

set -u  # Check for undefined variables

# Track temporary files so they are removed on exit, including on interrupt.
LOADED_LIBS_FILE=""
cleanup() {
  [ -n "${LOADED_LIBS_FILE:-}" ] && rm -f "$LOADED_LIBS_FILE"
}
trap cleanup EXIT INT TERM

die() {
  # Print a message and exit with code 1.
  #
  # Usage: die <error_message>
  #   e.g., die "Something bad happened."

  echo "$@" 1>&2
  exit 1
}

usage() {
  cat <<'USAGE'
Usage: tf_env_collect.sh [options]

Collect TensorFlow environment information for high-quality bug reports.

Options:
  -o, --output FILE   Write the human-readable report to FILE
                      (default: tf_env.txt).
      --json          Also write a machine-readable JSON summary alongside
                      the report (default: tf_env.json, derived from --output).
  -v, --verbose       Include extra diagnostics: loaded shared libraries,
                      accelerator toolchain versions, and full accelerator
                      environment variables.
  -h, --help          Show this help message and exit.
USAGE
}

# ----------------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------------
OUTPUT_FILE="tf_env.txt"
EMIT_JSON=0
VERBOSE=0

while [ $# -gt 0 ]; do
  case "$1" in
    -o|--output)
      shift
      [ $# -gt 0 ] || die "--output requires a file argument"
      OUTPUT_FILE="$1"
      ;;
    --json) EMIT_JSON=1 ;;
    -v|--verbose) VERBOSE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" 1>&2; usage; exit 1 ;;
  esac
  shift
done

# Derive the JSON path from the output path (foo.txt -> foo.json).
case "${OUTPUT_FILE##*/}" in
  *.*) JSON_FILE="${OUTPUT_FILE%.*}.json" ;;
  *)   JSON_FILE="${OUTPUT_FILE}.json" ;;
esac

echo "Collecting system information..."

PYTHON_BIN_PATH="$(command -v python || command -v python3 || die "Cannot find Python binary")"

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
have_cmd() {
  # Return success if the given command is available on PATH.
  command -v "$1" >/dev/null 2>&1
}

run_cmd() {
  # Run a command if it exists, otherwise note that it is missing instead of
  # erroring out. Captures stderr so the report stays readable.
  if have_cmd "$1"; then
    "$@" 2>&1
  else
    echo "$1 not found"
  fi
}

# Detect a usable pip front-end once. "python -m pip" is preferred because it
# is guaranteed to match the interpreter we are inspecting.
PIP_KIND=""
if "${PYTHON_BIN_PATH}" -m pip --version >/dev/null 2>&1; then
  PIP_KIND="module"
elif have_cmd pip; then
  PIP_KIND="pip"
elif have_cmd pip3; then
  PIP_KIND="pip3"
fi

pip_run() {
  case "$PIP_KIND" in
    module) "${PYTHON_BIN_PATH}" -m pip "$@" 2>&1 ;;
    pip)    pip "$@" 2>&1 ;;
    pip3)   pip3 "$@" 2>&1 ;;
    *)      echo "pip not found" 1>&2; return 1 ;;
  esac
}

# Packages whose simultaneous presence usually indicates a broken install.
TF_PKG_PATTERN='^(tensorflow|tf-nightly|tensorflow-cpu|tensorflow-gpu|tensorflow-rocm|tensorflow-macos|tensorflow-metal|intel-tensorflow)\b'

HEADER_WIDTH=68
# Create a string of HEADER_WIDTH "=" characters
HEADER=$(printf "%*s" "$HEADER_WIDTH" "" | sed 's/ /=/g')

print_header () {
  # This function simply prints the header with even spacing,
  # and also prints it to STDERR so that the human running
  # the script sees progress.
  local TITLE="$1"
  echo
  # This line is a bit cryptic, but it essentially
  # just pads the title with "=" to be the desired length.
  local PADDED_TITLE="== $TITLE ${HEADER:${#TITLE}+4}"
  # Echo to STDOUT
  echo "$PADDED_TITLE"
  # Echo to STDERR (to show progress to the user as it runs)
  echo "$PADDED_TITLE" 1>&2
}

# Clear the output file
echo > "$OUTPUT_FILE"

{
  # ==========================================================================
  # Section 1: host, Python, and OS environment
  # ==========================================================================
  print_header "report metadata"
  echo "tf_env_collect.sh report"
  echo "generated: $(date -u '+%Y-%m-%dT%H:%M:%SZ' 2>/dev/null || date)"
  echo "verbose: $([ "$VERBOSE" -eq 1 ] && echo yes || echo no)"

  print_header "check python"

  "${PYTHON_BIN_PATH}" <<EOF
import platform

print(f"""python version: {platform.python_version()}
python branch: {platform.python_branch()}
python build version: {platform.python_build()}
python compiler version: {platform.python_compiler()}
python implementation: {platform.python_implementation()}
""")
EOF

  print_header "check os platform"

  "${PYTHON_BIN_PATH}" <<EOF
import platform

PLATFORM_ENTRIES = [
    ("os", "system"),
    ("os kernel version", "version"),
    ("os release version", "release"),
    ("os platform", "platform"),
    ("freedesktop os release", "freedesktop_os_release"),
    ("mac version", "mac_ver"),
    ("uname", "uname"),
    ("architecture", "architecture"),
    ("machine", "machine"),
]

for label, function_name in PLATFORM_ENTRIES:
    if hasattr(platform, function_name):
        function = getattr(platform, function_name)
        result = function()  # Call the function
        print(f"{label}: {result}")
    else:
        print(f"{label}: N/A")

if platform.system() == "Darwin" and platform.machine() == "arm64":
    print("apple silicon: yes")
EOF


  print_header 'are we in docker'
  if grep -q docker /proc/1/cgroup 2>/dev/null || [ -f /.dockerenv ]; then
    echo "Yes"
  else
    echo "No"
  fi

  print_header 'c++ compiler'
  if have_cmd c++; then
    c++ --version 2>&1
  else
    echo "Not found"
  fi

  print_header 'check pips'
  pip_run list 2>&1 | grep -E 'proto|numpy|keras|tensorflow|tf_nightly|tf-nightly'


  print_header 'check for virtualenv'
  "${PYTHON_BIN_PATH}" <<EOF
import sys

if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print("Running inside a virtual environment.")
else:
    print("Not running inside a virtual environment.")
EOF

  # ==========================================================================
  # Section 2: TensorFlow installation and runtime
  # ==========================================================================
  print_header 'tensorflow package conflicts'
  # Multiple TensorFlow distributions in the same environment frequently cause
  # confusing import errors; surface them so triage can spot the conflict.
  TF_PKGS="$(pip_run list 2>/dev/null | grep -iE "$TF_PKG_PATTERN" || true)"
  if [ -z "$TF_PKGS" ]; then
    echo "No TensorFlow packages found via pip."
  else
    echo "$TF_PKGS"
    TF_COUNT="$(echo "$TF_PKGS" | grep -icE "$TF_PKG_PATTERN")"
    if [ "$TF_COUNT" -gt 1 ]; then
      echo "WARNING: multiple TensorFlow distributions detected; this can cause import conflicts."
    fi
  fi

  print_header 'tensorflow import'

  "${PYTHON_BIN_PATH}" <<EOF 2>&1
import tensorflow as tf;
print(f"""
tf.version.VERSION = {tf.version.VERSION}
tf.version.GIT_VERSION = {tf.version.GIT_VERSION}
tf.version.COMPILER_VERSION = {tf.version.COMPILER_VERSION}
""")
print(f"Sanity check: {tf.constant([1,2,3])[:1]!r}")
EOF

  print_header 'tensorflow visible devices'
  # Show the accelerators TensorFlow can actually see (CPU/GPU/TPU), which is
  # often more informative than vendor tools alone.
  "${PYTHON_BIN_PATH}" <<EOF 2>&1
try:
    import tensorflow as tf
    devices = tf.config.list_physical_devices()
    if not devices:
        print("No physical devices reported.")
    for d in devices:
        print(f"{d.device_type}: {d.name}")
except Exception as e:  # noqa: BLE001 - diagnostic best-effort
    print(f"Could not list devices: {e}")
EOF

  if [ "$VERBOSE" -eq 1 ]; then
    print_header 'loaded libraries (tensorflow import)'
    # Record shared libraries loaded by tensorflow. The mechanism differs
    # between Linux (LD_DEBUG) and macOS (DYLD_PRINT_LIBRARIES).
    LOADED_LIBS_FILE="$(mktemp 2>/dev/null || mktemp -t tfenv)"
    case "$(uname -s)" in
      Darwin)
        DYLD_PRINT_LIBRARIES=1 "${PYTHON_BIN_PATH}" -c "import tensorflow" \
          2>"$LOADED_LIBS_FILE" >/dev/null
        ;;
      *)
        LD_DEBUG=libs "${PYTHON_BIN_PATH}" -c "import tensorflow" \
          2>"$LOADED_LIBS_FILE" >/dev/null
        ;;
    esac
    if grep -qi 'cudnn' "$LOADED_LIBS_FILE"; then
      echo "libcudnn found"
    else
      echo "libcudnn not found"
    fi
    # Removed eagerly here; the EXIT trap also cleans up if we exit early.
    rm -f "$LOADED_LIBS_FILE"
    LOADED_LIBS_FILE=""
  fi

  # ==========================================================================
  # Section 3: accelerators and build / hermetic configuration
  # ==========================================================================
  print_header env

  # Note: the usage of "set -u" above would cause these to error if the
  #   basic form [[ -z $LD_LIBRARY_PATH ]] was used.
  if [ -z "${LD_LIBRARY_PATH+x}" ]; then
    echo "LD_LIBRARY_PATH is unset"
  else
    echo "LD_LIBRARY_PATH ${LD_LIBRARY_PATH}"
  fi
  if [ -z "${DYLD_LIBRARY_PATH+x}" ]; then
    echo "DYLD_LIBRARY_PATH is unset"
  else
    echo "DYLD_LIBRARY_PATH ${DYLD_LIBRARY_PATH}"
  fi

  print_header 'build / hermetic accelerator config'
  # Surface the environment variables that control modern (hermetic) CUDA and
  # ROCm builds. See .bazelrc for how these are consumed.
  for var in CC CXX \
             TF_NEED_CUDA TF_NEED_ROCM TF_CUDA_VERSION TF_CUDNN_VERSION \
             HERMETIC_CUDA_VERSION HERMETIC_CUDNN_VERSION \
             CUDA_HOME CUDA_PATH CUDA_TOOLKIT_PATH \
             ROCM_PATH HIP_PATH \
             XLA_FLAGS TF_XLA_FLAGS TPU_NAME; do
    eval "marker=\${$var+set} val=\"\$$var\""
    if [ "${marker:-}" = "set" ]; then
      echo "$var=$val"
    else
      echo "$var is unset"
    fi
  done

  print_header 'accelerator: nvidia gpu'
  run_cmd nvidia-smi

  print_header 'cuda libs'
  # Find cudart/cudnn files
  find /usr -type f -name 'libcud*'  2>/dev/null | grep -E 'cuda.*(cudart|cudnn)' | grep -v -F '.cache'
  if [ "$VERBOSE" -eq 1 ]; then
    print_header 'nvcc version'
    run_cmd nvcc --version
  fi

  print_header 'accelerator: amd / rocm gpu'
  run_cmd rocm-smi
  if [ "$VERBOSE" -eq 1 ]; then
    print_header 'rocminfo'
    run_cmd rocminfo
  fi
  print_header 'rocm libs'
  find /opt/rocm /usr -type f \( -name 'libhip*' -o -name 'libMIOpen*' -o -name 'librocm*' -o -name 'librccl*' \) \
    2>/dev/null | grep -v -F '.cache'

  print_header 'accelerator: apple metal'
  # tensorflow-metal is the PluggableDevice that enables GPU acceleration on
  # Apple Silicon; report whether it is installed and the GPU chipset.
  if [ "$(uname -s)" = "Darwin" ]; then
    if pip_run show tensorflow-metal >/dev/null 2>&1; then
      echo "tensorflow-metal installed:"
      pip_run show tensorflow-metal 2>&1 | grep -iE '^(Name|Version):'
    else
      echo "tensorflow-metal not installed"
    fi
    if [ "$VERBOSE" -eq 1 ] && have_cmd system_profiler; then
      system_profiler SPDisplaysDataType 2>/dev/null | grep -iE 'Chipset|Metal|Vendor' || true
    fi
  else
    echo "Not a macOS host; skipping Metal checks."
  fi

  print_header 'tensorflow installation'
  if ! pip_run show tensorflow >/dev/null 2>&1; then
    echo "tensorflow not found"
  else
    pip_run show tensorflow
  fi

  print_header 'tf_nightly installation'
  if ! pip_run show tf_nightly >/dev/null 2>&1; then
    echo "tf_nightly not found"
  else
    pip_run show tf_nightly
  fi

  print_header 'python version'
  echo '(major, minor, micro, releaselevel, serial)'
  "${PYTHON_BIN_PATH}" -c 'import sys; print(sys.version_info[:])'

  print_header 'bazel version'
  run_cmd bazel version

# Remove any lines with google.
} | grep -v -i google >> "$OUTPUT_FILE"

# ----------------------------------------------------------------------------
# Optional machine-readable JSON summary
# ----------------------------------------------------------------------------
if [ "$EMIT_JSON" -eq 1 ]; then
  # Detect accelerators in the shell and hand the booleans to Python, which
  # assembles a structured, easy-to-parse summary of the key facts.
  HAS_NVIDIA=0
  if have_cmd nvidia-smi && nvidia-smi >/dev/null 2>&1; then HAS_NVIDIA=1; fi
  HAS_ROCM=0
  if have_cmd rocm-smi && rocm-smi >/dev/null 2>&1; then HAS_ROCM=1; fi
  HAS_METAL=0
  if pip_run show tensorflow-metal >/dev/null 2>&1; then HAS_METAL=1; fi

  TFENV_HAS_NVIDIA="$HAS_NVIDIA" \
  TFENV_HAS_ROCM="$HAS_ROCM" \
  TFENV_HAS_METAL="$HAS_METAL" \
  TFENV_JSON_FILE="$JSON_FILE" \
  "${PYTHON_BIN_PATH}" <<'EOF'
import datetime
import json
import os
import platform
import sys


def tf_info():
    info = {"importable": False}
    try:
        import tensorflow as tf
        info["importable"] = True
        info["version"] = tf.version.VERSION
        info["git_version"] = tf.version.GIT_VERSION
        info["compiler_version"] = tf.version.COMPILER_VERSION
        try:
            info["physical_devices"] = [
                {"type": d.device_type, "name": d.name}
                for d in tf.config.list_physical_devices()
            ]
        except Exception as e:  # noqa: BLE001 - diagnostic best-effort
            info["physical_devices_error"] = str(e)
    except Exception as e:  # noqa: BLE001 - diagnostic best-effort
        info["import_error"] = str(e)
    return info


def redact(obj):
    """Recursively redact string values containing "google" (case-insensitive).

    Mirrors the `grep -v -i google` filter applied to the text report so the
    JSON summary cannot leak internal hostnames, depot paths, or build
    configurations when users upload diagnostic logs.
    """
    if isinstance(obj, str):
        return "[redacted]" if "google" in obj.lower() else obj
    if isinstance(obj, dict):
        return {k: redact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj


in_venv = hasattr(sys, "real_prefix") or (
    hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
)

relevant_env_keys = (
    "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",
    "CC", "CXX",
    "CUDA_HOME", "CUDA_PATH", "CUDA_TOOLKIT_PATH",
    "TF_NEED_CUDA", "TF_NEED_ROCM", "TF_CUDA_VERSION", "TF_CUDNN_VERSION",
    "HERMETIC_CUDA_VERSION", "HERMETIC_CUDNN_VERSION",
    "ROCM_PATH", "HIP_PATH",
    "XLA_FLAGS", "TF_XLA_FLAGS", "TPU_NAME",
)

report = {
    "schema_version": "1.0",
    "collected_at": datetime.datetime.now().astimezone().isoformat(),
    "python": {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "in_virtualenv": in_venv,
    },
    "os": {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "is_apple_silicon": (
            platform.system() == "Darwin" and platform.machine() == "arm64"
        ),
    },
    "accelerators": {
        "nvidia_smi": os.environ.get("TFENV_HAS_NVIDIA") == "1",
        "rocm_smi": os.environ.get("TFENV_HAS_ROCM") == "1",
        "tensorflow_metal": os.environ.get("TFENV_HAS_METAL") == "1",
    },
    "tensorflow": tf_info(),
    "relevant_env": {
        k: os.environ[k] for k in relevant_env_keys if k in os.environ
    },
}

# Apply the same `google` redaction the text report relies on before writing.
report = redact(report)

out_path = os.environ["TFENV_JSON_FILE"]
with open(out_path, "w") as f:
    json.dump(report, f, indent=2, sort_keys=True)
    f.write("\n")
print(f"Wrote JSON summary to {out_path}")
EOF
fi

echo "Wrote environment to ${OUTPUT_FILE}. You can review the contents of that file."
echo "and use it to populate the fields in the github issue template."
echo
echo "cat ${OUTPUT_FILE}"
echo
