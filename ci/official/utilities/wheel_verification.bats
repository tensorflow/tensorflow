# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
# Suite of verification tests for the SINGLE TensorFlow wheel in the "build"
# directory, or whatever path is set as $TF_WHEEL.

setup_file() {
    cd build
    if [[ -z "$TF_WHEEL" ]]; then
        export TF_WHEEL=$(find build -iname "*.whl")
    fi

    # Setup the env for the python import testing
    if [[ $TF_WHEEL == *"aarch64.whl" ]]; then
        python${TFCI_PYTHON_VERSION} -m venv "$BATS_FILE_TMPDIR/venv"
    else
        python3 -m venv "$BATS_FILE_TMPDIR/venv"
    fi
}

teardown_file() {
    rm -rf "$BATS_FILE_TMPDIR/venv"
}

@test "Wheel is manylinux2014 (manylinux_2_17) compliant" {
    python3 -m auditwheel show "$TF_WHEEL" > audit.txt
    # Verify wheel based upon name/architecture, fallback to x86
    if [[ $TF_WHEEL == *"aarch64.whl" ]]; then
        grep --quiet -zoP 'is consistent with the following platform tag:\n"manylinux_2_17_aarch64"\.' audit.txt
    else
        grep --quiet 'This constrains the platform tag to "manylinux_2_17_x86_64"' audit.txt
    fi
}

@test "Wheel conforms to upstream size limitations" {
    WHEEL_MEGABYTES=$(stat --format %s "$TF_WHEEL" | awk '{print int($1/(1024*1024))}')
    # Googlers: search for "test_tf_whl_size"
    case "$TF_WHEEL" in
        # CPU:
        *cpu*manylinux*) LARGEST_OK_SIZE=240 ;;
        # GPU:
        *manylinux*)     LARGEST_OK_SIZE=580 ;;
        # Unknown:
        *)
            echo "The wheel's name is in an unknown format."
            exit 1
            ;;
    esac
    # >&3 forces output in bats even if the test passes. See
    # https://bats-core.readthedocs.io/en/stable/writing-tests.html#printing-to-the-terminal
    echo "# Size of $TF_WHEEL is $WHEEL_MEGABYTES / $LARGEST_OK_SIZE megabytes." >&3
    test "$WHEEL_MEGABYTES" -le "$LARGEST_OK_SIZE"
}

# Note: this runs before the tests further down the file, so TF is installed in
# the venv and the venv is active when those tests run. The venv gets cleaned
# up in teardown_file() above.
@test "Wheel is installable" {
    source "$BATS_FILE_TMPDIR/venv/bin/activate"
    python3 -m pip install "$TF_WHEEL"
}

@test "TensorFlow is importable" {
    source "$BATS_FILE_TMPDIR/venv/bin/activate"
    python3 -c 'import tensorflow as tf; t1=tf.constant([1,2,3,4]); t2=tf.constant([5,6,7,8]); print(tf.add(t1,t2).shape)'
}

# Is this still useful?
@test "TensorFlow has Keras" {
    source "$BATS_FILE_TMPDIR/venv/bin/activate"
    python3 -c 'import sys; import tensorflow as tf; sys.exit(0 if "keras" in tf.keras.__name__ else 1)'
}

# Is this still useful?
@test "TensorFlow has Estimator" {
    source "$BATS_FILE_TMPDIR/venv/bin/activate"
    python3 -c 'import sys; import tensorflow as tf; sys.exit(0 if "_v2.estimator" in tf.estimator.__name__ else 1)'
}
