# vim: filetype=bash
#
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

setup_file() {
    cd /tf/tensorflow
    bazel version  # Start the bazel server
    # Without this, git errors if /tf/tensorflow directory owner is different
    git config --global --add safe.directory /tf/tensorflow
    # Note that you could generate a list of all the affected targets with e.g.:
    # bazel query $(paste -sd "+" $BATS_FILE_TMPDIR/changed_files) --keep_going
    # Only shows Added, Changed, Modified, Renamed, and Type-changed files
    if [[ "$(git rev-parse --abbrev-ref HEAD)" = "pull_branch" ]]; then
        # TF's CI runs 'git fetch origin "pull/PR#/merge:pull_branch"'
        # To get the as-merged branch during the CI tests
        git diff --diff-filter ACMRT --name-only pull_branch^ pull_branch > $BATS_FILE_TMPDIR/changed_files
    else
        # If the branch is not present, then diff against origin/master
        git diff --diff-filter ACMRT --name-only origin/master > $BATS_FILE_TMPDIR/changed_files
    fi
}

# Note: this is excluded on the full code base, since any submitted code must
# have passed Google's internal style guidelines.
@test "Check buildifier formatting on BUILD files" {
    echo "buildifier formatting is recommended. Here are the suggested fixes:"
    echo "============================="
    grep -e 'BUILD' $BATS_FILE_TMPDIR/changed_files \
        | xargs buildifier -v -mode=diff -diff_command="git diff --no-index"
}

# Note: this is excluded on the full code base, since any submitted code must
# have passed Google's internal style guidelines.
@test "Check formatting for C++ files" {
    skip "clang-format doesn't match internal clang-format checker"
    echo "clang-format is recommended. Here are the suggested changes:"
    echo "============================="
    grep -e '\.h$' -e '\.cc$' $BATS_FILE_TMPDIR/changed_files > $BATS_TEST_TMPDIR/files || true
    if [[ ! -s $BATS_TEST_TMPDIR/files ]]; then return 0; fi
    xargs -a $BATS_TEST_TMPDIR/files -i -n1 -P $(nproc --all) \
        bash -c 'clang-format-12 --style=Google {} | git diff --no-index {} -' \
        | tee $BATS_TEST_TMPDIR/needs_help.txt
    echo "You can use clang-format --style=Google -i <file> to apply changes to a file."
    [[ ! -s $BATS_TEST_TMPDIR/needs_help.txt ]]
}

# Note: this is excluded on the full code base, since any submitted code must
# have passed Google's internal style guidelines.
@test "Check pylint for Python files" {
    echo "Python formatting is recommended. Here are the pylint errors:"
    echo "============================="
    grep -e "\.py$" $BATS_FILE_TMPDIR/changed_files > $BATS_TEST_TMPDIR/files || true
    if [[ ! -s $BATS_TEST_TMPDIR/files ]]; then return 0; fi
    xargs -a $BATS_TEST_TMPDIR/files -n1 -P $(nproc --all) \
        python -m pylint --rcfile=tensorflow/tools/ci_build/pylintrc --score false \
        | grep -v "**** Module" \
        | tee $BATS_TEST_TMPDIR/needs_help.txt
    [[ ! -s $BATS_TEST_TMPDIR/needs_help.txt ]]
}

teardown_file() {
    bazel shutdown
}
