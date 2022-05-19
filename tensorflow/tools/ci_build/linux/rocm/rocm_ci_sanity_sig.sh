#!/usr/bin/env bash

export TEST_PATH=tensorflow/tools/ci_build/linux/rocm/

bats $TEST_PATH/code_check_changed_files.bats --timing --formatter junit
bats $TEST_PATH/code_check_full.bats --timing --formatter junit
