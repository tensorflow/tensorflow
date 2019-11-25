#!/bin/bash
# n.b. call this using . xcore_tests.sh, not ./xcore_tests.sh or .xcore_tests.sh
if [ -d ./tensorflow/lite/experimental/micro/tools/make/downloads/xtimecomposer/ ]; then
    pushd ./tensorflow/lite/experimental/micro/tools/make/downloads/xtimecomposer/
    source SetEnv
    popd
    xcc --version
    xsim --version
else
    make -f ./tensorflow/lite/experimental/micro/tools/make/Makefile TARGET="xcore" test
    ./xcore_test.sh
    #following line should prevent infinite recursion
    mkdir ./tensorflow/lite/experimental/micro/tools/make/downloads/xtimecomposer/
fi

