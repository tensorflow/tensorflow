#!/bin/bash

if [ -d ./tensorflow/lite/experimental/micro/tools/make/downloads/xtimecomposer/ ]; then
    pushd ./tensorflow/lite/experimental/micro/tools/make/downloads/xtimecomposer/
    source SetEnv
    popd
    exit 0
else
    make -f ./tensorflow/lite/experimental/micro/tools/make/Makefile TARGET="xcore" test
    xcore_test.sh
    #following line should prevent infinite recursion
    ./tensorflow/lite/experimental/micro/tools/make/downloads/xtimecomposer/
fi

