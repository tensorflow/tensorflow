#!/bin/bash
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

if [ ! -d "../Apollo3-SDK-2018.08.13" ]; then
    echo "Apollo 3 SDK does not exist"
    echo "Either the SDK has not been downloaded, or this script is not being done from the root of the repository"
else
    DEST_DIR="tensorflow/lite/experimental/micro/tools/make/targets/apollo3evb"
    AP3_DIR="../Apollo3-SDK-2018.08.13"
    cp "$AP3_DIR/boards/apollo3_evb/examples/hello_world/gcc/startup_gcc.c" "$DEST_DIR"
    cp "$AP3_DIR/utils/am_util_delay.c" "$DEST_DIR"
    cp "$AP3_DIR/utils/am_util_faultisr.c" "$DEST_DIR"
    cp "$AP3_DIR/utils/am_util_id.c" "$DEST_DIR"
    cp "$AP3_DIR/utils/am_util_stdio.c" "$DEST_DIR"
    cp "$AP3_DIR/boards/apollo3_evb/bsp/gcc/bin/libam_bsp.a" "$DEST_DIR"
    cp "$AP3_DIR/mcu/apollo3/hal/gcc/bin/libam_hal.a" "$DEST_DIR"
fi
