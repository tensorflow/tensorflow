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

AP3_DIR="tensorflow/lite/micro/tools/make/downloads/Apollo3-SDK-2018.08.13"
if [ ! -d $AP3_DIR ]; then
    echo "Apollo 3 SDK does not exist"
    echo "Either the SDK has not been downloaded, or this script is not being run from the root of the repository"
else
    DEST_DIR="tensorflow/lite/micro/tools/make/targets/apollo3evb"
    cp "$AP3_DIR/boards/apollo3_evb/examples/hello_world/gcc/startup_gcc.c" "$DEST_DIR"
    cp "$AP3_DIR/boards/apollo3_evb/examples/hello_world/gcc/hello_world.ld" "$DEST_DIR/apollo3evb.ld"
    sed -i -e '131s/1024/1024\*20/g' "$DEST_DIR/startup_gcc.c"
    sed -i -e 's/main/_main/g' "$DEST_DIR/startup_gcc.c"
    sed -i -e '3s/hello_world.ld/apollo3evb.ld/g' "$DEST_DIR/apollo3evb.ld"
    sed -i -e '3s/startup_gnu/startup_gcc/g' "$DEST_DIR/apollo3evb.ld"
    sed -i -e '6s/am_reset_isr/Reset_Handler/g' "$DEST_DIR/apollo3evb.ld"
    sed -i -e '22s/\*(.text\*)/\*(.text\*)\n\n\t\/\* These are the C++ global constructors.  Stick them all here and\n\t \* then walk through the array in main() calling them all.\n\t \*\/\n\t_init_array_start = .;\n\tKEEP (\*(SORT(.init_array\*)))\n\t_init_array_end = .;\n\n\t\/\* XXX Currently not doing anything for global destructors. \*\/\n/g' "$DEST_DIR/apollo3evb.ld"
    sed -i -e "70s/} > SRAM/} > SRAM\n    \/\* Add this to satisfy reference to symbol 'end' from libnosys.a(sbrk.o)\n     \* to denote the HEAP start.\n     \*\/\n   end = .;/g" "$DEST_DIR/apollo3evb.ld"
    echo "Finished preparing Apollo3 files"
    

fi
