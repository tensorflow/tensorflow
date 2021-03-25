# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(CMAKE_SYSTEM_NAME "Linux")
set(CMAKE_SYSTEM_PROCESSOR "riscv")

# Force object file extension to be .o
set(UNIX TRUE CACHE STRING "" FORCE)

if(APPLE)
  set(TOOLCHAIN_PREFIX_PATH ${CMAKE_SOURCE_DIR}/tools/cmake/riscv/Prebuilt/toolchain/clang/darwin/RISCV)
  set(QEMU_PREFIX_PATH ${CMAKE_SOURCE_DIR}/tools/cmake/riscv/Prebuilt/qemu/darwin/RISCV)
else()
  set(TOOLCHAIN_PREFIX_PATH ${CMAKE_SOURCE_DIR}/tools/cmake/riscv/Prebuilt/toolchain/clang/linux/RISCV)
  set(QEMU_PREFIX_PATH ${CMAKE_SOURCE_DIR}/tools/cmake/riscv/Prebuilt/qemu/linux/RISCV)
endif()

set(CMAKE_C_COMPILER "${TOOLCHAIN_PREFIX_PATH}/bin/clang")
set(CMAKE_CXX_COMPILER "${TOOLCHAIN_PREFIX_PATH}/bin/clang++")
# CMake will just use the host-side tools for the following tools, so we setup them here.
set(CMAKE_C_COMPILER_AR "${TOOLCHAIN_PREFIX_PATH}/bin/llvm-ar")
set(CMAKE_CXX_COMPILER_AR "${TOOLCHAIN_PREFIX_PATH}/bin/llvm-ar")
set(CMAKE_C_COMPILER_RANLIB "${TOOLCHAIN_PREFIX_PATH}/bin/llvm-ranlib")
set(CMAKE_CXX_COMPILER_RANLIB "${TOOLCHAIN_PREFIX_PATH}/bin/llvm-ranlib")
set(CMAKE_OBJDUMP "${TOOLCHAIN_PREFIX_PATH}/bin/llvm-objdump")
set(CMAKE_OBJCOPY "${TOOLCHAIN_PREFIX_PATH}/bin/llvm-objcopy")

add_compile_options(-mcpu=sifive-7-rv64)
# Use fused multiply-add operation.
add_compile_options(-ffp-contract=fast)
# The options for "--gc-sections" link option.
add_compile_options(-fdata-sections -ffunction-sections)

set(RISCV_COMPILE_DEFINITIONS "" CACHE STRING "The riscv specific compiling defs.")
add_compile_definitions(RISCV_COMPILE_DEFINITIONS)

set(RISCV_LINKER_FLAGS "" CACHE STRING "The riscv specific link flags.")
set(RISCV_LINKER_FLAGS "${RISCV_LINKER_FLAGS} -Wl,--gc-sections")
string(STRIP ${RISCV_LINKER_FLAGS} RISCV_LINKER_FLAGS)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${RISCV_LINKER_FLAGS}")

set(QEMU_PATH ${QEMU_PREFIX_PATH}/bin/qemu-riscv64)
set(QEMU_OPTION -cpu rv64,x-v=true,x-k=true,vlen=512,elen=64,vext_spec=v1.0 -L ${TOOLCHAIN_PREFIX_PATH}/sysroot)
# Use qemu for ctest.
set(CMAKE_CROSSCOMPILING_EMULATOR ${QEMU_PATH} ${QEMU_OPTION})

# The xnnpack haven't support riscv platform yet.
OPTION(TFLITE_ENABLE_XNNPACK OFF)
# There is no riscv asm impls in xnnpack.
OPTION(XNNPACK_ENABLE_ASSEMBLY OFF)
# There is no riscv specific timer in googlebenchmark project.
OPTION(XNNPACK_BUILD_BENCHMARKS OFF)
