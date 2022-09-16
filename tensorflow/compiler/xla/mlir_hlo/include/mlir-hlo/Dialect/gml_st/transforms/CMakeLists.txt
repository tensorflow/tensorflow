#
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
#

set(LLVM_TARGET_DEFINITIONS passes.td)
mlir_tablegen(passes.h.inc -gen-pass-decls -name GmlSt)
add_public_tablegen_target(MLIRGmlStPassIncGen)

set(LLVM_TARGET_DEFINITIONS test_passes.td)
mlir_tablegen(test_passes.h.inc -gen-pass-decls -name GmlStTest)
add_public_tablegen_target(MLIRGmlStTestPassIncGen)

set(LLVM_TARGET_DEFINITIONS tiling_interface.td)
mlir_tablegen(tiling_interface.h.inc -gen-op-interface-decls)
mlir_tablegen(tiling_interface.cc.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRGmlStTilingInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS fusion_interface.td)
mlir_tablegen(fusion_interface.h.inc -gen-op-interface-decls)
mlir_tablegen(fusion_interface.cc.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRGmlStFusionInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS compose_set_interface.td)
mlir_tablegen(compose_set_interface.h.inc -gen-op-interface-decls)
mlir_tablegen(compose_set_interface.cc.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRGmlStComposeSetInterfaceIncGen)
