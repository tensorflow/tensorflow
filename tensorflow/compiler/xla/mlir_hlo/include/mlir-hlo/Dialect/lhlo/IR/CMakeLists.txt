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
# Need a separate function because of the .cc vs .cpp used in the one provided by MLIR
function(add_mlir_hlo_dialect_separate_files dialect)
  set(LLVM_TARGET_DEFINITIONS ${dialect}.td)
  mlir_tablegen(${dialect}.h.inc -gen-op-decls)
  mlir_tablegen(${dialect}.cc.inc -gen-op-defs)
  set(LLVM_TARGET_DEFINITIONS ${dialect}_structs.td)
  mlir_tablegen(${dialect}_structs.h.inc -gen-attrdef-decls)
  mlir_tablegen(${dialect}_structs.cc.inc -gen-attrdef-defs)
  add_public_tablegen_target(MLIR${dialect}IncGen)
  add_dependencies(mlir-headers MLIR${dialect}IncGen)
endfunction()

add_mlir_hlo_dialect_separate_files(lhlo_ops)

add_mlir_interface(lhlo_structured_interface)
