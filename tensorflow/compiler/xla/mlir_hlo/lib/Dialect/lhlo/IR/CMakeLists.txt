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
include_directories(BEFORE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})

add_mlir_library(LmhloStructuredInterface
  lhlo_structured_interface.cc

  LINK_LIBS PUBLIC
  MLIRIR

  DEPENDS
  MLIRlhlo_structured_interfaceIncGen
)

add_mlir_dialect_library(LmhloDialect
  lhlo_ops.cc

  DEPENDS
  MLIRlhlo_opsIncGen

  LINK_LIBS PUBLIC
  HloOpsCommon
  LmhloStructuredInterface
  MhloDialect
  MLIRIR
)

