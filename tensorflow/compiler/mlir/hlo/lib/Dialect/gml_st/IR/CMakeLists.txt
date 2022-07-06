# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

include_directories(BEFORE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})

add_mlir_dialect_library(GmlStDialect
  gml_st_ops.cc

  DEPENDS
  MLIRgml_st_opsIncGen
  MLIRGmlStComposeSetInterfaceIncGen
  MLIRGmlStFusionInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRControlFlowInterfaces
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRLoopLikeInterface
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTensorDialect
  MLIRViewLikeInterface
)
