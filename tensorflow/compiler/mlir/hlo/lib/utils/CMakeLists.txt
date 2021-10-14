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

add_mlir_library(MLIRMhloUtils
  broadcast_utils.cc
  codegen_utils.cc
  convert_op_folder.cc
  cycle_detector.cc
  hlo_utils.cc

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRIR
  MLIRLLVMIR
  MLIRMemRef
  MLIRPass
  MLIRShape
  MLIRStandard
  MLIRSupport
  )
