/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// TFLite variables legalization patterns
include "mlir/IR/PatternBase.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_ops.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

def HasSupportedElementType :
  Constraint<CPred<"HasSupportedElementType($0.getDefiningOp())">>;

def IsSupportedElementType :
  Constraint<CPred<"IsSupportedElementType($0.getType().cast<ShapedType>())">>;

def LegalizeVarHandle : Pat<
  (TF_VarHandleOp:$result $container, $shared_name),
  (TFL_VarHandleOp $container, $shared_name),
  [(HasSupportedElementType $result)]>;

def LegalizeReadVariable : Pat<
  (TF_ReadVariableOp:$result $resource),
  (TFL_ReadVariableOp $resource),
  [(HasSupportedElementType $result)]>;

def LegalizeAssignVariable : Pat<
  (TF_AssignVariableOp $resource, $value, $validate_shape),
  (TFL_AssignVariableOp $resource, $value),
  [(IsSupportedElementType $value)]>;
