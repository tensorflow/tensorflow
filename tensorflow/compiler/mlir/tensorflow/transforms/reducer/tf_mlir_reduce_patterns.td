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
include "mlir/IR/OpBase.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

// Note that this may not be a valid transformation for general tensorflow
// program, but from the perspective of mlir-reduce, it's worth to give it a try
// because the validity will be checked seperately by the tester.
def : Pat<(TF_IdentityOp (TF_IdentityOp $arg)), (TF_IdentityOp $arg)>;
