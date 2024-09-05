/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "tensorflow/compiler/mlir/quantization/common/attrs_and_constraints.td"

// Combines the two variadic arguments ($in_tensors and $captured_tensors).
def GetBatchFunctionOpArgOperands:
    NativeCodeCall<"cast<TF::BatchFunctionOp>($0[0].getDefiningOp()).getArgOperands()">;

// Replaces `TF_BatchFunctionOp` into `TF_PartitionedCallOp` that calls the
// same $f. This may be required, for example, when inlining is desired,
// because `TF_BatchFunctionOp` doesn't have the `CallOpInterface` trait.
def ReplaceBatchFunctionOpToPartitionedCallOp : Pat<
  (TF_BatchFunctionOp:$src_op_res
      $_, $_, $f, $_, $_, $_, $_, $_, $_, $_, $_, $_, $_, $_, $_, $_, $_, $_),
  (TF_PartitionedCallOp
      (GetBatchFunctionOpArgOperands $src_op_res),
      $f,
      /*config=*/(CreateStringAttr<"">),
      /*config_proto=*/(CreateStringAttr<"">),
      /*executor_type=*/(CreateStringAttr<"">))>;
