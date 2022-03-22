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

// This is a legalization pattern definition file for TF to XLA (Collective Ops).

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

def RewriteCollectiveAssignGroupV2CollectiveReduceV2 :
  Pattern<
    (TF_CollectiveReduceV2Op:$src_op
       $input, $group_size, (TF_CollectiveAssignGroupV2Op $group_assignment, $device_index),
       $instance_key, $ordering_token,
       $merge_op, $final_op, $communication_hint, $timeout_seconds,
       $max_subdivs_per_device
    ),
    [(TF_XlaAllReduceOp
       $input, $group_assignment, $merge_op,
       ConstantAttr<StrAttr, "\"CrossReplicaAndPartition\"">
    )
    ]
   >;

