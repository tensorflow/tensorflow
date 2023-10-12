/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
include "mlir/IR/PatternBase.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"

// TODO(jpienaar): Move this somewhere general.
class GetI64ScalarElementsAttr<int value> :
  NativeCodeCall<"DenseElementsAttr::get<int64_t>(RankedTensorType::get({}, $_builder.getIntegerType(64)), " # value # ")">;

def FuseMapAndBatch : Pat<
  (TF_BatchDatasetV2Op
     (TF_MapDatasetOp $input_dataset, $other_arguments, $f, $output_types,
        $output_shapes, $use_inter_op_parallelism, $preserve_cardinality,
        $map_dataset_metadata),
     $batch_size, $drop_remainder, $parallel_copy, $batch_output_types,
     $batch_output_shapes, $unused_batch_dataset_metadata),
  (TF_MapAndBatchDatasetOp $input_dataset, $other_arguments, $batch_size,
     (TF_ConstOp (GetI64ScalarElementsAttr<1>)), $drop_remainder, $f,
        $batch_output_types, $batch_output_shapes, $preserve_cardinality,
        // TODO(kramm): Should we merge batch_dataset_metadata and
        //              map_dataset_metadata?
        $map_dataset_metadata)>;
