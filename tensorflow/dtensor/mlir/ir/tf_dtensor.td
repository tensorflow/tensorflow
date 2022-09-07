/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_DTENSOR_MLIR_IR_TF_DTENSOR_OPS
#define TENSORFLOW_DTENSOR_MLIR_IR_TF_DTENSOR_OPS

// Definitions for additional DTensor operations to add to the TF dialect.
// To update
// * Add //third_party/tensorflow/dtensor/cc:dtensor_ops to the internal dialect_generator_lib target.
// * Run `patch_tf_dialect.sh OpA,OpB`
// * Copy resulting output into this file.

include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// DTensor attribute definitions
//===----------------------------------------------------------------------===//

class DTensor_DTensorAttr <string name, string description> :
    Attr<CPred<"$_self.isa<mlir::dtensor::" # name # "Attr>()">,
         "DTensor " # description # " attribute">;

def DTensor_LayoutAttr : DTensor_DTensorAttr<"Layout", "layout"> {
  let returnType = "mlir::dtensor::LayoutAttr::Layout";
  let convertFromStorage = "$_self.cast<mlir::dtensor::LayoutAttr>().getValue()";
}

//===----------------------------------------------------------------------===//
// DTensor op definitions
//===----------------------------------------------------------------------===//

def Tf_DTensorSend : TF_Op<"DTensorSend", []> {
  let summary = "Sends data to different mesh.";

  let arguments = (ins
    TF_Tensor:$input,
    StrAttr:$key,
    DTensor_LayoutAttr:$target_layout
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr Tinput = TF_DerivedOperandTypeAttr<0>;
}

def Tf_DTensorRecv : TF_Op<"DTensorRecv", []> {
  let summary = "Receives data from different mesh.";

  let arguments = (ins
    StrAttr:$key,
    TF_ShapeAttr:$shape,
    DTensor_LayoutAttr:$layout
 );
 let results = (outs
    TF_Tensor:$output
 );

 TF_DerivedResultTypeListAttr Toutputs = TF_DerivedResultTypeListAttr<0>;
}

def TF_DTensorLayout: TF_Op<"DTensorLayout", [DeclareOpInterfaceMethods<InferTypeOpInterface>, NoSideEffect, TF_AllTypesMatch<["input", "output"]>, TF_NoConstantFold]> {
  let summary = [{
    Represents computational layout of an input tensor.
    }];

  let arguments = (ins
    TF_Tensor:$input,
    DTensor_LayoutAttr:$layout,
    TF_ShapeAttr:$global_shape
  );

  let results = (outs
    TF_Tensor:$output
  );
  let hasVerifier = 1;
}

def TF_DTensorEmbeddingEnqueueOp : TF_Op<"DTensorEmbeddingEnqueue", [SameVariadicOperandSize]> {
  let summary = "";

  let arguments = (ins
    Variadic<TF_Int32Tensor>:$embedding_indices,
    Variadic<TF_Float32Tensor>:$aggregation_weights,

    DefaultValuedOptionalAttr<StrAttr, "\"training\"">:$mode_override,
    DefaultValuedOptionalAttr<StrArrayAttr, "{}">:$combiners,
    StrAttr:$config
  );

  let results = (outs);

  TF_DerivedOperandSizeAttr num_inputs = TF_DerivedOperandSizeAttr<0>;
}

def TF_DTensorEmbeddingDequeueOp : TF_Op<"DTensorEmbeddingDequeue", [NoSideEffect]> {
  let summary = "DTensor TPU Embedding activation dequeue virtual op";

  let arguments = (ins
    StrAttr:$config,
    StrArrayAttr:$output_layouts
  );

  let results = (outs
    Variadic<TF_Float32Tensor>:$outputs
  );

  TF_DerivedResultSizeAttr num_outputs = TF_DerivedResultSizeAttr<0>;
}

def TF_RelayoutOp : TF_Op<"Relayout", [NoSideEffect, TF_AllTypesMatch<["input", "output"]>, TF_NoConstantFold]> {
  let summary = "Change layout of input to target layout inside the same mesh cluster.";

  let arguments = (ins
    TF_Tensor:$input,

    StrAttr:$layout
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_CopyToMeshOp : TF_Op<"CopyToMesh", [NoSideEffect]> {
  let summary = "";

  let arguments = (ins
    TF_Tensor:$input,

    StrAttr:$layout,

    StrAttr:$source_layout
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_SetEmbeddingConfigOp : TF_Op<"SetEmbeddingConfig", []> {
  let summary = "";

  let arguments = (ins
    TF_StrTensor:$config
  );

  let results = (outs);
}

def TF_GetEmbeddingConfigurationOp: TF_Op<"GetEmbeddingConfiguration", [AttrSizedOperandSegments, NoSideEffect]> {
  let summary = [{
    tf.GetEmbeddingConfiguration gathers information from the inputs and
    outputs a scalar string with the configuration to use when calling
    `initialize_system_for_tpu_embedding`.
    }];

  let arguments = (ins
    TF_Float32Tensor:$table,
    Variadic<TF_Float32Tensor>:$slots,
    TF_Int32Tensor:$ids,
    Variadic<TF_Float32Tensor>:$scalars,

    FlatSymbolRefAttr:$optimizer,
    DenseI32ArrayAttr:$operand_segment_sizes
  );

  let results = (outs
    TF_StrTensor:$config
  );

  TF_DerivedOperandTypeListAttr num_slots = TF_DerivedOperandTypeListAttr<1>;
  TF_DerivedOperandTypeListAttr num_scalars = TF_DerivedOperandTypeListAttr<3>;
}

// Op has no outputs, so we do not inlude the NosideEffect tag so that it will
// not get pruned.
def TF_LoadEmbeddingTableOp: TF_Op<"LoadEmbeddingTable", []> {
  let summary = [{
    tf.LoadEmbeddingTable loads an embedding table onto the tpu. Must be called
    after `initialize_system_for_tpu_embedding` and before any data is enqueued
    for embedding lookup.
    }];

  let arguments = (ins
    TF_Float32Tensor:$table,
    Variadic<TF_Float32Tensor>:$slots,

    StrAttr:$config
  );

  TF_DerivedOperandTypeListAttr num_slots = TF_DerivedOperandTypeListAttr<1>;
}

// This TF op is not public exported, add it here temporarily so we can easily
// construct one.
// Op has no outputs, so we do not inlude the NosideEffect tag so that it will
// not get pruned.
def TF_LoadAllTPUEmbeddingParametersOp : TF_Op<"LoadAllTPUEmbeddingParameters", [SameVariadicOperandSize]> {
  let summary = [{
An op that loads optimization parameters into embedding memory. Must be
  }];

  let description = [{
preceded by a ConfigureTPUEmbeddingHost op that sets up the correct embedding table
configuration. For example, this op is used to install parameters that are
loaded from a checkpoint before a training loop is executed.  For Adagrad,
auxiliary1 should be the accumulators. For SGD, all of the auxiliary* values
should be empty. For FTRL, auxiliary1 should be the accumulators and auxiliary2
should be the linear terms. For ADAM, auxiliary1 should be the momenta and
auxiliary2 should be the velocities.
  }];

  let arguments = (ins
    Arg<Variadic<TF_Float32Tensor>, [{A list of tensors, one for each embedding table,
containing the initial embedding table parameters to use in embedding
lookups.}]>:$parameters,
    Arg<Variadic<TF_Float32Tensor>, [{A list of tensors, one for each embedding table, containing the
initial values of the first auxiliary optimization parameter to use in embedding
training loop updates. The shape of each entry is ignored (and thus can be
empty) for those tables whose optimization algorithms do not have at least one
auxiliary parameter.}]>:$auxiliary1,
    Arg<Variadic<TF_Float32Tensor>, [{A list of tensors, one for each embedding table, containing the
initial values of the second auxiliary optimization parameter to use in
embedding training loop updates. The shape of each entry is ignored (and thus
can be empty) for those tables whose optimization algorithms do not have at
least two auxiliary}]>:$auxiliary2,
    Arg<Variadic<TF_Float32Tensor>, [{A list of tensors, one for each embedding table, containing the
initial values of the third auxiliary optimization parameter to use in embedding
training loop updates. The shape of each entry is ignored (and thus can be
empty) for those tables whose optimization algorithms do not have three
auxiliary parameters.}]>:$auxiliary3,
    Arg<Variadic<TF_Float32Tensor>, [{A list of tensors, one for each embedding table, containing the
initial values of the third auxiliary optimization parameter to use in embedding
training loop updates. The shape of each entry is ignored (and thus can be
empty) for those tables whose optimization algorithms do not have four
auxiliary parameters.}]>:$auxiliary4,
    Arg<Variadic<TF_Float32Tensor>, [{A list of tensors, one for each embedding table, containing the
initial values of the third auxiliary optimization parameter to use in embedding
training loop updates. The shape of each entry is ignored (and thus can be
empty) for those tables whose optimization algorithms do not have five
auxiliary parameters.}]>:$auxiliary5,
    Arg<Variadic<TF_Float32Tensor>, [{A list of tensors, one for each embedding table, containing the
initial values of the third auxiliary optimization parameter to use in embedding
training loop updates. The shape of each entry is ignored (and thus can be
empty) for those tables whose optimization algorithms do not have six
auxiliary parameters.}]>:$auxiliary6,
    Arg<Variadic<TF_Float32Tensor>, [{A list of tensors, one for each embedding table, containing the
initial values of the third auxiliary optimization parameter to use in embedding
training loop updates. The shape of each entry is ignored (and thus can be
empty) for those tables whose optimization algorithms do not have seven
auxiliary parameters.}]>:$auxiliary7,

    StrAttr:$config,
    I64Attr:$num_shards,
    I64Attr:$shard_id
  );

  let results = (outs);

  TF_DerivedOperandSizeAttr NumTables = TF_DerivedOperandSizeAttr<0>;
}

def TF_DTensorLoadTPUEmbeddingParametersOp : TF_Op<"DTensorLoadTPUEmbeddingParameters", [SameVariadicOperandSize]> {
  let summary = "";

  let arguments = (ins
    Variadic<TF_Float32Tensor>:$parameters,
    Variadic<TF_Float32Tensor>:$auxiliary1,
    Variadic<TF_Float32Tensor>:$auxiliary2,
    Variadic<TF_Float32Tensor>:$auxiliary3,
    Variadic<TF_Float32Tensor>:$auxiliary4,
    Variadic<TF_Float32Tensor>:$auxiliary5,
    Variadic<TF_Float32Tensor>:$auxiliary6,
    Variadic<TF_Float32Tensor>:$auxiliary7,

    StrAttr:$config,
    I64Attr:$num_shards,
    I64Attr:$shard_id
  );

  let results = (outs);

  TF_DerivedOperandSizeAttr num_tables = TF_DerivedOperandSizeAttr<0>;
}

def TF_RetrieveAllTPUEmbeddingParametersOp : TF_Op<"RetrieveAllTPUEmbeddingParameters", [SameVariadicResultSize]> {
  let summary = [{
An op that retrieves optimization parameters from embedding to host memory.
  }];

  let description = [{
Must be preceded by a ConfigureTPUEmbeddingHost op that sets up the correct
embedding table configuration. For example, this op is used to retrieve updated
parameters before saving a checkpoint.  For Adagrad, auxiliary1 will contain the
accumulators after running this op. For SGD, all of the auxiliary* values will
be empty (0x0 tensors for that table). For FTRL, auxiliary1 will contain the
accumulators and auxiliary2 will contain the linear terms. For ADAM, auxiliary1
will contain the momenta and auxiliary2 will contain the velocities.
  }];

  let arguments = (ins
    StrAttr:$config,
    I64Attr:$num_shards,
    I64Attr:$shard_id
  );

  let results = (outs
    Res<Variadic<TF_Float32Tensor>, [{A list of tensors, one for each embedding table, containing the
stored embedding table parameters.}]>:$parameters,
    Res<Variadic<TF_Float32Tensor>, [{A list of tensors, one for each embedding table, containing the
first auxiliary optimization parameter stored. Elements are
present in the list, but have zero size, for unused optimization parameters
(based on the algorithm in use for each table).}]>:$auxiliary1,
    Res<Variadic<TF_Float32Tensor>, [{A list of tensors, one for each embedding table, containing the
second auxiliary optimization parameter stored. Elements are
present in the list, but have zero size, for unused optimization parameters
(based on the algorithm in use for each table).}]>:$auxiliary2,
    Res<Variadic<TF_Float32Tensor>, [{A list of tensors, one for each embedding table, containing the
third auxiliary optimization parameter stored. Elements are
present in the list, but have zero size, for unused optimization parameters
(based on the algorithm in use for each table).}]>:$auxiliary3,
    Res<Variadic<TF_Float32Tensor>, [{A list of tensors, one for each embedding table, containing the
third auxiliary optimization parameter stored. Elements are
present in the list, but have zero size, for unused optimization parameters
(based on the algorithm in use for each table).}]>:$auxiliary4,
    Res<Variadic<TF_Float32Tensor>, [{A list of tensors, one for each embedding table, containing the
third auxiliary optimization parameter stored. Elements are
present in the list, but have zero size, for unused optimization parameters
(based on the algorithm in use for each table).}]>:$auxiliary5,
    Res<Variadic<TF_Float32Tensor>, [{A list of tensors, one for each embedding table, containing the
third auxiliary optimization parameter stored. Elements are
present in the list, but have zero size, for unused optimization parameters
(based on the algorithm in use for each table).}]>:$auxiliary6,
    Res<Variadic<TF_Float32Tensor>, [{A list of tensors, one for each embedding table, containing the
third auxiliary optimization parameter stored. Elements are
present in the list, but have zero size, for unused optimization parameters
(based on the algorithm in use for each table).}]>:$auxiliary7
  );

  TF_DerivedResultSizeAttr NumTables = TF_DerivedResultSizeAttr<0>;
}

def TF_DTensorRetrieveTPUEmbeddingParametersOp : TF_Op<"DTensorRetrieveTPUEmbeddingParameters", [NoSideEffect, SameVariadicResultSize]> {
  let summary = "";

  let arguments = (ins
    DefaultValuedOptionalAttr<I64Attr, "1">:$num_hosts,
    StrAttr:$config,
    I64Attr:$num_shards,
    I64Attr:$shard_id
  );

  let results = (outs
    Variadic<TF_Float32Tensor>:$parameters,
    Variadic<TF_Float32Tensor>:$auxiliary1,
    Variadic<TF_Float32Tensor>:$auxiliary2,
    Variadic<TF_Float32Tensor>:$auxiliary3,
    Variadic<TF_Float32Tensor>:$auxiliary4,
    Variadic<TF_Float32Tensor>:$auxiliary5,
    Variadic<TF_Float32Tensor>:$auxiliary6,
    Variadic<TF_Float32Tensor>:$auxiliary7
  );

  TF_DerivedResultSizeAttr num_tables = TF_DerivedResultSizeAttr<0>;
}

def TF_DTensorSendEmbeddingGradientsOp : TF_Op<"DTensorSendEmbeddingGradients", [AttrSizedOperandSegments]> {
  let summary = "";

  let arguments = (ins
    Variadic<TF_Float32Tensor>:$gradients,
    Variadic<TF_Float32Tensor>:$learning_rates,

    StrAttr:$config
  );

  let results = (outs);

  TF_DerivedOperandSizeAttr num_lr_tags = TF_DerivedOperandSizeAttr<1>;
  TF_DerivedOperandSizeAttr num_tables = TF_DerivedOperandSizeAttr<0>;
}

// Op has no outputs, so we do not inlude the NosideEffect tag so that it will
// not get pruned.
def TF_ApplyEmbeddingOptimizerOp : TF_Op<"ApplyEmbeddingOptimizer", []> {
  let summary = "";

  let arguments = (ins
    TF_Float32Tensor:$gradient,
    Variadic<TF_Float32Tensor>:$scalars,

    StrAttr:$config
  );

  let results = (outs);

  TF_DerivedOperandSizeAttr num_scalars = TF_DerivedOperandSizeAttr<1>;
}


def TF_ApplyEmbeddingOptimizerV2Op : TF_Op<"ApplyEmbeddingOptimizerV2", []> {
  let summary = "";

  let arguments = (ins
    TF_Float32Tensor:$gradient,
    TF_Float32Tensor:$variable,
    Variadic<TF_Float32Tensor>:$slot_variables,

    StrAttr:$optimization_parameters
  );

  let results = (outs);

  TF_DerivedOperandSizeAttr num_slots = TF_DerivedOperandSizeAttr<2>;
}

def TF_EmbeddingDequeueOp : TF_Op<"EmbeddingDequeue", [NoSideEffect]> {
  let summary = "";

  let arguments = (ins
    StrAttr:$config
  );

  let results = (outs
    TF_Float32Tensor:$result
  );
}

// Op has no outputs, so we do not inlude the NosideEffect tag so that it will
// not get pruned.
def TF_EmbeddingEnqueueOp : TF_Op<"EmbeddingEnqueue"> {
  let summary = "";

  let arguments = (ins
    TF_Int32Tensor:$ids,
    TF_StrTensor:$mode,

    StrAttr:$config
  );

  let results = (outs);
}

def TF_IsTPUEmbeddingInitializedOp : TF_Op<"IsTPUEmbeddingInitialized", []> {
  let summary = [{
Whether TPU Embedding is initialized in a distributed TPU system.
  }];

  let arguments = (ins
    DefaultValuedOptionalAttr<StrAttr, "\"\"">:$config
  );

  let results = (outs
    TF_BoolTensor:$is_tpu_embedding_initialized
  );
}

def TF_TPUDenseEmbeddingLookUpOp : TF_Op<"TPUDenseEmbeddingLookUp", [NoSideEffect]> {
  let summary = "";

  let arguments = (ins
    TF_Int32Tensor:$ids,
    TF_Float32Tensor:$table
  );

  let results = (outs
    TF_Float32Tensor:$output
  );
}

def TF_TPUDenseEmbeddingLookUpGradOp : TF_Op<"TPUDenseEmbeddingLookUpGrad", [NoSideEffect]> {
  let summary = "";

  let arguments = (ins
    TF_Float32Tensor:$gradients,
    TF_Int32Tensor:$ids,
    TF_Float32Tensor:$table
  );

  let results = (outs
    TF_Float32Tensor:$output
  );
}

def TF_DTensorAllReduceOp : TF_Op<"DTensorAllReduce", [NoSideEffect]> {
  let summary = "";

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Bool, TF_Float16, TF_Float32, TF_Int32, TF_Int64, TF_Uint32]>:$input,
    TF_Int32Tensor:$group_assignment,

    TF_AnyStrAttrOf<["Min", "Max", "Mul", "Add", "Mean", "Any", "All"]>:$reduce_op,
    StrAttr:$device_type
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Bool, TF_Float16, TF_Float32, TF_Int32, TF_Int64, TF_Uint32]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_DTensorReduceScatterOp : TF_Op<"DTensorReduceScatter", [NoSideEffect]> {
  let summary = "";

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Bool, TF_Float16, TF_Float32, TF_Int32, TF_Int64, TF_Uint32]>:$input,
    TF_Int32Tensor:$group_assignment,
    TF_Int32Tensor:$scatter_dimension,

    TF_AnyStrAttrOf<["Min", "Max", "Mul", "Add", "Mean", "Any", "All"]>:$reduce_op,
    StrAttr:$device_type
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Bool, TF_Float16, TF_Float32, TF_Int32, TF_Int64, TF_Uint32]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_DTensorShardedPrefixOp : TF_Op<"DTensorShardedPrefix", [NoSideEffect]> {
  let summary = "Queries the generated shard prefix from DTensor SaveV2.";

  let description = [{
DTensor SPMD will generates multiple SaveV2 ops from a single SaveV2 op. Each
generated SaveV2 op will have different prefix so that tensors can be saved
into multiple different files. This Op would query `all` the prefix genearetd.
Normally a MergeV2Checkpoints op would consume the output of this file so that
the checkpoint can cover all files.
  }];

  let arguments = (ins
    TF_StrTensor:$prefix,
    TF_StrTensor:$tensor_names,
    TF_StrTensor:$shape_and_slices,
    TF_StrTensor:$mesh,
    TF_StrTensor:$layouts,
    Variadic<TF_Tensor>:$tensors
  );

  let results = (outs
    TF_StrTensor:$generated_prefixes
  );
}

def TF_DTensorRestoreV2Op : TF_Op<"DTensorRestoreV2", [NoSideEffect]> {
  let summary = "DTensor RestoreV2 that takes extra shape and layouts.";

  let description = [{
DTensor RestoreV2 is needed so that we can restore tensors with layouts.
Ideally this can be done in tf.function style restore but it is currently
blocked by CopyToMesh. Note that this will be also used by name-based restore.
  }];

  let arguments = (ins
    TF_StrTensor:$prefix,
    TF_StrTensor:$tensor_names,
    TF_StrTensor:$shape_and_slices,
    TF_ShapeAttrArray:$input_shapes,
    StrArrayAttr:$input_layouts
  );

  let results = (outs
    Variadic<TF_Tensor>:$tensors
  );

  TF_DerivedResultTypeListAttr dtypes = TF_DerivedResultTypeListAttr<0>;
}

def TF_DTensorAllScatterOp : TF_Op<"DTensorAllScatter", [NoSideEffect]> {
  let summary = "Slices the input to the given layout.";

  let description = [{
This op takes both an input and an output layout. The output layout must be more
sharded than the input layout.
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Uint32, TF_Int64, TF_Bool]>:$input,
    DTensor_LayoutAttr:$input_layout,
    DTensor_LayoutAttr:$output_layout
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Float64, TF_Int32, TF_Uint32, TF_Int64, TF_Bool]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;
}

def TF_DTensorAllGatherOp : TF_Op<"DTensorAllGather", [NoSideEffect]> {
  let summary = "Concatenates the input to match the given layout.";

  let description = [{
This op takes both an input and an output layout. The output layout must be less
sharded than the input layout.
  }];

  let arguments = (ins
    TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Int32, TF_Uint32, TF_Int64, TF_Bool]>:$input,
    DTensor_LayoutAttr:$input_layout,
    DTensor_LayoutAttr:$output_layout
  );

  let results = (outs
    TensorOf<[TF_Bfloat16, TF_Float16, TF_Float32, TF_Int32, TF_Uint32, TF_Int64, TF_Bool]>:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;
}

#endif // TENSORFLOW_DTENSOR_MLIR_IR_TF_DTENSOR_OPS
