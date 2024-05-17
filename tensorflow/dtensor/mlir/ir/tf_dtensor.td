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

def DTensor_MeshAttr : DTensor_DTensorAttr<"Mesh", "mesh"> {
  let returnType = "mlir::dtensor::MeshAttr::Mesh";
  let convertFromStorage = "$_self.cast<mlir::dtensor::MeshAttr>().getValue()";
}

//===----------------------------------------------------------------------===//
// DTensor op definitions
//===----------------------------------------------------------------------===//

def Tf_DTensorSend : TF_Op<"DTensorSend", []> {
  let summary = "Sends data to different mesh.";

  let arguments = (ins
    TF_Tensor:$input,
    StrAttr:$key,
    DTensor_MeshAttr:$target_mesh
  );

  let results = (outs);

  TF_DerivedOperandTypeAttr Tinput = TF_DerivedOperandTypeAttr<0>;
}

def Tf_DTensorRecv : TF_Op<"DTensorRecv", []> {
  let summary = "Receives data from different mesh.";

  let arguments = (ins
    StrAttr:$key,
    TF_ShapeAttr:$shape,
    DTensor_MeshAttr:$mesh
 );
 let results = (outs
    TF_Tensor:$output
 );

 TF_DerivedResultTypeListAttr Toutputs = TF_DerivedResultTypeListAttr<0>;
}

def TF_DTensorLayout: TF_Op<"DTensorLayout", [DeclareOpInterfaceMethods<InferTypeOpInterface>, Pure, TF_AllTypesMatch<["input", "output"]>, TF_MustExecute, TF_NoConstantFold]> {
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

def TF_RelayoutOp : TF_Op<"Relayout", [Pure, TF_AllTypesMatch<["input", "output"]>, TF_NoConstantFold]> {
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

def TF_RelayoutLikeOp : TF_Op<"RelayoutLike", [Pure, TF_AllTypesMatch<["input", "output"]>, TF_NoConstantFold]> {
  let summary = "Change layout of the gradients to match the layout of the forward pass Relayout's input.";

  let arguments = (ins
    TF_Tensor:$input,

    TF_Tensor:$layout_input
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
  TF_DerivedOperandTypeAttr U = TF_DerivedOperandTypeAttr<1>;
}

def TF_CopyToMeshOp : TF_Op<"CopyToMesh", [Pure]> {
  let summary = "";

  let arguments = (ins
    TF_Tensor:$input,
    StrAttr:$mesh
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_CopyToMeshGradOp : TF_Op<"CopyToMeshGrad", [Pure]> {
  let summary = "";

  let arguments = (ins
    TF_Tensor:$input,
    TF_Tensor:$forward_input
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_DTensorAllReduceOp : TF_Op<"DTensorAllReduce", [Pure]> {
  let summary = "";

  let arguments = (ins
    TF_Tensor:$input,
    TF_Int32Tensor:$group_assignment,

    TF_AnyStrAttrOf<["Min", "Max", "Mul", "Add", "Mean", "Any", "All"]>:$reduce_op,
    StrAttr:$device_type
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_DTensorReduceScatterOp : TF_Op<"DTensorReduceScatter", [Pure]> {
  let summary = "";

  let arguments = (ins
    TF_Tensor:$input,
    TF_Int32Tensor:$group_assignment,
    TF_Int32Tensor:$scatter_dimension,

    TF_AnyStrAttrOf<["Min", "Max", "Mul", "Add", "Mean", "Any", "All"]>:$reduce_op,
    StrAttr:$device_type
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

def TF_DTensorShardedPrefixOp : TF_Op<"DTensorShardedPrefix", [Pure]> {
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

def TF_DTensorRestoreV2Op : TF_Op<"DTensorRestoreV2", [Pure]> {
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
def TF_DTensorAllScatterOp : TF_Op<"DTensorAllScatter", [Pure]> {
  let summary = "Slices the input to the given layout.";

  let description = [{
This op takes both an input and an output layout. The output layout must be more
sharded than the input layout.
  }];

  let arguments = (ins
    TF_Tensor:$input,
    DTensor_LayoutAttr:$input_layout,
    DTensor_LayoutAttr:$output_layout
  );

  let results = (outs
    TF_Tensor:$output
    );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;
}

def TF_DTensorAllGatherOp : TF_Op<"DTensorAllGather", [Pure]> {
  let summary = "Concatenates the input to match the given layout.";

  let description = [{
This op takes both an input and an output layout. The output layout must be less
sharded than the input layout.
  }];

  let arguments = (ins
    TF_Tensor:$input,
    DTensor_LayoutAttr:$input_layout,
    DTensor_LayoutAttr:$output_layout
  );

  let results = (outs
    TF_Tensor:$output
    );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;
}

def TF_DTensorAllToAllOp : TF_Op<"DTensorAllToAll", [Pure]> {
  let summary = "Mutually exchanges the input to match the given layout.";

  let description = [{"
This op takes both an input and an output layout. There can be one mesh
dimension which is becoming unsharded in one axis while becoming sharded 
in another axis."
  }];

  let arguments = (ins
    TF_Tensor:$input,
    DTensor_LayoutAttr:$input_layout,
    DTensor_LayoutAttr:$output_layout
  );

  let results = (outs
    TF_Tensor:$output
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;

  let hasVerifier = 1;
}

#endif // TENSORFLOW_DTENSOR_MLIR_IR_TF_DTENSOR_OPS
