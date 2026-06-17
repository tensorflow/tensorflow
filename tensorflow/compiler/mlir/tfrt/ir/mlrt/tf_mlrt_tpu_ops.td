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
#ifdef TF_MLRT_TPU_OPS
#else
#define TF_MLRT_TPU_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "tensorflow/compiler/mlir/tfrt/ir/mlrt/tf_mlrt_dialect.td"
include "tensorflow/compiler/mlir/tfrt/ir/mlrt/mlrt_dialect.td"

def TensorflowMlrtTpu_Dialect : Dialect {
  let name = "tf_mlrt_tpu";

  let description = [{
    The TF MLRT TPU Dialect.
  }];

  let cppNamespace = "::tensorflow::tf_mlrt_tpu";
}

class TensorflowMlrtTpu_Op<string mnemonic, list<Trait> traits = []> :
    Op<TensorflowMlrtTpu_Dialect, mnemonic, !listconcat(traits, [IsolatedFromAbove])> {
}

def GetTpuHostDeviceOp : TensorflowMlrtTpu_Op<"get_tpu_host_device", [Pure]> {
  let summary = "get the tpu host allocator that implements tensorflow::Device";

  let results = (outs
    TFDeviceType:$device
  );

  let assemblyFormat = "attr-dict";
}

def CompileAndExecuteOp : TensorflowMlrtTpu_Op<"compile_and_execute"> {
  let summary = "tpu compile and execute operation";
  let description = [{
    tf_mlrt_tpu.compile_and_execute compiles a mlir tpu program and executes the compiled tpu program.

    $mlir_module is a serialized MLIR module with a `main` function that contains target computation.
    $metadata is a serialized TPUCompileMetadataProto describing the shapes and types of the inputs to the computation, as well as a mapping onto the TPU pod topology.
    $constant_operand_indices are the indices of the inputs that are constant to the TPU program (e.g. weights in inference), the rest of the inputs are input tensors.
    constant_operand_indices is sorted in ascending order.
    $operands_with_static_shape are indices of operands that are tagged with a maximum static shape.
    $producer_name is a string describing the name of the framework that added support for running this portion of the model on TPUs.

    Example:
      %rendezvous_key_base, %result = tf_mlrt_tpu.compile_and_execute (%operands) constant_operand_indices = [1, 3] metadata = "metadata..." mlir_module = "mlir_module..."
  }];
  let arguments = (ins
    Variadic<TFTensorType>:$operands_and_static_shapes,
    DenseI32ArrayAttr:$constant_operand_indices,
    StrAttr:$metadata,
    StrAttr:$mlir_module,
    UI32Attr:$num_operands,
    DenseI32ArrayAttr:$operands_with_static_shape,
    StrAttr:$producer_name
  );

  let results = (outs
    TFTensorType:$rendezvous_key_base,
    Variadic<MlrtFutureType>:$results
  );

  let assemblyFormat = [{
    `(` $operands_and_static_shapes `)` attr-dict `:` functional-type($operands_and_static_shapes, results)
  }];
}

#endif
