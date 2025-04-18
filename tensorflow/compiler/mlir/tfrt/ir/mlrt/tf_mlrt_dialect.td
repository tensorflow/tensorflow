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
#ifdef TF_MLRT_DIALECT
#else
#define TF_MLRT_DIALECT

include "mlir/Interfaces/SideEffectInterfaces.td"
include "tensorflow/compiler/mlir/tfrt/ir/mlrt/mlrt_dialect.td"

// TODO(chky,rohitju): Unify this dialect with tfrt_fallback_sync dialect after
// vrooml is using the new interpreter.
def TensorflowMlrt_Dialect : Dialect {
  let name = "tf_mlrt";

  let description = [{
    The TF MLRT Dialect.
  }];

  let cppNamespace = "::tensorflow::tf_mlrt";
}

class TensorflowMlrt_Op<string mnemonic, list<Trait> traits = []> :
    Op<TensorflowMlrt_Dialect, mnemonic, !listconcat(traits, [IsolatedFromAbove])> {
}

// This corresponds to tensorflow::Tensor.
def TFTensorType : DialectType<TensorflowMlrt_Dialect,
    CPred<"isa<::tensorflow::tf_mlrt::TFTensorType>($_self)">, "!tf_mlrt.tensor type">,
    BuildableType<"$_builder.getType<::tensorflow::tf_mlrt::TFTensorType>()"> {
  let description = [{
    `!tf_mlrt.tensor type` represents a tensorflow::Tensor.
  }];
}

// This corresponds to tensorflow::Device* .
def TFDeviceType : DialectType<TensorflowMlrt_Dialect,
    CPred<"isa<::tensorflow::tf_mlrt::TFDeviceType>($_self)">, "!tf_mlrt.device type">,
    BuildableType<"$_builder.getType<::tensorflow::tf_mlrt::TFDeviceType>()"> {
  let description = [{
    `!tf_mlrt.device type` represents a tensorflow::device.
  }];
}

#endif
