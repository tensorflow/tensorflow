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

#ifdef TFRT_FALLBACK_OPS
#else
#define TFRT_FALLBACK_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"

def Fallback_Dialect : Dialect {
  let name = "tfrt_fallback";

  let description = [{
    The TFRT Fallback Dialect.
  }];

  let cppNamespace = "::tfrt::fallback";
}

// This corresponds to tensorflow::Tensor.
def TFTensorType : DialectType<Fallback_Dialect,
    CPred<"::llvm::isa<::tfrt::fallback::TFTensorType>($_self)">, "!tfrt_fallback.tf_tensor type">,
    BuildableType<"$_builder.getType<::tfrt::fallback::TFTensorType>()"> {
  let description = [{
    `!tfrt_fallback.tf_tensor type` represents a tensorflow::Tensor.
  }];
}

// This corresponds to tensorflow::Allocator.
def TFAllocatorType : DialectType<Fallback_Dialect,
    CPred<"::llvm::isa<::tfrt::fallback::TFAllocatorType>($_self)">, "!tfrt_fallback.tf_allocator type">,
    BuildableType<"$_builder.getType<::tfrt::fallback::TFAllocatorType>()"> {
  let description = [{
    `!tfrt_fallback.tf_tensor type` represents a tensorflow::Tensor.
  }];
}

#endif
