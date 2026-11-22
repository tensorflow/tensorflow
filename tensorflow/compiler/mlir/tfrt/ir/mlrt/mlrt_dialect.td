/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifdef MLRT_DIALECT
#else
#define MLRT_DIALECT

include "mlir/IR/OpBase.td"

def Mlrt_Dialect : Dialect {
  let name = "mlrt";

  let description = [{
    The MLRT Dialect.
  }];

  let cppNamespace = "::mlrt::compiler";
}

def MlrtFutureType : DialectType<Mlrt_Dialect,
    CPred<"::llvm::isa<::mlrt::compiler::FutureType>($_self)">, "!mlrt.future type">,
    BuildableType<"$_builder.getType<::mlrt::compiler::FutureType>()"> {
  let description = [{
    `!mlrt.future type` represents a C++ mlrt::Future.
  }];
}

def MlrtPromiseType : DialectType<Mlrt_Dialect,
    CPred<"::llvm::isa<::mlrt::compiler::PromiseType>($_self)">, "!mlrt.promise type">,
    BuildableType<"$_builder.getType<::mlrt::compiler::PromiseType>()"> {
  let description = [{
    `!mlrt.promise type` represents a C++ mlrt::Promise.
  }];
}

def MlrtAsyncHandleType : DialectType<Mlrt_Dialect,
    CPred<"::llvm::isa<::mlrt::compiler::AsyncHandleType>($_self)">, "!mlrt.async_handle type">,
    BuildableType<"$_builder.getType<::mlrt::compiler::AsyncHandleType>()"> {
  let description = [{
    `!mlrt.async_handle type` represents a C++ mlrt::AsyncHandle.
  }];
}

#endif
