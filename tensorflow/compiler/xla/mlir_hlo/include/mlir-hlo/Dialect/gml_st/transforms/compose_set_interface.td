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

#ifndef COMPOSE_SET_INTERFACE
#define COMPOSE_SET_INTERFACE

include "mlir/IR/OpBase.td"

def ComposeSetInterface : OpInterface<"ComposeSetInterface"> {
  let description = [{
    This interface should be implemented by all set operations that can be
    composed with their superset operand.
  }];
  let cppNamespace = "::mlir::gml_st";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns a composition of this set with its superset operand.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"compose",
      /*args=*/(ins "OpBuilder&":$builder)>,
  ];
}

#endif // COMPOSE_SET_INTERFACE
