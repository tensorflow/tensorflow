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

// This is the operation interface definition file for TensorFlow Lite.

#ifndef TFL_OP_INTERFACES
#define TFL_OP_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TFL op interface for stateful operands.

def TFL_StatefulOp : OpInterface<"StatefulOpInterface"> {
  let description = [{
    Interface for ops that are stateful and need to identify stateful operands.

    Stateful operands correspond to TF's variables semantics. An op that has 1
    or more stateful operands is a stateful op.
  }];

  let methods = [
    InterfaceMethod<
      [{Returns the indices of stateful operands.}],
      "std::vector<int>", "GetStatefulOperands", (ins)
    >,
  ];
}

//===----------------------------------------------------------------------===//
// TFL op interface for output channel index.

def TFL_ChannelDimIndexInterface : OpInterface<"ChannelDimIndexInterface"> {
  let description = [{
    Interface for defining the index of out channel index.
  }];

  let methods = [
    InterfaceMethod<
      [{Returns the dimension index of the output channels.}],
      "int", "GetChannelDimIndex", (ins)
    >,
  ];
}

#endif // TFL_OP_INTERFACES
