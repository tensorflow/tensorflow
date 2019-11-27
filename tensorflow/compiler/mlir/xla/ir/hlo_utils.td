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

// This is the utils file for the HLO dialect.

#ifndef HLO_UTILS
#define HLO_UTILS

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def NullArrayAttr : NativeCodeCall<"ArrayAttr()">;

def CastIntElementsAttr : NativeCodeCall<"$0.cast<DenseIntElementsAttr>()">;

class ConstantSplat<string value> : NativeCodeCall<
    "getSplat(&$_builder, $0, " # value # ")">;

def NullDenseIntElementsAttr : NativeCodeCall<"DenseIntElementsAttr()">;

def BinBroadcastDimensions : NativeCodeCall<
    "getBroadcastDimensionsAttr(&$_builder, $0, $1)">;

#endif // HLO_UTILS
