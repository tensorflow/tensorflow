/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.

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

#ifndef STABLEHLO_DIALECT_CHLO_ENUMS
#define STABLEHLO_DIALECT_CHLO_ENUMS

include "mlir/IR/EnumAttr.td"
include "mlir/IR/PatternBase.td"

//===----------------------------------------------------------------------===//
// Comparison op definitions.
//===----------------------------------------------------------------------===//

// These mirror the XLA ComparisonDirection enum.
def CHLO_COMPARISON_DIRECTION_EQ : I32EnumAttrCase<"EQ", 0>;
def CHLO_COMPARISON_DIRECTION_NE : I32EnumAttrCase<"NE", 1>;
def CHLO_COMPARISON_DIRECTION_GE : I32EnumAttrCase<"GE", 2>;
def CHLO_COMPARISON_DIRECTION_GT : I32EnumAttrCase<"GT", 3>;
def CHLO_COMPARISON_DIRECTION_LE : I32EnumAttrCase<"LE", 4>;
def CHLO_COMPARISON_DIRECTION_LT : I32EnumAttrCase<"LT", 5>;

def CHLO_ComparisonDirection : I32EnumAttr<"ComparisonDirection",
    "Which comparison operation to perform.",
    [
      CHLO_COMPARISON_DIRECTION_EQ,
      CHLO_COMPARISON_DIRECTION_NE,
      CHLO_COMPARISON_DIRECTION_GE,
      CHLO_COMPARISON_DIRECTION_GT,
      CHLO_COMPARISON_DIRECTION_LE,
      CHLO_COMPARISON_DIRECTION_LT
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::chlo";
}

def CHLO_ComparisonDirectionAttr : EnumAttr<CHLO_Dialect, CHLO_ComparisonDirection, "comparison_direction">;

def CHLO_DEFAULT_COMPARISON_TYPE : NativeCodeCall<"::mlir::chlo::ComparisonTypeAttr()">;
def CHLO_COMPARISON_TYPE_NOTYPE : I32EnumAttrCase<"NOTYPE", 0>;
def CHLO_COMPARISON_TYPE_FLOAT : I32EnumAttrCase<"FLOAT", 1>;
def CHLO_COMPARISON_TYPE_FLOAT_TOTAL_ORDER : I32EnumAttrCase<"TOTALORDER", 2>;
def CHLO_COMPARISON_TYPE_SIGNED : I32EnumAttrCase<"SIGNED", 3>;
def CHLO_COMPARISON_TYPE_UNSIGNED : I32EnumAttrCase<"UNSIGNED", 4>;

def CHLO_ComparisonType : I32EnumAttr<"ComparisonType",
    "Which comparison type to use.",
    [
      CHLO_COMPARISON_TYPE_NOTYPE,
      CHLO_COMPARISON_TYPE_FLOAT,
      CHLO_COMPARISON_TYPE_FLOAT_TOTAL_ORDER,
      CHLO_COMPARISON_TYPE_SIGNED,
      CHLO_COMPARISON_TYPE_UNSIGNED
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::chlo";
}

def CHLO_ComparisonTypeAttr : EnumAttr<CHLO_Dialect, CHLO_ComparisonType, "comparison_type">;

#endif // STABLEHLO_DIALECT_CHLO_ENUMS
