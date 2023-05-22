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

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Func/IR/FuncOps.td"

// The `returns` number indicates the number of results the function returns.
class LiftAsFunctionCall<string func_name, int returns = 1> :
  NativeCodeCall<"LiftAsFunctionCall($_builder, $_loc, "
                    "\""# func_name #"\", $0...)", returns>;

// Add the second argument to the first argument, which is expected to be an
// argument list.
// bias(einsum(inputs), bias) --> einsum_with_bias(AppendToVector(inputs, bias))
// Since inputs is a vector in case of einsum, we cannot use ArgumentList here.
def AppendToVector : NativeCodeCall<"AppendToVector($0, $1)">;

// The list of arguments of the composite function.
def ArgumentList : NativeCodeCall<"llvm::SmallVector<Value>{$0...}">;

// The list of results of the composite function.
def ResultList : NativeCodeCall<"llvm::SmallVector<Value>{$0...}">;

// Creates a list of NamedAttributes. An example usage would be:
// (NamedAttributeList (NamedAttr<"transpose_a"> $transpose_a))
def NamedAttributeList : NativeCodeCall<"llvm::SmallVector<NamedAttribute>{$0...}">;

// Creates a NamedAttribute given its name and value. Essentially creates
// a pair: {attribute_name, attribute_value}.
class NamedAttr<string attr_name> :
    NativeCodeCall<"NamedAttribute{$_builder.getStringAttr(\"" # attr_name # "\"), $0}">;

// Checks if the value is not defined inside a lifted function by checking the
// `tf_quant.composite_function` attribute.
def IsNotInLiftedFunc :
      Constraint<CPred<"!IsInLiftedFunc($0.getDefiningOp())">>;

// Checks if the given einsum op is supported for XlaDotV2 quantization.
def IsEinsumSupportedByXlaDotV2 :
      Constraint<CPred<"IsEinsumSupportedByXlaDotV2($0)">>;

// This attribute can be used in the `AttributeList` for missing attributes. It
// is necessary to keep other attributes in the same index as the quantized
// composite function.
def NullAttribute : NativeCodeCall<"$_builder.getStringAttr(\"N/A\")">;
