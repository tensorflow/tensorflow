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

// This is the operation enums definition file for TensorFlow Lite.

#ifndef TFL_OP_ENUMS
#define TFL_OP_ENUMS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "tensorflow/compiler/mlir/lite/ir/tfl_op_interfaces.td"

// A string attribute whose value are one of the values in `cases`.
// Referred TF_AnyStrAttrOf in tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td
class TFL_AnyStrAttrOf<list<string> cases> : StringBasedAttr<
  CPred<!foldl(
      "llvm::cast<StringAttr>($_self).getValue() == \"" # !head(cases) # "\"",
      !foreach(case, !tail(cases),
               "llvm::cast<StringAttr>($_self).getValue() == \"" # case # "\""),
      prev, cur, prev # " || " # cur)>,
  "string attribute whose value is " #
    !foldl(/*init*/!head(cases), /*list*/!tail(cases),
           prev, cur, prev # ", or " # cur)>;

// Attributes used for encoding sparse tensors.
// Please find detailed explanation of these parameters in the TFLite schema.
def TFL_DT_Dense : I32EnumAttrCase<"DENSE", 0>;
def TFL_DT_SparseCSR : I32EnumAttrCase<"SPARSE_CSR", 1>;

def TFL_DimensionType : I32EnumAttr<
    "DimensionType", "dimension_type", [TFL_DT_Dense, TFL_DT_SparseCSR]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::TFL";
}

def TFL_DimensionTypeAttr : EnumAttr<TFL_Dialect, TFL_DimensionType,
    "dimension_type_attr"> {
  let convertFromStorage = "$_self";
}

// Allowed activation function cases
// These should match the ActivationFunctionType enum in TFLite schema.
def TFL_AFEnum_None  : I32EnumAttrCase<"NONE", 0>;
def TFL_AFEnum_Relu  : I32EnumAttrCase<"RELU", 1>;
def TFL_AFEnum_Relu1 : I32EnumAttrCase<"RELU_N1_TO_1", 2>;
def TFL_AFEnum_Relu6 : I32EnumAttrCase<"RELU6", 3>;
def TFL_AFEnum_Tanh  : I32EnumAttrCase<"TANH", 4>;
def TFL_AFEnum_Sign  : I32EnumAttrCase<"SIGN_BIT", 5>;

def TFL_AFAttr : TFL_AnyStrAttrOf<[
      TFL_AFEnum_None.symbol,  TFL_AFEnum_Relu.symbol, TFL_AFEnum_Relu1.symbol,
      TFL_AFEnum_Relu6.symbol, TFL_AFEnum_Tanh.symbol, TFL_AFEnum_Sign.symbol
    ]>;

// Predefined constant attributes of activation function
def TFL_AF_None  : ConstantStrAttr<TFL_AFAttr, TFL_AFEnum_None.symbol>;
def TFL_AF_Relu  : ConstantStrAttr<TFL_AFAttr, TFL_AFEnum_Relu.symbol>;
def TFL_AF_Relu1 : ConstantStrAttr<TFL_AFAttr, TFL_AFEnum_Relu1.symbol>;
def TFL_AF_Relu6 : ConstantStrAttr<TFL_AFAttr, TFL_AFEnum_Relu6.symbol>;
def TFL_AF_Tanh  : ConstantStrAttr<TFL_AFAttr, TFL_AFEnum_Tanh.symbol>;
def TFL_AF_Sign  : ConstantStrAttr<TFL_AFAttr, TFL_AFEnum_Sign.symbol>;

// Allowed padding cases
// These should match the padding enum in TFLite schema.
def TFL_PADEnum_Same  : I32EnumAttrCase<"SAME", 0>;
def TFL_PADEnum_Valid : I32EnumAttrCase<"VALID", 1>;
def TFL_PaddingAttr : TFL_AnyStrAttrOf<[
      TFL_PADEnum_Same.symbol, TFL_PADEnum_Valid.symbol
    ]>;
def TFL_PAD_Same  : ConstantStrAttr<TFL_PaddingAttr, TFL_PADEnum_Same.symbol>;
def TFL_PAD_Valid : ConstantStrAttr<TFL_PaddingAttr, TFL_PADEnum_Valid.symbol>;

// FullyConnectedOptionsWeightFormat attributes
def TFL_FCWOEnum_Default         : I32EnumAttrCase<"DEFAULT", 0>;
def TFL_FCWOEnum_Shuffled4x16i8  : I32EnumAttrCase<"SHUFFLED4x16INT8", 1>;
def TFL_FullyConnectedOptionsWeightFormatAttr :
    TFL_AnyStrAttrOf<[
      TFL_FCWOEnum_Default.symbol,
      TFL_FCWOEnum_Shuffled4x16i8.symbol
    ]>;
def TFL_FCWO_Default        : ConstantStrAttr<
      TFL_FullyConnectedOptionsWeightFormatAttr, TFL_FCWOEnum_Default.symbol>;
def TFL_FCWO_Shuffled4x16i8 : ConstantStrAttr<
      TFL_FullyConnectedOptionsWeightFormatAttr, TFL_FCWOEnum_Shuffled4x16i8.symbol>;

// MirrorPadding type attributes
def TFL_MIRRORPAD_Reflect : I32EnumAttrCase<"REFLECT", 0>;
def TFL_MIRRORPAD_Symmetric : I32EnumAttrCase<"SYMMETRIC", 1>;
def TFL_MirrorPaddingType : I32EnumAttr<"MirrorPaddingType", "mirror_pad_enum", [
      TFL_MIRRORPAD_Reflect, TFL_MIRRORPAD_Symmetric
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::TFL";
}
def TFL_MirrorPaddingAttr : EnumAttr<TFL_Dialect, TFL_MirrorPaddingType,
    "mirror_pad_attr">;

// LSTM Kernel Type attributes
def TFL_LSTM_KT_FULL  : I32EnumAttrCase<"FULL", 0>;
def TFL_LSTM_KT_BASIC  : I32EnumAttrCase<"BASIC", 1>;
def TFL_LSTMKernelType : I32EnumAttr<"LSTMKernelType", "lstm_kernel_type",
   [
     TFL_LSTM_KT_FULL,  TFL_LSTM_KT_BASIC
   ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::TFL";
}
def TFL_LSTMKernelTypeAttr : EnumAttr<TFL_Dialect, TFL_LSTMKernelType,
    "lstm_kernel_type_attr">;

def I32ArrayParameter :
    AttrOrTypeParameter<"::llvm::ArrayRef<int32_t>", ""> {
  let allocator = [{$_dst = $_allocator.copyInto($_self);}];
  let cppStorageType = "::llvm::SmallVector<int32_t>";
  let parser = "::mlir::TFL::parseI32Array($_parser)";
  let printer = "$_printer << '[' << $_self << ']'";
}

def DimensionMetadataAttr : AttrDef<TFL_Dialect, "DimensionMetadata"> {
  let mnemonic = "dimension_metadata";
  let parameters = (ins
      TFL_DimensionTypeAttr:$format,
      "int32_t":$dense_size,
      I32ArrayParameter:$segments,
      I32ArrayParameter:$indices
  );
  let summary = "Dimension metadata.";
  let assemblyFormat = "`<` struct(params) `>`";
}

def SparsityParameterAttr : AttrDef<TFL_Dialect, "SparsityParameter"> {
  let mnemonic = "sparsity_parameter";
  let parameters = (ins
      I32ArrayParameter:$traversal_order,
      I32ArrayParameter:$block_map,
      ArrayRefParameter<"DimensionMetadataAttr">:$dim_metadata
  );
  let summary = "Sparsity parameter.";
  let assemblyFormat = "`<` struct(params) `>`";
}

def TFL_ConstBytesAttr : AttrDef<TFL_Dialect, "ConstBytes"> {
  let summary = "A string attribute representation of compiled bytes";
  let description = [{
    Syntax Examples:

    ```mlir
    #tfl<const_bytes : "0xDEADBEEF">
    ```
  }];
  let mnemonic = "const_bytes";
  let parameters = (ins StringRefParameter<"">:$value);
  let hasCustomAssemblyFormat = 1;
}

#endif // TFL_OP_ENUMS
