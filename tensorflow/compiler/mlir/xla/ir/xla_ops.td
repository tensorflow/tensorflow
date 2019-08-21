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

// This is the operation definition file for XLA.

#ifdef XLA_OPS
#else
#define XLA_OPS

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

#ifdef XLA_OPS_BASE
#else
include "tensorflow/compiler/mlir/xla/ir/xla_ops_base.td"
#endif

def XLA_Dialect : Dialect {
  let name = "xla_hlo";
  let cppNamespace = "XLA";
}

class XLA_Op<string mnemonic, list<OpTrait> traits> :
    Op<XLA_Dialect, mnemonic, traits> {
  // Whether this operation has a custom conversion to HLO or not.
  bit hasCustomHLOConverter = 0b0;
}

//===----------------------------------------------------------------------===//
// XLA type definitions.
//===----------------------------------------------------------------------===//

// Any integer tensor types
def XLA_IntTensor : StaticShapeTensorOf<[XLA_Int]>;

// Any floating-point tensor types
def XLA_FpTensor : StaticShapeTensorOf<[AnyFloat]>;

def XLA_PredTensor : StaticShapeTensorOf<[XLA_Pred]>;

// Any integer or floating-point tensor types
def XLA_IntOrFpTensor : StaticShapeTensorOf<[XLA_Int, AnyFloat]>;

def XLA_Tensor : StaticShapeTensorOf<[AnyFloat, AnyInteger]>;

def XLA_Tuple : NestedTupleOf<[XLA_Tensor]>;

def XLA_TensorOrTuple : AnyTypeOf<[XLA_Tensor, XLA_Tuple]>;

//===----------------------------------------------------------------------===//
// XLA nullary op definitions.
//===----------------------------------------------------------------------===//

def XLA_ConstOp : BXLA_ConstOp, XLA_Op<"constant", [NoSideEffect]> {
  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    XLA_Tensor:$output
  );

  let builders = [OpBuilder<
    "Builder *builder, OperationState *result, Attribute value"
  >];

  let hasFolder = 1;

  // Constant has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

def XLA_IotaOp : BXLA_IotaOp, XLA_Op<"iota", [NoSideEffect]> {
  let arguments = (ins I64Attr:$iota_dimension);

  let results = (outs XLA_Tensor:$output);

  let hasFolder = 1;

  // TODO(b/130357376): Iota has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

//===----------------------------------------------------------------------===//
// XLA unary elementwise op definitions.
//===----------------------------------------------------------------------===//
// See https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions
class XLA_UnaryElementwiseOp<string mnemonic, list<OpTrait> traits>:
    XLA_Op<mnemonic, traits> {

    let arguments = (ins XLA_Tensor);
    let results = (outs XLA_Tensor);
}

def XLA_AbsOp: XLA_UnaryElementwiseOp<"abs", [NoSideEffect, SameOperandsAndResultType]>, BXLA_AbsOp;

def XLA_ConvertOp : XLA_UnaryElementwiseOp<
      "convert", [NoSideEffect, SameOperandsAndResultShape]>, BXLA_ConvertOp {
  let hasFolder = 1;

  // TODO(b/130357376) Convert has a special constructor. Use a custom
  // HLO converter until we have a method to call the special constructor.
  let hasCustomHLOConverter = 1;
}

def XLA_ExpOp: XLA_UnaryElementwiseOp<"exp", [NoSideEffect, SameOperandsAndResultType]>, BXLA_ExpOp;

def XLA_NegOp: XLA_UnaryElementwiseOp<"neg", [NoSideEffect, SameOperandsAndResultType]>, BXLA_NegOp;

def XLA_SignOp: XLA_UnaryElementwiseOp<"sign", [NoSideEffect, SameOperandsAndResultShape]>, BXLA_SignOp;

def XLA_TanhOp: XLA_UnaryElementwiseOp<"tanh",
    [ResultsAreFloatLike, NoSideEffect, SameOperandsAndResultType]>, BXLA_TanhOp;

//===----------------------------------------------------------------------===//
// XLA binary elementwise op definitions.
//===----------------------------------------------------------------------===//

// See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations
class XLA_BinaryElementwiseOp<string mnemonic, list<OpTrait> traits> :
        XLA_Op<mnemonic, traits> {
  let arguments = (ins
      XLA_Tensor:$lhs,
      XLA_Tensor:$rhs,
      BroadcastDimAttr:$broadcast_dimensions
  );
  let results = (outs XLA_Tensor);
  let parser = [{ return mlir::impl::parseBinaryOp(parser, result); }];
  let printer = [{ return mlir::impl::printBinaryOp(getOperation(), p); }];
}

def XLA_AddOp : XLA_BinaryElementwiseOp<"add",
      [Commutative, NoSideEffect, SameOperandsAndResultElementType]>, BXLA_AddOp;

def XLA_DivOp : XLA_BinaryElementwiseOp<"div",
      [NoSideEffect, SameOperandsAndResultElementType]>, BXLA_DivOp;

def XLA_MaxOp : XLA_BinaryElementwiseOp<"max",
      [Commutative, NoSideEffect, SameOperandsAndResultElementType]>, BXLA_MaxOp;

def XLA_MinOp : XLA_BinaryElementwiseOp<"min",
      [Commutative, NoSideEffect, SameOperandsAndResultElementType]>, BXLA_MinOp;

def XLA_MulOp : XLA_BinaryElementwiseOp<"mul",
      [Commutative, NoSideEffect, SameOperandsAndResultElementType]>, BXLA_MulOp;

def XLA_SubOp : XLA_BinaryElementwiseOp<"sub",
      [NoSideEffect, SameOperandsAndResultElementType]>, BXLA_SubOp;

def XLA_AndOp: XLA_BinaryElementwiseOp<"and", [Commutative, NoSideEffect]>, BXLA_AndOp;

//===----------------------------------------------------------------------===//
// XLA control flow op definitions.
//===----------------------------------------------------------------------===//
def XLA_WhileOp: XLA_Op<"while", [NoSideEffect, SameOperandsAndResultType]> {
  string summary = "While operator";

  string description = [{
    Returns the result of executing a body function until the cond body returns
    true.

    See https://www.tensorflow.org/xla/operation_semantics#while.
  }];

  let arguments = (ins
    Variadic<XLA_TensorOrTuple>:$val,
    SymbolRefAttr:$cond,
    SymbolRefAttr:$body
  );

  let results = (outs Variadic<XLA_TensorOrTuple>);

  // TODO(b/129422361): WhileOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

def XLA_ReduceOp: XLA_Op<"reduce", [NoSideEffect]>, BXLA_ReduceOp {

  let arguments = (ins
    Variadic<XLA_TensorOrTuple>:$operands_and_init,
    SymbolRefAttr:$computation,
    ElementsAttr:$dimensions
  );

  let results = (outs Variadic<XLA_Tensor>);

  // TODO(b/129422361): ReduceOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

//===----------------------------------------------------------------------===//
// XLA tuple op definitions.
//===----------------------------------------------------------------------===//
def XLA_GetTupleElementOp: XLA_Op<"get_tuple_element", [NoSideEffect]>, BXLA_GetTupleElementOp {
  let arguments = (ins
    XLA_Tuple,
    I32Attr:$index
  );

  let results = (outs XLA_TensorOrTuple);

  // GetTupleElementOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

def XLA_TupleOp : XLA_Op<"tuple", [NoSideEffect]>, BXLA_TupleOp {
   let arguments = (ins Variadic<XLA_TensorOrTuple>:$val);
   let results = (outs XLA_Tuple);

  // TupleOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

def XLA_CompareOp: XLA_Op<"compare",
      [NoSideEffect, SameOperandsAndResultShape]>, BXLA_CompareOp {
  let arguments = (ins
      XLA_Tensor:$lhs,
      XLA_Tensor:$rhs,
      BroadcastDimAttr:$broadcast_dimensions,
      XLA_ComparisonDirectionAttr:$comparison_direction
  );
  let results = (outs XLA_PredTensor);
}

//===----------------------------------------------------------------------===//
// XLA Slice definitions.
//===----------------------------------------------------------------------===//

def XLA_SliceOp: XLA_Op<
      "slice",
      [NoSideEffect, SameOperandsAndResultElementType,
       AllTypesMatch<["start_indices", "limit_indices"]>]> {
  let arguments = (
    ins XLA_Tensor:$operand,
    ElementsAttr:$start_indices,
    ElementsAttr:$limit_indices
  );

  let results = (outs XLA_Tensor);

  // TODO(b/129422361) Two of the required arguments comes from the start and
  // limit indices which aren't handled by the codegen.
  let hasCustomHLOConverter = 1;
}

def XLA_DynamicUpdateSliceOp: XLA_Op<"dynamic-update-slice",
      [NoSideEffect, AllElementTypesMatch<["operand", "result"]>]> {
  let arguments = (ins
    XLA_Tensor:$operand,
    XLA_Tensor:$update,
    Variadic<XLA_Tensor>:$start_indices
  );

  let results = (outs XLA_Tensor:$result);

  // TODO(b/129422361) Requires a custom constructor.
  let hasCustomHLOConverter = 1;
}


//===----------------------------------------------------------------------===//
// XLA Other op definitions.
//===----------------------------------------------------------------------===//

def XLA_BatchNormInferenceOp : XLA_Op<"batch_norm_inference", [NoSideEffect]>,
    BXLA_BatchNormInferenceOp {

  let arguments = (ins
    XLA_Tensor:$operand,
    XLA_Tensor:$scale,
    XLA_Tensor:$offset,
    XLA_Tensor:$mean,
    XLA_Tensor:$variance,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );

  let results = (outs XLA_Tensor);
}

def XLA_BroadcastOp : XLA_Op<"broadcast",
      [NoSideEffect, SameOperandsAndResultElementType]>, BXLA_BroadcastOp {
  let arguments = (ins
    XLA_Tensor:$operand,
    ElementsAttr:$broadcast_sizes
  );

  let results = (outs XLA_Tensor);

  // TODO(b/129012527) These should be expressed as type constraints.
  let verifier = [{
    auto sizes = broadcast_sizes().dyn_cast<DenseIntElementsAttr>();
    if (!sizes) {
      return emitOpError(llvm::formatv(
          "broadcast_sizes must be a DenseIntElementsAttr; got {0}",
          broadcast_sizes()));
    }
    auto sizesType = sizes.getType().cast<RankedTensorType>();
    auto sizesRank = sizesType.getRank();
    if (sizesRank != 1) {
      return emitOpError(llvm::formatv(
          "broadcast_sizes has rank {0} instead of rank 1", sizesRank));
    }

    auto resultType = getResult()->getType().cast<RankedTensorType>();
    auto resultRank = resultType.getRank();
    auto operandType = operand()->getType().cast<RankedTensorType>();
    auto operandRank = operandType.getRank();
    auto sizesSize = sizesType.getNumElements();
    auto expectedRank = operandRank + sizesSize;

    if (resultRank != expectedRank) {
      return emitOpError(
          llvm::formatv("result rank ({0}) does not match operand rank "
                        "({2}) plus size of broadcast_sizes ({3})",
                        resultRank, operandRank, sizesSize));
    }

    llvm::SmallVector<int64_t, 10> expectedShape(sizes.getValues<int64_t>());

    auto operandShape = operandType.getShape();
    expectedShape.insert(expectedShape.end(), operandShape.begin(),
                         operandShape.end());

    auto resultShape = resultType.getShape();
    if (resultShape != llvm::makeArrayRef(expectedShape)) {
      return emitOpError(llvm::formatv(
          "result has shape [{0}"
          "] instead of [{1}"
          "]",
          llvm::make_range(resultShape.begin(), resultShape.end()),
          llvm::make_range(expectedShape.begin(), expectedShape.end())));
    }

    return success();
  }];
}

def XLA_BroadcastInDimOp : XLA_Op<"broadcast_in_dim",
      [NoSideEffect, SameOperandsAndResultElementType]>, BXLA_BroadcastInDimOp {
  let arguments = (ins
    XLA_Tensor:$operand,
    BroadcastDimAttr:$broadcast_dimensions
  );

  let results = (outs XLA_Tensor);

  // TODO(b/129012527) These should be expressed as type constraints.
  let verifier = [{
    auto operandType = operand()->getType().cast<RankedTensorType>();
    auto operandRank = operandType.getRank();
    if (!broadcast_dimensions()) {
      if (operandRank == 0) {
        return success();
      }
      return emitOpError(
          llvm::formatv("broadcast_dimensions is absent, but required because "
                        "operand has non-zero rank ({0})",
                        operandRank));
    }

    auto dimensions = broadcast_dimensions()->dyn_cast<DenseIntElementsAttr>();
    if (!dimensions) {
      return emitOpError(llvm::formatv(
          "broadcast_sizes must be a DenseIntElementsAttr; got {0}",
          broadcast_dimensions()));
    }

    auto dimensionsType = broadcast_dimensions()->getType().cast<RankedTensorType>();
    auto dimensionsRank = dimensionsType.getRank();
    if (dimensionsRank != 1) {
      return emitOpError(
          llvm::formatv("broadcast_dimensions has rank {0} instead of rank 1",
                        dimensionsRank));
    }

    auto dimensionsSize = dimensionsType.getNumElements();
    if (dimensionsSize != operandRank) {
      return emitOpError(llvm::formatv(
          "broadcast_dimensions size ({0}) does not match operand rank ({1})",
          dimensionsSize, operandRank));
    }

    auto resultType = getResult()->getType().cast<RankedTensorType>();
    auto resultRank = resultType.getRank();
    if (resultRank < operandRank) {
      return emitOpError(
          llvm::formatv("result rank ({0}) is less than operand rank ({1})",
                        resultRank, operandRank));
    }

    for (int i = 0; i != dimensionsSize; ++i) {
      auto dimIndex = dimensions.getValue<int64_t>(i);
      if (dimIndex >= resultRank) {
        return emitOpError(
            llvm::formatv("broadcast_dimensions contains invalid value {0} for "
                          "result result with rank {1}",
                          dimIndex, resultRank));
      }

      auto dimSize = operandType.getDimSize(i);
      auto resultDimSize = resultType.getDimSize(dimIndex);
      if (dimSize != 1 && dimSize != resultDimSize) {
        return emitOpError(
            llvm::formatv("size of operand dimension {0} ({1}) is not equal to "
                          "1 or size of result dimension {2} ({3})",
                          i, dimSize, dimIndex, resultDimSize));
      }
    }

    return success();
  }];

  // TODO(b/130357376): One of the arguments comes from the new shape, which is
  // not handled by the codegen.
  let hasCustomHLOConverter = 1;
}

def XLA_ClampOp : XLA_Op<"clamp",
      [NoSideEffect, SameOperandsAndResultElementType]>, BXLA_ClampOp {
  let arguments = (ins
    XLA_Tensor:$min,
    XLA_Tensor:$operand,
    XLA_Tensor:$max
  );

  let results = (outs XLA_Tensor);

  // TODO(b/129012527) These should be expressed as type constraints.
  let verifier = [{
    auto operandType = operand()->getType().cast<RankedTensorType>();
    auto operandShape = operandType.getShape();
    auto minType = min()->getType().cast<RankedTensorType>();

    auto minShape = minType.getShape();
    if (minShape != operandShape && minType.getRank() != 0) {
      return emitOpError(llvm::formatv(
          "min shape [{0}"
          "] is not scalar and does not match operand shape [{1}"
          "]",
          llvm::make_range(minShape.begin(), minShape.end()),
          llvm::make_range(operandShape.begin(), operandShape.end())));
    }

    auto maxType = max()->getType().cast<RankedTensorType>();
    auto maxShape = maxType.getShape();
    if (maxShape != operandShape && maxType.getRank() != 0) {
      return emitOpError(llvm::formatv(
          "max shape [{0}"
          "] is not scalar and does not match operand shape [{1}"
          "]",
          llvm::make_range(maxShape.begin(), maxShape.end()),
          llvm::make_range(operandShape.begin(), operandShape.end())));
    }

    return success();
  }];
}

def XLA_ConcatenateOp : XLA_Op<"concatenate",
      [NoSideEffect, SameOperandsAndResultElementType]>, BXLA_ConcatenateOp {

   let arguments = (
     ins Variadic<XLA_Tensor>:$val,
         I64Attr: $dimension
   );

   let verifier = [{
     auto firstType = getOperand(0)->getType().cast<RankedTensorType>();

     auto firstShape = firstType.getShape();
     int numOperands = getNumOperands();
     for (int i = 1; i < numOperands; i++) {
       auto secondType = getOperand(i)->getType().cast<RankedTensorType>();

       if (firstType.getRank() != secondType.getRank()) {
         return emitOpError(
             llvm::formatv("operands (0) and ({0}) do not match rank.", i));
       }

       auto secondShape = secondType.getShape();
       for (int d = 0; d < firstType.getRank(); ++d) {
         if (firstShape[d] != secondShape[d] && d != dimension()) {
           return emitOpError(llvm::formatv(
               "operands (0) and ({0}) non-concat dimensions do not match "
               "({1}) != ({2}).",
               i, llvm::make_range(firstShape.begin(), firstShape.end()),
               llvm::make_range(secondShape.begin(), secondShape.end())));
         }
       }
     }
     return success();
   }];

   let results = (outs XLA_Tensor);

  // TODO(b/129422361) ConcatOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

def XLA_ConvOp : XLA_Op<"conv", [NoSideEffect]>, BXLA_ConvOp {
  let arguments = (ins
    XLA_Tensor:$lhs,
    XLA_Tensor:$rhs
  );

  let results = (outs XLA_Tensor);

  // TODO(b/129422361) Needs additional work to handle attributes.
  // Conv has custom handling because its other args are passed as attributes
  let hasCustomHLOConverter = 1;
}

def XLA_CopyOp: XLA_Op<"copy", [NoSideEffect, SameOperandsAndResultType]> {
  string summary = "Copy operator";

  string description = [{
    Returns a copy of `operand`.
  }];

  let arguments = (ins XLA_Tensor);
  let results = (outs XLA_Tensor);

  // TODO(b/129422361) Implement special handling.
  // Copy has an HloOpcode, but is not one of the ops defined in xla_builder.
  let hasCustomHLOConverter = 1;
}

def XLA_DotOp: XLA_Op<"dot", [NoSideEffect]>, BXLA_DotOp {
  let arguments = (
        ins XLA_Tensor:$lhs,
        XLA_Tensor:$rhs,
        XLA_PrecisionConfigAttr:$precision_config
    );
  let results = (outs XLA_Tensor);
}

def XLA_GatherOp: XLA_Op<"gather", [NoSideEffect]>, BXLA_GatherOp {
  let arguments = (
      ins XLA_Tensor:$operand,
          XLA_IntTensor:$start_indices,
          I64Attr: $index_vector_dim,
          ElementsAttr: $offset_dims,
          ElementsAttr: $slice_sizes,
          ElementsAttr: $collapsed_slice_dims,
          ElementsAttr: $start_index_map
  );

  let results = (outs XLA_Tensor);

  // TODO(b/129422361) Attributes are not by the codegen. The optional argument
  // (dimensions) needs to be added as an attribute.
  let hasCustomHLOConverter = 1;
}

def XLA_ReshapeOp: XLA_Op<"reshape",
      [NoSideEffect, SameOperandsAndResultElementType]>, BXLA_ReshapeOp {
  let arguments = (ins XLA_Tensor:$operand);

  let results = (outs XLA_Tensor);
  let hasFolder = 1;

  // TODO(b/129422361) One of the required arguments comes from the new shape,
  // which isn't handled by the codegen. The optional argument (dimensions)
  // needs to be added as an attribute.
  let hasCustomHLOConverter = 1;
}


def XLA_SelectOp: XLA_Op<"select", [NoSideEffect]>, BXLA_SelectOp {
  let arguments = (ins
    XLA_PredTensor:$pred,
    XLA_Tensor:$on_true,
    XLA_Tensor:$on_false
  );

  let results = (outs XLA_Tensor);

  // TODO(b/129012527) These should be expressed as type constraints.
  let verifier = [{
    auto onTrueType = on_true()->getType().cast<RankedTensorType>();
    auto onFalseType = on_false()->getType().cast<RankedTensorType>();

    if (onTrueType != onFalseType) {
      return emitOpError(
          llvm::formatv("on_true type ({0}) does not match on_false type ({1})",
                        onTrueType, onFalseType));
    }

    auto predType = pred()->getType().cast<RankedTensorType>();
    auto predShape = predType.getShape();
    auto predRank = predType.getRank();
    auto selectShape = onTrueType.getShape();

    if (predRank != 0 && predShape != selectShape) {
      return emitOpError(llvm::formatv(
          "pred shape ([{0}"
          "]) is not scalar and does not match operand shapes ([{1}"
          "])",
          llvm::make_range(predShape.begin(), predShape.end()),
          llvm::make_range(selectShape.begin(), selectShape.end())));
    }

    return success();
  }];
}

def XLA_ReverseOp: XLA_Op<"reverse",
      [NoSideEffect, SameOperandsAndResultElementType]>, BXLA_ReverseOp {
  let arguments = (ins
    XLA_Tensor:$operand,
    ElementsAttr:$dimensions
  );

  let results = (outs XLA_Tensor);

  // TODO(b/129422361): ReverseOp has a custom constructor for HLO.
  let hasCustomHLOConverter = 1;
}

def XLA_PadOp: XLA_Op<"pad",
      [NoSideEffect, SameOperandsAndResultElementType]>, BXLA_PadOp {
  let arguments = (ins
    XLA_Tensor:$operand,
    XLA_Tensor:$padding_value,
    ElementsAttr: $edge_padding_low,
    ElementsAttr: $edge_padding_high,
    ElementsAttr: $interior_padding
  );

  let results = (outs XLA_Tensor);

  let description = [{
    Pads the `operand` according to TBD.
  }];

  let verifier = [{
    auto input_type = operand()->getType().cast<RankedTensorType>();
    auto pad_type = padding_value()->getType().cast<RankedTensorType>();

    if (pad_type.getRank() != 0) {
      return emitOpError(llvm::formatv("padding value type should be a rank-0 "
          "tensor, is rank {0}", pad_type.getRank()));
    }

    const auto& padding_low = edge_padding_low();
    if (padding_low.getType().getNumElements() != input_type.getRank()) {
      return emitOpError(llvm::formatv(
          "edge_padding_low length ({0}) must match operand rank ({1}).",
          padding_low.getType().getNumElements(), input_type.getRank()));
    }

    const auto& padding_high = edge_padding_high();
    if (padding_high.getType().getNumElements() != input_type.getRank()) {
      return emitOpError(llvm::formatv(
          "edge_padding_high length ({0}) must match operand rank ({1}).",
          padding_high.getType().getNumElements(), input_type.getRank()));
    }

    auto input_shape = input_type.getShape();
    auto output_shape = getResult()->getType().cast<RankedTensorType>().getShape();
    if (input_shape.size() != output_shape.size()) {
      return emitOpError(llvm::formatv(
          "Operand rank ({0}) and result rank({0}) should match",
          input_shape.size(), output_shape.size()));
    }

    for (int i = 0, e = input_shape.size(); i < e; i++) {
      int expected_output = input_shape[i]
          + padding_low.getValue<IntegerAttr>(i).getInt()
          + padding_high.getValue<IntegerAttr>(i).getInt();
      if (expected_output != output_shape[i]) {
        return emitOpError(llvm::formatv("Expected output shape ({0}) and "
            "output shape ({1}) should match.",
            expected_output, output_shape[i]));
      }
    }

    return success();
  }];

  // TODO(b/129422361): PadOp has a custom constructor for HLO.
  let hasCustomHLOConverter = 1;
}

def XLA_TransposeOp: XLA_Op<"transpose",
      [NoSideEffect, SameOperandsAndResultElementType]>, BXLA_TransposeOp {
  let arguments = (ins
    XLA_Tensor:$operand,
    ElementsAttr:$permutation
  );
  let results = (outs XLA_Tensor);

  let hasFolder = 1;

  // TODO(b/129012527) These should be expressed as type constraints.
  let verifier = [{
    if (!permutation().isa<DenseIntElementsAttr>()) {
      return emitOpError(
          llvm::formatv("permutation must be a DenseIntElementsAttr; got {0}",
                        permutation()));
    }

    auto permutationType = permutation().getType().cast<RankedTensorType>();
    auto permutationRank = permutationType.getRank();
    if (permutationRank != 1) {
      return emitOpError(llvm::formatv(
          "permutation has rank {0} instead of rank 1", permutationRank));
    }

    auto operandType = operand()->getType().cast<RankedTensorType>();
    auto operandRank = operandType.getRank();
    auto permutationSize = permutationType.getNumElements();
    if (permutationSize != operandRank) {
      return emitOpError(llvm::formatv(
          "permutation size ({0}) does not match operand rank ({1})",
          permutationSize, operandRank));
    }

    auto resultType = getResult()->getType().cast<RankedTensorType>();
    auto resultRank = resultType.getRank();
    if (resultRank != operandRank) {
      return emitOpError(
          llvm::formatv("result rank ({0}) does not match operand rank ({1})",
                        resultRank, operandRank));
    }

    auto resultShape = resultType.getShape();

    auto expectedShape = SmallVector<int64_t, 10>(operandRank);
    for (int i = 0; i != operandRank; ++i) {
      auto permutedDim = permutation().getValue<IntegerAttr>(i).getInt();
      expectedShape[i] = operandType.getDimSize(permutedDim);
    }

    if (resultShape != llvm::makeArrayRef(expectedShape)) {
      return emitOpError(llvm::formatv(
          "result shape is [{0}"
          "] instead of [{1}"
          "]",
          llvm::make_range(resultShape.begin(), resultShape.end()),
          llvm::make_range(expectedShape.begin(), expectedShape.end())));
    }

    return success();
  }];
}

#endif // XLA_OPS
