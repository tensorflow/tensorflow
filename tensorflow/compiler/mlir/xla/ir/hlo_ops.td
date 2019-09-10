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

#ifdef HLO_OPS
#else
#define HLO_OPS

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

#ifdef HLO_OPS_BASE
#else
include "tensorflow/compiler/mlir/xla/ir/hlo_ops_base.td"
#endif

def HLO_Dialect : Dialect {
  let name = "xla_hlo";
  let cppNamespace = "xla_hlo";
}

class HLO_Op<string mnemonic, list<OpTrait> traits> :
    Op<HLO_Dialect, mnemonic, traits> {
  // Whether this operation has a custom conversion to HLO or not.
  bit hasCustomHLOConverter = 0b0;
}

//===----------------------------------------------------------------------===//
// XLA type definitions.
//===----------------------------------------------------------------------===//

// Any integer tensor types
def HLO_IntTensor : StaticShapeTensorOf<[HLO_Int]>;

// Any floating-point tensor types
def HLO_FpTensor : StaticShapeTensorOf<[AnyFloat]>;

def HLO_PredTensor : StaticShapeTensorOf<[HLO_Pred]>;

// Any integer or floating-point tensor types
def HLO_IntOrFpTensor : StaticShapeTensorOf<[HLO_Int, AnyFloat]>;

def HLO_Tensor : StaticShapeTensorOf<[AnyFloat, AnyInteger]>;

def HLO_Tuple : NestedTupleOf<[HLO_Tensor]>;

def HLO_TensorOrTuple : AnyTypeOf<[HLO_Tensor, HLO_Tuple]>;

//===----------------------------------------------------------------------===//
// XLA nullary op definitions.
//===----------------------------------------------------------------------===//

def HLO_ConstOp : BASE_HLO_ConstOp, HLO_Op<"constant", [NoSideEffect]> {
  let arguments = (ins
    ElementsAttr:$value
  );

  let results = (outs
    HLO_Tensor:$output
  );

  let builders = [OpBuilder<
    "Builder *builder, OperationState *result, Attribute value"
  >];

  let hasFolder = 1;

  // Constant has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

def HLO_IotaOp : BASE_HLO_IotaOp, HLO_Op<"iota", [NoSideEffect]> {
  let arguments = (ins I64Attr:$iota_dimension);

  let results = (outs HLO_Tensor:$output);

  let hasFolder = 1;

  // TODO(b/130357376): Iota has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

//===----------------------------------------------------------------------===//
// XLA unary elementwise op definitions.
//===----------------------------------------------------------------------===//
// See https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions
class HLO_UnaryElementwiseOp<string mnemonic, list<OpTrait> traits>:
    HLO_Op<mnemonic, traits> {

    let arguments = (ins HLO_Tensor);
    let results = (outs HLO_Tensor);
}

def HLO_AbsOp: HLO_UnaryElementwiseOp<"abs", [NoSideEffect, SameOperandsAndResultType]>, BASE_HLO_AbsOp;

def HLO_ConvertOp : HLO_UnaryElementwiseOp<
      "convert", [NoSideEffect, SameOperandsAndResultShape]>, BASE_HLO_ConvertOp {
  let hasFolder = 1;

  // TODO(b/130357376) Convert has a special constructor. Use a custom
  // HLO converter until we have a method to call the special constructor.
  let hasCustomHLOConverter = 1;
}

def HLO_ExpOp: HLO_UnaryElementwiseOp<"exp", [NoSideEffect, SameOperandsAndResultType]>, BASE_HLO_ExpOp;

def HLO_FloorOp: HLO_UnaryElementwiseOp<"floor", [NoSideEffect, SameOperandsAndResultType]>, BASE_HLO_FloorOp;

def HLO_LogOp: HLO_UnaryElementwiseOp<"log", [NoSideEffect, SameOperandsAndResultType]>, BASE_HLO_LogOp;

def HLO_NegOp: HLO_UnaryElementwiseOp<"neg", [NoSideEffect, SameOperandsAndResultType]>, BASE_HLO_NegOp;

def HLO_RsqrtOp: HLO_UnaryElementwiseOp<"rsqrt", [NoSideEffect, SameOperandsAndResultType]>, BASE_HLO_RsqrtOp;

def HLO_SignOp: HLO_UnaryElementwiseOp<"sign", [NoSideEffect, SameOperandsAndResultShape]>, BASE_HLO_SignOp;

def HLO_TanhOp: HLO_UnaryElementwiseOp<"tanh",
    [ResultsAreFloatLike, NoSideEffect, SameOperandsAndResultType]>, BASE_HLO_TanhOp;

//===----------------------------------------------------------------------===//
// XLA binary elementwise op definitions.
//===----------------------------------------------------------------------===//

// See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations
class HLO_BinaryElementwiseOp<string mnemonic, list<OpTrait> traits> :
        HLO_Op<mnemonic, traits> {
  let arguments = (ins
      HLO_Tensor:$lhs,
      HLO_Tensor:$rhs,
      BroadcastDimAttr:$broadcast_dimensions
  );
  let results = (outs HLO_Tensor);
  let parser = [{ return mlir::impl::parseBinaryOp(parser, result); }];
  let printer = [{ return mlir::impl::printBinaryOp(getOperation(), p); }];
}

def HLO_AddOp : HLO_BinaryElementwiseOp<"add",
      [Commutative, NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_AddOp;

def HLO_DivOp : HLO_BinaryElementwiseOp<"div",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_DivOp;

def HLO_MaxOp : HLO_BinaryElementwiseOp<"max",
      [Commutative, NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_MaxOp;

def HLO_MinOp : HLO_BinaryElementwiseOp<"min",
      [Commutative, NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_MinOp;

def HLO_MulOp : HLO_BinaryElementwiseOp<"mul",
      [Commutative, NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_MulOp;

def HLO_SubOp : HLO_BinaryElementwiseOp<"sub",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_SubOp;

def HLO_AndOp: HLO_BinaryElementwiseOp<"and", [Commutative, NoSideEffect]>, BASE_HLO_AndOp;

//===----------------------------------------------------------------------===//
// XLA control flow op definitions.
//===----------------------------------------------------------------------===//
def HLO_WhileOp: HLO_Op<"while", [NoSideEffect, SameOperandsAndResultType]> {
  string summary = "While operator";

  string description = [{
    Returns the result of executing a body function until the cond body returns
    true.

    See https://www.tensorflow.org/xla/operation_semantics#while.
  }];

  let arguments = (ins
    Variadic<HLO_TensorOrTuple>:$val,
    SymbolRefAttr:$cond,
    SymbolRefAttr:$body
  );

  let results = (outs Variadic<HLO_TensorOrTuple>);

  // TODO(b/129422361): WhileOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

def HLO_ReduceOp: HLO_Op<"reduce", [
      NoSideEffect,
      SingleBlockImplicitTerminator<"ReturnOp">
    ]>, BASE_HLO_ReduceOp {
  // TODO(hinsu): Split operands and init into two separate variadic operands
  // by using SameVariadicOperandSize trait.
  let arguments = (ins
    Variadic<HLO_TensorOrTuple>:$operands_and_init,
    ElementsAttr:$dimensions
  );

  let results = (outs Variadic<HLO_Tensor>);

  // TODO(hinsu): Verify that the attached body arguments and results are
  // compatible with reduce op's operands.
  let regions = (region SizedRegion<1>:$body);

  // TODO(hinsu): Implement custom printer and parser.

  // TODO(b/129422361): ReduceOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

//===----------------------------------------------------------------------===//
// XLA tuple op definitions.
//===----------------------------------------------------------------------===//
def HLO_GetTupleElementOp: HLO_Op<"get_tuple_element", [NoSideEffect]>, BASE_HLO_GetTupleElementOp {
  let arguments = (ins
    HLO_Tuple,
    I32Attr:$index
  );

  let results = (outs HLO_TensorOrTuple);

  // GetTupleElementOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

def HLO_TupleOp : HLO_Op<"tuple", [NoSideEffect]>, BASE_HLO_TupleOp {
   let arguments = (ins Variadic<HLO_TensorOrTuple>:$val);
   let results = (outs HLO_Tuple);

  // TupleOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

def HLO_CompareOp: HLO_Op<"compare",
      [NoSideEffect, SameOperandsAndResultShape]>, BASE_HLO_CompareOp {
  let arguments = (ins
      HLO_Tensor:$lhs,
      HLO_Tensor:$rhs,
      BroadcastDimAttr:$broadcast_dimensions,
      HLO_ComparisonDirectionAttr:$comparison_direction
  );
  let results = (outs HLO_PredTensor);
}

//===----------------------------------------------------------------------===//
// XLA Slice definitions.
//===----------------------------------------------------------------------===//

def HLO_SliceOp: HLO_Op<
      "slice",
      [NoSideEffect, SameOperandsAndResultElementType,
       AllTypesMatch<["start_indices", "limit_indices"]>]> {
  let arguments = (
    ins HLO_Tensor:$operand,
    ElementsAttr:$start_indices,
    ElementsAttr:$limit_indices
  );

  let results = (outs HLO_Tensor);

  // TODO(b/129422361) Two of the required arguments comes from the start and
  // limit indices which aren't handled by the codegen.
  let hasCustomHLOConverter = 1;
}

def HLO_DynamicUpdateSliceOp: HLO_Op<"dynamic-update-slice",
      [NoSideEffect, AllElementTypesMatch<["operand", "result"]>]> {
  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_Tensor:$update,
    Variadic<HLO_Tensor>:$start_indices
  );

  let results = (outs HLO_Tensor:$result);

  // TODO(b/129422361) Requires a custom constructor.
  let hasCustomHLOConverter = 1;
}


//===----------------------------------------------------------------------===//
// XLA Other op definitions.
//===----------------------------------------------------------------------===//

def HLO_BatchNormInferenceOp : HLO_Op<"batch_norm_inference", [NoSideEffect]>,
    BASE_HLO_BatchNormInferenceOp {

  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_Tensor:$scale,
    HLO_Tensor:$offset,
    HLO_Tensor:$mean,
    HLO_Tensor:$variance,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );

  let results = (outs HLO_Tensor);
}

def HLO_BroadcastOp : HLO_Op<"broadcast",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_BroadcastOp {
  let arguments = (ins
    HLO_Tensor:$operand,
    ElementsAttr:$broadcast_sizes
  );

  let results = (outs HLO_Tensor);

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

def HLO_BroadcastInDimOp : HLO_Op<"broadcast_in_dim",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_BroadcastInDimOp {
  let arguments = (ins
    HLO_Tensor:$operand,
    BroadcastDimAttr:$broadcast_dimensions
  );

  let results = (outs HLO_Tensor);

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

def HLO_ClampOp : HLO_Op<"clamp",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_ClampOp {
  let arguments = (ins
    HLO_Tensor:$min,
    HLO_Tensor:$operand,
    HLO_Tensor:$max
  );

  let results = (outs HLO_Tensor);

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

def HLO_ConcatenateOp : HLO_Op<"concatenate",
    [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_ConcatenateOp {

  let arguments = (
    ins Variadic<HLO_Tensor>:$val,
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

  let results = (outs HLO_Tensor);

  let hasFolder = 1;

  // TODO(b/129422361) ConcatOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

def HLO_ConvOp : HLO_Op<"conv", [NoSideEffect]>, BASE_HLO_ConvOp {
  let arguments = (ins
    HLO_Tensor:$lhs,
    HLO_Tensor:$rhs
  );

  let results = (outs HLO_Tensor);

  // TODO(b/129422361) Needs additional work to handle attributes.
  // Conv has custom handling because its other args are passed as attributes
  let hasCustomHLOConverter = 1;
}

def HLO_CopyOp: HLO_Op<"copy", [NoSideEffect, SameOperandsAndResultType]> {
  string summary = "Copy operator";

  string description = [{
    Returns a copy of `operand`.
  }];

  let arguments = (ins HLO_Tensor);
  let results = (outs HLO_Tensor);

  // TODO(b/129422361) Implement special handling.
  // Copy has an HloOpcode, but is not one of the ops defined in xla_builder.
  let hasCustomHLOConverter = 1;
}

def HLO_DotOp: HLO_Op<"dot", [NoSideEffect]>, BASE_HLO_DotOp {
  let arguments = (
        ins HLO_Tensor:$lhs,
        HLO_Tensor:$rhs,
        HLO_PrecisionConfigAttr:$precision_config
    );
  let results = (outs HLO_Tensor);
}

def HLO_GatherOp: HLO_Op<"gather", [NoSideEffect]>, BASE_HLO_GatherOp {
  let arguments = (
      ins HLO_Tensor:$operand,
          HLO_IntTensor:$start_indices,
          I64Attr: $index_vector_dim,
          ElementsAttr: $offset_dims,
          ElementsAttr: $slice_sizes,
          ElementsAttr: $collapsed_slice_dims,
          ElementsAttr: $start_index_map
  );

  let results = (outs HLO_Tensor);

  // TODO(b/129422361) Attributes are not by the codegen. The optional argument
  // (dimensions) needs to be added as an attribute.
  let hasCustomHLOConverter = 1;
}

def HLO_ReshapeOp: HLO_Op<"reshape",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_ReshapeOp {
  let arguments = (ins HLO_Tensor:$operand);

  let results = (outs HLO_Tensor);
  let hasFolder = 1;

  // TODO(b/129422361) One of the required arguments comes from the new shape,
  // which isn't handled by the codegen. The optional argument (dimensions)
  // needs to be added as an attribute.
  let hasCustomHLOConverter = 1;
}


def HLO_SelectOp: HLO_Op<"select", [NoSideEffect]>, BASE_HLO_SelectOp {
  let arguments = (ins
    HLO_PredTensor:$pred,
    HLO_Tensor:$on_true,
    HLO_Tensor:$on_false
  );

  let results = (outs HLO_Tensor);

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

def HLO_ReverseOp: HLO_Op<"reverse",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_ReverseOp {
  let arguments = (ins
    HLO_Tensor:$operand,
    ElementsAttr:$dimensions
  );

  let results = (outs HLO_Tensor);

  // TODO(b/129422361): ReverseOp has a custom constructor for HLO.
  let hasCustomHLOConverter = 1;
}

def HLO_PadOp: HLO_Op<"pad",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_PadOp {
  let arguments = (ins
    HLO_Tensor:$operand,
    HLO_Tensor:$padding_value,
    ElementsAttr: $edge_padding_low,
    ElementsAttr: $edge_padding_high,
    ElementsAttr: $interior_padding
  );

  let results = (outs HLO_Tensor);

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

def HLO_TransposeOp: HLO_Op<"transpose",
      [NoSideEffect, SameOperandsAndResultElementType]>, BASE_HLO_TransposeOp {
  let arguments = (ins
    HLO_Tensor:$operand,
    ElementsAttr:$permutation
  );
  let results = (outs HLO_Tensor);

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

def HLO_ReduceWindowOp: HLO_Op<"reduce_window", [
      NoSideEffect,
      SameVariadicOperandSize,
      SingleBlockImplicitTerminator<"ReturnOp">
    ]>, BASE_HLO_ReduceWindowOp {

  // TODO(hinsu): Verify that padding attribute is 2-d and the remaining
  // attributes are 1-d. Attributes' leading dimension should match rank of the
  // inputs.
  let arguments = (ins
    Variadic<HLO_TensorOrTuple>:$operands,
    Variadic<HLO_TensorOrTuple>:$init_values,
    I64ElementsAttr:$window_dimensions,
    // If strides or dilations attributes are missing then the default value is
    // one for each of the input dimensions. Similarly, padding values are zero
    // for both low and high in each of the dimensions, if not specified.
    OptionalAttr<I64ElementsAttr>:$window_strides,
    OptionalAttr<I64ElementsAttr>:$base_dilations,
    OptionalAttr<I64ElementsAttr>:$window_dilations,
    OptionalAttr<I64ElementsAttr>:$padding
  );

  let results = (outs Variadic<HLO_Tensor>);

  // TODO(hinsu): Verify that the attached body arguments and results are
  // compatible with reduce op's operands.
  let regions = (region SizedRegion<1>:$body);

  // TODO(b/129422361): ReduceWindowOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;

  // TODO(hinsu): Implement custom printer and parser.
}

def HLO_ReturnOp : HLO_Op<"return", [Terminator]> {
  let summary = [{
    The `hlo.return` operation terminates a region and returns values.
  }];

  let arguments = (ins
    Variadic<HLO_TensorOrTuple>:$results
  );

  // Disable conversion operator for return op as the op is not an actual XLA
  // instruction and is only used as a terminator for regions.
  let hasCustomHLOConverter = 1;

  // TODO(hinsu): Implement custom printer and parser.
}

#endif // HLO_OPS
