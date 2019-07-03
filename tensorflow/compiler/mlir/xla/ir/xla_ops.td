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

def XLA_Dialect : Dialect {
  let name = "xla";
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

def XLA_Int : IntOfWidths<[8, 16, 32, 64]>;

// Any integer tensor types
def XLA_IntTensor : StaticShapeTensorOf<[XLA_Int]>;

// Any floating-point tensor types
def XLA_FpTensor : StaticShapeTensorOf<[AnyFloat]>;

def XLA_Pred : TypeAlias<I1, "pred (AKA boolean or 1-bit integer)">;

def XLA_PredTensor : StaticShapeTensorOf<[XLA_Pred]>;

// Any integer or floating-point tensor types
def XLA_IntOrFpTensor : StaticShapeTensorOf<[XLA_Int, AnyFloat]>;

def XLA_Tensor : StaticShapeTensorOf<[AnyFloat, AnyInteger]>;

def XLA_Tuple : NestedTupleOf<[XLA_Tensor]>;

def XLA_TensorOrTuple : AnyTypeOf<[XLA_Tensor, XLA_Tuple]>;

//===----------------------------------------------------------------------===//
// XLA nullary op definitions.
//===----------------------------------------------------------------------===//

def XLA_ConstOp : XLA_Op<"constant", [NoSideEffect]> {
  let summary = "Constant operator";

  let description = [{
    Represents a constant value.
  }];

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

def XLA_IotaOp : XLA_Op<"iota", [NoSideEffect]> {
  let summary = "Iota operator";

  let description = [{
    Creates a rank 1 array of values starting at zero and incrementing by one.
  }];

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
    XLA_Op<mnemonic, traits>, Arguments<(ins XLA_Tensor:$operand)>,
    Results<(outs XLA_Tensor:$res)>;

def XLA_AbsOp: XLA_UnaryElementwiseOp<"abs", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Absolute value operator";

  let description = [{
    Returns `abs(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

def XLA_ConvertOp : XLA_UnaryElementwiseOp<
      "convert", [NoSideEffect, SameOperandsAndResultShape]> {
  let summary = "Convert operator";

  let description = [{
    Performs element-wise conversion of values from one type to another, e.g.
    float to int.

    See https://www.tensorflow.org/xla/operation_semantics#convertelementtype.
  }];

  let hasFolder = 1;

  // TODO(b/130357376) Convert has a special constructor. Use a custom
  // HLO converter until we have a method to call the special constructor.
  let hasCustomHLOConverter = 1;
}

def XLA_NegOp: XLA_UnaryElementwiseOp<"neg", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Negation operator";

  let description = [{
    Returns `-operand` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

def XLA_SignOp: XLA_UnaryElementwiseOp<"sign", [NoSideEffect, SameOperandsAndResultShape]> {
  let summary = "Sign operator";

  let description = [{
    Returns `sign(operand)` element-wise, where

    ```
    sign(x) = -1  : x < 0
            = -0  : x = -0
            = NaN : x = NaN
            = +0  : x = +0
            = 1   : x > 0
    ```

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

def XLA_TanhOp: XLA_UnaryElementwiseOp<"tanh",
    [ResultsAreFloatLike, NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Tanh operator";

  let description = [{
    Returns `tanh(operand)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_unary_functions.
  }];
}

//===----------------------------------------------------------------------===//
// XLA binary elementwise op definitions.
//===----------------------------------------------------------------------===//

// The broadcasting dimensions correspond to a tuple that describes how a
// smaller rank shape is broadcast into a larger rank shape. For example,
// given a 2x3x4 cuboid and a 3x4 matrix, a broadcasting tuple (1,2) means
// matching the matrix to dimensions 1 and 2 of the cuboid.
def BroadcastDimAttr : OptionalAttr<ElementsAttr>;

// See https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations
class XLA_BinaryElementwiseOp<string mnemonic,
    list<OpTrait> traits, dag args = (ins)> :
        XLA_Op<mnemonic, traits>,
        Arguments<(
            ins XLA_Tensor:$lhs,
            XLA_Tensor:$rhs,
            BroadcastDimAttr:$broadcast_dimensions
        )>,
        Results<(outs XLA_Tensor:$res)> {
  let parser = [{ return mlir::impl::parseBinaryOp(parser, result); }];
  let printer = [{ return mlir::impl::printBinaryOp(getOperation(), p); }];
}

def XLA_AddOp : XLA_BinaryElementwiseOp<"add",
      [Commutative, NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "Addition operator";

  let description = [{
    Returns `lhs + rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def XLA_DivOp : XLA_BinaryElementwiseOp<"div",
      [NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "Division operator";

  let description = [{
    Returns `lhs / rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def XLA_MaxOp : XLA_BinaryElementwiseOp<"max",
      [Commutative, NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "Maximum operator";

  let description = [{
    Returns `max(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def XLA_MinOp : XLA_BinaryElementwiseOp<"min",
      [Commutative, NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "Minimum operator";

  let description = [{
    Returns `min(lhs, rhs)` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def XLA_MulOp : XLA_BinaryElementwiseOp<"mul",
      [Commutative, NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "Multiplication operator";

  let description = [{
    Returns `lhs * rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def XLA_SubOp : XLA_BinaryElementwiseOp<"sub",
      [NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "Subtraction operator";

  let description = [{
    Returns `lhs - rhs` element-wise.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_binary_arithmetic_operations.
  }];
}

def XLA_AndOp: XLA_BinaryElementwiseOp<"and", [Commutative, NoSideEffect]>;

//===----------------------------------------------------------------------===//
// XLA control flow op definitions.
//===----------------------------------------------------------------------===//
def XLA_WhileOp: XLA_Op<"while", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "While operator";

  let description = [{
    Returns the result of executing a body function until the cond body returns
    true.

    See https://www.tensorflow.org/xla/operation_semantics#while.
  }];

  let arguments = (ins
    Variadic<XLA_TensorOrTuple>:$val,
    FunctionAttr:$cond,
    FunctionAttr:$body
  );

  let results = (outs Variadic<XLA_TensorOrTuple>:$res);

  // TODO(b/129422361): WhileOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

def XLA_ReduceOp: XLA_Op<"reduce", [NoSideEffect]> {
  let summary = "Reduce operator";

  let description = [{
    Returns the result of executing a reduction function on one or more arrays
    in parallel.

    See https://www.tensorflow.org/xla/operation_semantics#reduce.
  }];

  let arguments = (ins
    Variadic<XLA_TensorOrTuple>:$operands_and_init,
    FunctionAttr:$computation,
    ElementsAttr:$dimensions
  );

  let results = (outs Variadic<XLA_Tensor>:$res);

  // TODO(b/129422361): ReduceOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

//===----------------------------------------------------------------------===//
// XLA tuple op definitions.
//===----------------------------------------------------------------------===//
def XLA_GetTupleElementOp: XLA_Op<"get_tuple_element", [NoSideEffect]> {
  let summary = "GetTupleElement operator";

  let description = [{
    Returns a member of a tuple specified by an index.

    See https://www.tensorflow.org/xla/operation_semantics#gettupleelement.
  }];

  let arguments = (ins
    XLA_Tuple,
    I32Attr:$index
  );

  let results = (outs XLA_TensorOrTuple);

  // GetTupleElementOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

def XLA_TupleOp : XLA_Op<"tuple", [NoSideEffect]> {
   let summary = "XLA's tuple op";

   let description = [{
     Groups a set of tensor inputs into a single tuple object.

     See https://www.tensorflow.org/xla/operation_semantics#tuple.
   }];

   let arguments = (ins Variadic<XLA_TensorOrTuple>:$val);
   let results = (outs XLA_Tuple:$res);

  // TupleOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

//===----------------------------------------------------------------------===//
// Precision Config enum definitions.
//===----------------------------------------------------------------------===//

// These mirror the XLA PrecisionConfig proto enum.
def XLA_PRECISION_DEFAULT : StrEnumAttrCase<"DEFAULT">;
def XLA_PRECISION_HIGH    : StrEnumAttrCase<"HIGH">;
def XLA_PRECISION_HIGHEST : StrEnumAttrCase<"HIGHEST">;

def XLA_PrecisionAttr : StrEnumAttr<"Precision",
    "XLA precision for an operand. Has backend specific meaning.",
    [XLA_PRECISION_DEFAULT,  XLA_PRECISION_HIGH, XLA_PRECISION_HIGHEST]>;

// TODO(b/129153247) See if it's possible to also validate the size.
def XLA_PrecisionConfigAttr:
    OptionalAttr<
          TypedArrayAttrBase<XLA_PrecisionAttr, "Precision Config attribute">>;

//===----------------------------------------------------------------------===//
// Comparison op definitions.
//===----------------------------------------------------------------------===//

// These mirror the XLA ComparisonDirection enum.
def XLA_COMPARISON_DIRECTION_EQ : StrEnumAttrCase<"EQ">;
def XLA_COMPARISON_DIRECTION_NE : StrEnumAttrCase<"NE">;
def XLA_COMPARISON_DIRECTION_GE : StrEnumAttrCase<"GE">;
def XLA_COMPARISON_DIRECTION_GT : StrEnumAttrCase<"GT">;
def XLA_COMPARISON_DIRECTION_LE : StrEnumAttrCase<"LE">;
def XLA_COMPARISON_DIRECTION_LT : StrEnumAttrCase<"LT">;

def XLA_ComparisonDirectionAttr : StrEnumAttr<"ComparisonDirection",
    "Which comparison operation to perform.",
    [
      XLA_COMPARISON_DIRECTION_EQ,
      XLA_COMPARISON_DIRECTION_NE,
      XLA_COMPARISON_DIRECTION_GE,
      XLA_COMPARISON_DIRECTION_GT,
      XLA_COMPARISON_DIRECTION_LE,
      XLA_COMPARISON_DIRECTION_LT
    ]>;

def XLA_CompareOp: XLA_Op<"compare",
      [NoSideEffect, SameOperandsAndResultShape]> {
  let arguments = (ins
      XLA_Tensor:$lhs,
      XLA_Tensor:$rhs,
      BroadcastDimAttr:$broadcast_dimensions,
      XLA_ComparisonDirectionAttr:$comparison_direction
  );
  let results = (outs I1Tensor:$res);
  let summary = "Comparison operator";

  let description = [{
    Compares `lhs` and `rhs` elementwise according to `comparison_direction`.

    See
    https://www.tensorflow.org/xla/operation_semantics#element-wise_comparison_operations.
  }];
}

//===----------------------------------------------------------------------===//
// XLA Slice definitions.
//===----------------------------------------------------------------------===//

def XLA_SliceOp: XLA_UnaryElementwiseOp<"slice",
      [NoSideEffect, SameOperandsAndResultElementType]> {
  let arguments = (
    ins XLA_Tensor:$operand,
    ElementsAttr:$start_indices,
    ElementsAttr:$limit_indices
  );

  let results = (outs XLA_Tensor:$res);

  let summary = "Slice operator";

  let description = [{
    Slices a portion of the `operand` into a new configuration.

    See https://www.tensorflow.org/xla/operation_semantics#slice.
  }];

  // TODO(b/129422361) Two of the required arguments comes from the start and
  // limit indices which aren't handled by the codegen.
  let hasCustomHLOConverter = 1;
}

def XLA_DynamicUpdateSliceOp: XLA_UnaryElementwiseOp<"dynamic-update-slice",
      [NoSideEffect, AllElementTypesMatch<["operand", "res"]>]> {
  let arguments = (ins
    XLA_Tensor:$operand,
    XLA_Tensor:$update,
    Variadic<XLA_Tensor>:$start_indices
  );

  let results = (outs XLA_Tensor:$res);

  let summary = "Dynamic Update Slice operator";

  let description = [{
    DynamicUpdateSlice generates a result which is the value of the input array
    operand, with a slice update overwritten at start_indices.

    See https://www.tensorflow.org/xla/operation_semantics#dynamicupdateslice.
  }];

  // TODO(b/129422361) Requires a custom constructor.
  let hasCustomHLOConverter = 1;
}


//===----------------------------------------------------------------------===//
// XLA Other op definitions.
//===----------------------------------------------------------------------===//

def XLA_BatchNormInferenceOp : XLA_Op<"batch_norm_inference", [NoSideEffect]> {
  let summary = "Batch Normalization for Inference";

  let description = [{
    Normalizes an array across batch and spatial dimensions.

    See https://www.tensorflow.org/xla/operation_semantics#batchnorminference
  }];

  let arguments = (ins
    XLA_Tensor:$operand,
    XLA_Tensor:$scale,
    XLA_Tensor:$offset,
    XLA_Tensor:$mean,
    XLA_Tensor:$variance,
    F32Attr:$epsilon,
    I64Attr:$feature_index
  );

  let results = (outs
    XLA_Tensor:$res
  );
}

def XLA_BroadcastOp : XLA_Op<"broadcast",
      [NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "Broadcast a tensor to a higher rank by prepending dimensions";

  let description = [{
    Broadcasts the operand tensor to a higher rank by prepending
    `broadcast_sizes` to the dimensions. The current values of the operand are
    copied into the other dimensions.

    This is a more limited form of broadcasting, that corresponds to the XLA
    client Broadcast method. For a more general form of broadcasting, see the
    BroadcastInDimOp.

    See https://www.tensorflow.org/xla/operation_semantics#broadcast.
  }];

  let arguments = (ins
    XLA_Tensor:$operand,
    ElementsAttr:$broadcast_sizes
  );

  let results = (outs XLA_Tensor:$res);

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

    auto resultType = res()->getType().cast<RankedTensorType>();
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

    auto raw_sizes = sizes.getValues<int64_t>();
    llvm::SmallVector<int64_t, 10> expectedShape(raw_sizes.begin(),
                                                 raw_sizes.end());
    if (sizes.isSplat()) {
      expectedShape.resize(sizesSize, raw_sizes.front());
    }

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
      [NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "Broadcast a tensor into the given shape by adding dimensions.";

  let description = [{
    Broadcasts the `operand` tensor to a higher rank. This is not the limited
    form of broadcasting exposed as the XLA client broadcast op, but rather the
    more powerful "InDim" broadcasting, which is closer to the HLO broadcast op
    and exposed in the XLA client BroadcastInDim method.

    `broadcast_dimensions` maps the operand dimension number to the target shape
    dimension number. It must have the same size as the rank of the operand. The
    mapped dimensions must either be the same size or the dimension being
    broadcast from must be size 1 (degenerate broadcasting).

    For a scalar (0D tensor) operand, `broadcast_dimensions` must be empty. The
    The scalar value will be broadcast to every element in the target shape.

    See https://www.tensorflow.org/xla/broadcasting.
  }];

  let arguments = (ins
    XLA_Tensor:$operand,
    BroadcastDimAttr:$broadcast_dimensions
  );

  let results = (outs XLA_Tensor:$res);

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

    auto resultType = res()->getType().cast<RankedTensorType>();
    auto resultRank = resultType.getRank();
    if (resultRank < operandRank) {
      return emitOpError(
          llvm::formatv("result rank ({0}) is less than operand rank ({1})",
                        resultRank, operandRank));
    }

    for (int i = 0; i != dimensionsSize; ++i) {
      auto dimIndex = dimensions.getValue(i).cast<IntegerAttr>().getInt();
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
      [NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "Clamp operator";

  let description = [{
    Clamps an operand to within the range between a minimum and maximum value.

    Note: All three arrays must be the same shape. Alternatively, as a
          restricted form of broadcasting, min and/or max can be a scalar (0D
          tensor) of the element type of the tensor operand.

    See https://www.tensorflow.org/xla/operation_semantics#clamp.
  }];

  let arguments = (ins
    XLA_Tensor:$min,
    XLA_Tensor:$operand,
    XLA_Tensor:$max
  );

  let results = (outs
    XLA_Tensor:$res
  );

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
      [NoSideEffect, SameOperandsAndResultElementType]> {
   let summary = "XLA's concantenate op";

   let description = [{
     Concatenates a set of tensors along the specified dimension.

     See https://www.tensorflow.org/xla/operation_semantics#concatenate.
   }];

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

   let results = (outs XLA_Tensor:$res);

  // TODO(b/129422361) ConcatOp has special conversion logic to HLO.
  let hasCustomHLOConverter = 1;
}

def XLA_ConvOp : XLA_Op<"conv", [NoSideEffect]> {
  let summary = "Convolution operator";

  let description = [{
    Computes a convolution of the kind used in neural networks.

    See https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
  }];

  let arguments = (ins
    XLA_Tensor:$lhs,
    XLA_Tensor:$rhs
  );

  let results = (outs XLA_Tensor:$res);

  // TODO(b/129422361) Needs additional work to handle attributes.
  // Conv has custom handling because its other args are passed as attributes
  let hasCustomHLOConverter = 1;
}

def XLA_CopyOp: XLA_UnaryElementwiseOp<"copy", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "Copy operator";

  let description = [{
    Returns a copy of `operand`.
  }];

  // TODO(b/129422361) Implement special handling.
  // Copy has an HloOpcode, but is not one of the ops defined in xla_builder.
  let hasCustomHLOConverter = 1;
}

def XLA_DotOp: XLA_Op<"dot", [NoSideEffect]> {
  let arguments = (
        ins XLA_Tensor:$lhs,
        XLA_Tensor:$rhs,
        XLA_PrecisionConfigAttr:$precision_config
    );
  let results = (outs XLA_Tensor:$res);

  let description = [{
    Performs dot products between vectors, vector/matrix and matrix/matrix
    multiplication.

    See https://www.tensorflow.org/xla/operation_semantics#dot.
  }];
}

def XLA_GatherOp: XLA_Op<"gather", [NoSideEffect]> {
  let arguments = (
      ins XLA_Tensor:$operand,
          XLA_IntTensor:$start_indices,
          I64Attr: $index_vector_dim,
          ElementsAttr: $offsets_dim,
          ElementsAttr: $slice_sizes,
          ElementsAttr: $collapsed_slice_sizes,
          ElementsAttr: $start_index_map
  );

  let results = (outs XLA_Tensor:$res);

  let summary = "Gather operator";

  let description = [{
    Stitches together several slices of an input array.

    See https://www.tensorflow.org/xla/operation_semantics#gather.
  }];

  // TODO(b/129422361) Attributes are not by the codegen. The optional argument
  // (dimensions) needs to be added as an attribute.
  let hasCustomHLOConverter = 1;
}

def XLA_ReshapeOp: XLA_Op<"reshape",
      [NoSideEffect, SameOperandsAndResultElementType]> {
  let arguments = (ins XLA_Tensor:$operand);

  let results = (outs XLA_Tensor:$res);

  let summary = "Reshape operator";

  let description = [{
    Reshapes the dimensions of `operand` into a new configuration.

    See https://www.tensorflow.org/xla/operation_semantics#reshape.
  }];

  let hasFolder = 1;

  // TODO(b/129422361) One of the required arguments comes from the new shape,
  // which isn't handled by the codegen. The optional argument (dimensions)
  // needs to be added as an attribute.
  let hasCustomHLOConverter = 1;
}


def XLA_SelectOp: XLA_Op<"select", [NoSideEffect]> {
  let summary = "Select operator";

  let description = [{
    Constructs an output tensor from the elements of `on_true` and `on_false`
    based on the values of `pred`.

    `on_true` and `on_false` must be the same shape. For each element of `pred`,
    `res` has the corresponding element of `on_true` or `on_false` depending on
    the value in `pred`. `pred` must be the same shape as `on_true` and
    `on_false` or a scalar, in which case `res` is equal to either `on_true` or
    `on_false`.

    See https://www.tensorflow.org/xla/operation_semantics#select.
  }];

  let arguments = (ins
    XLA_PredTensor:$pred,
    XLA_Tensor:$on_true,
    XLA_Tensor:$on_false
  );

  let results = (outs XLA_Tensor:$res);

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
      [NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "Reverse operator";

  let description = [{
    Reverses the specified dimensions of `operand` according to the given
    `dimensions`.

    See https://www.tensorflow.org/xla/operation_semantics#rev_reverse.
  }];

  let arguments = (ins
    XLA_Tensor:$operand,
    ElementsAttr:$dimensions
  );

  let results = (outs XLA_Tensor:$res);

  // TODO(b/129422361): ReverseOp has a custom constructor for HLO.
  let hasCustomHLOConverter = 1;
}

def XLA_PadOp: XLA_Op<"pad",
      [NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "Pad operator";

  let description = [{
    Pads the edges of `operand` with the `padding_value` and according to
    the passed configuration.

    See https://www.tensorflow.org/xla/operation_semantics#pad.
  }];

  let arguments = (ins
    XLA_Tensor:$operand,
    XLA_Tensor:$padding_value,
    ElementsAttr: $edge_padding_low,
    ElementsAttr: $edge_padding_high,
    ElementsAttr: $interior_padding
  );

  let results = (outs XLA_Tensor:$res);

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
          + padding_low.getValue(i).cast<IntegerAttr>().getInt()
          + padding_high.getValue(i).cast<IntegerAttr>().getInt();
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
      [NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = "Transpose operator";

  let description = [{
    Permutes the dimensions of `operand` according to the given `permutation`.

    `res_dimensions[i] = operand_dimensions[permutation[i]]`

    See https://www.tensorflow.org/xla/operation_semantics#transpose.
  }];

  let arguments = (ins
    XLA_Tensor:$operand,
    ElementsAttr:$permutation
  );
  let results = (outs XLA_Tensor:$res);

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

    auto resultType = res()->getType().cast<RankedTensorType>();
    auto resultRank = resultType.getRank();
    if (resultRank != operandRank) {
      return emitOpError(
          llvm::formatv("result rank ({0}) does not match operand rank ({1})",
                        resultRank, operandRank));
    }

    auto resultShape = resultType.getShape();

    auto expectedShape = SmallVector<int64_t, 10>(operandRank);
    for (int i = 0; i != operandRank; ++i) {
      auto permutedDim = permutation().getValue(i).cast<IntegerAttr>().getInt();
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
