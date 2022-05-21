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

// This is the operation definition file for ST ops.

#ifndef GML_ST_LEGACY_OPS
#define GML_ST_LEGACY_OPS

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def GMLST_LoopOp : GMLST_Op<"loop", [
     AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<LoopLikeOpInterface>,
     RecursiveSideEffects,
     SingleBlockImplicitTerminator<"gml_st::YieldOp">
    ]> {
  let summary = "Loop-like operation";
  let description = [{
    This is a loop-like operation with additional properties. The arguments
    also include the input and the output tensors or memrefs and the attributes
    to specify the iterator types.

    Parsing LoopOp will set all elements of the `iterator_types` attribute
    to "parallel" type, when it is absent from the custom format.

    Tensor-based version:

    The body region of the loop contains `extract_slice` operations applied to
    every tensor argument of LoopOp.

    The body region must contain exactly one block that terminates with
    `gml_st.yield` with the operands resulting from `insert_slice` operations.

    Example:

    ```mlir
    %0 = gml_st.loop (%i) = (%c0) to (%c24) step (%c4)
        ins(%lhs, %rhs : tensor<24x64xi8>, tensor<24x64xi8>)
        outs(%out : tensor<24x64xi8>)
        iterators("parallel")
        distribution("block_x") {
      %lhs_sub = tensor.extract_slice %lhs[%i, 0] [%c4, %c64] [1, 1]
          : tensor<24x64xi8> to tensor<?x?xi8>
      %rhs_sub = tensor.extract_slice %rhs[%i, 0] [%c4, %c64] [1, 1]
          : tensor<24x64xi8> to tensor<?x?xi8>
      %out_sub = tensor.extract_slice %out[%i, 0] [%c4, %c64] [1, 1]
          : tensor<24x64xi8> to tensor<?x?xi8>

      %result_sub = linalg.generic ...

      %result = tensor.insert_slice %result_sub into %out[%i, 0][%c4, %c64][1, 1]
        : tensor<?x?xi8> into tensor<24x64xi8>
      gml_st.yield %result : tensor<24x64xi8>
    }
    ```

    MemRef-based version:

    The body region of the loop contains `subview` operations applied to
    every memref argument of LoopOp.

    The body region must contain exactly one block that terminates with
    `gml_st.yield` with no operands.

    Example:

    ```mlir
    gml_st.loop (%i) = (%c0) to (%c24) step (%c4)
        ins(%lhs, %rhs : memref<24x64xi8>, memref<24x64xi8>)
        outs(%out : memref<24x64xi8>)
        iterators("parallel")
        distribution("block_x") {
      %lhs_sub = subview %lhs[%i, 0] [%c4, %c64] [1, 1]
          : memref<24x64xi8> to memref<?x?xi8>
      %rhs_sub = subview %rhs[%i, 0] [%c4, %c64] [1, 1]
          : memref<24x64xi8> to memref<?x?xi8>
      %out_sub = subview %out[%i, 0] [%c4, %c64] [1, 1]
          : memref<24x64xi8> to memref<?x?xi8>

      %result_sub = linalg.generic ...
      gml_st.yield
    }
    ```
  }];

  let arguments = (ins Variadic<Index>:$lowerBound,
                       Variadic<Index>:$upperBound,
                       Variadic<Index>:$step,
                       Variadic<AnyType>:$inputs,
                       Variadic<AnyShaped>:$outputs,
                       ArrayAttr:$iterator_types,
                       OptionalAttr<ArrayAttr>:$distribution_types);
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<(ins "ValueRange":$lowerBounds, "ValueRange":$upperBounds,
      "ValueRange":$steps, "ValueRange":$inputs, "ValueRange":$outputs,
      "ArrayAttr":$iteratorTypes, "Optional<ArrayAttr>":$distributionTypes,
      CArg<"function_ref<void (OpBuilder &, Location, /*ivs=*/ValueRange,"
        "/*inputs=*/ValueRange, /*outputs=*/ValueRange)>",
        "nullptr">:$bodyBuilderFn)>,
    OpBuilder<(ins "ValueRange":$lowerBounds, "ValueRange":$upperBounds,
      "ValueRange":$steps, "ValueRange":$inputs, "ValueRange":$outputs,
      "ArrayAttr":$iteratorTypes,
      CArg<"function_ref<void (OpBuilder &, Location, /*ivs=*/ValueRange,"
        "/*inputs=*/ValueRange, /*outputs=*/ValueRange)>",
        "nullptr">:$bodyBuilderFn)>,
  ];

  let extraClassDeclaration = [{
    /// Number of loops
    unsigned getNumLoops() { return step().size(); }

    /// Number of input operands
    unsigned getNumInputs() { return inputs().size(); }

    /// Number of output operands
    unsigned getNumOutputs() { return outputs().size(); }

    /// Number of operands controlling the loop: lbs, ubs, steps
    unsigned getNumControlOperands() { return 3 * getNumLoops(); }

    ValueRange getInductionVars() {
      return getBody()->getArguments().take_front(getNumLoops());
    }
    ValueRange getRegionInputArgs() {
      return getBody()->getArguments().slice(getNumLoops(), inputs().size());
    }
    ValueRange getRegionOutputArgs() {
      return getBody()->getArguments().take_back(outputs().size());
    }

    void setDistributionTypes(Builder& b, ArrayRef<StringRef> types) {
      assert(types.size() == getNumLoops() &&
             "expected distribution type for every dimension");
      distribution_typesAttr(b.getStrArrayAttr(types));
    }

    void setLowerBounds(ValueRange lowerBounds) {
      unsigned numLoops = getNumLoops();
      assert(lowerBounds.size() == numLoops &&
             "expected lower bounds for every loop dimension");
      for (unsigned i = 0; i < numLoops; ++i)
        setOperand(i, lowerBounds[i]);
    }

    void setUpperBounds(ValueRange upperBounds) {
      unsigned numLoops = getNumLoops();
      assert(upperBounds.size() == numLoops &&
             "expected upper bounds for every loop dimension");
      for (unsigned i = 0, pos = numLoops; i < numLoops; ++i, ++pos)
        setOperand(pos, upperBounds[i]);
    }

    void setSteps(ValueRange steps) {
      unsigned numLoops = getNumLoops();
      assert(steps.size() == numLoops &&
             "expected upper bounds for every loop dimension");
      for (unsigned i = 0, pos = 2 * numLoops; i < numLoops; ++i, ++pos)
        setOperand(pos, steps[i]);
    }

    /// Operand that corresponds to the `bbArg` block argument.
    OpOperand& getTiedOperand(BlockArgument& bbArg) {
      return getOperation()->getOpOperand(getNumControlOperands() +
                                          bbArg.getArgNumber() - getNumLoops());
    }

    /// Block argument that corresponds to the `input` or `output` operand.
    BlockArgument getTiedBlockArgument(OpOperand& operand) {
      auto operandIndex = operand.getOperandNumber();
      assert(
          operandIndex >= getNumControlOperands() &&
          operandIndex < getNumOperands() &&
          "tied block arg is defined only for `input` and `output` arguments");
      return getBody()->getArgument(operandIndex - 2 * getNumLoops());
    }

   /// Result that corresponds to the `outputs` argument of tensor type.
   OpResult getTiedOpResult(OpOperand& opOperand) {
      // No result can correspond to a memref argument.
      if (opOperand.get().getType().isa<MemRefType>()) return OpResult();

      // Check whether the operand index is in bounds of `outputs()` arg.
      int operandIndex = opOperand.getOperandNumber();
      int outputIndexStart =
          getNumControlOperands() + inputs().size();
      int outputIndexEnd = outputIndexStart + outputs().size();
      if (operandIndex < outputIndexStart || operandIndex >= outputIndexEnd)
        return OpResult();

      // Count tensor arguments in `outputs` to compute the result index.
      int tensorId = -1;
      for (int i = outputIndexStart; i <= operandIndex; ++i)
        tensorId += getOperand(i).getType().isa<RankedTensorType>();
      return getOperation()->getResult(tensorId);
    }

    /// Append `operand` to the `input` arguments.
    OpOperand& appendInputOperand(OpBuilder& builder, Value operand) {
      int numLoops = getNumLoops();
      int numInputs = getNumInputs();
      int numOutputs = getNumOutputs();

      getOperation()->insertOperands(getNumControlOperands() + numInputs,
                                     operand);
      getBody()->insertArgument(numLoops + numInputs, operand.getType(),
                                getLoc());
      getOperation()->setAttr(
          LoopOp::getOperandSegmentSizeAttr(),
          builder.getI32VectorAttr(
              {numLoops, numLoops, numLoops, numInputs + 1, numOutputs}));
      return getOperation()->getOpOperand(getNumControlOperands() + numInputs);
    }

    /// Append `operand` to the `output` arguments.
    OpOperand& appendOutputOperand(OpBuilder& builder, Value operand) {
      int numLoops = getNumLoops();
      int numInputs = getNumInputs();
      int numOutputs = getNumOutputs();

      getOperation()->insertOperands(
          getNumControlOperands() + numInputs + numOutputs, operand);
      getBody()->insertArgument(numLoops + numInputs + numOutputs,
                                operand.getType(), getLoc());
      getOperation()->setAttr(
          LoopOp::getOperandSegmentSizeAttr(),
          builder.getI32VectorAttr(
              {numLoops, numLoops, numLoops, numInputs, numOutputs + 1}));
      return getOperation()->getOpOperand(getNumControlOperands() + numInputs +
                                          numOutputs);
    }

    /// Erase `operand` from the `input` or `output` arguments.
    void eraseOperand(OpBuilder& builder, OpOperand& operand) {
      int numInputs = getNumInputs();
      int numLoops = getNumLoops();
      int numOutputs = getNumOutputs();
      int numControlOperands = getNumControlOperands();

      int operandIndex = operand.getOperandNumber();
      assert(operandIndex >= numControlOperands &&
             operandIndex < static_cast<int>(getNumOperands()) &&
             "Can erase only `input` or `output` operand");

      if (operandIndex >= numControlOperands + numInputs)
        --numOutputs;
      else
        --numInputs;

      getOperation()->eraseOperand(operandIndex);
      getBody()->eraseArgument(operandIndex - 2 * numLoops);
      getOperation()->setAttr(
          LoopOp::getOperandSegmentSizeAttr(),
          builder.getI32VectorAttr(
              {numLoops, numLoops, numLoops, numInputs, numOutputs}));
    }

    OpOperand* findInputOperand(Value value) {
      OperandRange::iterator it = llvm::find(inputs(), value);
      if (it == inputs().end()) return nullptr;
      return it.getBase();
    }

    OpOperand* findOutputOperand(Value value) {
      OperandRange::iterator it = llvm::find(outputs(), value);
      if (it == outputs().end()) return nullptr;
      return it.getBase();
    }

    /// Return whether the op has only MemRef input and outputs.
    bool hasBufferSemantics() {
      Operation* op = this->getOperation();
      return op->getNumResults() == 0 &&
             llvm::all_of(op->getOpOperands(), [&](OpOperand & operand) {
               return !operand.get().getType().template isa<ShapedType>() ||
                      operand.get().getType().template isa<MemRefType>();
             });
    }

    static constexpr StringRef getParallelIteratorTypeName() {
      return "parallel";
    }
    static constexpr StringRef getDistributionTypesAttrName() {
      return "distribution_types";
    }
    static constexpr StringRef getIteratorTypesAttrName() {
      return "iterator_types";
    }


    /// Return whether the loop dimension is parallel or not.
    bool isParallelDimension(unsigned dim) {
      StringAttr attr = this->iterator_types()[dim].cast<StringAttr>();
      return attr.getValue() == getParallelIteratorTypeName();
    }
  }];

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
}

def GMLST_YieldOp : GMLST_Op<"yield", [NoSideEffect, ReturnLike,
      Terminator, HasParent<"::mlir::gml_st::LoopOp">]>,
    Arguments<(ins Variadic<AnyType>:$values)> {
  let summary = "Yield operation";
  let description = [{
    `gml_st.yield` is a special terminator operation for `gml_st.loop` body.

    Example:

    ```mlir
    gml_st.yield %f0, %f1 : tensor<f32>, tensor<?xf32>
    ```
  }];
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
}

#endif // GML_ST_LEGACY_OPS
