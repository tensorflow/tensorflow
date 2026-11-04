/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_IR_IFRT_INTERFACES_TD_
#define XLA_PYTHON_IFRT_IR_IFRT_INTERFACES_TD_

include "mlir/IR/OpBase.td"

//===---------------------------------------------------------------------------
// Op interfaces
//===---------------------------------------------------------------------------

// Verifies op used in a FuncOp with `ifrt.function` attr.
def NestedInIfrtFunc : NativeOpTrait<"xla::ifrt::NestedInIfrtFuncTrait">;

// Verifies callee is a valid symbol ref to callee_op_type.
// Provides `callee_op_type getCalleeOp(SymbolTableCollection)`.
// Concrete type must have a SymbolRefAttr named `callee`.
class IfrtCallLike<string callee_op_type>
    : ParamNativeOpTrait<"xla::ifrt::IfrtCallLikeTrait", callee_op_type>;

class Ifrt_OpInterface<string name, list<Interface> baseInterfaces = []>
    : OpInterface<name, baseInterfaces> {
  let cppNamespace = "::xla::ifrt";
}

def Ifrt_ArrayOpInterface : Ifrt_OpInterface<"IfrtArrayOpInterface"> {
  let description = [{
    Interface for IFRT operations with variadic array inputs/outputs, variadic
    control inputs, and a single control output.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the array input operands.",
      /*retTy=*/"::mlir::Operation::operand_range",
      /*methodName=*/"getInputs"
    >,
    InterfaceMethod<
      /*desc=*/"Returns the mutable array input operands.",
      /*retTy=*/"::mlir::MutableOperandRange",
      /*methodName=*/"getInputsMutable"
    >,
    InterfaceMethod<
      /*desc=*/"Returns the array output results.",
      /*retTy=*/"::mlir::Operation::result_range",
      /*methodName=*/"getOutputs"
    >,
    InterfaceMethod<
      /*desc=*/"Returns the control input operands.",
      /*retTy=*/"::mlir::Operation::operand_range",
      /*methodName=*/"getControlInputs"
    >,
    InterfaceMethod<
      /*desc=*/"Returns the mutable control input operands.",
      /*retTy=*/"::mlir::MutableOperandRange",
      /*methodName=*/"getControlInputsMutable"
    >,
    InterfaceMethod<
      /*desc=*/"Returns the control output result.",
      /*retTy=*/"::mlir::TypedValue<::xla::ifrt::IfrtControlType>",
      /*methodName=*/"getControlOutput"
    >,
  ];
}

def Ifrt_PairwiseArrayOpInterface
    : Ifrt_OpInterface<"IfrtPairwiseArrayOpInterface",
                       [Ifrt_ArrayOpInterface]> {
  let description = [{
    Interface for IFRT operations whose array inputs map pairwise to array
    outputs (`inputs[i]` -> `outputs[i]`).
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns whether the input arrays are donated.",
      /*retTy=*/"bool",
      /*methodName=*/"getDonated"
    >,
    InterfaceMethod<
      /*desc=*/"Returns the pairwise input operand corresponding to array output `out`.",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"getInputForOutput",
      /*args=*/(ins "::mlir::OpResult":$out),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        assert(out.getOwner() == $_op.getOperation() &&
               out.getResultNumber() < $_op.getInputs().size());
        return $_op.getInputs()[out.getResultNumber()];
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Clones the op keeping only `(inputs[i], outputs[i])` where `keep[i]` is
        true. Returns nullptr if `keep.none()`.
      }],
      /*retTy=*/"::xla::ifrt::IfrtPairwiseArrayOpInterface",
      /*methodName=*/"cloneWithKeptOutputs",
      /*args=*/(ins
        "::mlir::RewriterBase&":$rewriter,
        "const ::llvm::BitVector&":$keep
      ),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return ::xla::ifrt::detail::ClonePairwiseArrayOpWithKeptOutputs(
            $_op, rewriter, keep);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Replaces the op keeping only `(inputs[i], outputs[i])` where `keep[i]`
        is true. If `keep.none()`, forwards `control_inputs` to users of
        `control_output`, erases the op, and returns nullptr.
      }],
      /*retTy=*/"::xla::ifrt::IfrtPairwiseArrayOpInterface",
      /*methodName=*/"replaceWithKeptOutputs",
      /*args=*/(ins
        "::mlir::RewriterBase&":$rewriter,
        "const ::llvm::BitVector&":$keep
      ),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return ::xla::ifrt::detail::ReplacePairwiseArrayOpWithKeptOutputs(
            $_op, rewriter, keep);
      }]
    >,
  ];
}

//===---------------------------------------------------------------------------
// Attribute interfaces
//===---------------------------------------------------------------------------

class Ifrt_AttrInterface<string name> : AttrInterface<name> {
  let cppNamespace = "::xla::ifrt";
}

def Ifrt_ShardingAttrInterface : Ifrt_AttrInterface<"IfrtShardingAttrInterface"> {
  let description = [{
    Interface that all IFRT IR sharding attributes must implement.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Verifies if the sharding can be applied to the array.",
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"CanApplyTo",
      /*args=*/(ins
        "::llvm::function_ref<mlir::InFlightDiagnostic()>":$emitError,
        "::mlir::RankedTensorType":$shape,
        "llvm::ArrayRef<int>":$device_ids
      )
    >,
    InterfaceMethod<
      /*desc=*/"Returns the shape of the global array from a local array shape.",
      /*retTy=*/"::absl::StatusOr<llvm::SmallVector<int64_t>>",
      /*methodName=*/"GlobalShapeFromLocalShape",
      /*args=*/(ins "llvm::ArrayRef<int64_t>":$local_shape)
    >,
    InterfaceMethod<
      /*desc=*/"Returns the shape of the local array from a global array shape.",
      /*retTy=*/"::absl::StatusOr<llvm::SmallVector<int64_t>>",
      /*methodName=*/"LocalShapeFromGlobalShape",
      /*args=*/(ins "llvm::ArrayRef<int64_t>":$global_shape)
    >,
    InterfaceMethod<
      /*desc=*/"Returns the number of devices this sharding applied to.",
      /*retTy=*/"int",
      /*methodName=*/"NumDevices",
      /*args=*/(ins)
    >
  ];
}

#endif // XLA_PYTHON_IFRT_IR_IFRT_INTERFACES_TD_
