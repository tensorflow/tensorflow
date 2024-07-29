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

def Ifrt_SpmdExpandableInterface : OpInterface<"IfrtSpmdExpandable"> {
  let cppNamespace = "::xla::ifrt";

  let description = [{
    Interface to conduct SPMD expansion on ops. It contains the SPMD expansion
    method as well as the layout computation methods as they are prerequisite
    for SPMD expansion.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Rewrite the global op to per-shard view.",
      /*retTy=*/"mlir::FailureOr<mlir::Operation*>",
      /*methodName=*/"SpmdExpand",
      /*args=*/(ins),
      /*methodBody=*/""
    >,
    InterfaceMethod<
      /*desc=*/"Compute sharding parameters for the outputs of op from input sharding parameters.",
      /*retTy=*/"mlir::FailureOr<llvm::DenseMap<int, xla::ifrt::ShardingParam>>",
      /*methodName=*/"ComputeShardingForward",
      /*args=*/(ins "const llvm::DenseMap<int, xla::ifrt::ShardingParam>&":$input_shardings
      ),
      /*methodBody=*/""
    >,
    InterfaceMethod<
      /*desc=*/"Compute sharding parameters for the inputs of op from output sharding parameters",
      /*retTy=*/"mlir::FailureOr<llvm::DenseMap<int, xla::ifrt::ShardingParam>>",
      /*methodName=*/"ComputeShardingBackward",
      /*args=*/(ins "const llvm::DenseMap<int, xla::ifrt::ShardingParam>&":$output_shardings
      ),
      /*methodBody=*/""
    >
  ];
}

// Verifies op used in a FuncOp with `ifrt.function` attr.
def NestedInIfrtFunc : NativeOpTrait<"xla::ifrt::NestedInIfrtFuncTrait">;

// Verifies callee is a valid symbol ref to callee_op_type.
// Provides `callee_op_type getCalleeOp(SymbolTableCollection)`.
// Concrete type must have a SymbolRefAttr named `callee`.
class IfrtCallLike<string callee_op_type>
    : ParamNativeOpTrait<"xla::ifrt::IfrtCallLikeTrait", callee_op_type>;

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
