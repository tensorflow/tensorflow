/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This is the operation interface definition file for TensorFlow.

#ifndef TF_OP_INTERFACES
#define TF_OP_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TensorFlow Layout Optimization Interfaces.
//===----------------------------------------------------------------------===//

def TF_LayoutSensitiveInterface : OpInterface<"LayoutSensitiveInterface"> {
  let description = [{
    A layout sensitive operation is one that depends on a `data_format` string
    attribute, that gives a meaning to the data inside arguments and results.

    Currently supported data formats (layouts):
      - NHWC : channels last  [batch, height, width, channels]
      - NCHW : channels first [batch, channels, height, width]

    Layout sensitive ops might have different preferred (and supported) layouts
    depending on arguments shape/type and execution device (CPU or GPU).
  }];

  let methods = [
    InterfaceMethod<
      [{Returns current operation data format (data layout).}],
      "StringRef", "data_format", (ins)
    >,
    InterfaceMethod<
      [{Returns indices of layout dependent arguments.}],
      "SmallVector<unsigned, 4>", "GetLayoutDependentArgs", (ins)
    >,
    InterfaceMethod<
      [{Returns indices of layout dependent results.}],
      "SmallVector<unsigned, 4>", "GetLayoutDependentResults", (ins)
    >,
    InterfaceMethod<
      [{Returns the optimal data layout based on the available devices.}],
      "StringRef", "GetOptimalLayout", (ins "const RuntimeDevices&":$devices)
    >,
    InterfaceMethod<
      [{Updates operation attributes and operands to account for the updated
        data format. If data format is not supported, must return failure.}],
      "LogicalResult", "UpdateDataFormat", (ins "StringRef":$data_format)
    >,
  ];

  let verify = [{
    return VerifyLayoutSensitiveInterface($_op);
  }];
}

def TF_FoldOperandsTransposeInterface : OpInterface<"FoldOperandsTransposeInterface"> {
  let description = [{
    Operation supports folding operand(s) transposes into the operation itself.

    (1) Operation might have layout dependent operands and results...

      Example:  MaxPool(Transpose($arg, $perm))
                  -> Transpose(MaxPool($arg, $perm))

    (2) ... or it might have only layout dependent operands:

      Example: Mean(Transpose($arg, $reduction_dims))
                 -> Mean($arg, Transpose($reduction_dims))
  }];

  let methods = [
    InterfaceMethod<
      [{Returns indices of layout dependent arguments.}],
      "SmallVector<unsigned, 4>", "GetLayoutDependentArgs", (ins)
    >,
    InterfaceMethod<
      [{Returns indices of layout dependent results.}],
      "SmallVector<unsigned, 4>", "GetLayoutDependentResults", (ins)
    >,
    InterfaceMethod<
      [{Updates operation attributes and operands to account for the folded
        permutation. If folding of permutation is not possible, must return
        failure.}],
      "LogicalResult", "FoldOperandsPermutation",
      (ins "ArrayRef<int64_t>":$permutation)
    >,
  ];

  let verify = [{
    return VerifyFoldOperandsTransposeInterface($_op);
  }];
}

//===----------------------------------------------------------------------===//
// TensorFlow Resource Handle Interfaces.
//===----------------------------------------------------------------------===//

def TF_ResourceHandleAllocatorInterface : OpInterface<"ResourceHandleAllocatorInterface"> {
  let description = [{
    A resource handle allocator operation is one that creates a resource handle,
    or looks up and reuses an existing resource handle.
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{Returns resource handle values and the unique ids associated with
                 the resource handles for this op. The handles should be created
                 for only the resource tensors in the results of the op. If a
                 resource handle is reused, then an existing id will be
                 returned. The order of the resource handles in the returned
                 vector are the order of those resources in the results of the
                 op.}],
      /*retTy=*/"llvm::SmallVector<ResourceHandleValueAndId, 4>",
      /*methodName=*/"GetResourceHandleValueAndIdList",
      /*args=*/(ins "llvm::SmallDenseMap<ResourceHandle, int64_t>&":$resource_handle_id_map,
                    "int64_t&":$next_id)
    >,
  ];
}

def TF_GetResourceInstanceInterface : OpInterface<"GetResourceInstanceInterface"> {
  let description = [{Returns an integer corresponding to the resource instance
                      accessed by this op}];

  let methods = [
    InterfaceMethod<
      /*desc=*/[{Returns an integer corresponding to the resource instance
                 accessed by this op. The implementation must guarantee that the
                 mapping between resource instances and integers is bijective,
                 i.e., two op instances should return the same integer if and
                 only if they access the same resource. The interface should
                 only be used for ops that access exactly one resource.}],
      /*retTy=*/"int64_t",
      /*methodName=*/"GetResourceInstanceId",
      /*args=*/(ins)
    >,
  ];
}

#endif // TF_OP_INTERFACES
