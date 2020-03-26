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

// WARNING: This Interface is experimental, DO NOT USE.

// This is the Target Hardware operation interfacea definition file
// for TensorFlow Lite.

#ifndef TFL_TARGET_HARDWARE_OP_INTERFACES
#define TFL_TARGET_HARDWARE_OP_INTERFACES

def TFL_CpuTargetOp : OpInterface<"CpuOpTargetInterface"> {
  let description = [{
    Interface for ops to run on CPU.
  }];

  let methods = [
    InterfaceMethod<
      [{Returns the cost of running this op on CPU.}],
      // TODO(karimnosseir): Change to return Cost object instead.
      "double", "GetOpCost", (ins "mlir::Operation*":$op_to_check), [{
        // TODO(karimnosseir): Consider changing to another way that doesn't
        // rely on template param name.
        return TFL::TFLiteCostEstimator<ConcreteOp, TFL::hardware::CPU>::GetCost(op_to_check);
      }]
    >,
    InterfaceMethod<
      [{Returns whether this op can be run on CPU.}],
      "bool", "IsSupported", (ins "mlir::Operation*":$op_to_check), [{
        // TODO(karimnosseir): Consider changing to another way that doesn't
        // rely on template param name.
        return TFL::TFLiteCostEstimator<ConcreteOp, TFL::hardware::CPU>::IsSupported(op_to_check);
      }]
    >,
  ];
}

def TFL_GpuTargetOp : OpInterface<"GpuOpTargetInterface"> {
  let description = [{
    Interface for ops to run on GPU.
  }];

  let methods = [
    InterfaceMethod<
      [{Returns the cost of running this op on GPU.}],
      // TODO(karimnosseir): Change to return Cost object instead.
      "double", "GetOpCost", (ins "Operation*":$op_to_check), [{
        // TODO(karimnosseir): Consider changing to another way that doesn't
        // rely on template param name.
        return TFL::TFLiteCostEstimator<ConcreteOp, TFL::hardware::GPU>::GetCost(op_to_check);
      }]
    >,
    InterfaceMethod<
      [{Returns whether this op can be run on GPU.}],
      "bool", "IsSupported", (ins "Operation*":$op_to_check), [{
        // TODO(karimnosseir): Consider changing to another way that doesn't
        // rely on template param name.
        return TFL::TFLiteCostEstimator<ConcreteOp, TFL::hardware::GPU>::IsSupported(op_to_check);
      }]
    >,
  ];
}

#endif // TFL_TARGET_HARDWARE_OP_INTERFACES
