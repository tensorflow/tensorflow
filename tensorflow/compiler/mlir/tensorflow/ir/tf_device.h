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

// This file defines the tf_device dialect: it contains operations that model
// TensorFlow's actions to launch computations on accelerator devices.

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_DEVICE_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_DEVICE_H_

#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Dialect.h"  // TF:llvm-project
#include "mlir/IR/OpDefinition.h"  // TF:llvm-project

namespace mlir {
namespace tf_device {

// The TensorFlow Device dialect.
//
// This dialect contains operations to describe/launch computations on devices.
// These operations do not map 1-1 to TensorFlow ops and requires a lowering
// pass later to transform them into Compile/Run op pairs, like XlaCompile and
// XlaRun.
class TensorFlowDeviceDialect : public Dialect {
 public:
  // Constructing TensorFlowDevice dialect under an non-null MLIRContext.
  explicit TensorFlowDeviceDialect(MLIRContext* context);
};

// Declares the operations for this dialect using the generated header.
#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h.inc"

// ParallelExecute op concurrently executes variadic number of regions. Regions
// must represent separate sets of instructions to execute concurrently. In
// order to represent concurrently executed regions with dependencies, multiple
// ParallelExecute ops can be used instead. As so, regions within
// ParallelExecute op must not have control/data dependencies. While explicit
// dependencies between regions are disallowed, ParallelExecute op does not
// prevent implicit communication between regions (e.g. communication via
// send/recvs). In this case, users of ParallelExecute op must provide correct
// control dependencies between regions to guarantee correctness. Regions in
// ParallelExecute may include Resource ops. In the case where different regions
// include ops access the same resource, the users of the ParallelExecute op
// must provide mechanism (via send/recvs or via control dependencies) to
// guarantee correct ordering. Sequential ordering of ops within a region is
// guaranteed. Also, sequential ordering of ops before/after ParallelExecute ops
// are guaranteed. That is, execution of regions inside ParallelExecute op is
// blocked until all inputs to all regions are materialized and ops following
// ParallelExecute op are blocked until all regions are executed.
class ParallelExecuteOp
    : public Op<ParallelExecuteOp,
                OpTrait::SingleBlockImplicitTerminator<ReturnOp>::Impl> {
 public:
  using Op::Op;

  static void build(Builder* builder, OperationState& state, int num_regions,
                    llvm::ArrayRef<Type> output_types);

  static StringRef getOperationName() { return "tf_device.parallel_execute"; }

  Operation::result_range getRegionOutputs(unsigned region_index);
  LogicalResult verify();
  Block& getRegionWithIndex(unsigned index);
};

}  // namespace tf_device
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_DEVICE_H_
