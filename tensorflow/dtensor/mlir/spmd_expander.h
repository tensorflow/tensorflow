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

#ifndef TENSORFLOW_DTENSOR_MLIR_SPMD_EXPANDER_H_
#define TENSORFLOW_DTENSOR_MLIR_SPMD_EXPANDER_H_

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "tensorflow/core/framework/registration/registration.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"

namespace tensorflow {
namespace dtensor {

// Base class for handling SPMD expansion of a MLIR TF Operation.
class SPMDExpanderBase {
 public:
  virtual ~SPMDExpanderBase() = default;

  // Converts `op` to a SPMD expanded form. SPMD expansion logic is
  // a function of op type, op output's layout, and layout of op's
  // inputs. Must return the `op` that is expanded as the final return value.
  virtual StatusOr<mlir::Operation*> ExpandOp(mlir::Operation* op) = 0;

  // Layout propagation functions.
  //
  // During the layout algorithm, for each op output we compute a layout by
  // merging the current layout request from the op producing the output and the
  // layout requests from the ops consuming the output. These merged layouts
  // represent the current state of layouts over the entire mlir module.
  //
  // For an op, if any of the merged layouts for the inputs or output are
  // updated, the ComputeLayoutForward and ComputeLayoutBackward functions will
  // be called with all the updated layout maps populated.
  //
  // ComputeLayoutForward should take the input layouts and determine which
  // output layout these inputs would produce. Likewise, ComputeLayoutBackward
  // should take the output layouts and determine the what layouts to propagate
  // to the inputs.
  //
  // In both cases the functions should choose layouts that reduce the amount of
  // cross device communication for the op.
  //
  // ComputeLayoutForward should not take into account the current output
  // layout(s) when computing the new ones. The merge algorithm will decide what
  // to do. There are only a very few cases where the current output layout may
  // need to propagated again, in which case those ops can override the
  // expanded ComputeLayout* functions. This similarly applies to
  // ComputeLayoutBackward.
  //
  // Note that for some ops, where the input layout does not determine output
  // layout (and visa versa), it is acceptable to either return a replicated
  // layout. E.g. for tf.Fill, ComputeLayoutForward can return a replicated
  // output layout and if a consumer requests a more sharded layout, then the
  // layout algorithm will merge the requests, resulting in the more sharded
  // layout.

  // Computes output layout(s) of `op` based on the current `input_layouts`
  // inferred from inputs of `op`. The `input_layouts` parameter maps input
  // indices to the corresponding layouts. It may be empty if the op has no
  // operands or if no input layouts have been inferred yet.
  virtual StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutForward(
      mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts);

  // Computes output layout(s) of `op` based on the current `input_layouts` and
  // `output_layouts` inferred from the inputs and outputs of `op`. Both
  // parameters maps input/output indices to the corresponding layouts. Either
  // may be empty.
  //
  // NOTE: The other ComputeLayoutForward function should be preferred since in
  // most cases the output layouts are only computed based on the input layouts.
  virtual StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutForward(
      mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts,
      const llvm::DenseMap<int, Layout>& output_layouts);

  // Computes input layout(s) of `op` based on the current `output_layouts`
  // inferred from outputs of `op`. The `output_layouts` parameter maps output
  // indices to the corresponding layouts. It may be empty if the op has no
  // outputs or if no output layouts have been inferred yet.
  virtual StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutBackward(
      mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts);

  // Computes input layout(s) of `op` based on the current `output_layouts` and
  // `input_layouts` inferred from the outputs and inputs of `op`. Both
  // parameters maps input/output indices to the corresponding layouts. Either
  // may be empty.
  //
  // NOTE: The other ComputeLayoutBackward function should be preferred since in
  // most cases the input layouts are only computed based on the output layouts.
  virtual StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutBackward(
      mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts,
      const llvm::DenseMap<int, Layout>& output_layouts);

  // Run ExpandOp() and set layout from the computed layout from original op.
  // Returns the expanded op in output.
  absl::Status ExpandOpAndSetLayout(mlir::Operation* op,
                                    mlir::Operation** output);
};

// Computes the SPMD expansion for `op`.
//
// Prior to this call, all inputs to `op` have been lowered to local operations
// & shapes. The lowered op must emit a type compatible with the local shape.
absl::Status RunSPMDExpansion(mlir::Operation* op, mlir::Operation** output);

// A registry of SPMD expanders. This map is statically stored and initialized
// with all the registered SPMD expanders.
class SPMDExpanderRegistry {
 public:
  ~SPMDExpanderRegistry() = default;

  // A singleton available at startup.
  static SPMDExpanderRegistry* Global();

  // Returns true if the op name is supported.
  // The name includes the "tf." prefix.
  bool IsOpSupported(const std::string& full_op_name) {
    return GetPropagateFnForFullOpName(full_op_name) != nullptr;
  }

  // Returns the expansion for the given operation (or nullptr if no expansion
  // has been registered).
  SPMDExpanderBase* GetPropagateFnForOp(mlir::Operation* op);

  // Returns the expansion for the given operation (or nullptr if no expansion
  // has been registered). The name is the full name with "tf." prefix.
  SPMDExpanderBase* GetPropagateFnForFullOpName(
      const std::string& full_op_name);

  // Registers an expander for the provided opName.
  InitOnStartupMarker RegisterPropagateFn(
      std::string opName, std::unique_ptr<SPMDExpanderBase> prop);

 private:
  absl::flat_hash_map<std::string, std::unique_ptr<SPMDExpanderBase>>
      op_to_propagate_fn_map_;
};

#define REGISTER_SPMD(name, op, prop, ...)                             \
  static ::tensorflow::InitOnStartupMarker const spmd_##name =         \
      InitOnStartupMarker{}                                            \
      << dtensor::SPMDExpanderRegistry::Global()->RegisterPropagateFn( \
             mlir::op::getOperationName().str(),                       \
             std::make_unique<prop>(__VA_ARGS__))

// Register the SPMD expander by ops string name.
// Comparing to REGISTER_SPMD, this macro allows registration for custom ops
// that isn't a MLIR op. Note that the op_name should start with "tf.", e.g
// REGISTER_SPMD_BY_OP_NAME(Foo, "tf.foo", expander_class).
#define REGISTER_SPMD_BY_OP_NAME(expander_name, op_name, prop, ...)     \
  static ::tensorflow::InitOnStartupMarker const spmd_##expander_name = \
      InitOnStartupMarker{}                                             \
      << dtensor::SPMDExpanderRegistry::Global()->RegisterPropagateFn(  \
             op_name, std::make_unique<prop>(__VA_ARGS__))

}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_MLIR_SPMD_EXPANDER_H_
