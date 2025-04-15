/* Copyright 2023 The StableHLO Authors.

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

#ifndef STABLEHLO_EXT_DIALECT_STABLEHLO_OPS_H
#define STABLEHLO_EXT_DIALECT_STABLEHLO_OPS_H

// This file supports XLA-specific extensions with the StableHLO opset.
// These are currently implemented as custom-call pseudo-ops, but it is likely
// that they will be upstreamed to StableHLO or CHLO in the future.

#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir {
namespace stablehlo_ext {

// The DynamicReduceWindowOp experiment provides a dynamic version of
// ReduceWindowOp. Once the dynamism RFC is figured out, we expect to have an
// upstream representation for this notion.
//
// Within this experiment, DynamicReduceWindowOp is represented via the
// `stablehlo.custom_call @stablehlo.dynamic_reduce_window` custom call.
// This custom call has the following operands which represent a dynamic version
// of operands and attributes of ReduceWindowOp:
//   * [0:N]   => inputs
//   * [N:2*N] => init_values
//   * [-5]    => window_dimensions
//   * [-4]    => window_strides
//   * [-3]    => base_dilations
//   * [-2]    => window_dilations
//   * [-1]    => padding
// Additionally, to represent the body of DynamicReduceWindowOp, the custom call
// has a satellite function attached to the custom call via called_computations.
//
// Semantics of DynamicReduceWindowOp are inherited from semantics of
// https://github.com/openxla/stablehlo/blob/main/docs/spec.md#reduce_window
// with the following exceptions:
//   1) All tensor constants, i.e. window_dimensions, window_strides,
//      base_dilations, window_dilations and padding, become tensors of
//      integer type.
//   2) As a result, some of the constraints can no longer be validated
//      statically. However, this operation still expects these constraints
//      to hold dynamically, and if they don't hold, the behavior is undefined.
class DynamicReduceWindowOpAdaptor {
 public:
  DynamicReduceWindowOpAdaptor(stablehlo::CustomCallOp op) : op_(op) {}
  operator Operation*() { return op_; }
  Operation* operator->() { return op_; }

  // Same accessors as for stablehlo::ReduceWindowOp, except that all the
  // std::optional<DenseIntElementsAttr> attributes have turned into values.
  // These accessors assume that the operation is well-formed (i.e. that it
  // can pass verification).
  ValueRange getInputs();
  ValueRange getInitValues();
  TypedValue<ShapedType> getWindowDimensions();
  TypedValue<ShapedType> getWindowStrides();
  TypedValue<ShapedType> getBaseDilations();
  TypedValue<ShapedType> getWindowDilations();
  TypedValue<ShapedType> getPadding();
  Region& getBody();
  ValueRange getResults();

  // Verifies the constraints documented above.
  // Emits errors if errors are detected.
  LogicalResult verify();

 private:
  stablehlo::CustomCallOp op_;
};

// Wraps a custom call in a DynamicReduceWindowOpAdaptor.
// Fails if the call_target_name of the custom call doesn't match
// "stablehlo.dynamic_reduce_window".
std::optional<DynamicReduceWindowOpAdaptor> getDynamicReduceWindowOp(
    stablehlo::CustomCallOp op);

// The DynamicRngBitGeneratorOp experiment provides a dynamic version of
// RngBitGeneratorOp. Once the dynamism RFC is figured out, we expect to have an
// upstream representation for this notion.
//
// Within this experiment, DynamicRngBitGeneratorOp is represented via the
// `stablehlo.custom_call @stablehlo.dynamic_rng_bit_generator` custom call.
// This custom call has the regular operand of RngBitGeneratorOp plus an
// additional `output_shape` operand that determines the shape of the output:
//   * [0] => initial_state
//   * [1] => output_shape
//
// Semantics of DynamicRngBitGeneratorOp are inherited from semantics of
// https://github.com/openxla/stablehlo/blob/main/docs/spec.md#rng_bit_generator
// extended with an additional input (I3) and an additional constraint (C3):
//
// #### Inputs
//
// | Label | Name            | Type                                         |
// |-------|-----------------|----------------------------------------------|
// | (I1)  | `rng_algorithm` | enum of `DEFAULT`, `THREE_FRY`, and `PHILOX` |
// | (I2)  | `initial_state` | 1-dimensional tensor of type `ui64`          |
// | (I3)  | `output_shape`  | 1-dimensional tensor of integer type         |
//
// #### Outputs
//
// | Name           | Type                                     |
// |----------------|------------------------------------------|
// | `output_state` | 1-dimensional tensor of type `ui64`      |
// | `output`       | tensor of integer or floating-point type |
//
// #### Constraints
//
// * (C1) `type(initial_state) = type(output_state)`.
// * (C2) `size(initial_state)` is defined as:
//   * implementation-defined if `rng_algorithm = DEFAULT`.
//   * `2` if `rng_algorithm = THREE_FRY`.
//   * `2` or `3` if `rng_algorithm = PHILOX`.
// * (C3) `shape(output) = output_shape`.
class DynamicRngBitGeneratorOpAdaptor {
 public:
  DynamicRngBitGeneratorOpAdaptor(stablehlo::CustomCallOp op) : op_(op) {}
  operator Operation*() { return op_; }
  Operation* operator->() { return op_; }

  // Same accessors as for stablehlo::RngBitGeneratorOp, extended with the
  // additional `output_shape` operand.
  // These accessors assume that the operation is well-formed (i.e. that it
  // can pass verification).
  stablehlo::RngAlgorithm getRngAlgorithm();
  TypedValue<ShapedType> getInitialState();
  TypedValue<ShapedType> getOutputShape();
  TypedValue<ShapedType> getOutputState();
  TypedValue<ShapedType> getOutput();

  // Verifies the constraints documented above.
  // Emits errors if errors are detected.
  LogicalResult verify();

 private:
  stablehlo::CustomCallOp op_;
};

// Wraps a custom call in a DynamicRngBitGeneratorOpAdaptor.
// Fails if the call_target_name of the custom call doesn't match
// "stablehlo.dynamic_rng_bit_generator".
std::optional<DynamicRngBitGeneratorOpAdaptor> getDynamicRngBitGeneratorOp(
    stablehlo::CustomCallOp op);

// The DynamicTopKOp experiment provides a dynamic version of
// TopKOp. Once the dynamism RFC is figured out, we expect to have an
// upstream representation for this notion.
//
// Within this experiment, DynamicTopKOp is represented via the
// `stablehlo.custom_call @stablehlo.dynamic_top_k` custom call.
// This custom call has the regular operand of TopKOp plus an
// additional `k` operand that determines the shape of the output.
//
// Semantics of DynamicTopKOp are inherited from semantics of Chlo.TopKOp.
//
// #### Inputs
//
// | Label | Name            | Type                                         |
// |-------|-----------------|----------------------------------------------|
// | (I1)  | `operand`       | tensor of integer or floating-point type     |
// | (I2)  | `k`             | 0-dimensional tensor of integer or index type|
//
// #### Outputs
//
// | Name           | Type                                     |
// |----------------|------------------------------------------|
// | `values`       | tensor of integer or floating-point type |
// | `indices`      | tensor of si32 type                      |
//
// #### Constraints
//
// * (C1) `shape(values)[:-1] = shape(operand)[:-1]`
// * (C2) `element_type(values) = element_type(operand)`
// * (C3) `shape(values)[-1] <= shape(operand)[-1]`
// * (C4) `shape(indices) = shape(values)`
class DynamicTopKOpAdaptor {
 public:
  DynamicTopKOpAdaptor(stablehlo::CustomCallOp op) : op_(op) {}
  operator Operation*() { return op_; }
  Operation* operator->() { return op_; }

  // These accessors assume that the operation is well-formed (i.e. that it
  // can pass verification).
  TypedValue<ShapedType> getOperand();
  TypedValue<ShapedType> getK();
  TypedValue<ShapedType> getValues();
  TypedValue<ShapedType> getIndices();

  // Verifies the constraints documented above.
  // Emits errors if errors are detected.
  LogicalResult verify();

 private:
  stablehlo::CustomCallOp op_;
};

// Wraps a custom call in a DynamicTopKOpAdaptor.
// Fails if the call_target_name of the custom call doesn't match
// "stablehlo.dynamic_top_k".
std::optional<DynamicTopKOpAdaptor> getDynamicTopKOp(
    stablehlo::CustomCallOp op);

// The DynamicApproxTopKOp experiment provides a dynamic version of
// ApproxTopKOp.
//
// Within this experiment, DynamicApproxTopKOp is represented via the
// `stablehlo.custom_call @stablehlo.dynamic_approx_top_k` custom call.
// This custom call has the regular operands of ApproxTopKOp plus an
// additional `k` operand that determines the shape of the output.
//
// Semantics of DynamicApproTopKOp are inherited from semantics of ApproxTopKOp.
//
// #### Inputs
//
// | Label | Name            | Type                                         |
// |-------|-----------------|----------------------------------------------|
// | (I1)  | `inputs`        | N tensors of integer or floating-point type   |
// | (I2)  | `initial_values`| N 0-dimensional tensors of same element type |
// |       |                 | as the corresponding input element type      |
// | (I3)  | `k`             | 0-dimensional tensor of integer or index type|
//
// #### Attributes
//
// * api_version: always 2 if present
// * has_side_effect: always False if present
// * called_computations: the comparator for scoring entries
// * mhlo.backend_config: does not include `top_k` and includes:
//   * `reduction_dim`
//
// #### Outputs
//
// | Name           | Type                                                |
// |----------------|-----------------------------------------------------|
// | `outputs`      | N tensor of same type as the corresponding input    |
//
// #### Constraints
//
// * (C1) the `mhlo.backend_config` attribute does not contain `top_k`
// * (C2) the `mhlo.backend_config` attribute contains `reduction_dim`
// * (C3) len(inputs) == len(initial_values) == len(outputs)
// * (C4) inputs have ranked type and have the same shape
// * (C5) initial_values have ranked type and have rank 0
// * (C6) initial_values have the same element type as the corresponding input
// * (C7) outputs have same shape
// * (C8) outputs have the same element type as the corresponding input
// * (C9) 0 <= reduction_dim < rank(inputs[0])
// * (C10) shape(inputs[0])[i] == shape(outputs[0])[i] except for i ==
// reduction_dim

class DynamicApproxTopKOpAdaptor {
 public:
  DynamicApproxTopKOpAdaptor(stablehlo::CustomCallOp op) : op_(op) {}
  operator Operation*() { return op_; }
  Operation* operator->() { return op_; }

  // These accessors assume that the operation is well-formed (i.e. that it
  // can pass verification).
  size_t getNumInputs();
  TypedValue<ShapedType> getInput(size_t idx);
  TypedValue<ShapedType> getInitialValue(size_t idx);
  TypedValue<ShapedType> getK();

  TypedValue<ShapedType> getOutput(size_t idx);

  // Verifies the constraints documented above.
  // Emits errors if errors are detected.
  LogicalResult verify();

 private:
  stablehlo::CustomCallOp op_;
};

// Wraps a custom call in a DynamicApproxTopKOpAdaptor.
// Fails if the call_target_name of the custom call doesn't match
// "stablehlo.dynamic_approx_top_k".
std::optional<DynamicApproxTopKOpAdaptor> getDynamicApproxTopKOp(
    stablehlo::CustomCallOp op);

}  // namespace stablehlo_ext
}  // namespace mlir

#endif  // STABLEHLO_EXT_DIALECT_STABLEHLO_OPS_H
