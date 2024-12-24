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

#ifndef TENSORFLOW_C_EAGER_GRADIENTS_H_
#define TENSORFLOW_C_EAGER_GRADIENTS_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/c/eager/abstract_context.h"
#include "tensorflow/c/eager/abstract_tensor_handle.h"
#include "tensorflow/c/eager/tape.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"

namespace tensorflow {
namespace gradients {

// =============== Experimental C++ API for computing gradients ===============

// Sample gradient function:
//
// class AddGradientFunction : public GradientFunction {
//  public:
//   Status Compute(Context* ctx,
//                  absl::Span<AbstractTensorHandle* const> grad_inputs,
//                  absl::Span<AbstractTensorHandle*> grad_outputs) override {
//     grad_outputs[0] = grad_inputs[0];
//     grad_outputs[1] = grad_inputs[0];
//     grad_outputs[0]->Ref();
//     grad_outputs[1]->Ref();
//     return OkStatus();
//   }
//   ~AddGradientFunction() override {}
// };
//
// GradientFunction* AddRegisterer(const ForwardOperation& op) {
//   // More complex gradient functions can use inputs/attrs etc. from the
//   // forward `op`.
//   return new AddGradientFunction;
// }
//
// Status RegisterGradients(GradientRegistry* registry) {
//   return registry->Register("Add", AddRegisterer);
// }
class GradientFunction {
 public:
  virtual absl::Status Compute(
      AbstractContext* ctx,
      absl::Span<AbstractTensorHandle* const> grad_outputs,
      absl::Span<AbstractTensorHandle*> grad_inputs) = 0;
  virtual ~GradientFunction() {}
};

// Metadata from the forward operation that is made available to the
// gradient registerer to instantiate a GradientFunction.
struct ForwardOperation {
 public:
  string op_name;
  std::vector<AbstractTensorHandle*> inputs;
  std::vector<AbstractTensorHandle*> outputs;
  std::vector<int64_t> skip_input_indices;
  AttrBuilder attrs;
};

using GradientFunctionFactory =
    std::function<GradientFunction*(const ForwardOperation& op)>;

// Map from op name to a `GradientFunctionFactory`.
class GradientRegistry {
 public:
  absl::Status Register(const string& op,
                        GradientFunctionFactory gradient_function_factory);
  absl::Status Lookup(
      const ForwardOperation& op,
      std::unique_ptr<GradientFunction>* gradient_function) const;

 private:
  absl::flat_hash_map<string, GradientFunctionFactory> registry_;
};

// TODO(srbs): Figure out if we can avoid declaring this in the public header.
// Wrapper for a tensor output of an operation executing under a tape.
//
// `GetID` returns a unique id for the wrapped tensor which is used to maintain
// a map (`tensorflow::eager::TensorTape`) from the wrapped tensor to the id of
// the op that produced it (or -1 if this tensor was watched using
// `GradientTape::Watch`.) The op_id is simply a unique index assigned to each
// op executed under the tape. A separate map (`tensorflow::eager::OpTape`)
// maintains the map from `op_id` to a `OpTapeEntry` which stores the `op_type`,
// inputs and outputs and the gradient function These data structures combined
// allow us to trace the data dependencies between operations and hence compute
// gradients.
//
// `ZerosLike` is not expected to be called and returns a nullptr. The creation
// of default zeros grads is handled by the `DefaultGradientFunction` registered
// for each op.
// TODO(srbs): We need to define `ZerosLike` here to keep the compiler happy.
// Figure out a way to avoid this.
// TODO(srbs): Should ZerosLike check-fail instead of returning nullptr?
class TapeTensor {
 public:
  explicit TapeTensor(AbstractTensorHandle* handle);
  TapeTensor(const TapeTensor& other);
  ~TapeTensor();

  int64_t GetID() const;
  tensorflow::DataType GetDType() const;

  AbstractTensorHandle* ZerosLike() const;

  AbstractTensorHandle* GetHandle() const;

 private:
  AbstractTensorHandle* handle_;
};

// A tracing/immediate-execution agnostic tape.
//
// Gradient functions defined for this tape must support handling null incoming
// gradients.
class Tape : protected eager::GradientTape<AbstractTensorHandle,
                                           GradientFunction, TapeTensor> {
 public:
  using GradientTape<AbstractTensorHandle, GradientFunction,
                     TapeTensor>::GradientTape;
  // Returns whether the tape is persistent, i.e., whether the tape will hold
  // onto its internal state after a call to `ComputeGradient`.
  using GradientTape<AbstractTensorHandle, GradientFunction,
                     TapeTensor>::IsPersistent;

  // Adds this tensor to the list of watched tensors.
  //
  // This is a no-op if the tensor is already being watched either from an
  // earlier call to `GradientTape::Watch` or being an output of an op with
  // watched inputs.
  void Watch(const AbstractTensorHandle*);
  // Records an operation with given inputs and outputs
  // on the tape and marks all its outputs as watched if at
  // least one input of the op is watched and has a trainable dtype.
  // op_name is optional and is used for debugging only.
  void RecordOperation(absl::Span<AbstractTensorHandle* const> inputs,
                       absl::Span<AbstractTensorHandle* const> outputs,
                       GradientFunction* gradient_function,
                       const string& op_name = "");
  // Returns whether any tensor in a list of tensors is being watched and has
  // a trainable dtype.
  bool ShouldRecord(
      absl::Span<const AbstractTensorHandle* const> tensors) const;
  // Unwatches this tensor on the tape. Mainly used for cleanup when deleting
  // eager tensors.
  void DeleteTrace(const AbstractTensorHandle*);

  // Consumes the internal state of the tape (so cannot be called more than
  // once unless the tape is persistent) and produces the gradient of the target
  // tensors with respect to the source tensors. The output gradients are used
  // if not empty and not null. The result is populated with one tensor per
  // target element.
  absl::Status ComputeGradient(
      AbstractContext* ctx, absl::Span<AbstractTensorHandle* const> targets,
      absl::Span<AbstractTensorHandle* const> sources,
      absl::Span<AbstractTensorHandle* const> output_gradients,
      absl::Span<AbstractTensorHandle*> result);
};

}  // namespace gradients
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_GRADIENTS_H_
