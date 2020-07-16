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
//   Status Compute(absl::Span<AbstractTensorHandle* const> grad_inputs,
//                  std::vector<AbstractTensorHandle*>* grad_outputs) override {
//     grad_outputs->resize(2);
//     (*grad_outputs)[0] = grad_inputs[0];
//     (*grad_outputs)[1] = grad_inputs[0];
//     return Status::OK();
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
  // TODO(srbs): How we support CompositeTensors e.g. IndexedSlices in
  // `grad_inputs`.
  virtual Status Compute(absl::Span<AbstractTensorHandle* const> grad_inputs,
                         std::vector<AbstractTensorHandle*>* grad_outputs) = 0;
  virtual ~GradientFunction() {}
};

// Metadata from the forward operation that is made available to the
// gradient registerer to instantiate a GradientFunction.
struct ForwardOperation {
 public:
  string op_name;
  std::vector<AbstractTensorHandle*> inputs;
  std::vector<AbstractTensorHandle*> outputs;
  AttrBuilder attrs;
  AbstractContext* ctx;
};

using GradientFunctionFactory =
    std::function<GradientFunction*(const ForwardOperation& op)>;

// Map from op name to a `GradientFunctionFactory`.
class GradientRegistry {
 public:
  Status Register(const string& op, GradientFunctionFactory factory);
  Status Lookup(const ForwardOperation& op,
                std::unique_ptr<GradientFunction>* grad_fn) const;

 private:
  absl::flat_hash_map<string, GradientFunctionFactory> registry_;
};

// Returns a unique id for the tensor which is used by the tape to build
// the gradient graph. See documentation of `TapeTensor` for more details.
int64 ToId(AbstractTensorHandle* t);

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
// This also implements `ZerosLike` and `OnesLike` to create the default
// incoming gradients for tensors which do not already have an incoming
// gradient.
class TapeTensor {
 public:
  TapeTensor(AbstractTensorHandle* handle, AbstractContext* ctx);
  TapeTensor(const TapeTensor& other);
  ~TapeTensor();

  tensorflow::int64 GetID() const;
  tensorflow::DataType GetDType() const;

  AbstractTensorHandle* OnesLike() const;
  AbstractTensorHandle* ZerosLike() const;

 private:
  AbstractTensorHandle* handle_;
  // The context where OnesLike and ZerosLike ops are to be created.
  AbstractContext* ctx_;
};

// Vector space for actually computing gradients. Implements methods for calling
// the backward function with incoming gradients and returning the outgoing
// gradient and for performing gradient aggregation.
// See `tensorflow::eager::VSpace` for more details.
class TapeVSpace
    : public eager::VSpace<AbstractTensorHandle, GradientFunction, TapeTensor> {
 public:
  explicit TapeVSpace(AbstractContext* ctx) : ctx_(ctx) {}
  ~TapeVSpace() override {}

  // Returns the number of elements in the gradient tensor.
  int64 NumElements(AbstractTensorHandle* tensor) const override;

  // Consumes references to the tensors in the gradient_tensors list and returns
  // a tensor with the result.
  AbstractTensorHandle* AggregateGradients(
      gtl::ArraySlice<AbstractTensorHandle*> gradient_tensors) const override;

  // Calls the passed-in backward function.
  Status CallBackwardFunction(
      GradientFunction* backward_function,
      const std::vector<int64>& unneeded_gradients,
      gtl::ArraySlice<AbstractTensorHandle*> output_gradients,
      std::vector<AbstractTensorHandle*>* result) const override;

  // Looks up the ID of a Gradient.
  int64 TensorId(AbstractTensorHandle* tensor) const override;

  // Converts a Gradient to a TapeTensor.
  TapeTensor TapeTensorFromGradient(AbstractTensorHandle* g) const override;

  void MarkAsResult(AbstractTensorHandle* gradient) const override;

  void DeleteGradient(AbstractTensorHandle* gradient) const override;

 private:
  // The context where the aggregation op `Add` is to be created.
  AbstractContext* ctx_;
};

// A tracing/immediate-execution agnostic tape.
using Tape = tensorflow::eager::GradientTape<AbstractTensorHandle,
                                             GradientFunction, TapeTensor>;

}  // namespace gradients
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_GRADIENTS_H_
