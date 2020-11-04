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
struct Context {
 public:
  AbstractContext* ctx;
};

class IncomingGradients {
 public:
  virtual AbstractTensorHandle* operator[](int i) const = 0;
  virtual size_t size() const = 0;
  virtual ~IncomingGradients() {}
};

class GradientFunction {
 public:
  // TODO(srbs): How we support CompositeTensors e.g. IndexedSlices in
  // `grad_inputs`.
  virtual Status Compute(Context* ctx, const IncomingGradients& grad_inputs,
                         std::vector<AbstractTensorHandle*>* grad_outputs) = 0;
  virtual ~GradientFunction() {}
};

// Metadata from the forward operation that is made available to the
// gradient registerer to instantiate a BackwardFunction.
struct ForwardOperation {
 public:
  string op_name;
  std::vector<AbstractTensorHandle*> inputs;
  std::vector<AbstractTensorHandle*> outputs;
  AttrBuilder attrs;
};

// Interface for building default zeros gradients for op outputs which are
// missing incoming gradients. Custom implementations of this can be used to
// control which of the forward op's output tensors/their metadata needs to
// be kept around in memory to build the default zeros grad.
//
// Some common helper implementations are provided below.
class DefaultGradientFunction {
 public:
  virtual AbstractTensorHandle* get(
      Context* ctx, absl::Span<AbstractTensorHandle* const> grad_inputs,
      int i) = 0;
  virtual ~DefaultGradientFunction() {}
};

// Returns zeros for any `nullptr` in `grad_inputs`.
//
// This may require keeping track of all of forward op's output
// tensors and hence may incur a higher memory footprint. Use sparingly.
//
// Multiple calls to `AllZerosDefaultGradients::get` return the same tensor
// handle.
//
// The destructor of this class `Unref`'s any cached tensor handles so users of
// those tensor handles should `Ref` them in order to keep them alive if needed.
class AllZerosDefaultGradients : public DefaultGradientFunction {
 public:
  explicit AllZerosDefaultGradients(const ForwardOperation& op);
  AbstractTensorHandle* get(Context* ctx,
                            absl::Span<AbstractTensorHandle* const> grad_inputs,
                            int i) override;

 private:
  // TODO(srbs): We do not always need to keep the tensors around. In immediate
  // execution mode we just need to store the shape and dtype. During tracing
  // we may need to keep the tensor around if the shape is not full defined.
  std::vector<AbstractTensorHandle*> outputs_;
  std::vector<AbstractTensorHandlePtr> cached_default_grads_;
};

// Passes through `grad_inputs` as-is. The `GradientFunction`
// will be expected to deal with nullptr in `grad_inputs` if any.
class PassThroughDefaultGradients : public DefaultGradientFunction {
 public:
  explicit PassThroughDefaultGradients(const ForwardOperation& op);
  AbstractTensorHandle* get(Context* ctx,
                            absl::Span<AbstractTensorHandle* const> grad_inputs,
                            int i) override;
};

// A `BackwardFunction` wraps a `GradientFunction` and a
// `DefaultGradientFunction`. Both are owned by this class' instance.
class BackwardFunction {
 public:
  BackwardFunction(GradientFunction* gradient_function,
                   DefaultGradientFunction* default_gradients)
      : gradient_function_(gradient_function),
        default_gradients_(default_gradients) {}
  GradientFunction* GetGradientFunction() { return gradient_function_.get(); }
  DefaultGradientFunction* GetDefaultGradientFunction() {
    return default_gradients_.get();
  }

 private:
  std::unique_ptr<GradientFunction> gradient_function_;
  std::unique_ptr<DefaultGradientFunction> default_gradients_;
};

using BackwardFunctionFactory =
    std::function<BackwardFunction*(const ForwardOperation& op)>;

// Map from op name to a `BackwardFunctionFactory`.
class GradientRegistry {
 public:
  Status Register(const string& op,
                  BackwardFunctionFactory backward_function_factory);
  Status Lookup(const ForwardOperation& op,
                std::unique_ptr<BackwardFunction>* backward_function) const;

 private:
  absl::flat_hash_map<string, BackwardFunctionFactory> registry_;
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

  tensorflow::int64 GetID() const;
  tensorflow::DataType GetDType() const;

  AbstractTensorHandle* ZerosLike() const;

  AbstractTensorHandle* GetHandle() const;

 private:
  AbstractTensorHandle* handle_;
};

// Vector space for actually computing gradients. Implements methods for calling
// the backward function with incoming gradients and returning the outgoing
// gradient and for performing gradient aggregation.
// See `tensorflow::eager::VSpace` for more details.
class TapeVSpace
    : public eager::VSpace<AbstractTensorHandle, BackwardFunction, TapeTensor> {
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
      BackwardFunction* backward_function,
      const std::vector<int64>& unneeded_gradients,
      gtl::ArraySlice<AbstractTensorHandle*> output_gradients,
      std::vector<AbstractTensorHandle*>* result) const override;

  // Builds a tensor filled with ones with the same shape and dtype as `t`.
  Status BuildOnesLike(const TapeTensor& t,
                       AbstractTensorHandle** result) const override;

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
//
// Gradient functions defined for this library support handling null incoming
// gradients. `Tape::ComputeGradient` should be called with
// `build_default_zeros_grads=false`. Calling with
// `build_default_zeros_grads=true` (the default) is equivalent but just results
// in extra work because `TapeTensor::ZerosLike` returns a `nullptr` anyway.
using Tape = tensorflow::eager::GradientTape<AbstractTensorHandle,
                                             BackwardFunction, TapeTensor>;

}  // namespace gradients
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EAGER_GRADIENTS_H_
