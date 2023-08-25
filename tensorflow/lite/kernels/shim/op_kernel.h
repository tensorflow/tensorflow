/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_SHIM_OP_KERNEL_H_
#define TENSORFLOW_LITE_KERNELS_SHIM_OP_KERNEL_H_

// This file defines a shim layer on top of TF and TFLite custom op APIs.
// The goal is for a custom op to be written once and used for both runtimes
//
// It consists of two pieces:
//   * A set of *context* interfaces:
//     ** InvokeContext, InitContext, ShapeInferenceContext
//     These are passed on to the custom op implementation to read/write
//     tensors, etc.
//
//   * An OpKernelShim interface:
//     This is what a custom op needs to implement. By using that interface the
//     custom op can then be easily converted to both a TF op kernel and a
//     TFLite op kernel.

#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/variant.h"
#include "tensorflow/lite/kernels/shim/shape.h"
#include "tensorflow/lite/kernels/shim/status_macros.h"
#include "tensorflow/lite/kernels/shim/tensor_view.h"

namespace tflite {
namespace shim {

// List of the TF custom op APIs this shim library is abstracting away.
//
// This enum is used as the template parameter in various places in
// order to pick the correct set of types (eg. TfInvokeContext vs.
// TfLiteInvokeContext) in the op implementation.
enum class Runtime { kTf, kTfLite };

// TensorView or error
using TensorViewOr = absl::StatusOr<std::unique_ptr<TensorView>>;
using ConstTensorViewOr = absl::StatusOr<std::unique_ptr<const TensorView>>;

// Below are the interfaces for various "Context" objects to abstract away the
// TF and TFLite differences.
//
// The interfaces are static and use the CRTP pattern instead of virtual
// methods.

// The attribute dictionary passed to the op
using AttrValue = absl::variant<bool, int64_t, float, absl::string_view>;

// The interface for available methods during an op kernel initialization
template <typename SubType>
class InitContext {
 public:
  // Read the given attribute and populate the given value.
  template <typename AttrType>
  absl::Status GetAttr(const std::string& attr_name, AttrType* value) const;

 protected:
  // Read a given attribute or return error
  absl::StatusOr<AttrValue> GetAttr(const std::string& attr_name) const {
    return static_cast<const SubType&>(*this).GetAttr(attr_name);
  }
};

// The interface for available methods during an op kernel invocation
template <typename SubType>
class InvokeContext {
 public:
  // Read an input tensor
  ConstTensorViewOr GetInput(const int idx) const {
    return static_cast<const SubType&>(*this).GetInput(idx);
  }
  // Get a mutable output tensor
  TensorViewOr GetOutput(const int idx, const Shape& shape) const {
    return static_cast<const SubType&>(*this).GetOutput(idx, shape);
  }
  // Number of input tensors
  int NumInputs() const {
    return static_cast<const SubType&>(*this).NumInputs();
  }
  // Number of output tensors
  int NumOutputs() const {
    return static_cast<const SubType&>(*this).NumOutputs();
  }
};

// The interface for available methods during shape inference
template <typename SubType>
class ShapeInferenceContext {
 public:
  // Read an input tensor shape
  ShapeOr GetInputShape(const int idx) const {
    return static_cast<const SubType&>(*this).GetInputShape(idx);
  }
  // Set an output tensor shape
  absl::Status SetOutputShape(const int idx, const Shape& shape) {
    return static_cast<SubType&>(*this).SetOutputShape(idx, shape);
  }
  // Read an input tensor during shape inference
  ConstTensorViewOr GetInputTensor(const int idx) const {
    return static_cast<const SubType&>(*this).GetInputTensor(idx);
  }
  // Number of input tensors
  int NumInputs() const {
    return static_cast<const SubType&>(*this).NumInputs();
  }
  // Number of output tensors
  int NumOutputs() const {
    return static_cast<const SubType&>(*this).NumOutputs();
  }
  // Read the given attribute and populate the given value.
  template <typename AttrType>
  absl::Status GetAttr(const std::string& attr_name, AttrType* value) const;

 protected:
  // Read a given attribute or return error
  absl::StatusOr<AttrValue> GetAttr(const std::string& attr_name) const {
    return static_cast<const SubType&>(*this).GetAttr(attr_name);
  }
};

// Maps the Runtime to the correct context types.
// eg. ContextTypeForRuntime<Runtime::Tf>  -->
//       { TfInitContext, TfInvokeContext, TfShapreInferenceContext }
template <Runtime Rt>
struct ContextTypeForRuntime {
  // * Init
  // * Invoke
  // * ShapeInference
};

// A Tensorflow operation interface which is then adapted to both TF and TFLite
// runtimes.
//
// Example usage:
//
//   template<Runtime R>
//   class MyOp : public OpKernelShim<MyOp, R> {
//
//     // Attributes declaration
//     // (syntax: https://www.tensorflow.org/guide/create_op)
//     static std::vector<std::string> Attrs();
//
//     // Input tensors declaration
//     // (syntax: https://www.tensorflow.org/guide/create_op)
//     static std::vector<std::string> Inputs();
//
//     // Output tensors declaration
//     // (syntax: https://www.tensorflow.org/guide/create_op)
//     static std::vector<std::string> Outputs();
//
//     // Initializes the op
//     absl::Status Init(InitContext* ctx);
//
//     // Runs the operation
//     absl::Status Invoke(InvokeContext* ctx);
//
//     // Shape inference
//     static absl::Status ShapeInference(ShapeInferenceContext* ctx);
//
//   };
//
// WARNING: Experimental interface, subject to change
template <template <Runtime, typename...> typename SubType, Runtime Rt,
          typename... Ts>
class OpKernelShim {
 public:
  // Some typedefs for convenience
  using Shape = ::tflite::shim::Shape;
  using InitContext =
      ::tflite::shim::InitContext<typename ContextTypeForRuntime<Rt>::Init>;
  using InvokeContext =
      ::tflite::shim::InvokeContext<typename ContextTypeForRuntime<Rt>::Invoke>;
  using ShapeInferenceContext = ::tflite::shim::ShapeInferenceContext<
      typename ContextTypeForRuntime<Rt>::ShapeInference>;

  // Needed because the pointer to this class is stored
  virtual ~OpKernelShim() = default;

  // If the operation has any attributes they are passed here.
  absl::Status Init(InitContext* ctx) {
    return static_cast<SubType<Rt, Ts...>&>(*this).Init(ctx);
  }

  // The actual computations of the operation
  absl::Status Invoke(InvokeContext* ctx) {
    return static_cast<SubType<Rt, Ts...>&>(*this).Invoke(ctx);
  }

  // Shape inference
  static absl::Status ShapeInference(ShapeInferenceContext* ctx) {
    return SubType<Rt, Ts...>::ShapeInference(ctx);
  }

 protected:
  OpKernelShim() = default;

  // Convience method for filling a single dimension output tensor.
  template <typename BufferType, typename DType>
  absl::Status FillOutputTensor(const std::vector<BufferType>& buffer,
                                int index, InvokeContext* context) const;
};

/////////////////////// Implementations

namespace internal {
// Extract the given AttrType from the AttrValue variant or returns error.
template <typename AttrType>
absl::Status GetAttr(const std::string& attr_name,
                     const absl::StatusOr<AttrValue> attr_value_or,
                     AttrType* value) {
  if (!attr_value_or.ok()) return attr_value_or.status();
  const AttrValue& attr_value = attr_value_or.value();
  if (!absl::holds_alternative<AttrType>(attr_value)) {
    return absl::InternalError(
        absl::StrCat("The attribute type does not match the provided "
                     "type: attr_name: ",
                     attr_name));
  }
  *value = absl::get<AttrType>(attr_value);
  return absl::OkStatus();
}
}  // namespace internal

template <typename SubType>
template <typename AttrType>
absl::Status InitContext<SubType>::GetAttr(const std::string& attr_name,
                                           AttrType* value) const {
  const auto attr_value_or = GetAttr(attr_name);
  return internal::GetAttr<AttrType>(attr_name, attr_value_or, value);
}

template <typename SubType>
template <typename AttrType>
absl::Status ShapeInferenceContext<SubType>::GetAttr(
    const std::string& attr_name, AttrType* value) const {
  const auto attr_value_or = GetAttr(attr_name);
  return internal::GetAttr<AttrType>(attr_name, attr_value_or, value);
}

template <template <Runtime, typename...> typename SubType, Runtime Rt,
          typename... Ts>
template <typename BufferType, typename DType>
absl::Status OpKernelShim<SubType, Rt, Ts...>::FillOutputTensor(
    const std::vector<BufferType>& buffer, const int index,
    tflite::shim::InvokeContext<typename ContextTypeForRuntime<Rt>::Invoke>*
        context) const {
  SH_ASSIGN_OR_RETURN(
      const auto tensorview,
      context->GetOutput(
          index, tflite::shim::Shape({static_cast<int>(buffer.size())})));
  auto data = tensorview->template As<DType, 1>();
  for (int i = 0; i < buffer.size(); ++i) data(i) = buffer.at(i);
  return absl::OkStatus();
}

}  // namespace shim
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_SHIM_OP_KERNEL_H_
