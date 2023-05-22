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
#ifndef TENSORFLOW_LITE_KERNELS_SHIM_TFLITE_OP_SHIM_H_
#define TENSORFLOW_LITE_KERNELS_SHIM_TFLITE_OP_SHIM_H_

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/shape.h"
#include "tensorflow/lite/kernels/shim/tensor_view.h"
#include "tensorflow/lite/kernels/shim/tflite_tensor_view.h"
#include "tensorflow/lite/mutable_op_resolver.h"

// This file contains the TFLite adapter. That is, it takes a `OpKernelShim`
// class and provides a TFLite kernel out of it.

namespace tflite {
namespace shim {

// TfLite implementation of the methods during an op kernel initialization
class TfLiteInitContext : public InitContext<TfLiteInitContext> {
 public:
  TfLiteInitContext(const TfLiteContext* context,
                    const flexbuffers::Map* attr_map);
  // Read a given attribute
  absl::StatusOr<AttrValue> GetAttr(const std::string& attr_name) const;

 private:
  const flexbuffers::Map* attr_map_;
};

// TfLite implementation of the methods during an op kernel invocation
class TfLiteInvokeContext : public InvokeContext<TfLiteInvokeContext> {
 public:
  TfLiteInvokeContext(TfLiteContext* context_, TfLiteNode* node_);
  // Read an input tensor
  ConstTensorViewOr GetInput(const int idx) const;
  // Get a mutable output tensor. For output string tensors, this should only
  // be called once.
  TensorViewOr GetOutput(const int idx, const Shape& shape) const;
  // Number of input tensors
  int NumInputs() const;
  // Number of output tensors
  int NumOutputs() const;

 private:
  absl::Status AssertShapesEqual(const TfLiteIntArray* dims,
                                 const Shape& output_shape) const;

  std::string ShapeMismatchErrorMsg(const TfLiteIntArray* actual_shape,
                                    const Shape& expected_shape) const;

  TfLiteContext* context_;
  TfLiteNode* node_;
};

// TfLite implementation of the methods during shape inference
class TfLiteShapeInferenceContext
    : public ShapeInferenceContext<TfLiteShapeInferenceContext> {
 public:
  TfLiteShapeInferenceContext(TfLiteContext* context, TfLiteNode* node,
                              const flexbuffers::Map* attr_map,
                              std::vector<Shape>* inferred_shapes);
  // Read an input tensor shape
  ShapeOr GetInputShape(const int idx) const;
  // Set an output tensor shape
  absl::Status SetOutputShape(const int idx, const Shape& shape);
  // Read an input tensor during shape inference
  ConstTensorViewOr GetInputTensor(const int idx) const;
  // Read a given attribute
  absl::StatusOr<AttrValue> GetAttr(const std::string& attr_name) const;
  // Number of input tensors
  int NumInputs() const;
  // Number of output tensors
  int NumOutputs() const;

 private:
  TfLiteContext* context_;
  TfLiteNode* node_;
  const flexbuffers::Map* attr_map_;
  std::vector<Shape>* inferred_shapes_;
};

// Convert the absl::Status to a TfLiteStatus and report the error message.
TfLiteStatus StatusToTfLiteStatus(TfLiteContext* context,
                                  const absl::Status& status);

// Converts a vector of dims into an int array for TFLite use.
TfLiteIntArray* ShapeToTfLiteShape(const std::vector<int>& shape);

// Converts an int array representing shape in TFLite to Shape.
Shape TfLiteShapeToShape(const TfLiteIntArray* tflite_shape);

// An op kernel base class which is an adapter between an Op implementation
// (OpKernelShim subclass) and TFLite runtime
template <template <Runtime, typename...> typename Impl, typename... Ts>
class TfLiteOpKernel {
 public:
  using ImplType = Impl<Runtime::kTfLite, Ts...>;

  // Builds a TfLiteRegistration object to register this with the TfLite runtime
  static TfLiteRegistration* GetTfLiteRegistration() {
    static TfLiteRegistration r =
        TfLiteRegistration{Init, Free, Prepare, Invoke};
    return &r;
  }

  // Adds this op kernel to the passed in op resolver
  static void Add(MutableOpResolver* resolver) {
    resolver->AddCustom(ImplType::OpName(), GetTfLiteRegistration());
  }

  // The operation name
  static const char* OpName() { return ImplType::OpName(); }

 protected:
  // The data that is stored in node::user_data.
  struct UserData {
    UserData(const char* buffer, size_t length) {
      impl = new ImplType;
      attr_map = new flexbuffers::Map(
          flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(buffer), length)
              .AsMap());
    }

    // An instance of OpKernelShim<TF or TFLite>.
    // This is so that the Invoke(), Prepare(), etc. can call Invoke(),
    // ShapeInference(), ... on the kernel defined using this library.
    ImplType* impl = nullptr;
    // Attribute map for the op kernel.
    // The map needs to be accessible because the library provides
    // GetAttr() during ShapeInference() which is called during Prepare(). So
    // this needs to be accessible at that point.
    const flexbuffers::Map* attr_map = nullptr;

    ~UserData() {
      if (impl) delete impl;
      if (attr_map) delete attr_map;
    }
  };

  static void* Init(TfLiteContext* context, const char* buffer, size_t length) {
    auto* user_data = new UserData(buffer, length);
    TfLiteInitContext ctx(context, user_data->attr_map);
    auto status = user_data->impl->Init(&ctx);
    StatusToTfLiteStatus(context, status);
    return user_data;
  }

  static void Free(TfLiteContext* context, void* buffer) {
    if (buffer) delete static_cast<UserData*>(buffer);
  }

  // Resizes the Output Tensor to their shape. There are two cases:
  //
  // case 1: output shape is known after ShapeInference() was called during
  //     Prepare()
  //   ResizeTensor(output_of_shape_inference)
  // case 2: output shape is not fully defined even after shape inference
  //   SetTensorToDynamic(...)
  static TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
    const size_t num_outputs = ::tflite::NumOutputs(node);
    std::vector<Shape> inferred_output_shapes(num_outputs);
    const auto* attr_map = static_cast<UserData*>(node->user_data)->attr_map;
    TfLiteShapeInferenceContext ctx(context, node, attr_map,
                                    &inferred_output_shapes);
    auto status = ImplType::ShapeInference(&ctx);
    TF_LITE_ENSURE_STATUS(StatusToTfLiteStatus(context, status));
    // Output shapes.
    for (int output_idx = 0; output_idx < num_outputs; ++output_idx) {
      TfLiteTensor* output_tensor =
          tflite::GetOutput(context, node, output_idx);
      TF_LITE_ENSURE(context, output_tensor != nullptr);
      if (inferred_output_shapes[output_idx].FullyDefined()) {
        // Case: output shape can be inferred during `Prepare`
        TF_LITE_ENSURE_OK(context,
                          context->ResizeTensor(
                              context, output_tensor,
                              ShapeToTfLiteShape(
                                  inferred_output_shapes[output_idx].value())));
      } else {
        // Case: output shape is dynamic
        tflite::SetTensorToDynamic(output_tensor);
      }
    }
    return kTfLiteOk;
  }

  static TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node) {
    TfLiteInvokeContext ctx(context, node);
    return StatusToTfLiteStatus(
        context, static_cast<UserData*>(node->user_data)->impl->Invoke(&ctx));
  }
};

template <>
struct ContextTypeForRuntime<Runtime::kTfLite> {
  using Init = TfLiteInitContext;
  using Invoke = TfLiteInvokeContext;
  using ShapeInference = TfLiteShapeInferenceContext;
};

}  // namespace shim
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_SHIM_TFLITE_OP_SHIM_H_
