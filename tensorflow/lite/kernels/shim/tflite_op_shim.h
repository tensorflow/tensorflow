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
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/shape.h"
#include "tensorflow/lite/kernels/shim/tensor_view.h"
#include "tensorflow/lite/kernels/shim/tflite_tensor_view.h"
#include "tensorflow/lite/mutable_op_resolver.h"

namespace tflite {
namespace shim {

// TfLite implementation of the methods during an op kernel initialization
class TfLiteInitContext : public InitContext<TfLiteInitContext> {
 public:
  TfLiteInitContext(const TfLiteContext* context, const char* buffer,
                    const size_t length);
  // Read a given attribute
  absl::StatusOr<AttrValue> GetAttr(const std::string& attr_name) const;

 private:
  const flexbuffers::Map attr_map_;
};

// TfLite implementation of the methods during an op kernel invocation
class TfLiteInvokeContext : public InvokeContext<TfLiteInvokeContext> {
 public:
  TfLiteInvokeContext(TfLiteContext* context_, TfLiteNode* node_,
                      const std::vector<bool>& is_static_output);
  // Read an input tensor
  ConstTensorViewOr GetInput(const int idx) const;
  // Get a mutable output tensor
  TensorViewOr GetOutput(const int idx, const Shape& shape) const;

 private:
  absl::Status AssertShapesEqual(const TfLiteIntArray* dims,
                                 const Shape& output_shape) const;

  std::string ShapeMismatchErrorMsg(const TfLiteIntArray* actual_shape,
                                    const Shape& expected_shape) const;

  TfLiteContext* context_;
  TfLiteNode* node_;
  const std::vector<bool>& is_static_output_;
};

// TfLite implementation of the methods during shape inference
class TfLiteShapeInferenceContext
    : public ShapeInferenceContext<TfLiteShapeInferenceContext> {
 public:
  TfLiteShapeInferenceContext(TfLiteContext* context, TfLiteNode* node,
                              std::vector<Shape>* inferred_shapes);
  // Read an input tensor shape
  ShapeOr GetInputShape(const int idx) const;
  // Set an output tensor shape
  absl::Status SetOutputShape(const int idx, const Shape& shape);
  // Read an input tensor during shape inference
  ConstTensorViewOr GetInputTensor(const int idx) const;

 private:
  TfLiteContext* context_;
  TfLiteNode* node_;
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
template <template <Runtime> typename Impl>
class TfLiteOpKernel {
 public:
  using ImplType = Impl<Runtime::kTfLite>;

  // Builds a TfLiteRegistration object to register this with the TfLite runtime
  static TfLiteRegistration* GetTfLiteRegistration() {
    static TfLiteRegistration r =
        TfLiteRegistration{Init, Free, Prepare, Invoke};
    return &r;
  }

  // Adds this op kernel to the passed in op resolver
  static void Add(MutableOpResolver* resolver) {
    resolver->AddCustom(ImplType::kOpName, GetTfLiteRegistration());
  }

  // A boolean indicator for each output whether its shape is fully known or
  // not.
  static const std::vector<bool>& StaticOutputShapeIndicator();

  // The operation name
  static const char* OpName() { return ImplType::kOpName; }

 protected:
  static void* Init(TfLiteContext* context, const char* buffer, size_t length) {
    ImplType* impl = new ImplType;
    TfLiteInitContext ctx(context, buffer, length);
    auto status = impl->Init(&ctx);
    StatusToTfLiteStatus(context, status);
    return impl;
  }

  static void Free(TfLiteContext* context, void* buffer) {
    if (buffer != nullptr) delete static_cast<ImplType*>(buffer);
  }

  static TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
    return ResizeOutputTensorsWithKnownShape(context, node);
  }

  static TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node) {
    TfLiteInvokeContext ctx(context, node, StaticOutputShapeIndicator());
    return StatusToTfLiteStatus(
        context, static_cast<ImplType*>(node->user_data)->Invoke(&ctx));
  }

  static TfLiteStatus ResizeOutputTensorsWithKnownShape(TfLiteContext* context,
                                                        TfLiteNode* node);
};

template <>
struct ContextTypeForRuntime<Runtime::kTfLite> {
  using Init = TfLiteInitContext;
  using Invoke = TfLiteInvokeContext;
  using ShapeInference = TfLiteShapeInferenceContext;
};

////////////////////////////////////////////
///////////////////////////// Implementation

template <template <Runtime> typename Impl>
const std::vector<bool>& TfLiteOpKernel<Impl>::StaticOutputShapeIndicator() {
  static std::vector<bool>* ret = []() {
    auto ret = new std::vector<bool>;
    const auto outputs_decl = ImplType::Outputs();
    ret->reserve(outputs_decl.size());
    for (const TensorDeclaration& output_decl : outputs_decl) {
      ret->emplace_back(output_decl.shape.FullyDefined());
    }
    return ret;
  }();
  return *ret;
}

// Resizes the Output Tensor to their shape. It goes over three cases:
//
// case 1: output shape is statically set in the op declaration
//   ResizeTensor(static_shape)
// case 2: output shape is known after ShapeInference() was called during
//     Prepare()
//   ResizeTensor(output_of_shape_inference)
// case 3: output shape is not fully defined even after shape inference
//   SetTensorToDynamic(...)
template <template <Runtime> typename Impl>
TfLiteStatus TfLiteOpKernel<Impl>::ResizeOutputTensorsWithKnownShape(
    TfLiteContext* context, TfLiteNode* node) {
  // Whether all output shapes are static or there's a need to run shape
  // inference.
  static const auto& is_static_output = StaticOutputShapeIndicator();
  static const bool all_outputs_static =
      std::all_of(is_static_output.cbegin(), is_static_output.cend(),
                  [](bool b) { return b; });
  const size_t num_outputs = ::tflite::NumOutputs(node);
  std::vector<Shape> inferred_output_shapes(num_outputs);
  TfLiteShapeInferenceContext ctx(context, node, &inferred_output_shapes);
  if (!all_outputs_static) {
    auto status = ImplType::ShapeInference(&ctx);
    TF_LITE_ENSURE_STATUS(StatusToTfLiteStatus(context, status));
  }
  // Output shapes.
  const auto outputs_decl = ImplType::Outputs();
  TF_LITE_ENSURE_EQ(context, num_outputs, outputs_decl.size());
  for (int output_idx = 0; output_idx < num_outputs; ++output_idx) {
    TfLiteTensor* output_tensor = tflite::GetOutput(context, node, output_idx);
    TF_LITE_ENSURE(context, output_tensor != nullptr);
    const TensorDeclaration& output_decl = outputs_decl[output_idx];
    if (is_static_output[output_idx]) {
      // Case: output shape is static
      TF_LITE_ENSURE_OK(
          context,
          context->ResizeTensor(context, output_tensor,
                                ShapeToTfLiteShape(output_decl.shape.value())));
    } else if (inferred_output_shapes[output_idx].FullyDefined()) {
      // Case: output shape can be inferred during `Prepare`
      TF_LITE_ENSURE_OK(
          context,
          context->ResizeTensor(
              context, output_tensor,
              ShapeToTfLiteShape(inferred_output_shapes[output_idx].value())));
    } else {
      // Case: output shape is dynamic
      tflite::SetTensorToDynamic(output_tensor);
    }
  }
  return kTfLiteOk;
}

}  // namespace shim
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_SHIM_TFLITE_OP_SHIM_H_
