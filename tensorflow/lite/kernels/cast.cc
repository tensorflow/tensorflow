/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <algorithm>
#include <complex>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace cast {
constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // TODO(ahentz): these two checks would make the new implementation
  // incompatible with some existing models, where params is not specified. It
  // is OK not to have them because toco would have set input and output types
  // to match the parameters.
  // auto* params = reinterpret_cast<TfLiteCastParams*>(node->builtin_data);
  // TF_LITE_ENSURE_EQ(context, input->type, params->in_data_type);
  // TF_LITE_ENSURE_EQ(context, output->type, params->out_data_type);

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

template <typename FromT, typename ToT>
void copyCast(const FromT* in, ToT* out, int num_elements) {
  std::transform(in, in + num_elements, out,
                 [](FromT a) { return static_cast<ToT>(a); });
}

template <typename ToT>
void copyCast(const std::complex<float>* in, ToT* out, int num_elements) {
  std::transform(in, in + num_elements, out, [](std::complex<float> a) {
    return static_cast<ToT>(std::real(a));
  });
}

template <>
void copyCast(const std::complex<float>* in, std::complex<float>* out,
              int num_elements) {
  std::transform(in, in + num_elements, out,
                 [](std::complex<float> a) { return a; });
}

template <typename ToT>
void copyCast(const Eigen::half* in, ToT* out, int num_elements) {
  std::transform(in, in + num_elements, out, [](Eigen::half a) {
    return static_cast<ToT>(Eigen::half_impl::half_to_float(a));
  });
}

template <>
void copyCast(const Eigen::half* in, std::complex<float>* out,
              int num_elements) {
  std::transform(in, in + num_elements, out, [](Eigen::half a) {
    return std::complex<float>(Eigen::half_impl::half_to_float(a));
  });
}

template <typename FromT>
void copyCastToFloat16(const FromT* in, Eigen::half* out, int num_elements) {
  std::transform(in, in + num_elements, out, [](FromT a) {
    return Eigen::half_impl::float_to_half_rtne(static_cast<float>(a));
  });
}

template <>
void copyCastToFloat16(const std::complex<float>* in, Eigen::half* out,
                       int num_elements) {
  std::transform(in, in + num_elements, out, [](std::complex<float> a) {
    return Eigen::half_impl::float_to_half_rtne(std::real(a));
  });
}

template <>
void copyCastToFloat16(const Eigen::half* in, Eigen::half* out,
                       int num_elements) {
  std::transform(in, in + num_elements, out, [](Eigen::half a) { return a; });
}

template <typename FromT>
TfLiteStatus copyToTensor(TfLiteContext* context, const FromT* in,
                          TfLiteTensor* out, int num_elements) {
  switch (out->type) {
    case kTfLiteInt64:
      copyCast(in, out->data.i64, num_elements);
      break;
    case kTfLiteInt32:
      copyCast(in, out->data.i32, num_elements);
      break;
    case kTfLiteUInt32:
      copyCast(in, out->data.u32, num_elements);
      break;
    case kTfLiteInt16:
      copyCast(in, out->data.i16, num_elements);
      break;
    case kTfLiteUInt16:
      copyCast(in, out->data.ui16, num_elements);
      break;
    case kTfLiteUInt8:
      copyCast(in, out->data.uint8, num_elements);
      break;
    case kTfLiteInt8:
      copyCast(in, out->data.int8, num_elements);
      break;
    case kTfLiteFloat16:
      copyCastToFloat16(in, reinterpret_cast<Eigen::half*>(out->data.f16),
                        num_elements);
      break;
    case kTfLiteFloat32:
      copyCast(in, GetTensorData<float>(out), num_elements);
      break;
    case kTfLiteFloat64:
      copyCast(in, out->data.f64, num_elements);
      break;
    case kTfLiteBool:
      copyCast(in, out->data.b, num_elements);
      break;
    case kTfLiteComplex64:
      copyCast(in, reinterpret_cast<std::complex<float>*>(out->data.c64),
               num_elements);
      break;
    default:
      // Unsupported type.
      TF_LITE_UNSUPPORTED_TYPE(context, out->type, "Cast");
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  const int num_elements = NumElements(input);
  TF_LITE_ENSURE_EQ(context, num_elements, NumElements(output));
  switch (input->type) {
    case kTfLiteInt64:
      return copyToTensor(context, input->data.i64, output, num_elements);
    case kTfLiteInt32:
      return copyToTensor(context, input->data.i32, output, num_elements);
    case kTfLiteUInt32:
      return copyToTensor(context, input->data.u32, output, num_elements);
    case kTfLiteUInt16:
      return copyToTensor(context, input->data.ui16, output, num_elements);
    case kTfLiteInt16:
      return copyToTensor(context, input->data.i16, output, num_elements);
    case kTfLiteUInt8:
      return copyToTensor(context, input->data.uint8, output, num_elements);
    case kTfLiteInt8:
      return copyToTensor(context, input->data.int8, output, num_elements);
    case kTfLiteFloat16:
      return copyToTensor(context,
                          reinterpret_cast<Eigen::half*>(input->data.f16),
                          output, num_elements);
    case kTfLiteFloat32:
      return copyToTensor(context, GetTensorData<float>(input), output,
                          num_elements);
    case kTfLiteFloat64:
      return copyToTensor(context, input->data.f64, output, num_elements);
    case kTfLiteBool:
      return copyToTensor(context, input->data.b, output, num_elements);
    case kTfLiteComplex64:
      return copyToTensor(
          context, reinterpret_cast<std::complex<float>*>(input->data.c64),
          output, num_elements);
    default:
      // Unsupported type.
      TF_LITE_UNSUPPORTED_TYPE(context, input->type, "Cast");
  }
}
}  // namespace cast

TfLiteRegistration* Register_CAST() {
  static TfLiteRegistration r = {nullptr, nullptr, cast::Prepare, cast::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
