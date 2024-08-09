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

#include <complex>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace complex {

static const int kInputTensor = 0;
static const int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);

  TF_LITE_ENSURE(context, input->type == kTfLiteComplex64 ||
                              input->type == kTfLiteComplex128);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  if (input->type == kTfLiteComplex64) {
    TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteFloat32);
  } else {
    TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteFloat64);
  }

  TfLiteIntArray* output_shape = TfLiteIntArrayCopy(input->dims);
  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus PrepareComplex(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* real_input = GetInput(context, node, 0);
  const TfLiteTensor* imag_input = GetInput(context, node, 1);

  // Ensure the real input and the imaginary input are the same types and shapes
  TF_LITE_ENSURE(context, real_input->type == imag_input->type &&
                              (real_input->type == kTfLiteFloat32 ||
                               real_input->type == kTfLiteFloat64));
  TF_LITE_ENSURE_EQ(context, NumDimensions(real_input), NumDimensions(imag_input));
  for (auto d = 0; d < NumDimensions(real_input); ++d) {
    TF_LITE_ENSURE_EQ(context, real_input->dims->data[d], imag_input->dims->data[d]);
  }
  
  TfLiteTensor* output = GetOutput(context, node, 0);

  if (real_input->type == kTfLiteFloat32) {
    TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteComplex64);
  } else {
    TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteComplex128);
  }
  
  TfLiteIntArray* output_shape = TfLiteIntArrayCopy(real_input->dims);
  return context->ResizeTensor(context, output, output_shape);
}

template <typename T, typename ExtractF>
void ExtractData(const TfLiteTensor* input, ExtractF extract_func,
                 TfLiteTensor* output) {
  const std::complex<T>* input_data = GetTensorData<std::complex<T>>(input);
  T* output_data = GetTensorData<T>(output);
  const int input_size = NumElements(input);
  for (int i = 0; i < input_size; ++i) {
    *output_data++ = extract_func(*input_data++);
  }
}

template <typename T>
void ConvertToComplex(const TfLiteTensor* real_input,
                      const TfLiteTensor* imag_input, TfLiteTensor* output) {
  const T* real_input_data = GetTensorData<T>(real_input);
  const T* imag_input_data = GetTensorData<T>(imag_input);
  std::complex<T>* output_data = GetTensorData<std::complex<T>>(output);
  const int input_size = NumElements(real_input);
  for (auto i = 0; i < input_size; ++i) {
    output_data[i] = std::complex<T>(real_input_data[i], imag_input_data[i]);
  }
}

TfLiteStatus EvalComplex(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* real_input = GetInput(context, node, 0);
  const TfLiteTensor* imag_input = GetInput(context, node, 1);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (real_input->type) {
    case kTfLiteFloat32: {
      ConvertToComplex<float>(
        real_input,
        imag_input,
        output
      );
      break;
    }
    case kTfLiteFloat64: {
      ConvertToComplex<double>(
        real_input,
        imag_input,
        output
      );
      break;
    }
    default: {
      TF_LITE_KERNEL_LOG(context,
                         "Unsupported input type, Complex op only supports "
                         "real valued float32 or float64 inputs, but got: ",
                         TfLiteTypeGetName(real_input->type), " and ",
                         TfLiteTypeGetName(imag_input->type));
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

TfLiteStatus EvalReal(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (input->type) {
    case kTfLiteComplex64: {
      ExtractData<float>(
          input,
          static_cast<float (*)(const std::complex<float>&)>(std::real<float>),
          output);
      break;
    }
    case kTfLiteComplex128: {
      ExtractData<double>(input,
                          static_cast<double (*)(const std::complex<double>&)>(
                              std::real<double>),
                          output);
      break;
    }
    default: {
      TF_LITE_KERNEL_LOG(context,
                         "Unsupported input type, Real op only supports "
                         "complex input, but got: %s",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

TfLiteStatus EvalImag(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (input->type) {
    case kTfLiteComplex64: {
      ExtractData<float>(
          input,
          static_cast<float (*)(const std::complex<float>&)>(std::imag<float>),
          output);
      break;
    }
    case kTfLiteComplex128: {
      ExtractData<double>(input,
                          static_cast<double (*)(const std::complex<double>&)>(
                              std::imag<double>),
                          output);
      break;
    }
    default: {
      TF_LITE_KERNEL_LOG(context,
                         "Unsupported input type, Imag op only supports "
                         "complex input, but got: %s",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

TfLiteStatus EvalAbs(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (input->type) {
    case kTfLiteComplex64: {
      ExtractData<float>(
          input,
          static_cast<float (*)(const std::complex<float>&)>(std::abs<float>),
          output);
      break;
    }
    case kTfLiteComplex128: {
      ExtractData<double>(input,
                          static_cast<double (*)(const std::complex<double>&)>(
                              std::abs<double>),
                          output);
      break;
    }
    default: {
      TF_LITE_KERNEL_LOG(context,
                         "Unsupported input type, ComplexAbs op only supports "
                         "complex input, but got: %s",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

}  // namespace complex

TfLiteRegistration* Register_COMPLEX() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                  complex::PrepareComplex, complex::EvalComplex};
  return &r;
}
TfLiteRegistration* Register_REAL() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 complex::Prepare, complex::EvalReal};
  return &r;
}

TfLiteRegistration* Register_IMAG() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 complex::Prepare, complex::EvalImag};
  return &r;
}

TfLiteRegistration* Register_COMPLEX_ABS() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 complex::Prepare, complex::EvalAbs};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
