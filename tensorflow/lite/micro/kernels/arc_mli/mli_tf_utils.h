/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_ARC_MLI_TF_UTILS_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_ARC_MLI_TF_UTILS_H_

#include "mli_api.h"  // NOLINT
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

constexpr int kFracBitsQ15 = 15;
constexpr int kFracBitsQ31 = 31;

namespace tflite {
namespace ops {
namespace micro {

inline void ConvertToMliTensorData(const TfLiteTensor* tfT, mli_tensor* mliT) {
  // Data is NULL until MliTensorAttachBuffer is called.
  mliT->data = NULL;
  if (tfT->type == kTfLiteInt8) {
    mliT->el_type = MLI_EL_ASYM_I8;
  } else if (tfT->type == kTfLiteInt32) {
    mliT->el_type = MLI_EL_ASYM_I32;
  } else {
    TF_LITE_FATAL("Wrong data type. Expected int8_t or int32_t.");
  }

  mliT->capacity = tfT->bytes;
  mliT->rank = GetTensorShape(tfT).DimensionsCount();
  for (int i = 0; i < GetTensorShape(tfT).DimensionsCount(); i++) {
    mliT->shape[i] = GetTensorShape(tfT).Dims(i);
  }
}

inline void ConvertToMliQuantParams(const TfLiteTensor* tfT, mli_tensor* mliT) {
  mliT->el_params.asym.dim = -1;
  mliT->el_params.asym.zero_point.i16 = tfT->params.zero_point;
  float fscale = tfT->params.scale;
  int exp;
  frexpf(fscale, &exp);
  int frac_bits = kFracBitsQ31 - exp;
  int32_t iscale = (int32_t)((1ll << frac_bits) * fscale + 0.5f);
  mliT->el_params.asym.scale_frac_bits = frac_bits;
  mliT->el_params.asym.scale.i32 = (int32_t)iscale;
}

inline void ConvertToMliQuantParamsPerChannel(const TfLiteTensor* tfT,
                                              mli_tensor* mliT) {
  // mli tensor scale and zero_point arrays should be allocated at this point
  TFLITE_DCHECK_NE(mliT->el_params.asym.scale.pi16, 0);
  TFLITE_DCHECK_NE(mliT->el_params.asym.zero_point.pi16, 0);

  // get per channel quantization parameters
  const auto* affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(tfT->quantization.params);
  mliT->el_params.asym.dim = affine_quantization->quantized_dimension;

  // find frac_bits
  const int num_channels =
      mliT->shape[affine_quantization->quantized_dimension];
  int min_frac_bits;
  float* fscale = affine_quantization->scale->data;
  for (int i = 0; i < num_channels; i++) {
    int exp;
    frexpf(fscale[i], &exp);
    int cur_frac_bits = kFracBitsQ31 - exp;
    if (i == 0) {
      min_frac_bits = cur_frac_bits;
    } else {
      min_frac_bits =
          min_frac_bits < cur_frac_bits ? min_frac_bits : cur_frac_bits;
    }
  }
  mliT->el_params.asym.scale_frac_bits = min_frac_bits;

  for (int i = 0; i < num_channels; i++) {
    int32_t iscale = (int32_t)((1ll << min_frac_bits) * fscale[i] + 0.5f);
    mliT->el_params.asym.scale.pi32[i] = iscale;
  }
}

template <typename datatype>
inline void MliTensorAttachBuffer(const TfLiteEvalTensor* tfT,
                                  mli_tensor* mliT) {
  // "const_cast" here used to attach const data buffer to the initially
  // non-const mli_tensor. This is required by current implementation of MLI
  // backend and planned for redesign due to this and some other aspects.
  mliT->data = const_cast<void*>(
      static_cast<const void*>(tflite::micro::GetTensorData<datatype>(tfT)));
}

inline void ConvertToMliTensor(const TfLiteTensor* tfT, mli_tensor* mliT) {
  ConvertToMliTensorData(tfT, mliT);
  ConvertToMliQuantParams(tfT, mliT);
}

inline void ConvertToMliTensorPerChannel(const TfLiteTensor* tfT,
                                         mli_tensor* mliT) {
  ConvertToMliTensorData(tfT, mliT);
  ConvertToMliQuantParamsPerChannel(tfT, mliT);
}
}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_ARC_MLI_TF_UTILS_H_
