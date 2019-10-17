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

#ifndef MLI_TF_UTILS_H_
#define MLI_TF_UTILS_H_

#include "mli_api.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include <math.h>

#define Q15_FRAC_BITS 15

namespace tflite {
namespace ops {
namespace micro {

template <typename datatype>
static void TfLiteTensor2mli_tensor(const TfLiteTensor* tfT, mli_tensor* mliT) {
  mliT->data = (void*)GetTensorData<datatype>(tfT);
  mliT->capacity = tfT->bytes;
  for (int i = 0; i <  GetTensorShape(tfT).DimensionsCount(); i++) {
    mliT->shape[i] =  GetTensorShape(tfT).Dims(i);
  }
  mliT->rank = GetTensorShape(tfT).DimensionsCount();
  if (tfT->type == kTfLiteInt8) {
    mliT->el_type = MLI_EL_ASYM_I8;
  } else if (tfT->type == kTfLiteInt32) {
    mliT->el_type = MLI_EL_ASYM_I32;
  } else {
    //return kTfLiteError;
  }
  // for now only support per tensor quantization paramters
  mliT->el_params.asym.dim = -1;
  mliT->el_params.asym.zero_point.i16 = tfT->params.zero_point;
  float fscale = tfT->params.scale;
  int exp;
  frexpf(fscale, &exp);
  int frac_bits = Q15_FRAC_BITS - exp;
  int32_t iscale = (1<<frac_bits) * fscale + 0.5f;
  mliT->el_params.asym.scale_frac_bits = frac_bits;
  mliT->el_params.asym.scale.i16 = (int16_t)iscale;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite

#endif  // MLI_TF_UTILS_H_
