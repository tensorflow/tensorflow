/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_YNNPACK_ATTENTION_MODEL_H_
#define TENSORFLOW_LITE_DELEGATES_YNNPACK_ATTENTION_MODEL_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/ynnpack/ynnpack_delegate.h"
#include "tensorflow/lite/kernels/test_util.h"

namespace tflite {
namespace ynnpack {

TfLiteRegistration* Register_RuntimeBmm();

class AttentionModel : public MultiOpModel {
 public:
  AttentionModel(int b, int t, int s, int h, int n, float scale,
                 bool transpose_io, bool use_delegate,
                 const TfLiteYNNPackDelegateOptions& delegate_options);

  int query() const { return query_id_; }
  int key() const { return key_id_; }
  int value() const { return value_id_; }
  int runtime_bmm_params() const { return runtime_bmm_params_id_; }
  int mask() const { return mask_id_; }
  int output() const { return output_id_; }

 private:
  int query_id_;
  int key_id_;
  int value_id_;
  int runtime_bmm_params_id_;
  int mask_id_;
  int output_id_;
};

}  // namespace ynnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_YNNPACK_ATTENTION_MODEL_H_
