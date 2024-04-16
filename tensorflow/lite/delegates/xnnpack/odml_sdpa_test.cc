/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/delegates/xnnpack/odml_sdpa_tester.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace tflite {
namespace xnnpack {

TEST(ODMLSDPA, MQA) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  const auto batch = 1;
  const auto input_seq_len = 1;
  const auto max_seq_len = 500;
  const auto q_heads = 32;
  const auto kv_heads = 1;
  const auto head_dim = 4;  // embedding_dim//q_heads

  ODMLSDPATester()
      .QueryShape({batch, input_seq_len, q_heads, head_dim})  // q
      .KeyShape({batch, max_seq_len, kv_heads, head_dim})     // k
      .ValueShape({batch, max_seq_len, kv_heads, head_dim})   // v
      .MaskShape({batch, 1, input_seq_len, max_seq_len})      // mask
      .Test(xnnpack_delegate.get());
}

TEST(ODMLSDPA, MHA) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  const auto batch = 1;
  const auto input_seq_len = 1;
  const auto max_seq_len = 500;
  const auto q_heads = 32;
  const auto kv_heads = 32;
  const auto head_dim = 4;  // embedding_dim//q_heads

  ODMLSDPATester()
      .QueryShape({batch, input_seq_len, q_heads, head_dim})  // q
      .KeyShape({batch, max_seq_len, kv_heads, head_dim})     // k
      .ValueShape({batch, max_seq_len, kv_heads, head_dim})   // v
      .MaskShape({batch, 1, input_seq_len, max_seq_len})      // mask
      .Test(xnnpack_delegate.get());
}

}  // namespace xnnpack
}  // namespace tflite
