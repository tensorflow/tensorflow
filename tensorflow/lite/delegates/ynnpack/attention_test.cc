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

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/ynnpack/attention_model.h"
#include "tensorflow/lite/delegates/ynnpack/ynnpack_delegate.h"

namespace tflite {
namespace ynnpack {
namespace {

TEST(AttentionTest, Correctness) {
  // Small hardcoded shapes
  const int b = 1;
  const int t = 4;
  const int s = 8;
  const int h = 16;
  const int n = 2;
  const int s_active = 5;
  const float scale = 1.0f / std::sqrt(static_cast<float>(h));

  TfLiteYNNPackDelegateOptions options = TfLiteYNNPackDelegateOptionsDefault();
  options.num_threads = 1;
  options.static_shape = true;

  // Initialize input data with some hardcoded values
  std::vector<float> q_data(b * n * t * h);
  std::vector<float> k_data(b * n * s * h);
  std::vector<float> v_data(b * n * h * s);
  std::vector<float> mask_data(b * 1 * t * s);

  for (size_t i = 0; i < q_data.size(); ++i) {
    q_data[i] = 0.1f * (i % 10);
  }
  for (size_t i = 0; i < k_data.size(); ++i) {
    k_data[i] = 0.2f * (i % 10);
  }
  for (size_t i = 0; i < v_data.size(); ++i) {
    v_data[i] = 0.3f * (i % 10);
  }
  for (int i = 0; i < b * t; ++i) {
    for (int j = 0; j < s; ++j) {
      mask_data[i * s + j] = (j < s_active) ? 0.0f : -1e9f;
    }
  }

  // Run reference model (CPU)
  AttentionModel model_ref(b, t, s, h, n, scale, /*transpose_io=*/false,
                           /*use_delegate=*/false, options);
  model_ref.PopulateTensor(model_ref.query(), q_data);
  model_ref.PopulateTensor(model_ref.key(), k_data);
  model_ref.PopulateTensor(model_ref.value(), v_data);
  model_ref.PopulateTensor(model_ref.runtime_bmm_params(), {s_active});
  model_ref.PopulateTensor(model_ref.mask(), mask_data);
  ASSERT_EQ(model_ref.Invoke(), kTfLiteOk);

  // Run delegate model (YNNPACK)
  AttentionModel model(b, t, s, h, n, scale, /*transpose_io=*/false,
                       /*use_delegate=*/true, options);
  model.PopulateTensor(model.query(), q_data);
  model.PopulateTensor(model.key(), k_data);
  model.PopulateTensor(model.value(), v_data);
  model.PopulateTensor(model.runtime_bmm_params(), {s_active});
  model.PopulateTensor(model.mask(), mask_data);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  // Compare outputs
  auto out_delegate = model.ExtractVector<float>(model.output());
  auto out_ref = model_ref.ExtractVector<float>(model_ref.output());
  ASSERT_EQ(out_delegate.size(), out_ref.size());

  for (size_t i = 0; i < out_delegate.size(); ++i) {
    EXPECT_FALSE(std::isnan(out_delegate[i]));
    EXPECT_FALSE(std::isnan(out_ref[i]));
    EXPECT_NEAR(out_delegate[i], out_ref[i], 1e-3f);
  }
}

TEST(AttentionTest, CorrectnessTransposed) {
  // Test with transpose_io = true
  const int b = 1;
  const int t = 4;
  const int s = 8;
  const int h = 16;
  const int n = 2;
  const int s_active = 5;
  const float scale = 1.0f / std::sqrt(static_cast<float>(h));

  TfLiteYNNPackDelegateOptions options = TfLiteYNNPackDelegateOptionsDefault();
  options.num_threads = 1;
  options.static_shape = true;

  std::vector<float> q_data(b * n * t * h);
  std::vector<float> k_data(b * n * s * h);
  std::vector<float> v_data(b * n * h * s);
  std::vector<float> mask_data(b * 1 * t * s);

  for (size_t i = 0; i < q_data.size(); ++i) {
    q_data[i] = 0.1f * (i % 10);
  }
  for (size_t i = 0; i < k_data.size(); ++i) {
    k_data[i] = 0.2f * (i % 10);
  }
  for (size_t i = 0; i < v_data.size(); ++i) {
    v_data[i] = 0.3f * (i % 10);
  }
  for (int i = 0; i < b * t; ++i) {
    for (int j = 0; j < s; ++j) {
      mask_data[i * s + j] = (j < s_active) ? 0.0f : -1e9f;
    }
  }

  // Run reference model (CPU)
  AttentionModel model_ref(b, t, s, h, n, scale, /*transpose_io=*/true,
                           /*use_delegate=*/false, options);
  model_ref.PopulateTensor(model_ref.query(), q_data);
  model_ref.PopulateTensor(model_ref.key(), k_data);
  model_ref.PopulateTensor(model_ref.value(), v_data);
  model_ref.PopulateTensor(model_ref.runtime_bmm_params(), {s_active});
  model_ref.PopulateTensor(model_ref.mask(), mask_data);
  ASSERT_EQ(model_ref.Invoke(), kTfLiteOk);

  // Run delegate model (YNNPACK)
  AttentionModel model(b, t, s, h, n, scale, /*transpose_io=*/true,
                       /*use_delegate=*/true, options);
  model.PopulateTensor(model.query(), q_data);
  model.PopulateTensor(model.key(), k_data);
  model.PopulateTensor(model.value(), v_data);
  model.PopulateTensor(model.runtime_bmm_params(), {s_active});
  model.PopulateTensor(model.mask(), mask_data);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  auto out_delegate = model.ExtractVector<float>(model.output());
  auto out_ref = model_ref.ExtractVector<float>(model_ref.output());
  ASSERT_EQ(out_delegate.size(), out_ref.size());

  for (size_t i = 0; i < out_delegate.size(); ++i) {
    EXPECT_FALSE(std::isnan(out_delegate[i]));
    EXPECT_FALSE(std::isnan(out_ref[i]));
    EXPECT_NEAR(out_delegate[i], out_ref[i], 1e-3f);
  }
}

TEST(AttentionTest, LargeShapeBounds) {
  const int b = 1;
  const int t = 1;
  const int s = 4096;
  const int h = 64;
  const int n = 32;
  const int s_active = 41;
  const float scale = 1.0f / std::sqrt(static_cast<float>(h));

  TfLiteYNNPackDelegateOptions options = TfLiteYNNPackDelegateOptionsDefault();
  options.num_threads = 1;
  options.static_shape = true;

  std::vector<float> q_data(b * n * t * h);
  for (size_t i = 0; i < q_data.size(); ++i) q_data[i] = std::sin(i);
  std::vector<float> k_data(b * n * s * h);
  for (size_t i = 0; i < k_data.size(); ++i) k_data[i] = std::sin(i * 2);
  std::vector<float> v_data(b * n * h * s);
  for (size_t i = 0; i < v_data.size(); ++i) v_data[i] = std::sin(i * 3);

  std::vector<float> mask_data(b * 1 * t * s);
  for (int i = 0; i < b * 1 * t * s; ++i) {
    int s_idx = i % s;
    if (s_idx >= s_active) {
      mask_data[i] = -std::numeric_limits<float>::infinity();
    } else {
      mask_data[i] = 0.0f;
    }
  }

  AttentionModel model(b, t, s, h, n, scale, /*transpose_io=*/false,
                       /*use_delegate=*/true, options);
  model.PopulateTensor(model.query(), q_data);
  model.PopulateTensor(model.key(), k_data);
  model.PopulateTensor(model.value(), v_data);
  model.PopulateTensor(model.runtime_bmm_params(), {s_active});
  model.PopulateTensor(model.mask(), mask_data);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  AttentionModel model_ref(b, t, s, h, n, scale, /*transpose_io=*/false,
                           /*use_delegate=*/false, options);
  model_ref.PopulateTensor(model_ref.query(), q_data);
  model_ref.PopulateTensor(model_ref.key(), k_data);
  model_ref.PopulateTensor(model_ref.value(), v_data);
  model_ref.PopulateTensor(model_ref.runtime_bmm_params(), {s_active});
  model_ref.PopulateTensor(model_ref.mask(), mask_data);
  ASSERT_EQ(model_ref.Invoke(), kTfLiteOk);

  auto out_delegate = model.ExtractVector<float>(model.output());
  auto out_ref = model_ref.ExtractVector<float>(model_ref.output());
  ASSERT_EQ(out_delegate.size(), out_ref.size());

  for (size_t i = 0; i < out_delegate.size(); ++i) {
    EXPECT_FALSE(std::isnan(out_delegate[i]));
    EXPECT_FALSE(std::isnan(out_ref[i]));
    EXPECT_NEAR(out_delegate[i], out_ref[i], 1e-3f);
  }
}

}  // namespace
}  // namespace ynnpack
}  // namespace tflite
