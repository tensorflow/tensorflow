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
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/ynnpack/attention_model.h"
#include "tensorflow/lite/delegates/ynnpack/ynnpack_delegate.h"

namespace tflite {
namespace ynnpack {
namespace {

class AttentionTest : public ::testing::TestWithParam<AttentionImpl> {};

TEST_P(AttentionTest, Correctness) {
  AttentionImpl impl = GetParam();
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

  for (size_t i = 0; i < q_data.size(); ++i) q_data[i] = 0.1f * (i % 10);
  for (size_t i = 0; i < k_data.size(); ++i) k_data[i] = 0.2f * (i % 10);
  for (size_t i = 0; i < v_data.size(); ++i) v_data[i] = 0.3f * (i % 10);
  for (int i = 0; i < b * t; ++i) {
    for (int j = 0; j < s; ++j) {
      mask_data[i * s + j] = (j < s_active) ? 0.0f : -1e9f;
    }
  }

  AttentionImpl ref_impl = (impl == AttentionImpl::kOdmlSdpa)
                               ? AttentionImpl::kOdmlRuntimeBmm
                               : impl;

  AttentionModel model_ref(b, t, s, h, n, scale, /*transpose_io=*/false,
                           /*use_delegate=*/false, options, ref_impl);
  model_ref.PopulateTensor(model_ref.query(), q_data);
  model_ref.PopulateTensor(model_ref.key(), k_data);
  model_ref.PopulateTensor(model_ref.value(), v_data);
  if (model_ref.runtime_bmm_params() != -1) {
    model_ref.PopulateTensor(model_ref.runtime_bmm_params(), {s_active});
  }
  model_ref.PopulateTensor(model_ref.mask(), mask_data);
  ASSERT_EQ(model_ref.Invoke(), kTfLiteOk);

  AttentionModel model(b, t, s, h, n, scale, /*transpose_io=*/false,
                       /*use_delegate=*/true, options, impl);
  model.PopulateTensor(model.query(), q_data);
  model.PopulateTensor(model.key(), k_data);
  model.PopulateTensor(model.value(), v_data);
  if (model.runtime_bmm_params() != -1) {
    model.PopulateTensor(model.runtime_bmm_params(), {s_active});
  }
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

TEST_P(AttentionTest, CorrectnessTransposed) {
  AttentionImpl impl = GetParam();
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

  std::vector<float> q_data(b * t * n * h);
  std::vector<float> k_data(b * s * n * h);
  std::vector<float> mask_data(b * 1 * t * s);

  for (size_t i = 0; i < q_data.size(); ++i) q_data[i] = 0.1f * (i % 10);
  for (size_t i = 0; i < k_data.size(); ++i) k_data[i] = 0.2f * (i % 10);
  for (int i = 0; i < b * t; ++i) {
    for (int j = 0; j < s; ++j) {
      mask_data[i * s + j] = (j < s_active) ? 0.0f : -1e9f;
    }
  }

  AttentionImpl ref_impl = (impl == AttentionImpl::kOdmlSdpa)
                               ? AttentionImpl::kOdmlRuntimeBmm
                               : impl;

  std::vector<float> v_data_model;
  std::vector<float> v_data_ref;

  if (impl == AttentionImpl::kOdmlSdpa) {
    v_data_ref.resize(b * n * h * s);
    for (size_t i = 0; i < v_data_ref.size(); ++i)
      v_data_ref[i] = 0.3f * (i % 10);

    v_data_model.resize(b * s * n * h);
    for (int ib = 0; ib < b; ++ib) {
      for (int is = 0; is < s; ++is) {
        for (int in = 0; in < n; ++in) {
          for (int ih = 0; ih < h; ++ih) {
            int attn_idx = ((ib * h + ih) * n + in) * s + is;
            int sdpa_idx = ((ib * s + is) * n + in) * h + ih;
            v_data_model[sdpa_idx] = v_data_ref[attn_idx];
          }
        }
      }
    }
  } else {
    int size = b * n * h * s;
    v_data_model.resize(size);
    for (size_t i = 0; i < v_data_model.size(); ++i)
      v_data_model[i] = 0.3f * (i % 10);
    v_data_ref = v_data_model;
  }

  AttentionModel model_ref(b, t, s, h, n, scale, /*transpose_io=*/true,
                           /*use_delegate=*/false, options, ref_impl);
  model_ref.PopulateTensor(model_ref.query(), q_data);
  model_ref.PopulateTensor(model_ref.key(), k_data);
  model_ref.PopulateTensor(model_ref.value(), v_data_ref);
  if (model_ref.runtime_bmm_params() != -1) {
    model_ref.PopulateTensor(model_ref.runtime_bmm_params(), {s_active});
  }
  model_ref.PopulateTensor(model_ref.mask(), mask_data);
  ASSERT_EQ(model_ref.Invoke(), kTfLiteOk);

  AttentionModel model(b, t, s, h, n, scale, /*transpose_io=*/true,
                       /*use_delegate=*/true, options, impl);
  model.PopulateTensor(model.query(), q_data);
  model.PopulateTensor(model.key(), k_data);
  model.PopulateTensor(model.value(), v_data_model);
  if (model.runtime_bmm_params() != -1) {
    model.PopulateTensor(model.runtime_bmm_params(), {s_active});
  }
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

TEST_P(AttentionTest, LargeShapeBounds) {
  AttentionImpl impl = GetParam();
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

  AttentionImpl ref_impl = (impl == AttentionImpl::kOdmlSdpa)
                               ? AttentionImpl::kOdmlRuntimeBmm
                               : impl;

  AttentionModel model(b, t, s, h, n, scale, /*transpose_io=*/false,
                       /*use_delegate=*/true, options, impl);
  model.PopulateTensor(model.query(), q_data);
  model.PopulateTensor(model.key(), k_data);
  model.PopulateTensor(model.value(), v_data);
  if (model.runtime_bmm_params() != -1) {
    model.PopulateTensor(model.runtime_bmm_params(), {s_active});
  }
  model.PopulateTensor(model.mask(), mask_data);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  AttentionModel model_ref(b, t, s, h, n, scale, /*transpose_io=*/false,
                           /*use_delegate=*/false, options, ref_impl);
  model_ref.PopulateTensor(model_ref.query(), q_data);
  model_ref.PopulateTensor(model_ref.key(), k_data);
  model_ref.PopulateTensor(model_ref.value(), v_data);
  if (model_ref.runtime_bmm_params() != -1) {
    model_ref.PopulateTensor(model_ref.runtime_bmm_params(), {s_active});
  }
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

std::string PrintAttentionImplName(
    const ::testing::TestParamInfo<AttentionImpl>& info) {
  switch (info.param) {
    case AttentionImpl::kFullSequence:
      return "kFullSequence";
    case AttentionImpl::kOdmlRuntimeBmm:
      return "kOdmlRuntimeBmm";
    case AttentionImpl::kOdmlSdpa:
      return "kOdmlSdpa";
  }
}

INSTANTIATE_TEST_SUITE_P(AttentionTest, AttentionTest,
                         ::testing::Values(AttentionImpl::kFullSequence,
                                           AttentionImpl::kOdmlRuntimeBmm,
                                           AttentionImpl::kOdmlSdpa),
                         PrintAttentionImplName);

}  // namespace
}  // namespace ynnpack
}  // namespace tflite
