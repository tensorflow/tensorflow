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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "Eigen/Core"  // from @eigen_archive
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "tensorflow/lite/delegates/ynnpack/ynnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/types/half.h"

namespace tflite {
namespace ynnpack {
namespace {

class MoeModel : public SingleOpModel {
 public:
  MoeModel(int B, int N, int D_in, int D_mid, int E, int K, bool use_delegate,
           bool scalar_scale = false, bool static_shape = true)
      : w_gate_data_(D_mid * E * D_in),
        w_up_data_(D_mid * E * D_in),
        w_down_data_(D_in * E * D_mid),
        scale_data_(scalar_scale ? 1 : E) {
    std::vector<int> tokens_shape = {B, N, D_in};
    std::vector<int> rw_shape = {B, N, K};
    std::vector<int> ei_shape = {B, N, K};
    std::vector<int> w_gate_shape = {D_mid, E, 1, D_in};
    std::vector<int> w_up_shape = {D_mid, E, 1, D_in};
    std::vector<int> w_down_shape = {D_in, E, 1, D_mid};
    std::vector<int> scale_shape =
        scalar_scale ? std::vector<int>{1} : std::vector<int>{1, 1, 1, E};
    std::vector<int> out_shape = {B, N, D_in};

    tokens_id_ = AddInput({TensorType_FLOAT32, tokens_shape});
    rw_id_ = AddInput({TensorType_FLOAT32, rw_shape});
    ei_id_ = AddInput({TensorType_INT32, ei_shape});

    for (size_t i = 0; i < w_gate_data_.size(); ++i) {
      w_gate_data_[i] = 0.1f * std::sin(static_cast<float>(i));
    }
    for (size_t i = 0; i < w_up_data_.size(); ++i) {
      w_up_data_[i] = 0.1f * std::sin(static_cast<float>(i) + 1.0f);
    }
    for (size_t i = 0; i < w_down_data_.size(); ++i) {
      w_down_data_[i] = 0.1f * std::sin(static_cast<float>(i) + 2.0f);
    }
    for (size_t i = 0; i < scale_data_.size(); ++i) {
      scale_data_[i] = 1.0f + 0.1f * std::sin(static_cast<float>(i) + 3.0f);
    }

    w_gate_id_ = AddConstInput(TensorData{TensorType_FLOAT32, w_gate_shape},
                               w_gate_data_);
    w_up_id_ =
        AddConstInput(TensorData{TensorType_FLOAT32, w_up_shape}, w_up_data_);
    w_down_id_ = AddConstInput(TensorData{TensorType_FLOAT32, w_down_shape},
                               w_down_data_);
    scale_id_ =
        AddConstInput(TensorData{TensorType_FLOAT32, scale_shape}, scale_data_);

    out_id_ = AddOutput({TensorType_FLOAT32, out_shape});

    std::vector<uint8_t> empty_attrs;
    flatbuffers::Offset<StableHLOCompositeOptions> options =
        CreateStableHLOCompositeOptionsDirect(
            builder_, "odml.moe_experts",
            /*decomposition_subgraph_index=*/1, &empty_attrs);

    SetBuiltinOp(BuiltinOperator_STABLEHLO_COMPOSITE,
                 BuiltinOptions2_StableHLOCompositeOptions, options.Union());

    BuildInterpreter({tokens_shape, rw_shape, ei_shape}, -1, false, false,
                     /*allocate_and_delegate=*/false);

    if (use_delegate) {
      TfLiteYNNPackDelegateOptions delegate_options =
          TfLiteYNNPackDelegateOptionsDefault();
      delegate_options.num_threads = 1;
      delegate_options.static_shape = static_shape;
      SetDelegate(Interpreter::TfLiteDelegatePtr(
          TfLiteYNNPackDelegateCreate(&delegate_options),
          TfLiteYNNPackDelegateDelete));
      ApplyDelegate();
    }
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
      fprintf(stderr, "Failed to allocate tensors\n");
    }
  }

  int tokens() const { return tokens_id_; }
  int rw() const { return rw_id_; }
  int ei() const { return ei_id_; }
  int w_gate() const { return w_gate_id_; }
  int w_up() const { return w_up_id_; }
  int w_down() const { return w_down_id_; }
  int scale() const { return scale_id_; }
  int out() const { return out_id_; }

  const std::vector<float>& w_gate_data() const { return w_gate_data_; }
  const std::vector<float>& w_up_data() const { return w_up_data_; }
  const std::vector<float>& w_down_data() const { return w_down_data_; }
  const std::vector<float>& scale_data() const { return scale_data_; }

  TfLiteStatus ResizeInputTensor(int id, const std::vector<int>& dims) {
    return interpreter_->ResizeInputTensor(id, dims);
  }
  TfLiteStatus AllocateTensors() { return interpreter_->AllocateTensors(); }

 private:
  int tokens_id_;
  int rw_id_;
  int ei_id_;
  int w_gate_id_;
  int w_up_id_;
  int w_down_id_;
  int scale_id_;
  int out_id_;

  std::vector<float> w_gate_data_;
  std::vector<float> w_up_data_;
  std::vector<float> w_down_data_;
  std::vector<float> scale_data_;
};

void RunReferenceMoe(int B, int N, int D_in, int D_mid, int E, int K,
                     const float* tokens, const float* rw, const int32_t* ei,
                     const float* w_gate, const float* w_up,
                     const float* w_down, const float* scale,
                     size_t num_scale_elements, float* out) {
  std::fill_n(out, B * N * D_in, 0.0f);

  int total_tokens = B * N;
  for (int i = 0; i < total_tokens; ++i) {
    const float* token = tokens + i * D_in;
    float* token_out = out + i * D_in;

    for (int k = 0; k < K; ++k) {
      int expert = ei[i * K + k];
      float weight = rw[i * K + k];
      float s = (num_scale_elements == 1) ? scale[0] : scale[expert];

      std::vector<float> gate_val(D_mid, 0.0f);
      std::vector<float> up_val(D_mid, 0.0f);

      for (int d_mid = 0; d_mid < D_mid; ++d_mid) {
        float sum_g = 0.0f;
        float sum_u = 0.0f;
        for (int d_in = 0; d_in < D_in; ++d_in) {
          int src_idx = d_mid * E * D_in + expert * D_in + d_in;
          sum_g += token[d_in] * w_gate[src_idx];
          sum_u += token[d_in] * w_up[src_idx];
        }
        float x = sum_g;
        float act = x * 0.5f * (1.0f + std::erf(x * 0.70710678118654752440f));
        gate_val[d_mid] = act * sum_u;
      }

      for (int d_in = 0; d_in < D_in; ++d_in) {
        float sum_d = 0.0f;
        for (int d_mid = 0; d_mid < D_mid; ++d_mid) {
          int src_idx = d_in * E * D_mid + expert * D_mid + d_mid;
          sum_d += gate_val[d_mid] * w_down[src_idx] * s;
        }
        token_out[d_in] += weight * sum_d;
      }
    }
  }
}

TEST(MoeTest, SingleTokenEvaluation) {
  int B = 1, N = 1, D_in = 32, D_mid = 64, E = 4, K = 2;

  MoeModel model_del(B, N, D_in, D_mid, E, K, /*use_delegate=*/true);

  std::vector<float> tokens(B * N * D_in);
  std::vector<float> rw(B * N * K);
  std::vector<int32_t> ei(B * N * K);

  for (size_t i = 0; i < tokens.size(); ++i) {
    tokens[i] = 0.5f * std::sin(static_cast<float>(i));
  }
  for (size_t i = 0; i < rw.size(); ++i) {
    rw[i] = 0.5f;
  }
  ei[0] = 1;
  ei[1] = 3;

  model_del.PopulateTensor(model_del.tokens(), tokens);
  model_del.PopulateTensor(model_del.rw(), rw);
  model_del.PopulateTensor(model_del.ei(), ei);

  ASSERT_EQ(model_del.Invoke(), kTfLiteOk);

  std::vector<float> expected_out(B * N * D_in);
  RunReferenceMoe(B, N, D_in, D_mid, E, K, tokens.data(), rw.data(), ei.data(),
                  model_del.w_gate_data().data(), model_del.w_up_data().data(),
                  model_del.w_down_data().data(), model_del.scale_data().data(),
                  model_del.scale_data().size(), expected_out.data());

  std::vector<float> del_out = model_del.ExtractVector<float>(model_del.out());

  for (size_t i = 0; i < expected_out.size(); ++i) {
    EXPECT_NEAR(del_out[i], expected_out[i], 1e-4f) << "Mismatch at " << i;
  }
}

TEST(MoeTest, BatchTokenEvaluation) {
  int B = 1, N = 4, D_in = 32, D_mid = 64, E = 4, K = 2;

  MoeModel model_del(B, N, D_in, D_mid, E, K, /*use_delegate=*/true);

  std::vector<float> tokens(B * N * D_in);
  std::vector<float> rw(B * N * K);
  std::vector<int32_t> ei(B * N * K);

  for (size_t i = 0; i < tokens.size(); ++i) {
    tokens[i] = 0.5f * std::sin(static_cast<float>(i));
  }
  for (size_t i = 0; i < rw.size(); ++i) {
    rw[i] = 0.5f;
  }
  for (size_t i = 0; i < ei.size(); ++i) {
    ei[i] = static_cast<int32_t>(i % E);
  }

  model_del.PopulateTensor(model_del.tokens(), tokens);
  model_del.PopulateTensor(model_del.rw(), rw);
  model_del.PopulateTensor(model_del.ei(), ei);

  ASSERT_EQ(model_del.Invoke(), kTfLiteOk);

  std::vector<float> expected_out(B * N * D_in);
  RunReferenceMoe(B, N, D_in, D_mid, E, K, tokens.data(), rw.data(), ei.data(),
                  model_del.w_gate_data().data(), model_del.w_up_data().data(),
                  model_del.w_down_data().data(), model_del.scale_data().data(),
                  model_del.scale_data().size(), expected_out.data());

  std::vector<float> del_out = model_del.ExtractVector<float>(model_del.out());

  for (size_t i = 0; i < expected_out.size(); ++i) {
    EXPECT_NEAR(del_out[i], expected_out[i], 1e-4f) << "Mismatch at " << i;
  }
}

TEST(MoeTest, ScalarScaleEvaluation) {
  int B = 1, N = 2, D_in = 16, D_mid = 32, E = 4, K = 2;

  MoeModel model_del(B, N, D_in, D_mid, E, K, /*use_delegate=*/true,
                     /*scalar_scale=*/true);

  std::vector<float> tokens(B * N * D_in);
  std::vector<float> rw(B * N * K);
  std::vector<int32_t> ei(B * N * K);

  for (size_t i = 0; i < tokens.size(); ++i) {
    tokens[i] = 0.3f * std::sin(static_cast<float>(i));
  }
  for (size_t i = 0; i < rw.size(); ++i) {
    rw[i] = 0.5f;
  }
  for (size_t i = 0; i < ei.size(); ++i) {
    ei[i] = static_cast<int32_t>(i % E);
  }

  model_del.PopulateTensor(model_del.tokens(), tokens);
  model_del.PopulateTensor(model_del.rw(), rw);
  model_del.PopulateTensor(model_del.ei(), ei);

  ASSERT_EQ(model_del.Invoke(), kTfLiteOk);

  std::vector<float> expected_out(B * N * D_in);
  RunReferenceMoe(B, N, D_in, D_mid, E, K, tokens.data(), rw.data(), ei.data(),
                  model_del.w_gate_data().data(), model_del.w_up_data().data(),
                  model_del.w_down_data().data(), model_del.scale_data().data(),
                  model_del.scale_data().size(), expected_out.data());

  std::vector<float> del_out = model_del.ExtractVector<float>(model_del.out());

  for (size_t i = 0; i < expected_out.size(); ++i) {
    EXPECT_NEAR(del_out[i], expected_out[i], 1e-4f) << "Mismatch at " << i;
  }
}

TEST(MoeTest, SingleExpertEvaluation) {
  int B = 1, N = 2, D_in = 16, D_mid = 32, E = 1, K = 1;

  MoeModel model_del(B, N, D_in, D_mid, E, K, /*use_delegate=*/true);

  std::vector<float> tokens(B * N * D_in);
  std::vector<float> rw(B * N * K);
  std::vector<int32_t> ei(B * N * K);

  for (size_t i = 0; i < tokens.size(); ++i) {
    tokens[i] = 0.2f * std::sin(static_cast<float>(i));
  }
  for (size_t i = 0; i < rw.size(); ++i) {
    rw[i] = 1.0f;
  }
  for (size_t i = 0; i < ei.size(); ++i) {
    ei[i] = 0;
  }

  model_del.PopulateTensor(model_del.tokens(), tokens);
  model_del.PopulateTensor(model_del.rw(), rw);
  model_del.PopulateTensor(model_del.ei(), ei);

  ASSERT_EQ(model_del.Invoke(), kTfLiteOk);

  std::vector<float> expected_out(B * N * D_in);
  RunReferenceMoe(B, N, D_in, D_mid, E, K, tokens.data(), rw.data(), ei.data(),
                  model_del.w_gate_data().data(), model_del.w_up_data().data(),
                  model_del.w_down_data().data(), model_del.scale_data().data(),
                  model_del.scale_data().size(), expected_out.data());

  std::vector<float> del_out = model_del.ExtractVector<float>(model_del.out());

  for (size_t i = 0; i < expected_out.size(); ++i) {
    EXPECT_NEAR(del_out[i], expected_out[i], 1e-4f) << "Mismatch at " << i;
  }
}

TEST(MoeTest, MultiBatchEvaluation) {
  int B = 2, N = 3, D_in = 16, D_mid = 32, E = 4, K = 2;

  MoeModel model_del(B, N, D_in, D_mid, E, K, /*use_delegate=*/true);

  std::vector<float> tokens(B * N * D_in);
  std::vector<float> rw(B * N * K);
  std::vector<int32_t> ei(B * N * K);

  for (size_t i = 0; i < tokens.size(); ++i) {
    tokens[i] = 0.3f * std::sin(static_cast<float>(i));
  }
  for (size_t i = 0; i < rw.size(); ++i) {
    rw[i] = 0.5f;
  }
  for (size_t i = 0; i < ei.size(); ++i) {
    ei[i] = static_cast<int32_t>(i % E);
  }

  model_del.PopulateTensor(model_del.tokens(), tokens);
  model_del.PopulateTensor(model_del.rw(), rw);
  model_del.PopulateTensor(model_del.ei(), ei);

  ASSERT_EQ(model_del.Invoke(), kTfLiteOk);

  std::vector<float> expected_out(B * N * D_in);
  RunReferenceMoe(B, N, D_in, D_mid, E, K, tokens.data(), rw.data(), ei.data(),
                  model_del.w_gate_data().data(), model_del.w_up_data().data(),
                  model_del.w_down_data().data(), model_del.scale_data().data(),
                  model_del.scale_data().size(), expected_out.data());

  std::vector<float> del_out = model_del.ExtractVector<float>(model_del.out());

  for (size_t i = 0; i < expected_out.size(); ++i) {
    EXPECT_NEAR(del_out[i], expected_out[i], 1e-4f) << "Mismatch at " << i;
  }
}

TEST(MoeTest, ResizeTensorEvaluation) {
  int B = 1, N = 2, D_in = 16, D_mid = 32, E = 4, K = 2;

  MoeModel model_del(B, N, D_in, D_mid, E, K, /*use_delegate=*/true);

  std::vector<float> tokens(B * N * D_in);
  std::vector<float> rw(B * N * K);
  std::vector<int32_t> ei(B * N * K);

  for (size_t i = 0; i < tokens.size(); ++i) {
    tokens[i] = 0.3f * std::sin(static_cast<float>(i));
  }
  for (size_t i = 0; i < rw.size(); ++i) {
    rw[i] = 0.5f;
  }
  for (size_t i = 0; i < ei.size(); ++i) {
    ei[i] = static_cast<int32_t>(i % E);
  }

  model_del.PopulateTensor(model_del.tokens(), tokens);
  model_del.PopulateTensor(model_del.rw(), rw);
  model_del.PopulateTensor(model_del.ei(), ei);

  ASSERT_EQ(model_del.Invoke(), kTfLiteOk);
  EXPECT_EQ(model_del.GetTensorShape(model_del.out()),
            std::vector<int>({B, N, D_in}));

  // Resize tensor inputs to B = 2, N = 4.
  int B_new = 2;
  int N_new = 4;
  ASSERT_EQ(
      model_del.ResizeInputTensor(model_del.tokens(), {B_new, N_new, D_in}),
      kTfLiteOk);
  ASSERT_EQ(model_del.ResizeInputTensor(model_del.rw(), {B_new, N_new, K}),
            kTfLiteOk);
  ASSERT_EQ(model_del.ResizeInputTensor(model_del.ei(), {B_new, N_new, K}),
            kTfLiteOk);
  ASSERT_EQ(model_del.AllocateTensors(), kTfLiteOk);
  EXPECT_EQ(model_del.GetTensorShape(model_del.out()),
            std::vector<int>({B_new, N_new, D_in}));

  std::vector<float> new_tokens(B_new * N_new * D_in);
  std::vector<float> new_rw(B_new * N_new * K);
  std::vector<int32_t> new_ei(B_new * N_new * K);

  for (size_t i = 0; i < new_tokens.size(); ++i) {
    new_tokens[i] = 0.4f * std::sin(static_cast<float>(i) + 0.5f);
  }
  for (size_t i = 0; i < new_rw.size(); ++i) {
    new_rw[i] = 0.5f;
  }
  for (size_t i = 0; i < new_ei.size(); ++i) {
    new_ei[i] = static_cast<int32_t>((i + 1) % E);
  }

  model_del.PopulateTensor(model_del.tokens(), new_tokens);
  model_del.PopulateTensor(model_del.rw(), new_rw);
  model_del.PopulateTensor(model_del.ei(), new_ei);

  ASSERT_EQ(model_del.Invoke(), kTfLiteOk);

  std::vector<float> expected_out(B_new * N_new * D_in);
  RunReferenceMoe(B_new, N_new, D_in, D_mid, E, K, new_tokens.data(),
                  new_rw.data(), new_ei.data(), model_del.w_gate_data().data(),
                  model_del.w_up_data().data(), model_del.w_down_data().data(),
                  model_del.scale_data().data(), model_del.scale_data().size(),
                  expected_out.data());

  std::vector<float> del_out = model_del.ExtractVector<float>(model_del.out());

  for (size_t i = 0; i < expected_out.size(); ++i) {
    EXPECT_NEAR(del_out[i], expected_out[i], 1e-4f) << "Mismatch at " << i;
  }
}

TEST(MoeTest, ResizeTensorDefaultOptions) {
  int B = 1, N = 2, D_in = 16, D_mid = 32, E = 4, K = 2;

  MoeModel model_del(B, N, D_in, D_mid, E, K, /*use_delegate=*/true,
                     /*scalar_scale=*/false, /*static_shape=*/false);

  std::vector<float> tokens(B * N * D_in);
  std::vector<float> rw(B * N * K);
  std::vector<int32_t> ei(B * N * K);

  for (size_t i = 0; i < tokens.size(); ++i) {
    tokens[i] = 0.3f * std::sin(static_cast<float>(i));
  }
  for (size_t i = 0; i < rw.size(); ++i) {
    rw[i] = 0.5f;
  }
  for (size_t i = 0; i < ei.size(); ++i) {
    ei[i] = static_cast<int32_t>(i % E);
  }

  model_del.PopulateTensor(model_del.tokens(), tokens);
  model_del.PopulateTensor(model_del.rw(), rw);
  model_del.PopulateTensor(model_del.ei(), ei);

  ASSERT_EQ(model_del.Invoke(), kTfLiteOk);
  EXPECT_EQ(model_del.GetTensorShape(model_del.out()),
            std::vector<int>({B, N, D_in}));

  // Resize sequence length N from 2 to 4 with static_shape = false.
  int B_new = 1;
  int N_new = 4;
  ASSERT_EQ(
      model_del.ResizeInputTensor(model_del.tokens(), {B_new, N_new, D_in}),
      kTfLiteOk);
  ASSERT_EQ(model_del.ResizeInputTensor(model_del.rw(), {B_new, N_new, K}),
            kTfLiteOk);
  ASSERT_EQ(model_del.ResizeInputTensor(model_del.ei(), {B_new, N_new, K}),
            kTfLiteOk);
  ASSERT_EQ(model_del.AllocateTensors(), kTfLiteOk);
  EXPECT_EQ(model_del.GetTensorShape(model_del.out()),
            std::vector<int>({B_new, N_new, D_in}));

  std::vector<float> new_tokens(B_new * N_new * D_in);
  std::vector<float> new_rw(B_new * N_new * K);
  std::vector<int32_t> new_ei(B_new * N_new * K);

  for (size_t i = 0; i < new_tokens.size(); ++i) {
    new_tokens[i] = 0.4f * std::sin(static_cast<float>(i) + 0.5f);
  }
  for (size_t i = 0; i < new_rw.size(); ++i) {
    new_rw[i] = 0.5f;
  }
  for (size_t i = 0; i < new_ei.size(); ++i) {
    new_ei[i] = static_cast<int32_t>((i + 1) % E);
  }

  model_del.PopulateTensor(model_del.tokens(), new_tokens);
  model_del.PopulateTensor(model_del.rw(), new_rw);
  model_del.PopulateTensor(model_del.ei(), new_ei);

  ASSERT_EQ(model_del.Invoke(), kTfLiteOk);

  std::vector<float> expected_out(B_new * N_new * D_in);
  RunReferenceMoe(B_new, N_new, D_in, D_mid, E, K, new_tokens.data(),
                  new_rw.data(), new_ei.data(), model_del.w_gate_data().data(),
                  model_del.w_up_data().data(), model_del.w_down_data().data(),
                  model_del.scale_data().data(), model_del.scale_data().size(),
                  expected_out.data());

  std::vector<float> del_out = model_del.ExtractVector<float>(model_del.out());

  for (size_t i = 0; i < expected_out.size(); ++i) {
    EXPECT_NEAR(del_out[i], expected_out[i], 1e-4f) << "Mismatch at " << i;
  }
}

template <typename InT, typename WeightT, typename OutT = InT>
class FloatMoeModel : public SingleOpModel {
 public:
  FloatMoeModel(int B, int N, int D_in, int D_mid, int E, int K,
                bool use_delegate)
      : w_gate_ref_(D_mid * E * D_in),
        w_up_ref_(D_mid * E * D_in),
        w_down_ref_(D_in * E * D_mid),
        scale_ref_(E) {
    std::vector<int> tokens_shape = {B, N, D_in};
    std::vector<int> rw_shape = {B, N, K};
    std::vector<int> ei_shape = {B, N, K};
    std::vector<int> w_gate_shape = {D_mid, E, 1, D_in};
    std::vector<int> w_up_shape = {D_mid, E, 1, D_in};
    std::vector<int> w_down_shape = {D_in, E, 1, D_mid};
    std::vector<int> scale_shape = {1, 1, 1, E};
    std::vector<int> out_shape = {B, N, D_in};

    tokens_id_ = AddInput({GetTensorType<InT>(), tokens_shape});
    rw_id_ = AddInput({GetTensorType<InT>(), rw_shape});
    ei_id_ = AddInput({TensorType_INT32, ei_shape});

    std::vector<WeightT> w_gate_data(w_gate_ref_.size());
    std::vector<WeightT> w_up_data(w_up_ref_.size());
    std::vector<WeightT> w_down_data(w_down_ref_.size());
    std::vector<InT> scale_data(scale_ref_.size());

    for (size_t i = 0; i < w_gate_data.size(); ++i) {
      float v = 0.1f * std::sin(static_cast<float>(i));
      w_gate_data[i] = static_cast<WeightT>(v);
      w_gate_ref_[i] = static_cast<float>(w_gate_data[i]);
    }
    for (size_t i = 0; i < w_up_data.size(); ++i) {
      float v = 0.1f * std::sin(static_cast<float>(i) + 1.0f);
      w_up_data[i] = static_cast<WeightT>(v);
      w_up_ref_[i] = static_cast<float>(w_up_data[i]);
    }
    for (size_t i = 0; i < w_down_data.size(); ++i) {
      float v = 0.1f * std::sin(static_cast<float>(i) + 2.0f);
      w_down_data[i] = static_cast<WeightT>(v);
      w_down_ref_[i] = static_cast<float>(w_down_data[i]);
    }
    for (size_t i = 0; i < scale_data.size(); ++i) {
      float v = 1.0f + 0.1f * std::sin(static_cast<float>(i) + 3.0f);
      scale_data[i] = static_cast<InT>(v);
      scale_ref_[i] = static_cast<float>(scale_data[i]);
    }

    w_gate_id_ = AddConstInput(
        TensorData{GetTensorType<WeightT>(), w_gate_shape}, w_gate_data);
    w_up_id_ = AddConstInput(TensorData{GetTensorType<WeightT>(), w_up_shape},
                             w_up_data);
    w_down_id_ = AddConstInput(
        TensorData{GetTensorType<WeightT>(), w_down_shape}, w_down_data);
    scale_id_ = AddConstInput(TensorData{GetTensorType<InT>(), scale_shape},
                              scale_data);

    out_id_ = AddOutput({GetTensorType<OutT>(), out_shape});

    std::vector<uint8_t> empty_attrs;
    flatbuffers::Offset<StableHLOCompositeOptions> options =
        CreateStableHLOCompositeOptionsDirect(
            builder_, "odml.moe_experts",
            /*decomposition_subgraph_index=*/1, &empty_attrs);

    SetBuiltinOp(BuiltinOperator_STABLEHLO_COMPOSITE,
                 BuiltinOptions2_StableHLOCompositeOptions, options.Union());

    BuildInterpreter({tokens_shape, rw_shape, ei_shape}, -1, false, false,
                     /*allocate_and_delegate=*/false);

    if (use_delegate) {
      TfLiteYNNPackDelegateOptions delegate_options =
          TfLiteYNNPackDelegateOptionsDefault();
      delegate_options.num_threads = 1;
      delegate_options.static_shape = true;
      SetDelegate(Interpreter::TfLiteDelegatePtr(
          TfLiteYNNPackDelegateCreate(&delegate_options),
          TfLiteYNNPackDelegateDelete));
      ApplyDelegate();
    }
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
      fprintf(stderr, "Failed to allocate tensors\n");
    }
  }

  int tokens() const { return tokens_id_; }
  int rw() const { return rw_id_; }
  int ei() const { return ei_id_; }
  int out() const { return out_id_; }

  const std::vector<float>& w_gate_ref() const { return w_gate_ref_; }
  const std::vector<float>& w_up_ref() const { return w_up_ref_; }
  const std::vector<float>& w_down_ref() const { return w_down_ref_; }
  const std::vector<float>& scale_ref() const { return scale_ref_; }

 private:
  int tokens_id_;
  int rw_id_;
  int ei_id_;
  int w_gate_id_;
  int w_up_id_;
  int w_down_id_;
  int scale_id_;
  int out_id_;

  std::vector<float> w_gate_ref_;
  std::vector<float> w_up_ref_;
  std::vector<float> w_down_ref_;
  std::vector<float> scale_ref_;
};

// Number of values LiteRT packs into a single byte for `type`.
int ValuesPerByte(TensorType type) {
  switch (type) {
    case TensorType_INT4:
      return 2;
    case TensorType_INT2:
      return 4;
    default:
      return 1;
  }
}

// Pseudo-random quantized weight values covering the representable range of
// `type`: [-15, 15] for int8 (the exact range does not matter, it just has to
// be symmetric), [-7, 7] for int4 as the LiteRT quantization spec prescribes,
// and [-2, 1] for int2, which is a plain two's complement 2-bit value.
//
// The values must not be a short cyclic pattern: a period that divides the
// contraction dimension makes every row of the weight matrix identical, and
// the reference output then collapses to near zero and compares equal to
// anything, including a delegate that returns all zeros.
std::vector<int8_t> MakeQuantizedWeights(size_t count, uint32_t seed,
                                         TensorType type) {
  int min_value = -15;
  int max_value = 15;
  switch (type) {
    case TensorType_INT4:
      min_value = -7;
      max_value = 7;
      break;
    case TensorType_INT2:
      min_value = -2;
      max_value = 1;
      break;
    default:
      break;
  }
  const uint32_t range = max_value - min_value + 1;
  std::vector<int8_t> values(count);
  uint32_t state = seed;
  for (size_t i = 0; i < count; ++i) {
    state = state * 1664525u + 1013904223u;
    values[i] = static_cast<int8_t>(min_value +
                                    static_cast<int>((state >> 16) % range));
  }
  return values;
}

// Scale of the quantized weights, chosen per type so that the MoE output has a
// comparable magnitude whatever the weight precision is.
float WeightScaleFor(TensorType type) {
  switch (type) {
    case TensorType_INT4:
      return 0.04f;
    case TensorType_INT2:
      return 0.15f;
    default:
      return 0.02f;
  }
}

// Packs `values` the way LiteRT stores sub-byte tensors: the lowest-index value
// goes into the least significant bits of the byte.
std::vector<uint8_t> PackQuantizedWeights(const std::vector<int8_t>& values,
                                          TensorType type) {
  const int values_per_byte = ValuesPerByte(type);
  const int bits = 8 / values_per_byte;
  const int mask = (1 << bits) - 1;
  std::vector<uint8_t> packed(
      (values.size() + values_per_byte - 1) / values_per_byte, 0);
  for (size_t i = 0; i < values.size(); ++i) {
    packed[i / values_per_byte] |= static_cast<uint8_t>(
        (values[i] & mask) << (bits * (i % values_per_byte)));
  }
  return packed;
}

class QuantizedMoeModel : public SingleOpModel {
 public:
  QuantizedMoeModel(int B, int N, int D_in, int D_mid, int E, int K,
                    bool use_delegate, bool explicit_scales = true,
                    TensorType scale_type = TensorType_FLOAT32,
                    TensorType weight_type = TensorType_INT8)
      : weight_scale_(WeightScaleFor(weight_type)) {
    std::vector<int> tokens_shape = {B, N, D_in};
    std::vector<int> rw_shape = {B, N, K};
    std::vector<int> ei_shape = {B, N, K};
    std::vector<int> w_gate_shape = {D_mid, E, 1, D_in};
    std::vector<int> gate_scale_shape = {D_mid, E, 1, 1};
    std::vector<int> w_up_shape = {D_mid, E, 1, D_in};
    std::vector<int> up_scale_shape = {D_mid, E, 1, 1};
    std::vector<int> w_down_shape = {D_in, E, 1, D_mid};
    std::vector<int> down_scale_shape = {D_in, E, 1, 1};
    std::vector<int> scale_shape = {1, 1, 1, E};
    std::vector<int> out_shape = {B, N, D_in};

    tokens_id_ = AddInput({TensorType_FLOAT32, tokens_shape});
    rw_id_ = AddInput({TensorType_FLOAT32, rw_shape});
    ei_id_ = AddInput({TensorType_INT32, ei_shape});

    const size_t gate_up_size = static_cast<size_t>(D_mid) * E * D_in;
    const size_t down_size = static_cast<size_t>(D_in) * E * D_mid;
    const std::vector<int8_t> w_gate_data =
        MakeQuantizedWeights(gate_up_size, /*seed=*/1, weight_type);
    const std::vector<int8_t> w_up_data =
        MakeQuantizedWeights(gate_up_size, /*seed=*/2, weight_type);
    const std::vector<int8_t> w_down_data =
        MakeQuantizedWeights(down_size, /*seed=*/3, weight_type);

    w_gate_ref_ = Dequantize(w_gate_data);
    w_up_ref_ = Dequantize(w_up_data);
    w_down_ref_ = Dequantize(w_down_data);
    scale_ref_.assign(E, 1.0f);

    const std::vector<uint8_t> w_gate_packed =
        PackQuantizedWeights(w_gate_data, weight_type);
    const std::vector<uint8_t> w_up_packed =
        PackQuantizedWeights(w_up_data, weight_type);
    const std::vector<uint8_t> w_down_packed =
        PackQuantizedWeights(w_down_data, weight_type);

    if (explicit_scales) {
      w_gate_id_ =
          AddConstInput(TensorData{weight_type, w_gate_shape}, w_gate_packed);
      gate_scale_id_ =
          AddScaleInput(scale_type, gate_scale_shape, gate_up_size / D_in);
      w_up_id_ =
          AddConstInput(TensorData{weight_type, w_up_shape}, w_up_packed);
      up_scale_id_ =
          AddScaleInput(scale_type, up_scale_shape, gate_up_size / D_in);
      w_down_id_ =
          AddConstInput(TensorData{weight_type, w_down_shape}, w_down_packed);
      down_scale_id_ =
          AddScaleInput(scale_type, down_scale_shape, down_size / D_mid);
    } else {
      w_gate_id_ = AddConstInput(
          TensorData{weight_type, w_gate_shape, 0.0f, 0.0f, weight_scale_, 0},
          w_gate_packed);
      w_up_id_ = AddConstInput(
          TensorData{weight_type, w_up_shape, 0.0f, 0.0f, weight_scale_, 0},
          w_up_packed);
      w_down_id_ = AddConstInput(
          TensorData{weight_type, w_down_shape, 0.0f, 0.0f, weight_scale_, 0},
          w_down_packed);
    }
    scale_id_ =
        AddConstInput(TensorData{TensorType_FLOAT32, scale_shape}, scale_ref_);

    out_id_ = AddOutput({TensorType_FLOAT32, out_shape});

    std::vector<uint8_t> empty_attrs;
    flatbuffers::Offset<StableHLOCompositeOptions> options =
        CreateStableHLOCompositeOptionsDirect(
            builder_, "odml.moe_experts",
            /*decomposition_subgraph_index=*/1, &empty_attrs);

    SetBuiltinOp(BuiltinOperator_STABLEHLO_COMPOSITE,
                 BuiltinOptions2_StableHLOCompositeOptions, options.Union());

    BuildInterpreter({tokens_shape, rw_shape, ei_shape}, -1, false, false,
                     /*allocate_and_delegate=*/false);

    if (use_delegate) {
      TfLiteYNNPackDelegateOptions delegate_options =
          TfLiteYNNPackDelegateOptionsDefault();
      delegate_options.num_threads = 1;
      delegate_options.static_shape = true;
      SetDelegate(Interpreter::TfLiteDelegatePtr(
          TfLiteYNNPackDelegateCreate(&delegate_options),
          TfLiteYNNPackDelegateDelete));
      ApplyDelegate();
    }
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
      fprintf(stderr, "Failed to allocate tensors\n");
    }
  }

  int tokens() const { return tokens_id_; }
  int rw() const { return rw_id_; }
  int ei() const { return ei_id_; }
  int out() const { return out_id_; }

  // The weights the delegate is expected to see, i.e. the quantized values
  // scaled back to float.
  const std::vector<float>& w_gate_ref() const { return w_gate_ref_; }
  const std::vector<float>& w_up_ref() const { return w_up_ref_; }
  const std::vector<float>& w_down_ref() const { return w_down_ref_; }
  const std::vector<float>& scale_ref() const { return scale_ref_; }

 private:
  // The weights as the delegate should see them: every element shares
  // `weight_scale_`, so dequantizing is a single multiply.
  std::vector<float> Dequantize(const std::vector<int8_t>& values) const {
    std::vector<float> dequantized(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      dequantized[i] = static_cast<float>(values[i]) * weight_scale_;
    }
    return dequantized;
  }

  int AddScaleInput(TensorType scale_type, const std::vector<int>& shape,
                    size_t count) {
    if (scale_type == TensorType_FLOAT16) {
      return AddConstInput(TensorData{TensorType_FLOAT16, shape},
                           std::vector<half>(count, half(weight_scale_)));
    }
    return AddConstInput(TensorData{TensorType_FLOAT32, shape},
                         std::vector<float>(count, weight_scale_));
  }

  const float weight_scale_;

  int tokens_id_;
  int rw_id_;
  int ei_id_;
  int w_gate_id_;
  int gate_scale_id_ = -1;
  int w_up_id_;
  int up_scale_id_ = -1;
  int w_down_id_;
  int down_scale_id_ = -1;
  int scale_id_;
  int out_id_;

  std::vector<float> w_gate_ref_;
  std::vector<float> w_up_ref_;
  std::vector<float> w_down_ref_;
  std::vector<float> scale_ref_;
};

// Runs the quantized model and compares it against the float reference
// evaluated on the dequantized weights. The activations are quantized to int8
// by the delegate, so the bound is relative to the magnitude of the reference
// output and has to absorb that error as well.
void ExpectQuantizedMoeNearReference(
    int B, int N, int D_in, int D_mid, int E, int K,
    const std::vector<float>& mock_tokens, const std::vector<int32_t>& mock_ei,
    bool explicit_scales, TensorType scale_type, TensorType weight_type,
    float relative_tolerance) {
  QuantizedMoeModel model(B, N, D_in, D_mid, E, K, /*use_delegate=*/true,
                          explicit_scales, scale_type, weight_type);

  const std::vector<float> mock_rw(B * N * K, 0.5f);

  model.PopulateTensor(model.tokens(), mock_tokens);
  model.PopulateTensor(model.rw(), mock_rw);
  model.PopulateTensor(model.ei(), mock_ei);

  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  std::vector<float> expected_out(B * N * D_in);
  RunReferenceMoe(B, N, D_in, D_mid, E, K, mock_tokens.data(), mock_rw.data(),
                  mock_ei.data(), model.w_gate_ref().data(),
                  model.w_up_ref().data(), model.w_down_ref().data(),
                  model.scale_ref().data(), model.scale_ref().size(),
                  expected_out.data());

  float max_abs_expected = 0.0f;
  for (float expected : expected_out) {
    max_abs_expected = std::max(max_abs_expected, std::abs(expected));
  }
  // A comparison against a reference that is ~0 would pass for any result, a
  // delegate returning all zeros included.
  ASSERT_GT(max_abs_expected, 1e-3f);
  const float tolerance = relative_tolerance * max_abs_expected;

  std::vector<float> delegate_output = model.ExtractVector<float>(model.out());
  for (int i = 0; i < B * N * D_in; ++i) {
    EXPECT_NEAR(delegate_output[i], expected_out[i], tolerance) << "at " << i;
  }
}

// Tokens that exercise a range of magnitudes rather than a single value, so
// the dynamic quantization of the activations is not degenerate.
std::vector<float> MakeTokens(int count) {
  std::vector<float> tokens(count);
  for (int i = 0; i < count; ++i) {
    tokens[i] = 0.2f * std::sin(static_cast<float>(i));
  }
  return tokens;
}

TEST(MoeTest, QuantizedInt8MoeDecode) {
  const int B = 1, N = 1, D_in = 32, D_mid = 16, E = 4, K = 2;
  ExpectQuantizedMoeNearReference(
      B, N, D_in, D_mid, E, K, std::vector<float>(B * N * D_in, 0.5f),
      /*mock_ei=*/{1, 3}, /*explicit_scales=*/true,
      /*scale_type=*/TensorType_FLOAT32, /*weight_type=*/TensorType_INT8,
      /*relative_tolerance=*/0.01f);
}

TEST(MoeTest, QuantizedInt8MoePrefill) {
  const int B = 1, N = 4, D_in = 32, D_mid = 16, E = 4, K = 2;
  std::vector<int32_t> mock_ei(B * N * K);
  for (size_t i = 0; i < mock_ei.size(); ++i) {
    mock_ei[i] = (i * 2 + 1) % E;
  }
  ExpectQuantizedMoeNearReference(
      B, N, D_in, D_mid, E, K, MakeTokens(B * N * D_in), mock_ei,
      /*explicit_scales=*/true, /*scale_type=*/TensorType_FLOAT32,
      /*weight_type=*/TensorType_INT8, /*relative_tolerance=*/0.01f);
}

TEST(MoeTest, AffineQuantizedInt8MoeDecode) {
  const int B = 1, N = 1, D_in = 32, D_mid = 16, E = 4, K = 2;
  ExpectQuantizedMoeNearReference(
      B, N, D_in, D_mid, E, K, std::vector<float>(B * N * D_in, 0.5f),
      /*mock_ei=*/{1, 3}, /*explicit_scales=*/false,
      /*scale_type=*/TensorType_FLOAT32, /*weight_type=*/TensorType_INT8,
      /*relative_tolerance=*/0.01f);
}

TEST(MoeTest, QuantizedInt4MoeDecode) {
  const int B = 1, N = 1, D_in = 32, D_mid = 16, E = 4, K = 2;
  ExpectQuantizedMoeNearReference(
      B, N, D_in, D_mid, E, K, std::vector<float>(B * N * D_in, 0.5f),
      /*mock_ei=*/{1, 3}, /*explicit_scales=*/true,
      /*scale_type=*/TensorType_FLOAT32, /*weight_type=*/TensorType_INT4,
      /*relative_tolerance=*/0.01f);
}

TEST(MoeTest, QuantizedInt4MoePrefill) {
  const int B = 1, N = 4, D_in = 32, D_mid = 16, E = 4, K = 2;
  std::vector<int32_t> mock_ei(B * N * K);
  for (size_t i = 0; i < mock_ei.size(); ++i) {
    mock_ei[i] = (i * 2 + 1) % E;
  }
  ExpectQuantizedMoeNearReference(
      B, N, D_in, D_mid, E, K, MakeTokens(B * N * D_in), mock_ei,
      /*explicit_scales=*/true, /*scale_type=*/TensorType_FLOAT32,
      /*weight_type=*/TensorType_INT4, /*relative_tolerance=*/0.01f);
}

TEST(MoeTest, QuantizedInt2MoeDecode) {
  const int B = 1, N = 1, D_in = 32, D_mid = 16, E = 4, K = 2;
  ExpectQuantizedMoeNearReference(
      B, N, D_in, D_mid, E, K, std::vector<float>(B * N * D_in, 0.5f),
      /*mock_ei=*/{1, 3}, /*explicit_scales=*/true,
      /*scale_type=*/TensorType_FLOAT32, /*weight_type=*/TensorType_INT2,
      /*relative_tolerance=*/0.01f);
}

TEST(MoeTest, QuantizedInt2MoePrefill) {
  const int B = 1, N = 4, D_in = 32, D_mid = 16, E = 4, K = 2;
  std::vector<int32_t> mock_ei(B * N * K);
  for (size_t i = 0; i < mock_ei.size(); ++i) {
    mock_ei[i] = (i * 2 + 1) % E;
  }
  ExpectQuantizedMoeNearReference(
      B, N, D_in, D_mid, E, K, MakeTokens(B * N * D_in), mock_ei,
      /*explicit_scales=*/true, /*scale_type=*/TensorType_FLOAT32,
      /*weight_type=*/TensorType_INT2, /*relative_tolerance=*/0.01f);
}

TEST(MoeTest, AffineQuantizedInt2MoeDecode) {
  const int B = 1, N = 1, D_in = 32, D_mid = 16, E = 4, K = 2;
  ExpectQuantizedMoeNearReference(
      B, N, D_in, D_mid, E, K, std::vector<float>(B * N * D_in, 0.5f),
      /*mock_ei=*/{1, 3}, /*explicit_scales=*/false,
      /*scale_type=*/TensorType_FLOAT32, /*weight_type=*/TensorType_INT2,
      /*relative_tolerance=*/0.01f);
}

TEST(MoeTest, Float16WeightsMoe) {
  int B = 1, N = 2, D_in = 16, D_mid = 32, E = 4, K = 2;
  FloatMoeModel<float, half, float> model(B, N, D_in, D_mid, E, K,
                                          /*use_delegate=*/true);

  std::vector<float> tokens(B * N * D_in);
  std::vector<float> rw(B * N * K);
  std::vector<int32_t> ei = {0, 2, 1, 3};

  for (size_t i = 0; i < tokens.size(); ++i) {
    tokens[i] = 0.3f * std::sin(static_cast<float>(i));
  }
  for (size_t i = 0; i < rw.size(); ++i) {
    rw[i] = 0.5f;
  }

  model.PopulateTensor(model.tokens(), tokens);
  model.PopulateTensor(model.rw(), rw);
  model.PopulateTensor(model.ei(), ei);

  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  std::vector<float> expected_out(B * N * D_in);
  RunReferenceMoe(B, N, D_in, D_mid, E, K, tokens.data(), rw.data(), ei.data(),
                  model.w_gate_ref().data(), model.w_up_ref().data(),
                  model.w_down_ref().data(), model.scale_ref().data(),
                  model.scale_ref().size(), expected_out.data());

  std::vector<float> del_out = model.ExtractVector<float>(model.out());
  for (size_t i = 0; i < expected_out.size(); ++i) {
    EXPECT_NEAR(del_out[i], expected_out[i], 1e-3f) << "Mismatch at " << i;
  }
}

TEST(MoeTest, PureFloat16Moe) {
  int B = 1, N = 2, D_in = 16, D_mid = 32, E = 4, K = 2;
  FloatMoeModel<half, half, half> model(B, N, D_in, D_mid, E, K,
                                        /*use_delegate=*/true);

  std::vector<half> tokens(B * N * D_in);
  std::vector<float> tokens_ref(tokens.size());
  std::vector<half> rw(B * N * K);
  std::vector<float> rw_ref(rw.size());
  std::vector<int32_t> ei = {0, 2, 1, 3};

  for (size_t i = 0; i < tokens.size(); ++i) {
    tokens[i] = static_cast<half>(0.3f * std::sin(static_cast<float>(i)));
    tokens_ref[i] = static_cast<float>(tokens[i]);
  }
  for (size_t i = 0; i < rw.size(); ++i) {
    rw[i] = static_cast<half>(0.5f);
    rw_ref[i] = static_cast<float>(rw[i]);
  }

  model.PopulateTensor<half>(model.tokens(), tokens);
  model.PopulateTensor<half>(model.rw(), rw);
  model.PopulateTensor(model.ei(), ei);

  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  std::vector<float> expected_out(B * N * D_in);
  RunReferenceMoe(B, N, D_in, D_mid, E, K, tokens_ref.data(), rw_ref.data(),
                  ei.data(), model.w_gate_ref().data(), model.w_up_ref().data(),
                  model.w_down_ref().data(), model.scale_ref().data(),
                  model.scale_ref().size(), expected_out.data());

  std::vector<half> del_out = model.ExtractVector<half>(model.out());
  for (size_t i = 0; i < expected_out.size(); ++i) {
    EXPECT_NEAR(static_cast<float>(del_out[i]), expected_out[i], 1e-3f)
        << "Mismatch at " << i;
  }
}

TEST(MoeTest, BFloat16WeightsMoe) {
  int B = 1, N = 2, D_in = 16, D_mid = 32, E = 4, K = 2;
  FloatMoeModel<float, Eigen::bfloat16, float> model(B, N, D_in, D_mid, E, K,
                                                     /*use_delegate=*/true);

  std::vector<float> tokens(B * N * D_in);
  std::vector<float> rw(B * N * K);
  std::vector<int32_t> ei = {0, 2, 1, 3};

  for (size_t i = 0; i < tokens.size(); ++i) {
    tokens[i] = 0.3f * std::sin(static_cast<float>(i));
  }
  for (size_t i = 0; i < rw.size(); ++i) {
    rw[i] = 0.5f;
  }

  model.PopulateTensor(model.tokens(), tokens);
  model.PopulateTensor(model.rw(), rw);
  model.PopulateTensor(model.ei(), ei);

  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  std::vector<float> expected_out(B * N * D_in);
  RunReferenceMoe(B, N, D_in, D_mid, E, K, tokens.data(), rw.data(), ei.data(),
                  model.w_gate_ref().data(), model.w_up_ref().data(),
                  model.w_down_ref().data(), model.scale_ref().data(),
                  model.scale_ref().size(), expected_out.data());

  std::vector<float> del_out = model.ExtractVector<float>(model.out());
  for (size_t i = 0; i < expected_out.size(); ++i) {
    EXPECT_NEAR(del_out[i], expected_out[i], 1e-2f) << "Mismatch at " << i;
  }
}

TEST(MoeTest, PureBFloat16Moe) {
  int B = 1, N = 2, D_in = 16, D_mid = 32, E = 4, K = 2;
  FloatMoeModel<Eigen::bfloat16, Eigen::bfloat16, Eigen::bfloat16> model(
      B, N, D_in, D_mid, E, K, /*use_delegate=*/true);

  std::vector<Eigen::bfloat16> tokens(B * N * D_in);
  std::vector<float> tokens_ref(tokens.size());
  std::vector<Eigen::bfloat16> rw(B * N * K);
  std::vector<float> rw_ref(rw.size());
  std::vector<int32_t> ei = {0, 2, 1, 3};

  for (size_t i = 0; i < tokens.size(); ++i) {
    tokens[i] =
        static_cast<Eigen::bfloat16>(0.3f * std::sin(static_cast<float>(i)));
    tokens_ref[i] = static_cast<float>(tokens[i]);
  }
  for (size_t i = 0; i < rw.size(); ++i) {
    rw[i] = static_cast<Eigen::bfloat16>(0.5f);
    rw_ref[i] = static_cast<float>(rw[i]);
  }

  model.PopulateTensor<Eigen::bfloat16>(model.tokens(), tokens);
  model.PopulateTensor<Eigen::bfloat16>(model.rw(), rw);
  model.PopulateTensor(model.ei(), ei);

  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  std::vector<float> expected_out(B * N * D_in);
  RunReferenceMoe(B, N, D_in, D_mid, E, K, tokens_ref.data(), rw_ref.data(),
                  ei.data(), model.w_gate_ref().data(), model.w_up_ref().data(),
                  model.w_down_ref().data(), model.scale_ref().data(),
                  model.scale_ref().size(), expected_out.data());

  std::vector<Eigen::bfloat16> del_out =
      model.ExtractVector<Eigen::bfloat16>(model.out());
  for (size_t i = 0; i < expected_out.size(); ++i) {
    EXPECT_NEAR(static_cast<float>(del_out[i]), expected_out[i], 1e-2f)
        << "Mismatch at " << i;
  }
}

TEST(MoeTest, QuantizedInt8MoeFloat16Scale) {
  const int B = 1, N = 2, D_in = 32, D_mid = 16, E = 4, K = 2;
  ExpectQuantizedMoeNearReference(
      B, N, D_in, D_mid, E, K, MakeTokens(B * N * D_in),
      /*mock_ei=*/{0, 2, 1, 3}, /*explicit_scales=*/true,
      /*scale_type=*/TensorType_FLOAT16, /*weight_type=*/TensorType_INT8,
      /*relative_tolerance=*/0.01f);
}

}  // namespace
}  // namespace ynnpack
}  // namespace tflite
