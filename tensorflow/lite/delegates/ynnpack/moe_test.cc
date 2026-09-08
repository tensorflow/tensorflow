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
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "tensorflow/lite/delegates/ynnpack/ynnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/schema/schema_generated.h"

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
        float gelu = x * 0.5f * (1.0f + std::erf(x * 0.70710678118654752440f));
        gate_val[d_mid] = gelu * sum_u;
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

}  // namespace
}  // namespace ynnpack
}  // namespace tflite
