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
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "Eigen/Core"  // from @eigen_archive
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
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
  // If `ei_from_reshape`, the expert indices go through a reshape before the
  // MoE op, so that they are computed inside the delegated partition.
  MoeModel(int B, int N, int D_in, int D_mid, int E, int K, bool use_delegate,
           bool scalar_scale = false, bool static_shape = true,
           bool ei_from_reshape = false, bool dispatch_moe_on_host = false)
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

    if (ei_from_reshape) {
      const int ei_reshaped_id =
          AddTensor<int32_t>({TensorType_INT32, ei_shape}, {}, false);
      opcodes_.push_back(CreateOperatorCode(
          builder_,
          static_cast<int8_t>(BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES),
          /*custom_code=*/0, /*version=*/1,
          BuiltinOperator_STABLEHLO_COMPOSITE));
      opcodes_.push_back(
          CreateOperatorCode(builder_, BuiltinOperator_RESHAPE, 0, 0));
      operators_.push_back(
          CreateOperator(builder_, /*opcode_index=*/1,
                         builder_.CreateVector<int32_t>({ei_id_}),
                         builder_.CreateVector<int32_t>({ei_reshaped_id}),
                         BuiltinOptions_ReshapeOptions,
                         CreateReshapeOptions(
                             builder_, builder_.CreateVector<int32_t>(ei_shape))
                             .Union()));
      const std::vector<int32_t> moe_inputs = {
          tokens_id_, rw_id_,     ei_reshaped_id, w_gate_id_,
          w_up_id_,   w_down_id_, scale_id_};
      operators_.push_back(CreateOperator(
          builder_, /*opcode_index=*/0,
          builder_.CreateVector<int32_t>(moe_inputs),
          builder_.CreateVector<int32_t>({out_id_}), BuiltinOptions_NONE,
          /*builtin_options=*/0, /*custom_options=*/0,
          CustomOptionsFormat_FLEXBUFFERS, /*mutating_variable_inputs=*/0,
          /*intermediates=*/0, /*large_custom_options_offset=*/0,
          /*large_custom_options_size=*/0,
          BuiltinOptions2_StableHLOCompositeOptions, options.Union()));
    } else {
      SetBuiltinOp(BuiltinOperator_STABLEHLO_COMPOSITE,
                   BuiltinOptions2_StableHLOCompositeOptions, options.Union());
    }

    BuildInterpreter({tokens_shape, rw_shape, ei_shape}, -1, false, false,
                     /*allocate_and_delegate=*/false);

    if (use_delegate) {
      TfLiteYNNPackDelegateOptions delegate_options =
          TfLiteYNNPackDelegateOptionsDefault();
      delegate_options.num_threads = 1;
      delegate_options.static_shape = static_shape;
      delegate_options.dispatch_moe_on_host = dispatch_moe_on_host;
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
  size_t NumNodesInExecutionPlan() const {
    return interpreter_->execution_plan().size();
  }

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

  for (bool dispatch_moe_on_host : {false, true}) {
    SCOPED_TRACE(dispatch_moe_on_host ? "dispatch_moe_on_host"
                                      : "weight_gather");
    MoeModel model_del(B, N, D_in, D_mid, E, K, /*use_delegate=*/true,
                       /*scalar_scale=*/false, /*static_shape=*/true,
                       /*ei_from_reshape=*/false, dispatch_moe_on_host);

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
    RunReferenceMoe(
        B, N, D_in, D_mid, E, K, tokens.data(), rw.data(), ei.data(),
        model_del.w_gate_data().data(), model_del.w_up_data().data(),
        model_del.w_down_data().data(), model_del.scale_data().data(),
        model_del.scale_data().size(), expected_out.data());

    std::vector<float> del_out =
        model_del.ExtractVector<float>(model_del.out());

    for (size_t i = 0; i < expected_out.size(); ++i) {
      EXPECT_NEAR(del_out[i], expected_out[i], 1e-4f) << "Mismatch at " << i;
    }
  }
}

TEST(MoeTest, ScalarScaleEvaluation) {
  int B = 1, N = 2, D_in = 16, D_mid = 32, E = 4, K = 2;

  for (bool dispatch_moe_on_host : {false, true}) {
    SCOPED_TRACE(dispatch_moe_on_host ? "dispatch_moe_on_host"
                                      : "weight_gather");
    MoeModel model_del(B, N, D_in, D_mid, E, K, /*use_delegate=*/true,
                       /*scalar_scale=*/true, /*static_shape=*/true,
                       /*ei_from_reshape=*/false, dispatch_moe_on_host);

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
    RunReferenceMoe(
        B, N, D_in, D_mid, E, K, tokens.data(), rw.data(), ei.data(),
        model_del.w_gate_data().data(), model_del.w_up_data().data(),
        model_del.w_down_data().data(), model_del.scale_data().data(),
        model_del.scale_data().size(), expected_out.data());

    std::vector<float> del_out =
        model_del.ExtractVector<float>(model_del.out());

    for (size_t i = 0; i < expected_out.size(); ++i) {
      EXPECT_NEAR(del_out[i], expected_out[i], 1e-4f) << "Mismatch at " << i;
    }
  }
}

TEST(MoeTest, SingleExpertEvaluation) {
  int B = 1, N = 2, D_in = 16, D_mid = 32, E = 1, K = 1;

  for (bool dispatch_moe_on_host : {false, true}) {
    SCOPED_TRACE(dispatch_moe_on_host ? "dispatch_moe_on_host"
                                      : "weight_gather");
    MoeModel model_del(B, N, D_in, D_mid, E, K, /*use_delegate=*/true,
                       /*scalar_scale=*/false, /*static_shape=*/true,
                       /*ei_from_reshape=*/false, dispatch_moe_on_host);

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
    RunReferenceMoe(
        B, N, D_in, D_mid, E, K, tokens.data(), rw.data(), ei.data(),
        model_del.w_gate_data().data(), model_del.w_up_data().data(),
        model_del.w_down_data().data(), model_del.scale_data().data(),
        model_del.scale_data().size(), expected_out.data());

    std::vector<float> del_out =
        model_del.ExtractVector<float>(model_del.out());

    for (size_t i = 0; i < expected_out.size(); ++i) {
      EXPECT_NEAR(del_out[i], expected_out[i], 1e-4f) << "Mismatch at " << i;
    }
  }
}

TEST(MoeTest, MultiBatchEvaluation) {
  int B = 2, N = 3, D_in = 16, D_mid = 32, E = 4, K = 2;

  for (bool dispatch_moe_on_host : {false, true}) {
    SCOPED_TRACE(dispatch_moe_on_host ? "dispatch_moe_on_host"
                                      : "weight_gather");
    MoeModel model_del(B, N, D_in, D_mid, E, K, /*use_delegate=*/true,
                       /*scalar_scale=*/false, /*static_shape=*/true,
                       /*ei_from_reshape=*/false, dispatch_moe_on_host);

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
    RunReferenceMoe(
        B, N, D_in, D_mid, E, K, tokens.data(), rw.data(), ei.data(),
        model_del.w_gate_data().data(), model_del.w_up_data().data(),
        model_del.w_down_data().data(), model_del.scale_data().data(),
        model_del.scale_data().size(), expected_out.data());

    std::vector<float> del_out =
        model_del.ExtractVector<float>(model_del.out());

    for (size_t i = 0; i < expected_out.size(); ++i) {
      EXPECT_NEAR(del_out[i], expected_out[i], 1e-4f) << "Mismatch at " << i;
    }
  }
}

TEST(MoeTest, ResizeTensorEvaluation) {
  int B = 1, N = 2, D_in = 16, D_mid = 32, E = 4, K = 2;

  for (bool dispatch_moe_on_host : {false, true}) {
    SCOPED_TRACE(dispatch_moe_on_host ? "dispatch_moe_on_host"
                                      : "weight_gather");
    MoeModel model_del(B, N, D_in, D_mid, E, K, /*use_delegate=*/true,
                       /*scalar_scale=*/false, /*static_shape=*/true,
                       /*ei_from_reshape=*/false, dispatch_moe_on_host);

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
    RunReferenceMoe(
        B_new, N_new, D_in, D_mid, E, K, new_tokens.data(), new_rw.data(),
        new_ei.data(), model_del.w_gate_data().data(),
        model_del.w_up_data().data(), model_del.w_down_data().data(),
        model_del.scale_data().data(), model_del.scale_data().size(),
        expected_out.data());

    std::vector<float> del_out =
        model_del.ExtractVector<float>(model_del.out());

    for (size_t i = 0; i < expected_out.size(); ++i) {
      EXPECT_NEAR(del_out[i], expected_out[i], 1e-4f) << "Mismatch at " << i;
    }
  }
}

TEST(MoeTest, ResizeTensorDefaultOptions) {
  int B = 1, N = 2, D_in = 16, D_mid = 32, E = 4, K = 2;

  for (bool dispatch_moe_on_host : {false, true}) {
    SCOPED_TRACE(dispatch_moe_on_host ? "dispatch_moe_on_host"
                                      : "weight_gather");
    MoeModel model_del(B, N, D_in, D_mid, E, K, /*use_delegate=*/true,
                       /*scalar_scale=*/false, /*static_shape=*/false,
                       /*ei_from_reshape=*/false, dispatch_moe_on_host);

    for (int N_cur : {2, 4, 1}) {
      ASSERT_EQ(
          model_del.ResizeInputTensor(model_del.tokens(), {B, N_cur, D_in}),
          kTfLiteOk);
      ASSERT_EQ(model_del.ResizeInputTensor(model_del.rw(), {B, N_cur, K}),
                kTfLiteOk);
      ASSERT_EQ(model_del.ResizeInputTensor(model_del.ei(), {B, N_cur, K}),
                kTfLiteOk);
      ASSERT_EQ(model_del.AllocateTensors(), kTfLiteOk);
      EXPECT_EQ(model_del.GetTensorShape(model_del.out()),
                std::vector<int>({B, N_cur, D_in}));

      std::vector<float> tokens(B * N_cur * D_in);
      std::vector<float> rw(B * N_cur * K);
      std::vector<int32_t> ei(B * N_cur * K);

      for (size_t i = 0; i < tokens.size(); ++i) {
        tokens[i] = 0.4f * std::sin(static_cast<float>(i) + 0.5f);
      }
      for (size_t i = 0; i < rw.size(); ++i) {
        rw[i] = 0.5f;
      }
      for (size_t i = 0; i < ei.size(); ++i) {
        ei[i] = static_cast<int32_t>((i + 1) % E);
      }

      model_del.PopulateTensor(model_del.tokens(), tokens);
      model_del.PopulateTensor(model_del.rw(), rw);
      model_del.PopulateTensor(model_del.ei(), ei);

      ASSERT_EQ(model_del.Invoke(), kTfLiteOk);

      std::vector<float> expected_out(B * N_cur * D_in);
      RunReferenceMoe(
          B, N_cur, D_in, D_mid, E, K, tokens.data(), rw.data(), ei.data(),
          model_del.w_gate_data().data(), model_del.w_up_data().data(),
          model_del.w_down_data().data(), model_del.scale_data().data(),
          model_del.scale_data().size(), expected_out.data());

      std::vector<float> del_out =
          model_del.ExtractVector<float>(model_del.out());

      for (size_t i = 0; i < expected_out.size(); ++i) {
        EXPECT_NEAR(del_out[i], expected_out[i], 1e-4f) << "Mismatch at " << i;
      }
    }
  }
}

// Invokes `model` with each of `routings` as the expert indices in turn, and
// checks the output against the reference each time.
void CheckRoutings(MoeModel& model, int B, int N, int D_in, int D_mid, int E,
                   int K, const std::vector<std::vector<int32_t>>& routings) {
  std::vector<float> tokens(B * N * D_in);
  std::vector<float> rw(B * N * K);
  for (size_t i = 0; i < tokens.size(); ++i) {
    tokens[i] = 0.5f * std::sin(static_cast<float>(i));
  }
  for (size_t i = 0; i < rw.size(); ++i) {
    rw[i] = 0.25f + 0.5f * std::cos(static_cast<float>(i));
  }
  model.PopulateTensor(model.tokens(), tokens);
  model.PopulateTensor(model.rw(), rw);

  for (size_t r = 0; r < routings.size(); ++r) {
    const std::vector<int32_t>& ei = routings[r];
    ASSERT_EQ(ei.size(), B * N * K);
    model.PopulateTensor(model.ei(), ei);
    ASSERT_EQ(model.Invoke(), kTfLiteOk) << "routing " << r;

    std::vector<float> expected_out(B * N * D_in);
    RunReferenceMoe(B, N, D_in, D_mid, E, K, tokens.data(), rw.data(),
                    ei.data(), model.w_gate_data().data(),
                    model.w_up_data().data(), model.w_down_data().data(),
                    model.scale_data().data(), model.scale_data().size(),
                    expected_out.data());
    std::vector<float> out = model.ExtractVector<float>(model.out());
    for (size_t i = 0; i < expected_out.size(); ++i) {
      ASSERT_NEAR(out[i], expected_out[i], 1e-4f)
          << "routing " << r << ", mismatch at " << i;
    }
  }
}

// Routings of 8 tokens to K = 2 of E = 4 experts, which route between 0 and
// 16 (token, expert) pairs to one expert.
std::vector<std::vector<int32_t>> TestRoutings() {
  return {
      // Balanced: 4 pairs per expert.
      {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},
      // Skewed: 11 pairs to expert 2, none to expert 0.
      {2, 1, 2, 3, 2, 1, 2, 3, 2, 2, 2, 2, 2, 3, 2, 2},
      // Every token selects expert 3 twice: 16 pairs to one expert.
      {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3},
      // Balanced again.
      {3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0, 3, 2, 1, 0},
  };
}

TEST(MoeTest, RoutingChangesBetweenInvocations) {
  const int B = 2, N = 4, D_in = 32, D_mid = 64, E = 4, K = 2;
  for (bool dispatch_moe_on_host : {false, true}) {
    SCOPED_TRACE(dispatch_moe_on_host ? "dispatch_moe_on_host"
                                      : "weight_gather");
    for (bool static_shape : {true, false}) {
      SCOPED_TRACE(static_shape ? "static_shape" : "dynamic_shape");
      MoeModel model(B, N, D_in, D_mid, E, K, /*use_delegate=*/true,
                     /*scalar_scale=*/false, static_shape,
                     /*ei_from_reshape=*/false, dispatch_moe_on_host);
      CheckRoutings(model, B, N, D_in, D_mid, E, K, TestRoutings());
    }
  }
}

TEST(MoeTest, ExpertIndicesComputedInPartition) {
  const int B = 2, N = 4, D_in = 32, D_mid = 64, E = 4, K = 2;
  for (bool dispatch_moe_on_host : {false, true}) {
    SCOPED_TRACE(dispatch_moe_on_host ? "dispatch_moe_on_host"
                                      : "weight_gather");
    for (bool static_shape : {true, false}) {
      SCOPED_TRACE(static_shape ? "static_shape" : "dynamic_shape");
      MoeModel model(B, N, D_in, D_mid, E, K, /*use_delegate=*/true,
                     /*scalar_scale=*/false, static_shape,
                     /*ei_from_reshape=*/true, dispatch_moe_on_host);
      // The reshape is delegated together with the MoE op, so the expert
      // indices are not available on the host when the partition is invoked.
      ASSERT_EQ(model.NumNodesInExecutionPlan(), 1);
      CheckRoutings(model, B, N, D_in, D_mid, E, K, TestRoutings());
    }
  }
}

template <typename InT, typename WeightT, typename OutT = InT>
class FloatMoeModel : public SingleOpModel {
 public:
  FloatMoeModel(int B, int N, int D_in, int D_mid, int E, int K,
                bool use_delegate, bool dispatch_moe_on_host = false)
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
      delegate_options.dispatch_moe_on_host = dispatch_moe_on_host;
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

// `ActT` is the type of the tokens, the routing weights and the output. If
// `multi_axis_quantization` is true, the weights carry a multi-axis
// quantization that refers to their explicit scales, or, if
// `mismatched_multi_axis_scales`, the gate weights refer to the up scales.
template <typename ActT>
class QuantizedMoeModel : public SingleOpModel {
 public:
  QuantizedMoeModel(int B, int N, int D_in, int D_mid, int E, int K,
                    bool use_delegate, bool explicit_scales = true,
                    TensorType scale_type = TensorType_FLOAT32,
                    TensorType weight_type = TensorType_INT8,
                    bool multi_axis_quantization = false,
                    bool mismatched_multi_axis_scales = false,
                    bool dispatch_moe_on_host = false)
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

    tokens_id_ = AddInput({GetTensorType<ActT>(), tokens_shape});
    rw_id_ = AddInput({GetTensorType<ActT>(), rw_shape});
    ei_id_ = AddInput({TensorType_INT32, ei_shape});

    const size_t gate_up_size = static_cast<size_t>(D_mid) * E * D_in;
    const size_t down_size = static_cast<size_t>(D_in) * E * D_mid;
    const std::vector<int8_t> w_gate_data =
        MakeQuantizedWeights(gate_up_size, /*seed=*/1, weight_type);
    const std::vector<int8_t> w_up_data =
        MakeQuantizedWeights(gate_up_size, /*seed=*/2, weight_type);
    const std::vector<int8_t> w_down_data =
        MakeQuantizedWeights(down_size, /*seed=*/3, weight_type);

    const std::vector<uint8_t> w_gate_packed =
        PackQuantizedWeights(w_gate_data, weight_type);
    const std::vector<uint8_t> w_up_packed =
        PackQuantizedWeights(w_up_data, weight_type);
    const std::vector<uint8_t> w_down_packed =
        PackQuantizedWeights(w_down_data, weight_type);

    // Distinct but uniform gate and up scales detect a gate/up scale swap
    // without widening the per-row range of the `gelu(gate) * up` activations,
    // which the delegate requantizes to int8 before the down projection. The
    // down scales vary per (channel, expert) to detect a misplaced scale.
    std::vector<float> gate_scales(
        D_mid * E, explicit_scales ? 0.75f * weight_scale_ : weight_scale_);
    std::vector<float> up_scales(
        D_mid * E, explicit_scales ? 1.25f * weight_scale_ : weight_scale_);
    std::vector<float> down_scales =
        explicit_scales ? MakeWeightScales(D_in * E, /*seed=*/3)
                        : std::vector<float>(D_in * E, weight_scale_);

    if (explicit_scales) {
      w_gate_id_ =
          AddConstInput(TensorData{weight_type, w_gate_shape}, w_gate_packed);
      gate_scale_id_ = AddScaleInput(scale_type, gate_scale_shape, gate_scales);
      w_up_id_ =
          AddConstInput(TensorData{weight_type, w_up_shape}, w_up_packed);
      up_scale_id_ = AddScaleInput(scale_type, up_scale_shape, up_scales);
      w_down_id_ =
          AddConstInput(TensorData{weight_type, w_down_shape}, w_down_packed);
      down_scale_id_ = AddScaleInput(scale_type, down_scale_shape, down_scales);
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

    // After `AddScaleInput`, so that the reference uses the scales as the
    // delegate sees them.
    w_gate_ref_ = Dequantize(w_gate_data, gate_scales, /*inner=*/D_in);
    w_up_ref_ = Dequantize(w_up_data, up_scales, /*inner=*/D_in);
    w_down_ref_ = Dequantize(w_down_data, down_scales, /*inner=*/D_mid);

    scale_ref_.assign(E, 1.0f);
    scale_id_ =
        AddConstInput(TensorData{TensorType_FLOAT32, scale_shape}, scale_ref_);

    out_id_ = AddOutput({GetTensorType<ActT>(), out_shape});

    std::vector<uint8_t> empty_attrs;
    flatbuffers::Offset<StableHLOCompositeOptions> options =
        CreateStableHLOCompositeOptionsDirect(
            builder_, "odml.moe_experts",
            /*decomposition_subgraph_index=*/1, &empty_attrs);

    SetBuiltinOp(BuiltinOperator_STABLEHLO_COMPOSITE,
                 BuiltinOptions2_StableHLOCompositeOptions, options.Union());

    BuildInterpreter({tokens_shape, rw_shape, ei_shape}, -1, false, false,
                     /*allocate_and_delegate=*/false);

    if (multi_axis_quantization && explicit_scales) {
      SetMultiAxisQuantization(w_gate_id_, mismatched_multi_axis_scales
                                               ? up_scale_id_
                                               : gate_scale_id_);
      SetMultiAxisQuantization(w_up_id_, up_scale_id_);
      SetMultiAxisQuantization(w_down_id_, down_scale_id_);
    }

    if (use_delegate) {
      TfLiteYNNPackDelegateOptions delegate_options =
          TfLiteYNNPackDelegateOptionsDefault();
      delegate_options.num_threads = 1;
      delegate_options.static_shape = true;
      delegate_options.dispatch_moe_on_host = dispatch_moe_on_host;
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

  // Whether the delegate replaced the whole model.
  bool IsFullyDelegated() const {
    const std::vector<int>& plan = interpreter_->execution_plan();
    return plan.size() == 1 &&
           interpreter_->node_and_registration(plan[0])->second.builtin_code ==
               kTfLiteBuiltinDelegate;
  }

  // The weights the delegate is expected to see, i.e. the quantized values
  // scaled back to float.
  const std::vector<float>& w_gate_ref() const { return w_gate_ref_; }
  const std::vector<float>& w_up_ref() const { return w_up_ref_; }
  const std::vector<float>& w_down_ref() const { return w_down_ref_; }
  const std::vector<float>& scale_ref() const { return scale_ref_; }

 private:
  // The weights as the delegate should see them: every element of row
  // `(channel, expert)`, i.e. of `inner` consecutive values, shares one scale.
  std::vector<float> Dequantize(const std::vector<int8_t>& values,
                                const std::vector<float>& scales,
                                int inner) const {
    std::vector<float> dequantized(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      dequantized[i] = static_cast<float>(values[i]) * scales[i / inner];
    }
    return dequantized;
  }

  // Per (output channel, expert) scales of the down projection, for its
  // [D_in, E, 1, 1] scale tensor: `weight_scale_` times a factor in
  // {0.5, 0.75, 1, 1.25, 1.5} that varies with the channel and expert, so a
  // misplaced scale changes the result.
  std::vector<float> MakeWeightScales(size_t count, int seed) const {
    std::vector<float> scales(count);
    for (size_t i = 0; i < count; ++i) {
      scales[i] = weight_scale_ * (0.5f + 0.25f * ((i * 7 + seed) % 5));
    }
    return scales;
  }

  // Replaces the quantization of tensor `tensor_id` with a per output channel
  // and expert multi-axis quantization with the scales `scale_tensor_id`.
  void SetMultiAxisQuantization(int tensor_id, int scale_tensor_id) {
    TfLiteTensor* tensor = interpreter_->tensor(tensor_id);
    TfLiteQuantizationFree(&tensor->quantization);
    auto* params = static_cast<TfLiteMultiAxisQuantization*>(
        malloc(sizeof(TfLiteMultiAxisQuantization)));
    params->scales = scale_tensor_id;
    params->zero_points = -1;
    params->blocksize = 0;
    params->quantized_dimensions = TfLiteIntArrayCreate(2);
    params->quantized_dimensions->data[0] = 0;
    params->quantized_dimensions->data[1] = 1;
    tensor->quantization.type = kTfLiteMultiAxisQuantization;
    tensor->quantization.params = params;
  }

  // Adds the constant scale tensor `scales` of type `scale_type`. For fp16,
  // rounds `scales` in place to the values the delegate sees.
  int AddScaleInput(TensorType scale_type, const std::vector<int>& shape,
                    std::vector<float>& scales) {
    if (scale_type == TensorType_FLOAT16) {
      std::vector<half> scales_f16(scales.size());
      for (size_t i = 0; i < scales.size(); ++i) {
        scales_f16[i] = half(scales[i]);
        scales[i] = static_cast<float>(scales_f16[i]);
      }
      return AddConstInput(TensorData{TensorType_FLOAT16, shape}, scales_f16);
    }
    return AddConstInput(TensorData{TensorType_FLOAT32, shape}, scales);
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

template <typename T>
std::vector<T> CastVector(const std::vector<float>& values) {
  std::vector<T> cast(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    cast[i] = static_cast<T>(values[i]);
  }
  return cast;
}

template <typename T>
std::vector<float> ToFloatVector(const std::vector<T>& values) {
  std::vector<float> as_float(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    as_float[i] = static_cast<float>(values[i]);
  }
  return as_float;
}

// Runs the quantized model and compares it against the float reference
// evaluated on the dequantized weights. The activations are quantized to int8
// by the delegate, so the bound is relative to the magnitude of the reference
// output and has to absorb that error as well.
template <typename ActT = float>
void ExpectQuantizedMoeNearReference(
    int B, int N, int D_in, int D_mid, int E, int K,
    const std::vector<float>& mock_tokens, const std::vector<int32_t>& mock_ei,
    bool explicit_scales, TensorType scale_type, TensorType weight_type,
    float relative_tolerance, bool multi_axis_quantization = false) {
  for (bool dispatch_moe_on_host : {false, true}) {
    SCOPED_TRACE(dispatch_moe_on_host ? "dispatch_moe_on_host"
                                      : "weight_gather");
    QuantizedMoeModel<ActT> model(
        B, N, D_in, D_mid, E, K, /*use_delegate=*/true, explicit_scales,
        scale_type, weight_type, multi_axis_quantization,
        /*mismatched_multi_axis_scales=*/false, dispatch_moe_on_host);
    ASSERT_TRUE(model.IsFullyDelegated());

    const std::vector<ActT> tokens = CastVector<ActT>(mock_tokens);
    const std::vector<ActT> rw =
        CastVector<ActT>(std::vector<float>(B * N * K, 0.5f));

    model.template PopulateTensor<ActT>(model.tokens(), tokens);
    model.template PopulateTensor<ActT>(model.rw(), rw);
    model.PopulateTensor(model.ei(), mock_ei);

    ASSERT_EQ(model.Invoke(), kTfLiteOk);

    // The reference runs on the activations as the model stores them, so the
    // bound does not have to absorb their rounding as well.
    const std::vector<float> tokens_ref = ToFloatVector(tokens);
    const std::vector<float> rw_ref = ToFloatVector(rw);

    std::vector<float> expected_out(B * N * D_in);
    RunReferenceMoe(B, N, D_in, D_mid, E, K, tokens_ref.data(), rw_ref.data(),
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

    const std::vector<float> delegate_output =
        ToFloatVector(model.template ExtractVector<ActT>(model.out()));
    for (int i = 0; i < B * N * D_in; ++i) {
      EXPECT_NEAR(delegate_output[i], expected_out[i], tolerance) << "at " << i;
    }
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

TEST(MoeTest, MultiAxisQuantizedInt4MoeDecode) {
  const int B = 1, N = 1, D_in = 32, D_mid = 16, E = 4, K = 2;
  ExpectQuantizedMoeNearReference(
      B, N, D_in, D_mid, E, K, std::vector<float>(B * N * D_in, 0.5f),
      /*mock_ei=*/{1, 3}, /*explicit_scales=*/true,
      /*scale_type=*/TensorType_FLOAT32, /*weight_type=*/TensorType_INT4,
      /*relative_tolerance=*/0.01f, /*multi_axis_quantization=*/true);
}

TEST(MoeTest, MultiAxisQuantizedInt4MoePrefill) {
  const int B = 1, N = 4, D_in = 32, D_mid = 16, E = 4, K = 2;
  std::vector<int32_t> mock_ei(B * N * K);
  for (size_t i = 0; i < mock_ei.size(); ++i) {
    mock_ei[i] = (i * 2 + 1) % E;
  }
  ExpectQuantizedMoeNearReference(
      B, N, D_in, D_mid, E, K, MakeTokens(B * N * D_in), mock_ei,
      /*explicit_scales=*/true, /*scale_type=*/TensorType_FLOAT32,
      /*weight_type=*/TensorType_INT4, /*relative_tolerance=*/0.01f,
      /*multi_axis_quantization=*/true);
}

TEST(MoeTest, NotDelegatedIfMultiAxisScalesMismatch) {
  const int B = 1, N = 4, D_in = 32, D_mid = 16, E = 4, K = 2;
  QuantizedMoeModel<float> model(B, N, D_in, D_mid, E, K,
                                 /*use_delegate=*/true,
                                 /*explicit_scales=*/true, TensorType_FLOAT32,
                                 TensorType_INT4,
                                 /*multi_axis_quantization=*/true,
                                 /*mismatched_multi_axis_scales=*/true);
  EXPECT_FALSE(model.IsFullyDelegated());
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
  for (bool dispatch_moe_on_host : {false, true}) {
    SCOPED_TRACE(dispatch_moe_on_host ? "dispatch_moe_on_host"
                                      : "weight_gather");
    FloatMoeModel<float, half, float> model(B, N, D_in, D_mid, E, K,
                                            /*use_delegate=*/true,
                                            dispatch_moe_on_host);

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
    RunReferenceMoe(B, N, D_in, D_mid, E, K, tokens.data(), rw.data(),
                    ei.data(), model.w_gate_ref().data(),
                    model.w_up_ref().data(), model.w_down_ref().data(),
                    model.scale_ref().data(), model.scale_ref().size(),
                    expected_out.data());

    std::vector<float> del_out = model.ExtractVector<float>(model.out());
    for (size_t i = 0; i < expected_out.size(); ++i) {
      EXPECT_NEAR(del_out[i], expected_out[i], 1e-3f) << "Mismatch at " << i;
    }
  }
}

TEST(MoeTest, PureFloat16Moe) {
  int B = 1, N = 2, D_in = 16, D_mid = 32, E = 4, K = 2;
  for (bool dispatch_moe_on_host : {false, true}) {
    SCOPED_TRACE(dispatch_moe_on_host ? "dispatch_moe_on_host"
                                      : "weight_gather");
    FloatMoeModel<half, half, half> model(B, N, D_in, D_mid, E, K,
                                          /*use_delegate=*/true,
                                          dispatch_moe_on_host);

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
                    ei.data(), model.w_gate_ref().data(),
                    model.w_up_ref().data(), model.w_down_ref().data(),
                    model.scale_ref().data(), model.scale_ref().size(),
                    expected_out.data());

    std::vector<half> del_out = model.ExtractVector<half>(model.out());
    for (size_t i = 0; i < expected_out.size(); ++i) {
      EXPECT_NEAR(static_cast<float>(del_out[i]), expected_out[i], 1e-3f)
          << "Mismatch at " << i;
    }
  }
}

TEST(MoeTest, BFloat16WeightsMoe) {
  int B = 1, N = 2, D_in = 16, D_mid = 32, E = 4, K = 2;
  for (bool dispatch_moe_on_host : {false, true}) {
    SCOPED_TRACE(dispatch_moe_on_host ? "dispatch_moe_on_host"
                                      : "weight_gather");
    FloatMoeModel<float, Eigen::bfloat16, float> model(B, N, D_in, D_mid, E, K,
                                                       /*use_delegate=*/true,
                                                       dispatch_moe_on_host);

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
    RunReferenceMoe(B, N, D_in, D_mid, E, K, tokens.data(), rw.data(),
                    ei.data(), model.w_gate_ref().data(),
                    model.w_up_ref().data(), model.w_down_ref().data(),
                    model.scale_ref().data(), model.scale_ref().size(),
                    expected_out.data());

    std::vector<float> del_out = model.ExtractVector<float>(model.out());
    for (size_t i = 0; i < expected_out.size(); ++i) {
      EXPECT_NEAR(del_out[i], expected_out[i], 1e-2f) << "Mismatch at " << i;
    }
  }
}

TEST(MoeTest, PureBFloat16Moe) {
  int B = 1, N = 2, D_in = 16, D_mid = 32, E = 4, K = 2;
  for (bool dispatch_moe_on_host : {false, true}) {
    SCOPED_TRACE(dispatch_moe_on_host ? "dispatch_moe_on_host"
                                      : "weight_gather");
    FloatMoeModel<Eigen::bfloat16, Eigen::bfloat16, Eigen::bfloat16> model(
        B, N, D_in, D_mid, E, K, /*use_delegate=*/true, dispatch_moe_on_host);

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
                    ei.data(), model.w_gate_ref().data(),
                    model.w_up_ref().data(), model.w_down_ref().data(),
                    model.scale_ref().data(), model.scale_ref().size(),
                    expected_out.data());

    std::vector<Eigen::bfloat16> del_out =
        model.ExtractVector<Eigen::bfloat16>(model.out());
    for (size_t i = 0; i < expected_out.size(); ++i) {
      EXPECT_NEAR(static_cast<float>(del_out[i]), expected_out[i], 1e-2f)
          << "Mismatch at " << i;
    }
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

// 16-bit activations are quantized to int8 directly, with nothing promoting
// them to fp32 first.
TEST(MoeTest, QuantizedInt8MoeFloat16Activations) {
  const int B = 1, N = 2, D_in = 32, D_mid = 16, E = 4, K = 2;
  ExpectQuantizedMoeNearReference<half>(
      B, N, D_in, D_mid, E, K, MakeTokens(B * N * D_in),
      /*mock_ei=*/{0, 2, 1, 3}, /*explicit_scales=*/true,
      /*scale_type=*/TensorType_FLOAT32, /*weight_type=*/TensorType_INT8,
      /*relative_tolerance=*/0.01f);
}

TEST(MoeTest, QuantizedInt8MoeBFloat16Activations) {
  const int B = 1, N = 2, D_in = 32, D_mid = 16, E = 4, K = 2;
  ExpectQuantizedMoeNearReference<Eigen::bfloat16>(
      B, N, D_in, D_mid, E, K, MakeTokens(B * N * D_in),
      /*mock_ei=*/{0, 2, 1, 3}, /*explicit_scales=*/true,
      /*scale_type=*/TensorType_FLOAT32, /*weight_type=*/TensorType_INT8,
      /*relative_tolerance=*/0.02f);
}

}  // namespace
}  // namespace ynnpack
}  // namespace tflite
