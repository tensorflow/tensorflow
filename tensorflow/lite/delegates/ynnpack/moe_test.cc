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
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "Eigen/Core"  // from @eigen_archive
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/delegates/ynnpack/ynnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/test_util.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/types/half.h"

namespace tflite {
namespace ynnpack {
namespace {

class MoeModel : public SingleOpModel {
 public:
  // If `ei_from_reshape`, the expert indices go through a reshape before the
  // MoE op, so that `expert_indices` is a partition-internal view of the
  // external partition input.
  MoeModel(int B, int N, int D_in, int D_mid, int E, int K, bool use_delegate,
           bool scalar_scale = false, bool static_shape = true,
           bool ei_from_reshape = false)
      : w_gate_data_(D_mid * E * D_in),
        w_up_data_(D_mid * E * D_in),
        w_down_data_(D_in * E * D_mid),
        scale_data_(scalar_scale ? 1 : E) {
    std::vector<int> tokens_shape = {B, N, D_in};
    std::vector<int> rw_shape = {B, N, K};
    std::vector<int> ei_shape = {B, N, K};
    std::vector<int> ei_in_shape =
        ei_from_reshape ? std::vector<int>{B * N, K} : ei_shape;
    std::vector<int> w_gate_shape = {D_mid, E, 1, D_in};
    std::vector<int> w_up_shape = {D_mid, E, 1, D_in};
    std::vector<int> w_down_shape = {D_in, E, 1, D_mid};
    std::vector<int> scale_shape =
        scalar_scale ? std::vector<int>{1} : std::vector<int>{1, 1, 1, E};
    std::vector<int> out_shape = {B, N, D_in};

    tokens_id_ = AddInput({TensorType_FLOAT32, tokens_shape});
    rw_id_ = AddInput({TensorType_FLOAT32, rw_shape});
    ei_id_ = AddInput({TensorType_INT32, ei_in_shape});

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

    BuildInterpreter({tokens_shape, rw_shape, ei_in_shape}, -1, false, false,
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

  for (int B_cur : {1, 2}) {
    for (int N_cur : {2, 4, 1}) {
      for (int K_cur : {2, 3, 1}) {
        ASSERT_EQ(model_del.ResizeInputTensor(model_del.tokens(),
                                              {B_cur, N_cur, D_in}),
                  kTfLiteOk);
        ASSERT_EQ(
            model_del.ResizeInputTensor(model_del.rw(), {B_cur, N_cur, K_cur}),
            kTfLiteOk);
        ASSERT_EQ(
            model_del.ResizeInputTensor(model_del.ei(), {B_cur, N_cur, K_cur}),
            kTfLiteOk);
        ASSERT_EQ(model_del.AllocateTensors(), kTfLiteOk);
        EXPECT_EQ(model_del.GetTensorShape(model_del.out()),
                  std::vector<int>({B_cur, N_cur, D_in}));

        std::vector<float> tokens(B_cur * N_cur * D_in);
        std::vector<float> rw(B_cur * N_cur * K_cur);
        std::vector<int32_t> ei(B_cur * N_cur * K_cur);

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

        std::vector<float> expected_out(B_cur * N_cur * D_in);
        RunReferenceMoe(B_cur, N_cur, D_in, D_mid, E, K_cur, tokens.data(),
                        rw.data(), ei.data(), model_del.w_gate_data().data(),
                        model_del.w_up_data().data(),
                        model_del.w_down_data().data(),
                        model_del.scale_data().data(),
                        model_del.scale_data().size(), expected_out.data());

        std::vector<float> del_out =
            model_del.ExtractVector<float>(model_del.out());

        for (size_t i = 0; i < expected_out.size(); ++i) {
          EXPECT_NEAR(del_out[i], expected_out[i], 1e-4f)
              << "Mismatch at " << i;
        }
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
  for (bool static_shape : {true, false}) {
    SCOPED_TRACE(static_shape ? "static_shape" : "dynamic_shape");
    MoeModel model(B, N, D_in, D_mid, E, K, /*use_delegate=*/true,
                   /*scalar_scale=*/false, static_shape);
    CheckRoutings(model, B, N, D_in, D_mid, E, K, TestRoutings());
  }
}

TEST(MoeTest, ExpertIndicesComputedInPartition) {
  const int B = 2, N = 4, D_in = 32, D_mid = 64, E = 4, K = 2;
  for (bool static_shape : {true, false}) {
    SCOPED_TRACE(static_shape ? "static_shape" : "dynamic_shape");
    MoeModel model(B, N, D_in, D_mid, E, K, /*use_delegate=*/true,
                   /*scalar_scale=*/false, static_shape,
                   /*ei_from_reshape=*/true);
    // The reshape is delegated together with the MoE op in a single partition.
    ASSERT_EQ(model.NumNodesInExecutionPlan(), 1u);
    CheckRoutings(model, B, N, D_in, D_mid, E, K, TestRoutings());
  }
}

TEST(MoeTest, OutOfRangeExpertIndexFails) {
  const int B = 1, N = 2, D_in = 16, D_mid = 32, E = 4, K = 2;
  MoeModel model(B, N, D_in, D_mid, E, K, /*use_delegate=*/true);
  model.PopulateTensor(model.tokens(), std::vector<float>(B * N * D_in, 0.5f));
  model.PopulateTensor(model.rw(), std::vector<float>(B * N * K, 0.5f));

  model.PopulateTensor(model.ei(), std::vector<int32_t>{0, -1, 1, 2});
  EXPECT_EQ(model.Invoke(), kTfLiteError);

  model.PopulateTensor(model.ei(), std::vector<int32_t>{0, 1, E, 2});
  EXPECT_EQ(model.Invoke(), kTfLiteError);
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
                    bool mismatched_multi_axis_scales = false)
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
  QuantizedMoeModel<ActT> model(B, N, D_in, D_mid, E, K, /*use_delegate=*/true,
                                explicit_scales, scale_type, weight_type,
                                multi_axis_quantization);
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

std::vector<float> DequantizeBlockwise(const std::vector<int8_t>& values,
                                       const std::vector<float>& scales,
                                       const std::vector<int32_t>& zero_points,
                                       int D_out, int E, int D_in,
                                       int block_size) {
  int num_blocks = D_in / block_size;
  std::vector<float> dequantized(values.size());
  for (int d = 0; d < D_out; ++d) {
    for (int e = 0; e < E; ++e) {
      for (int k = 0; k < D_in; ++k) {
        int val_idx = (d * E + e) * D_in + k;
        int b = k / block_size;
        int scale_idx = (d * E + e) * num_blocks + b;
        float s = scales[scale_idx];
        float zp = zero_points.empty()
                       ? 0.0f
                       : static_cast<float>(zero_points[scale_idx]);
        dequantized[val_idx] = s * (static_cast<float>(values[val_idx]) - zp);
      }
    }
  }
  return dequantized;
}

template <typename ActT = float>
class BlockwiseQuantizedMoeModel {
 public:
  BlockwiseQuantizedMoeModel(int B, int N, int D_in, int D_mid, int E, int K,
                             int block_size, bool use_delegate,
                             TensorType scale_type = TensorType_FLOAT32,
                             TensorType weight_type = TensorType_INT8,
                             bool has_zero_points = false,
                             bool enable_blockwise_quantization = true,
                             bool use_wire_default_zero_zp = false,
                             bool static_shape = true, int num_threads = 1)
      : block_size_(block_size) {
    std::vector<int32_t> tokens_shape = {B, N, D_in};
    std::vector<int32_t> rw_shape = {B, N, K};
    std::vector<int32_t> ei_shape = {B, N, K};
    std::vector<int32_t> w_gate_shape = {D_mid, E, 1, D_in};
    std::vector<int32_t> w_up_shape = {D_mid, E, 1, D_in};
    std::vector<int32_t> w_down_shape = {D_in, E, 1, D_mid};
    std::vector<int32_t> scale_shape = {1, 1, 1, E};
    std::vector<int32_t> out_shape = {B, N, D_in};

    const size_t gate_up_size = static_cast<size_t>(D_mid) * E * D_in;
    const size_t down_size = static_cast<size_t>(D_in) * E * D_mid;

    const std::vector<int8_t> w_gate_data =
        MakeQuantizedWeights(gate_up_size, /*seed=*/1, weight_type);
    const std::vector<int8_t> w_up_data =
        MakeQuantizedWeights(gate_up_size, /*seed=*/2, weight_type);
    const std::vector<int8_t> w_down_data =
        MakeQuantizedWeights(down_size, /*seed=*/3, weight_type);

    const int num_blocks_gate_up = D_in / block_size;
    const int num_blocks_down = D_mid / block_size;
    const size_t gate_up_scale_count =
        static_cast<size_t>(D_mid) * E * num_blocks_gate_up;
    const size_t down_scale_count =
        static_cast<size_t>(D_in) * E * num_blocks_down;

    const float base_scale = WeightScaleFor(weight_type);
    std::vector<float> gate_scales(gate_up_scale_count);
    std::vector<float> up_scales(gate_up_scale_count);
    std::vector<float> down_scales(down_scale_count);

    for (size_t i = 0; i < gate_up_scale_count; ++i) {
      gate_scales[i] =
          base_scale * (0.8f + 0.4f * std::sin(static_cast<float>(i)));
      up_scales[i] =
          base_scale * (0.8f + 0.4f * std::sin(static_cast<float>(i) + 1.0f));
    }
    for (size_t i = 0; i < down_scale_count; ++i) {
      down_scales[i] =
          base_scale * (0.8f + 0.4f * std::sin(static_cast<float>(i) + 2.0f));
    }

    if (scale_type == TensorType_FLOAT16) {
      for (size_t i = 0; i < gate_up_scale_count; ++i) {
        gate_scales[i] = static_cast<float>(half(gate_scales[i]));
        up_scales[i] = static_cast<float>(half(up_scales[i]));
      }
      for (size_t i = 0; i < down_scale_count; ++i) {
        down_scales[i] = static_cast<float>(half(down_scales[i]));
      }
    }

    std::vector<int32_t> gate_zp;
    std::vector<int32_t> up_zp;
    std::vector<int32_t> down_zp;
    if (has_zero_points) {
      gate_zp.resize(gate_up_scale_count);
      up_zp.resize(gate_up_scale_count);
      down_zp.resize(down_scale_count);
      for (size_t i = 0; i < gate_up_scale_count; ++i) {
        gate_zp[i] = static_cast<int32_t>((i % 5) - 2);
        up_zp[i] = static_cast<int32_t>(((i + 1) % 5) - 2);
      }
      for (size_t i = 0; i < down_scale_count; ++i) {
        down_zp[i] = static_cast<int32_t>(((i + 2) % 5) - 2);
      }
    }

    w_gate_ref_ = DequantizeBlockwise(w_gate_data, gate_scales, gate_zp, D_mid,
                                      E, D_in, block_size);
    w_up_ref_ = DequantizeBlockwise(w_up_data, up_scales, up_zp, D_mid, E, D_in,
                                    block_size);
    w_down_ref_ = DequantizeBlockwise(w_down_data, down_scales, down_zp, D_in,
                                      E, D_mid, block_size);
    scale_ref_.assign(E, 1.0f);

    const std::vector<uint8_t> w_gate_packed =
        PackQuantizedWeights(w_gate_data, weight_type);
    const std::vector<uint8_t> w_up_packed =
        PackQuantizedWeights(w_up_data, weight_type);
    const std::vector<uint8_t> w_down_packed =
        PackQuantizedWeights(w_down_data, weight_type);

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<Tensor>> tensors;
    std::vector<flatbuffers::Offset<Buffer>> buffers;
    buffers.push_back(CreateBufferDirect(builder));

    auto add_buffer = [&](const void* data, size_t bytes) -> int {
      int id = buffers.size();
      buffers.push_back(CreateBuffer(
          builder,
          builder.CreateVector(reinterpret_cast<const uint8_t*>(data), bytes)));
      return id;
    };

    tokens_id_ = tensors.size();
    tensors.push_back(CreateTensorDirect(builder, &tokens_shape,
                                         GetTensorType<ActT>(), 0, "tokens"));

    rw_id_ = tensors.size();
    tensors.push_back(
        CreateTensorDirect(builder, &rw_shape, GetTensorType<ActT>(), 0, "rw"));

    ei_id_ = tensors.size();
    tensors.push_back(
        CreateTensorDirect(builder, &ei_shape, TensorType_INT32, 0, "ei"));

    auto add_blockwise_weight = [&](const std::vector<int32_t>& shape,
                                    const std::vector<uint8_t>& packed_data,
                                    const std::vector<float>& scales,
                                    const std::vector<int32_t>& zp,
                                    const char* name) -> int {
      int weight_id = tensors.size();
      std::vector<int32_t> s_shape = shape;
      s_shape.back() /= block_size;

      int scale_tensor_id = weight_id + 1;
      int zp_tensor_id =
          zp.empty() ? (use_wire_default_zero_zp ? 0 : -1) : (weight_id + 2);

      auto bq = CreateBlockwiseQuantizationDirect(builder, scale_tensor_id,
                                                  zp_tensor_id, block_size);
      auto q_params = CreateQuantizationParameters(
          builder, 0, 0, 0, 0, QuantizationDetails_BlockwiseQuantization,
          bq.Union(), shape.size() - 1);

      int w_buf = add_buffer(packed_data.data(), packed_data.size());
      tensors.push_back(CreateTensorDirect(builder, &shape, weight_type, w_buf,
                                           name, q_params));

      if (scale_type == TensorType_FLOAT16) {
        std::vector<half> fp16_scales(scales.size());
        for (size_t i = 0; i < scales.size(); ++i) {
          fp16_scales[i] = half(scales[i]);
        }
        int s_buf =
            add_buffer(fp16_scales.data(), fp16_scales.size() * sizeof(half));
        tensors.push_back(
            CreateTensorDirect(builder, &s_shape, TensorType_FLOAT16, s_buf));
      } else {
        int s_buf = add_buffer(scales.data(), scales.size() * sizeof(float));
        tensors.push_back(
            CreateTensorDirect(builder, &s_shape, TensorType_FLOAT32, s_buf));
      }

      if (!zp.empty()) {
        int zp_buf = add_buffer(zp.data(), zp.size() * sizeof(int32_t));
        tensors.push_back(
            CreateTensorDirect(builder, &s_shape, TensorType_INT32, zp_buf));
      }
      return weight_id;
    };

    w_gate_id_ = add_blockwise_weight(w_gate_shape, w_gate_packed, gate_scales,
                                      gate_zp, "w_gate");
    w_up_id_ =
        add_blockwise_weight(w_up_shape, w_up_packed, up_scales, up_zp, "w_up");
    w_down_id_ = add_blockwise_weight(w_down_shape, w_down_packed, down_scales,
                                      down_zp, "w_down");

    int scale_buf =
        add_buffer(scale_ref_.data(), scale_ref_.size() * sizeof(float));
    scale_id_ = tensors.size();
    tensors.push_back(CreateTensorDirect(
        builder, &scale_shape, TensorType_FLOAT32, scale_buf, "scale"));

    out_id_ = tensors.size();
    tensors.push_back(CreateTensorDirect(builder, &out_shape,
                                         GetTensorType<ActT>(), 0, "out"));

    std::vector<flatbuffers::Offset<OperatorCode>> opcodes = {
        CreateOperatorCode(builder,
                           BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES, 0,
                           1, BuiltinOperator_STABLEHLO_COMPOSITE)};

    std::vector<uint8_t> empty_attrs;
    auto composite_opts = CreateStableHLOCompositeOptionsDirect(
        builder, "odml.moe_experts", 1, &empty_attrs);

    std::vector<int32_t> op_inputs = {tokens_id_, rw_id_,   ei_id_,
                                      w_gate_id_, w_up_id_, w_down_id_,
                                      scale_id_};
    std::vector<int32_t> op_outputs = {out_id_};

    std::vector<flatbuffers::Offset<Operator>> operators = {CreateOperator(
        builder, 0, builder.CreateVector(op_inputs),
        builder.CreateVector(op_outputs), BuiltinOptions_NONE, 0, 0,
        CustomOptionsFormat_FLEXBUFFERS, 0, 0, 0, 0,
        BuiltinOptions2_StableHLOCompositeOptions, composite_opts.Union())};

    std::vector<int32_t> subgraph_inputs = {tokens_id_, rw_id_, ei_id_};
    std::vector<int32_t> subgraph_outputs = {out_id_};

    std::vector<flatbuffers::Offset<SubGraph>> subgraphs = {
        CreateSubGraphDirect(builder, &tensors, &subgraph_inputs,
                             &subgraph_outputs, &operators)};

    auto model = CreateModelDirect(builder, /*version=*/3, &opcodes, &subgraphs,
                                   "blockwise_moe", &buffers);
    FinishModelBuffer(builder, model);

    model_buffer_.assign(
        reinterpret_cast<const uint8_t*>(builder.GetBufferPointer()),
        reinterpret_cast<const uint8_t*>(builder.GetBufferPointer()) +
            builder.GetSize());

    model_ = FlatBufferModel::BuildFromBuffer(
        reinterpret_cast<const char*>(model_buffer_.data()),
        model_buffer_.size());
    if (!model_) {
      fprintf(stderr, "Failed to build FlatBufferModel\n");
      return;
    }

    TfLiteRegistration dummy_reg{};
    SingleOpResolver resolver(BuiltinOperator_STABLEHLO_COMPOSITE, &dummy_reg);
    if (InterpreterBuilder(*model_, resolver)(&interpreter_) != kTfLiteOk) {
      fprintf(stderr, "Failed to build interpreter\n");
      return;
    }

    if (use_delegate) {
      TfLiteYNNPackDelegateOptions delegate_options =
          TfLiteYNNPackDelegateOptionsDefault();
      delegate_options.num_threads = num_threads;
      delegate_options.static_shape = static_shape;
      delegate_options.enable_blockwise_quantization =
          enable_blockwise_quantization;
      delegate_ = Interpreter::TfLiteDelegatePtr(
          TfLiteYNNPackDelegateCreate(&delegate_options),
          TfLiteYNNPackDelegateDelete);
      if (interpreter_->ModifyGraphWithDelegate(delegate_.get()) == kTfLiteOk) {
        ++num_applied_delegates_;
      }
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

  template <typename T>
  void PopulateTensor(int index, const std::vector<T>& data) {
    TfLiteTensor* tensor = interpreter_->tensor(index);
    std::memcpy(tensor->data.raw, data.data(), data.size() * sizeof(T));
  }

  template <typename T>
  std::vector<T> ExtractVector(int index) const {
    const TfLiteTensor* tensor = interpreter_->tensor(index);
    const T* data = reinterpret_cast<const T*>(tensor->data.raw);
    int num_elements = tflite::NumElements(tensor);
    return std::vector<T>(data, data + num_elements);
  }

  TfLiteStatus ResizeInputTensor(int tensor_index,
                                 const std::vector<int>& new_shape) {
    return interpreter_->ResizeInputTensor(tensor_index, new_shape);
  }
  TfLiteStatus AllocateTensors() { return interpreter_->AllocateTensors(); }
  std::vector<int> GetTensorShape(int index) const {
    const TfLiteTensor* tensor = interpreter_->tensor(index);
    return std::vector<int>(tensor->dims->data,
                            tensor->dims->data + tensor->dims->size);
  }
  TfLiteStatus Invoke() { return interpreter_->Invoke(); }
  int GetNumberOfAppliedDelegates() const { return num_applied_delegates_; }

 private:
  const int block_size_;
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

  std::vector<uint8_t> model_buffer_;
  std::unique_ptr<FlatBufferModel> model_;
  Interpreter::TfLiteDelegatePtr delegate_{nullptr, nullptr};
  std::unique_ptr<Interpreter> interpreter_;
  int num_applied_delegates_ = 0;
};

template <typename ActT = float>
void ExpectBlockwiseQuantizedMoeNearReference(
    int B, int N, int D_in, int D_mid, int E, int K, int block_size,
    const std::vector<float>& mock_tokens, const std::vector<int32_t>& mock_ei,
    TensorType scale_type, TensorType weight_type, bool has_zero_points,
    float relative_tolerance) {
  BlockwiseQuantizedMoeModel<ActT> model(
      B, N, D_in, D_mid, E, K, block_size,
      /*use_delegate=*/true, scale_type, weight_type, has_zero_points,
      /*enable_blockwise_quantization=*/true);

  const std::vector<ActT> tokens = CastVector<ActT>(mock_tokens);
  const std::vector<ActT> rw =
      CastVector<ActT>(std::vector<float>(B * N * K, 0.5f));

  model.template PopulateTensor<ActT>(model.tokens(), tokens);
  model.template PopulateTensor<ActT>(model.rw(), rw);
  model.PopulateTensor(model.ei(), mock_ei);

  ASSERT_EQ(model.Invoke(), kTfLiteOk);

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
  ASSERT_GT(max_abs_expected, 1e-3f);
  const float tolerance = relative_tolerance * max_abs_expected;

  const std::vector<float> delegate_output =
      ToFloatVector(model.template ExtractVector<ActT>(model.out()));
  for (int i = 0; i < B * N * D_in; ++i) {
    EXPECT_NEAR(delegate_output[i], expected_out[i], tolerance) << "at " << i;
  }
}

TEST(MoeTest, BlockwiseInt8MoeDecode) {
  const int B = 1, N = 1, D_in = 64, D_mid = 128, E = 4, K = 2, block_size = 32;
  ExpectBlockwiseQuantizedMoeNearReference(
      B, N, D_in, D_mid, E, K, block_size,
      std::vector<float>(B * N * D_in, 0.5f),
      /*mock_ei=*/{1, 3}, /*scale_type=*/TensorType_FLOAT32,
      /*weight_type=*/TensorType_INT8, /*has_zero_points=*/false,
      /*relative_tolerance=*/0.02f);
}

TEST(MoeTest, BlockwiseInt8MoePrefill) {
  const int B = 1, N = 4, D_in = 64, D_mid = 128, E = 4, K = 2, block_size = 32;
  std::vector<int32_t> mock_ei(B * N * K);
  for (size_t i = 0; i < mock_ei.size(); ++i) {
    mock_ei[i] = (i * 2 + 1) % E;
  }
  ExpectBlockwiseQuantizedMoeNearReference(
      B, N, D_in, D_mid, E, K, block_size, MakeTokens(B * N * D_in), mock_ei,
      /*scale_type=*/TensorType_FLOAT32, /*weight_type=*/TensorType_INT8,
      /*has_zero_points=*/false, /*relative_tolerance=*/0.02f);
}

TEST(MoeTest, BlockwiseInt4MoeDecode) {
  const int B = 1, N = 1, D_in = 64, D_mid = 128, E = 4, K = 2, block_size = 32;
  ExpectBlockwiseQuantizedMoeNearReference(
      B, N, D_in, D_mid, E, K, block_size,
      std::vector<float>(B * N * D_in, 0.5f),
      /*mock_ei=*/{1, 3}, /*scale_type=*/TensorType_FLOAT32,
      /*weight_type=*/TensorType_INT4, /*has_zero_points=*/false,
      /*relative_tolerance=*/0.02f);
}

TEST(MoeTest, BlockwiseInt4MoePrefill) {
  const int B = 1, N = 4, D_in = 64, D_mid = 128, E = 4, K = 2, block_size = 32;
  std::vector<int32_t> mock_ei(B * N * K);
  for (size_t i = 0; i < mock_ei.size(); ++i) {
    mock_ei[i] = (i * 2 + 1) % E;
  }
  ExpectBlockwiseQuantizedMoeNearReference(
      B, N, D_in, D_mid, E, K, block_size, MakeTokens(B * N * D_in), mock_ei,
      /*scale_type=*/TensorType_FLOAT32, /*weight_type=*/TensorType_INT4,
      /*has_zero_points=*/false, /*relative_tolerance=*/0.02f);
}

TEST(MoeTest, BlockwiseInt8MoeWithZeroPoints) {
  const int B = 1, N = 2, D_in = 64, D_mid = 128, E = 4, K = 2, block_size = 32;
  std::vector<int32_t> mock_ei = {0, 2, 1, 3};
  ExpectBlockwiseQuantizedMoeNearReference(
      B, N, D_in, D_mid, E, K, block_size, MakeTokens(B * N * D_in), mock_ei,
      /*scale_type=*/TensorType_FLOAT32, /*weight_type=*/TensorType_INT8,
      /*has_zero_points=*/true, /*relative_tolerance=*/0.02f);
}

TEST(MoeTest, BlockwiseInt8MoeFloat16Scale) {
  const int B = 1, N = 2, D_in = 64, D_mid = 128, E = 4, K = 2, block_size = 32;
  std::vector<int32_t> mock_ei = {0, 2, 1, 3};
  ExpectBlockwiseQuantizedMoeNearReference(
      B, N, D_in, D_mid, E, K, block_size, MakeTokens(B * N * D_in), mock_ei,
      /*scale_type=*/TensorType_FLOAT16, /*weight_type=*/TensorType_INT8,
      /*has_zero_points=*/false, /*relative_tolerance=*/0.02f);
}

TEST(MoeTest, BlockwiseInt2MoeDecode) {
  const int B = 1, N = 1, D_in = 64, D_mid = 128, E = 4, K = 2, block_size = 32;
  ExpectBlockwiseQuantizedMoeNearReference(
      B, N, D_in, D_mid, E, K, block_size,
      std::vector<float>(B * N * D_in, 0.5f),
      /*mock_ei=*/{1, 3}, /*scale_type=*/TensorType_FLOAT32,
      /*weight_type=*/TensorType_INT2, /*has_zero_points=*/false,
      /*relative_tolerance=*/0.05f);
}

TEST(MoeTest, BlockwiseInt2MoePrefill) {
  const int B = 1, N = 4, D_in = 64, D_mid = 128, E = 4, K = 2, block_size = 32;
  std::vector<int32_t> mock_ei(B * N * K);
  for (size_t i = 0; i < mock_ei.size(); ++i) {
    mock_ei[i] = (i * 2 + 1) % E;
  }
  ExpectBlockwiseQuantizedMoeNearReference(
      B, N, D_in, D_mid, E, K, block_size, MakeTokens(B * N * D_in), mock_ei,
      /*scale_type=*/TensorType_FLOAT32, /*weight_type=*/TensorType_INT2,
      /*has_zero_points=*/false, /*relative_tolerance=*/0.05f);
}

TEST(MoeTest, BlockwiseInt4MoeDynamicShapeAndRoutingChanges) {
  const int B = 2, N = 4, D_in = 64, D_mid = 128, E = 4, K = 2, block_size = 32;
  // Also exercise `num_threads = 2` so `~Interpreter()` tears down the kernel
  // against a live `thread_pool_` on `delegate_`.
  BlockwiseQuantizedMoeModel<float> model(
      B, N, D_in, D_mid, E, K, block_size, /*use_delegate=*/true,
      TensorType_FLOAT32, TensorType_INT4, /*has_zero_points=*/true,
      /*enable_blockwise_quantization=*/true,
      /*use_wire_default_zero_zp=*/false, /*static_shape=*/false,
      /*num_threads=*/2);
  ASSERT_EQ(model.GetNumberOfAppliedDelegates(), 1);

  auto verify_invocation = [&](int B_cur, int N_cur, int K_cur,
                               const std::vector<int32_t>& ei) {
    const std::vector<float> tokens = MakeTokens(B_cur * N_cur * D_in);
    const std::vector<float> rw(B_cur * N_cur * K_cur,
                                1.0f / static_cast<float>(K_cur));
    model.PopulateTensor<float>(model.tokens(), tokens);
    model.PopulateTensor<float>(model.rw(), rw);
    model.PopulateTensor<int32_t>(model.ei(), ei);
    ASSERT_EQ(model.Invoke(), kTfLiteOk);

    std::vector<float> expected_out(B_cur * N_cur * D_in);
    RunReferenceMoe(B_cur, N_cur, D_in, D_mid, E, K_cur, tokens.data(),
                    rw.data(), ei.data(), model.w_gate_ref().data(),
                    model.w_up_ref().data(), model.w_down_ref().data(),
                    model.scale_ref().data(), model.scale_ref().size(),
                    expected_out.data());
    float max_abs_expected = 0.0f;
    for (float expected : expected_out) {
      max_abs_expected = std::max(max_abs_expected, std::abs(expected));
    }
    ASSERT_GT(max_abs_expected, 1e-3f);
    const float tolerance = 0.03f * max_abs_expected;
    const std::vector<float> actual = model.ExtractVector<float>(model.out());
    for (size_t i = 0; i < expected_out.size(); ++i) {
      EXPECT_NEAR(actual[i], expected_out[i], tolerance) << "at " << i;
    }
  };

  for (const std::vector<int32_t>& routing : TestRoutings()) {
    verify_invocation(B, N, K, routing);
  }

  for (auto [B_cur, N_cur, K_cur] :
       {std::tuple<int, int, int>{2, 2, 2}, std::tuple<int, int, int>{1, 5, 3},
        std::tuple<int, int, int>{1, 1, 1}}) {
    ASSERT_EQ(model.ResizeInputTensor(model.tokens(), {B_cur, N_cur, D_in}),
              kTfLiteOk);
    ASSERT_EQ(model.ResizeInputTensor(model.rw(), {B_cur, N_cur, K_cur}),
              kTfLiteOk);
    ASSERT_EQ(model.ResizeInputTensor(model.ei(), {B_cur, N_cur, K_cur}),
              kTfLiteOk);
    ASSERT_EQ(model.AllocateTensors(), kTfLiteOk);
    ASSERT_EQ(model.GetTensorShape(model.out()),
              (std::vector<int>{B_cur, N_cur, D_in}));
    std::vector<int32_t> ei(B_cur * N_cur * K_cur);
    for (size_t i = 0; i < ei.size(); ++i) {
      ei[i] = static_cast<int32_t>((i * 3 + 1) % E);
    }
    verify_invocation(B_cur, N_cur, K_cur, ei);
  }
}

TEST(MoeTest, BlockwiseQuantizationDisabledFallback) {
  const int B = 1, N = 1, D_in = 64, D_mid = 128, E = 4, K = 2, block_size = 32;
  BlockwiseQuantizedMoeModel<float> model(
      B, N, D_in, D_mid, E, K, block_size, /*use_delegate=*/true,
      TensorType_FLOAT32, TensorType_INT8, /*has_zero_points=*/false,
      /*enable_blockwise_quantization=*/false);
  EXPECT_EQ(model.GetNumberOfAppliedDelegates(), 0);
}

TEST(MoeTest, BlockwiseInt8MoeWireDefaultZeroPointIndex) {
  const int B = 1, N = 1, D_in = 64, D_mid = 128, E = 4, K = 2, block_size = 32;
  BlockwiseQuantizedMoeModel<float> model(
      B, N, D_in, D_mid, E, K, block_size, /*use_delegate=*/true,
      TensorType_FLOAT32, TensorType_INT8, /*has_zero_points=*/false,
      /*enable_blockwise_quantization=*/true,
      /*use_wire_default_zero_zp=*/true);
  EXPECT_EQ(model.GetNumberOfAppliedDelegates(), 1);

  const std::vector<float> tokens(B * N * D_in, 0.5f);
  const std::vector<float> rw(B * N * K, 0.5f);
  const std::vector<int32_t> mock_ei = {1, 3};
  model.PopulateTensor<float>(model.tokens(), tokens);
  model.PopulateTensor<float>(model.rw(), rw);
  model.PopulateTensor(model.ei(), mock_ei);
  ASSERT_EQ(model.Invoke(), kTfLiteOk);

  std::vector<float> expected_out(B * N * D_in);
  RunReferenceMoe(B, N, D_in, D_mid, E, K, tokens.data(), rw.data(),
                  mock_ei.data(), model.w_gate_ref().data(),
                  model.w_up_ref().data(), model.w_down_ref().data(),
                  model.scale_ref().data(), model.scale_ref().size(),
                  expected_out.data());
  const std::vector<float> delegate_output =
      model.ExtractVector<float>(model.out());
  float max_abs_expected = 0.0f;
  for (float expected : expected_out) {
    max_abs_expected = std::max(max_abs_expected, std::abs(expected));
  }
  const float tolerance = 0.02f * max_abs_expected;
  for (int i = 0; i < B * N * D_in; ++i) {
    EXPECT_NEAR(delegate_output[i], expected_out[i], tolerance) << "at " << i;
  }
}

TEST(MoeTest, InvalidMoeShapesFallback) {
  // Unaligned sub-byte INT4 D_in (33 % 2 != 0) must be rejected cleanly.
  BlockwiseQuantizedMoeModel<float> unaligned_int4(
      /*B=*/1, /*N=*/1, /*D_in=*/33, /*D_mid=*/66, /*E=*/4, /*K=*/2,
      /*block_size=*/33, /*use_delegate=*/true, TensorType_FLOAT32,
      TensorType_INT4);
  EXPECT_EQ(unaligned_int4.GetNumberOfAppliedDelegates(), 0);

  // Unaligned sub-byte INT4 D_mid (dims[0] = 33 % 2 != 0) must be rejected.
  BlockwiseQuantizedMoeModel<float> unaligned_d_mid(
      /*B=*/1, /*N=*/1, /*D_in=*/64, /*D_mid=*/33, /*E=*/4, /*K=*/2,
      /*block_size=*/1, /*use_delegate=*/true, TensorType_FLOAT32,
      TensorType_INT4);
  EXPECT_EQ(unaligned_d_mid.GetNumberOfAppliedDelegates(), 0);
}

TEST(MoeTest, SymmetricPerChannelAffineQuantizedInt8Moe) {
  const int B = 1, N = 2, D_in = 32, D_mid = 16, E = 4, K = 2;
  const std::vector<int32_t> tokens_shape = {B, N, D_in};
  const std::vector<int32_t> rw_shape = {B, N, K};
  const std::vector<int32_t> ei_shape = {B, N, K};
  const std::vector<int32_t> w_gate_shape = {D_mid, E, 1, D_in};
  const std::vector<int32_t> w_up_shape = {D_mid, E, 1, D_in};
  const std::vector<int32_t> w_down_shape = {D_in, E, 1, D_mid};
  const std::vector<int32_t> scale_shape = {1, 1, 1, E};
  const std::vector<int32_t> out_shape = {B, N, D_in};

  const std::vector<int8_t> w_gate_q =
      MakeQuantizedWeights(D_mid * E * D_in, /*seed=*/1, TensorType_INT8);
  const std::vector<int8_t> w_up_q =
      MakeQuantizedWeights(D_mid * E * D_in, /*seed=*/2, TensorType_INT8);
  const std::vector<int8_t> w_down_q =
      MakeQuantizedWeights(D_in * E * D_mid, /*seed=*/3, TensorType_INT8);

  auto make_scales = [](int out_dim, int seed) {
    std::vector<float> scales(out_dim);
    for (int i = 0; i < out_dim; ++i) {
      scales[i] = 0.02f * (0.75f + 0.25f * ((i + seed) % 3));
    }
    return scales;
  };
  auto dequant_per_channel = [&](const std::vector<int8_t>& q, int out_dim,
                                 int in_dim, const std::vector<float>& scales) {
    std::vector<float> ref(q.size());
    for (int o = 0; o < out_dim; ++o) {
      for (int e = 0; e < E; ++e) {
        for (int k = 0; k < in_dim; ++k) {
          const size_t idx = (static_cast<size_t>(o) * E + e) * in_dim + k;
          ref[idx] = static_cast<float>(q[idx]) * scales[o];
        }
      }
    }
    return ref;
  };

  const std::vector<float> g_scales = make_scales(D_mid, 1);
  const std::vector<float> u_scales = make_scales(D_mid, 2);
  const std::vector<float> d_scales = make_scales(D_in, 3);
  const std::vector<float> w_gate_ref =
      dequant_per_channel(w_gate_q, D_mid, D_in, g_scales);
  const std::vector<float> w_up_ref =
      dequant_per_channel(w_up_q, D_mid, D_in, u_scales);
  const std::vector<float> w_down_ref =
      dequant_per_channel(w_down_q, D_in, D_mid, d_scales);
  const std::vector<float> out_scale(E, 1.0f);

  auto build_and_try_delegate = [&](int quantized_dim, int64_t zp_val,
                                    std::vector<uint8_t>* buf_out,
                                    std::unique_ptr<FlatBufferModel>* fb_out,
                                    std::unique_ptr<Interpreter>* interp_out,
                                    Interpreter::TfLiteDelegatePtr* del_out) {
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<Tensor>> tensors;
    std::vector<flatbuffers::Offset<Buffer>> buffers;
    buffers.push_back(CreateBufferDirect(builder));
    auto add_buffer = [&](const void* data, size_t bytes) -> int {
      int id = buffers.size();
      buffers.push_back(CreateBuffer(
          builder,
          builder.CreateVector(reinterpret_cast<const uint8_t*>(data), bytes)));
      return id;
    };
    tensors.push_back(CreateTensorDirect(builder, &tokens_shape,
                                         TensorType_FLOAT32, 0, "tokens"));
    tensors.push_back(
        CreateTensorDirect(builder, &rw_shape, TensorType_FLOAT32, 0, "rw"));
    tensors.push_back(
        CreateTensorDirect(builder, &ei_shape, TensorType_INT32, 0, "ei"));

    auto add_weight = [&](const std::vector<int32_t>& shape,
                          const std::vector<int8_t>& q,
                          const std::vector<float>& scales, const char* name) {
      const int n_params = quantized_dim == 0 ? shape[0] : E;
      std::vector<float> s_padded(
          scales.begin(),
          scales.begin() + std::min<int>(n_params, scales.size()));
      s_padded.resize(n_params, 0.02f);
      const std::vector<int64_t> zps(n_params, zp_val);
      auto qp = CreateQuantizationParametersDirect(
          builder, nullptr, nullptr, &s_padded, &zps, QuantizationDetails_NONE,
          0, quantized_dim);
      tensors.push_back(CreateTensorDirect(builder, &shape, TensorType_INT8,
                                           add_buffer(q.data(), q.size()), name,
                                           qp));
    };
    add_weight(w_gate_shape, w_gate_q, g_scales, "wg");
    add_weight(w_up_shape, w_up_q, u_scales, "wu");
    add_weight(w_down_shape, w_down_q, d_scales, "wd");
    tensors.push_back(CreateTensorDirect(
        builder, &scale_shape, TensorType_FLOAT32,
        add_buffer(out_scale.data(), out_scale.size() * sizeof(float)),
        "scale"));
    tensors.push_back(
        CreateTensorDirect(builder, &out_shape, TensorType_FLOAT32, 0, "out"));

    std::vector<uint8_t> empty_attrs;
    auto comp_opts = CreateStableHLOCompositeOptionsDirect(
        builder, "odml.moe_experts", /*decomposition_subgraph_index=*/1,
        &empty_attrs);
    std::vector<flatbuffers::Offset<OperatorCode>> opcodes = {
        CreateOperatorCode(
            builder,
            static_cast<int8_t>(
                BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES),
            0, 1, BuiltinOperator_STABLEHLO_COMPOSITE),
        CreateOperatorCodeDirect(builder, BuiltinOperator_CUSTOM,
                                 "moe_fallback")};
    std::vector<int32_t> ins = {0, 1, 2, 3, 4, 5, 6};
    std::vector<int32_t> outs = {7};
    std::vector<flatbuffers::Offset<Operator>> ops = {CreateOperator(
        builder, 0, builder.CreateVector(ins), builder.CreateVector(outs),
        BuiltinOptions_NONE, 0, 0, CustomOptionsFormat_FLEXBUFFERS, 0, 0, 0, 0,
        BuiltinOptions2_StableHLOCompositeOptions, comp_opts.Union())};
    std::vector<int32_t> sg_ins = {0, 1, 2};

    std::vector<flatbuffers::Offset<Tensor>> decomp_tensors = {
        CreateTensorDirect(builder, &tokens_shape, TensorType_FLOAT32, 0, "t0"),
        CreateTensorDirect(builder, &rw_shape, TensorType_FLOAT32, 0, "t1"),
        CreateTensorDirect(builder, &ei_shape, TensorType_INT32, 0, "t2"),
        CreateTensorDirect(builder, &w_gate_shape, TensorType_INT8, 0, "t3"),
        CreateTensorDirect(builder, &w_up_shape, TensorType_INT8, 0, "t4"),
        CreateTensorDirect(builder, &w_down_shape, TensorType_INT8, 0, "t5"),
        CreateTensorDirect(builder, &scale_shape, TensorType_FLOAT32, 0, "t6"),
        CreateTensorDirect(builder, &out_shape, TensorType_FLOAT32, 0, "t7")};
    std::vector<flatbuffers::Offset<Operator>> decomp_ops = {CreateOperator(
        builder, 1, builder.CreateVector(ins), builder.CreateVector(outs))};
    std::vector<flatbuffers::Offset<SubGraph>> subgraphs = {
        CreateSubGraphDirect(builder, &tensors, &sg_ins, &outs, &ops),
        CreateSubGraphDirect(builder, &decomp_tensors, &ins, &outs,
                             &decomp_ops)};
    FinishModelBuffer(builder, CreateModelDirect(builder, 3, &opcodes,
                                                 &subgraphs, "m", &buffers));

    buf_out->assign(builder.GetBufferPointer(),
                    builder.GetBufferPointer() + builder.GetSize());
    *fb_out = FlatBufferModel::BuildFromBuffer(
        reinterpret_cast<const char*>(buf_out->data()), buf_out->size());
    ASSERT_NE(*fb_out, nullptr);

    class CompositeFallbackResolver : public OpResolver {
     public:
      const TfLiteRegistration* FindOp(BuiltinOperator op, int) const override {
        if (op == BuiltinOperator_STABLEHLO_COMPOSITE) {
          static TfLiteRegistration reg = []() {
            TfLiteRegistration r{};
            r.builtin_code = kTfLiteBuiltinStablehloComposite;
            return r;
          }();
          return &reg;
        }
        return nullptr;
      }
      const TfLiteRegistration* FindOp(const char* op, int) const override {
        if (op != nullptr && std::strcmp(op, "moe_fallback") == 0) {
          static TfLiteRegistration reg = []() {
            TfLiteRegistration r{};
            r.builtin_code = kTfLiteBuiltinCustom;
            r.custom_name = "moe_fallback";
            return r;
          }();
          return &reg;
        }
        return nullptr;
      }
    };
    CompositeFallbackResolver resolver;
    ASSERT_EQ(InterpreterBuilder(**fb_out, resolver)(interp_out), kTfLiteOk);

    TfLiteYNNPackDelegateOptions opts = TfLiteYNNPackDelegateOptionsDefault();
    opts.num_threads = 1;
    opts.static_shape = true;
    *del_out = Interpreter::TfLiteDelegatePtr(
        TfLiteYNNPackDelegateCreate(&opts), TfLiteYNNPackDelegateDelete);
    ASSERT_EQ((*interp_out)->ModifyGraphWithDelegate(del_out->get()),
              kTfLiteOk);
  };

  // Symmetric per-channel (`quantized_dim == 0`, `zp == 0`) is delegated.
  {
    std::vector<uint8_t> model_buf;
    std::unique_ptr<FlatBufferModel> fb;
    Interpreter::TfLiteDelegatePtr delegate(nullptr, nullptr);
    std::unique_ptr<Interpreter> interpreter;
    build_and_try_delegate(/*quantized_dim=*/0, /*zp_val=*/0, &model_buf, &fb,
                           &interpreter, &delegate);
    ASSERT_EQ(interpreter->execution_plan().size(), 1u);
    const auto* n0 =
        interpreter->node_and_registration(interpreter->execution_plan()[0]);
    ASSERT_NE(n0, nullptr);
    ASSERT_STREQ(n0->second.custom_name, "YNNPackDelegate");
    ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

    const std::vector<float> tokens = MakeTokens(B * N * D_in);
    const std::vector<float> rw(B * N * K, 0.5f);
    const std::vector<int32_t> ei = {0, 2, 1, 3};
    std::memcpy(interpreter->typed_tensor<float>(0), tokens.data(),
                tokens.size() * sizeof(float));
    std::memcpy(interpreter->typed_tensor<float>(1), rw.data(),
                rw.size() * sizeof(float));
    std::memcpy(interpreter->typed_tensor<int32_t>(2), ei.data(),
                ei.size() * sizeof(int32_t));
    ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

    std::vector<float> expected(B * N * D_in);
    RunReferenceMoe(B, N, D_in, D_mid, E, K, tokens.data(), rw.data(),
                    ei.data(), w_gate_ref.data(), w_up_ref.data(),
                    w_down_ref.data(), out_scale.data(), out_scale.size(),
                    expected.data());
    float max_abs = 0.0f;
    for (float v : expected) max_abs = std::max(max_abs, std::abs(v));
    ASSERT_GT(max_abs, 1e-3f);
    const float* actual = interpreter->typed_tensor<float>(7);
    for (int i = 0; i < B * N * D_in; ++i) {
      EXPECT_NEAR(actual[i], expected[i], 0.02f * max_abs) << "at " << i;
    }
  }

  // Non-zero affine zero points and `quantized_dimension == 1` are rejected.
  for (auto [q_dim, zp] :
       {std::pair<int, int64_t>{0, 2}, std::pair<int, int64_t>{1, 0}}) {
    std::vector<uint8_t> model_buf;
    std::unique_ptr<FlatBufferModel> fb;
    Interpreter::TfLiteDelegatePtr delegate(nullptr, nullptr);
    std::unique_ptr<Interpreter> interpreter;
    build_and_try_delegate(q_dim, zp, &model_buf, &fb, &interpreter, &delegate);
    ASSERT_EQ(interpreter->execution_plan().size(), 1u);
    const auto* n0 =
        interpreter->node_and_registration(interpreter->execution_plan()[0]);
    ASSERT_NE(n0, nullptr);
    EXPECT_STREQ(n0->second.custom_name, "moe_fallback");
  }
}

TEST(MoeTest, NonViewExpertIndicesProducerFallback) {
  // When `expert_indices` is produced by a non-view op that YNNPACK supports
  // (e.g. `GATHER`), `SetupMoeDispatch` cannot read it on the host before
  // invoking the partition; the delegate must reject the MoE node cleanly
  // rather than failing `Init()` / `Eval()`.
  const int B = 1, N = 2, D_in = 16, D_mid = 32, E = 4, K = 2;
  const std::vector<int32_t> tokens_shape = {B, N, D_in};
  const std::vector<int32_t> rw_shape = {B, N, K};
  const std::vector<int32_t> table_shape = {4, N, K};
  const std::vector<int32_t> idx_shape = {B};
  const std::vector<int32_t> ei_shape = {B, N, K};
  const std::vector<int32_t> wg_shape = {D_mid, E, 1, D_in};
  const std::vector<int32_t> wd_shape = {D_in, E, 1, D_mid};
  const std::vector<int32_t> s_shape = {1};

  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<Tensor>> tensors;
  std::vector<flatbuffers::Offset<Buffer>> buffers;
  buffers.push_back(CreateBufferDirect(builder));
  auto add_buffer = [&](const void* data, size_t bytes) -> int {
    int id = buffers.size();
    buffers.push_back(CreateBuffer(
        builder,
        builder.CreateVector(reinterpret_cast<const uint8_t*>(data), bytes)));
    return id;
  };
  tensors.push_back(CreateTensorDirect(builder, &tokens_shape,
                                       TensorType_FLOAT32, 0, "tokens"));
  tensors.push_back(
      CreateTensorDirect(builder, &rw_shape, TensorType_FLOAT32, 0, "rw"));
  tensors.push_back(
      CreateTensorDirect(builder, &table_shape, TensorType_INT32, 0, "table"));
  tensors.push_back(
      CreateTensorDirect(builder, &idx_shape, TensorType_INT32, 0, "idx"));
  tensors.push_back(
      CreateTensorDirect(builder, &ei_shape, TensorType_INT32, 0, "ei"));

  const std::vector<float> wg(D_mid * E * D_in, 0.1f);
  const std::vector<float> wd(D_in * E * D_mid, 0.1f);
  const std::vector<float> s = {1.0f};
  tensors.push_back(CreateTensorDirect(
      builder, &wg_shape, TensorType_FLOAT32,
      add_buffer(wg.data(), wg.size() * sizeof(float)), "wg"));
  tensors.push_back(CreateTensorDirect(
      builder, &wg_shape, TensorType_FLOAT32,
      add_buffer(wg.data(), wg.size() * sizeof(float)), "wu"));
  tensors.push_back(CreateTensorDirect(
      builder, &wd_shape, TensorType_FLOAT32,
      add_buffer(wd.data(), wd.size() * sizeof(float)), "wd"));
  tensors.push_back(CreateTensorDirect(
      builder, &s_shape, TensorType_FLOAT32,
      add_buffer(s.data(), s.size() * sizeof(float)), "scale"));
  tensors.push_back(
      CreateTensorDirect(builder, &tokens_shape, TensorType_FLOAT32, 0, "out"));

  class GatherAndCustomMoeResolver : public OpResolver {
   public:
    const TfLiteRegistration* FindOp(BuiltinOperator op, int) const override {
      if (op == BuiltinOperator_GATHER) {
        static TfLiteRegistration reg = []() {
          TfLiteRegistration r{};
          r.builtin_code = kTfLiteBuiltinGather;
          return r;
        }();
        return &reg;
      }
      if (op == BuiltinOperator_STABLEHLO_COMPOSITE) {
        static TfLiteRegistration reg = []() {
          TfLiteRegistration r{};
          r.builtin_code = kTfLiteBuiltinStablehloComposite;
          return r;
        }();
        return &reg;
      }
      return nullptr;
    }
    const TfLiteRegistration* FindOp(const char* op, int) const override {
      if (op != nullptr && std::strcmp(op, "moe_fallback") == 0) {
        static TfLiteRegistration reg = []() {
          TfLiteRegistration r{};
          r.builtin_code = kTfLiteBuiltinCustom;
          r.custom_name = "moe_fallback";
          return r;
        }();
        return &reg;
      }
      return nullptr;
    }
  };

  std::vector<uint8_t> empty_attrs;
  auto comp_opts = CreateStableHLOCompositeOptionsDirect(
      builder, "odml.moe_experts", /*decomposition_subgraph_index=*/1,
      &empty_attrs);
  std::vector<flatbuffers::Offset<OperatorCode>> opcodes = {
      CreateOperatorCode(builder, BuiltinOperator_GATHER),
      CreateOperatorCode(
          builder,
          static_cast<int8_t>(BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES),
          0, 1, BuiltinOperator_STABLEHLO_COMPOSITE),
      CreateOperatorCodeDirect(builder, BuiltinOperator_CUSTOM,
                               "moe_fallback")};
  std::vector<int32_t> g_in = {2, 3};
  std::vector<int32_t> g_out = {4};
  std::vector<int32_t> m_in = {0, 1, 4, 5, 6, 7, 8};
  std::vector<int32_t> m_out = {9};
  std::vector<flatbuffers::Offset<Operator>> ops = {
      CreateOperator(builder, 0, builder.CreateVector(g_in),
                     builder.CreateVector(g_out), BuiltinOptions_GatherOptions,
                     CreateGatherOptions(builder, /*axis=*/0).Union()),
      CreateOperator(
          builder, 1, builder.CreateVector(m_in), builder.CreateVector(m_out),
          BuiltinOptions_NONE, 0, 0, CustomOptionsFormat_FLEXBUFFERS, 0, 0, 0,
          0, BuiltinOptions2_StableHLOCompositeOptions, comp_opts.Union())};
  std::vector<int32_t> sg_ins = {0, 1, 2, 3};

  // Decomposition subgraph 1 for un-delegated `odml.moe_experts` fallback.
  std::vector<flatbuffers::Offset<Tensor>> decomp_tensors = {
      CreateTensorDirect(builder, &tokens_shape, TensorType_FLOAT32, 0, "t0"),
      CreateTensorDirect(builder, &rw_shape, TensorType_FLOAT32, 0, "t1"),
      CreateTensorDirect(builder, &ei_shape, TensorType_INT32, 0, "t2"),
      CreateTensorDirect(builder, &wg_shape, TensorType_FLOAT32, 0, "t3"),
      CreateTensorDirect(builder, &wg_shape, TensorType_FLOAT32, 0, "t4"),
      CreateTensorDirect(builder, &wd_shape, TensorType_FLOAT32, 0, "t5"),
      CreateTensorDirect(builder, &s_shape, TensorType_FLOAT32, 0, "t6"),
      CreateTensorDirect(builder, &tokens_shape, TensorType_FLOAT32, 0, "t7")};
  std::vector<int32_t> d_in = {0, 1, 2, 3, 4, 5, 6};
  std::vector<int32_t> d_out = {7};
  std::vector<flatbuffers::Offset<Operator>> decomp_ops = {CreateOperator(
      builder, 2, builder.CreateVector(d_in), builder.CreateVector(d_out))};
  std::vector<flatbuffers::Offset<SubGraph>> subgraphs = {
      CreateSubGraphDirect(builder, &tensors, &sg_ins, &m_out, &ops),
      CreateSubGraphDirect(builder, &decomp_tensors, &d_in, &d_out,
                           &decomp_ops)};
  FinishModelBuffer(builder, CreateModelDirect(builder, 3, &opcodes, &subgraphs,
                                               "m", &buffers));

  std::vector<uint8_t> model_buf(
      builder.GetBufferPointer(),
      builder.GetBufferPointer() + builder.GetSize());
  auto fb_model = FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(model_buf.data()), model_buf.size());
  ASSERT_NE(fb_model, nullptr);
  GatherAndCustomMoeResolver resolver;
  TfLiteYNNPackDelegateOptions opts = TfLiteYNNPackDelegateOptionsDefault();
  opts.num_threads = 1;
  opts.static_shape = true;
  Interpreter::TfLiteDelegatePtr delegate(TfLiteYNNPackDelegateCreate(&opts),
                                          TfLiteYNNPackDelegateDelete);
  std::unique_ptr<Interpreter> interpreter;
  ASSERT_EQ(InterpreterBuilder(*fb_model, resolver)(&interpreter), kTfLiteOk);
  ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate.get()), kTfLiteOk);
  // GATHER is delegated while `odml.moe_experts` is rejected and decomposed to
  // `moe_fallback`, leaving 2 nodes in the execution plan instead of merging
  // both into a broken 1-node partition.
  ASSERT_EQ(interpreter->execution_plan().size(), 2u);
  const auto* n0 =
      interpreter->node_and_registration(interpreter->execution_plan()[0]);
  const auto* n1 =
      interpreter->node_and_registration(interpreter->execution_plan()[1]);
  ASSERT_NE(n0, nullptr);
  ASSERT_NE(n1, nullptr);
  EXPECT_STREQ(n0->second.custom_name, "YNNPackDelegate");
  EXPECT_STREQ(n1->second.custom_name, "moe_fallback");
}

enum class EiViewKind {
  kExpandDimsPosAxis,
  kReshapeBatchDynamic,
  kReshapeSeqDynamic,
  kMultiHopOneInputReshapeAndNegExpandDims,
};

class InternalInputsMoeModel : public SingleOpModel {
 public:
  InternalInputsMoeModel(int B, int N, int D_in, int D_mid, int E, int K,
                         bool static_shape, EiViewKind view_kind)
      : w_gate_data_(D_mid * E * D_in),
        w_up_data_(D_mid * E * D_in),
        w_down_data_(D_in * E * D_mid),
        scale_data_(E) {
    const std::vector<int> tokens_shape = {B, N, D_in};
    const std::vector<int> rw_shape = {B, N, K};
    const std::vector<int> ei_shape = {B, N, K};
    std::vector<int> ei_in_shape;
    switch (view_kind) {
      case EiViewKind::kExpandDimsPosAxis:
        ei_in_shape = {B, N};
        break;
      case EiViewKind::kReshapeBatchDynamic:
      case EiViewKind::kReshapeSeqDynamic:
        ei_in_shape = {B * N, K};
        break;
      case EiViewKind::kMultiHopOneInputReshapeAndNegExpandDims:
        ei_in_shape = {B * N};
        break;
    }
    const std::vector<int> w_gate_shape = {D_mid, E, 1, D_in};
    const std::vector<int> w_down_shape = {D_in, E, 1, D_mid};
    const std::vector<int> scale_shape = {1, 1, 1, E};

    x_id_ = AddInput({TensorType_FLOAT32, tokens_shape});
    residual_id_ = AddInput({TensorType_FLOAT32, tokens_shape});
    logits_id_ = AddInput({TensorType_FLOAT32, rw_shape});
    ei_id_ = AddInput({TensorType_INT32, ei_in_shape});

    for (size_t i = 0; i < w_gate_data_.size(); ++i) {
      w_gate_data_[i] = 0.1f * std::sin(static_cast<float>(i));
      w_up_data_[i] = 0.1f * std::sin(static_cast<float>(i) + 1.0f);
    }
    for (size_t i = 0; i < w_down_data_.size(); ++i) {
      w_down_data_[i] = 0.1f * std::sin(static_cast<float>(i) + 2.0f);
    }
    for (size_t i = 0; i < scale_data_.size(); ++i) {
      scale_data_[i] = 1.0f + 0.1f * std::sin(static_cast<float>(i) + 3.0f);
    }

    const int w_gate_id = AddConstInput(
        TensorData{TensorType_FLOAT32, w_gate_shape}, w_gate_data_);
    const int w_up_id =
        AddConstInput(TensorData{TensorType_FLOAT32, w_gate_shape}, w_up_data_);
    const int w_down_id = AddConstInput(
        TensorData{TensorType_FLOAT32, w_down_shape}, w_down_data_);
    const int scale_id =
        AddConstInput(TensorData{TensorType_FLOAT32, scale_shape}, scale_data_);

    const int tokens_internal_id =
        AddTensor<float>({TensorType_FLOAT32, tokens_shape}, {}, false);
    const int rw_internal_id =
        AddTensor<float>({TensorType_FLOAT32, rw_shape}, {}, false);
    const int ei_internal_id =
        AddTensor<int32_t>({TensorType_INT32, ei_shape}, {}, false);
    out_id_ = AddOutput({TensorType_FLOAT32, tokens_shape});

    opcodes_.push_back(CreateOperatorCode(builder_, BuiltinOperator_ADD, 0, 1));
    opcodes_.push_back(
        CreateOperatorCode(builder_, BuiltinOperator_SOFTMAX, 0, 1));
    opcodes_.push_back(
        CreateOperatorCode(builder_, BuiltinOperator_EXPAND_DIMS, 0, 1));
    opcodes_.push_back(
        CreateOperatorCode(builder_, BuiltinOperator_RESHAPE, 0, 1));
    opcodes_.push_back(CreateOperatorCode(
        builder_,
        static_cast<int8_t>(BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES),
        /*custom_code=*/0, /*version=*/1, BuiltinOperator_STABLEHLO_COMPOSITE));

    operators_.push_back(CreateOperator(
        builder_, /*opcode_index=*/0,
        builder_.CreateVector<int32_t>({x_id_, residual_id_}),
        builder_.CreateVector<int32_t>({tokens_internal_id}),
        BuiltinOptions_AddOptions,
        CreateAddOptions(builder_, ActivationFunctionType_NONE).Union()));

    operators_.push_back(
        CreateOperator(builder_, /*opcode_index=*/1,
                       builder_.CreateVector<int32_t>({logits_id_}),
                       builder_.CreateVector<int32_t>({rw_internal_id}),
                       BuiltinOptions_SoftmaxOptions,
                       CreateSoftmaxOptions(builder_, /*beta=*/1.0f).Union()));

    switch (view_kind) {
      case EiViewKind::kExpandDimsPosAxis: {
        const int axis_id = AddConstInput(TensorData{TensorType_INT32, {1}},
                                          std::vector<int32_t>{2});
        operators_.push_back(
            CreateOperator(builder_, /*opcode_index=*/2,
                           builder_.CreateVector<int32_t>({ei_id_, axis_id}),
                           builder_.CreateVector<int32_t>({ei_internal_id}),
                           BuiltinOptions_ExpandDimsOptions,
                           CreateExpandDimsOptions(builder_).Union()));
        break;
      }
      case EiViewKind::kReshapeBatchDynamic:
      case EiViewKind::kReshapeSeqDynamic: {
        const std::vector<int32_t> target_shape =
            (view_kind == EiViewKind::kReshapeBatchDynamic)
                ? std::vector<int32_t>{-1, N, K}
                : std::vector<int32_t>{B, -1, K};
        const int shape_id =
            AddConstInput(TensorData{TensorType_INT32, {3}}, target_shape);
        operators_.push_back(CreateOperator(
            builder_, /*opcode_index=*/3,
            builder_.CreateVector<int32_t>({ei_id_, shape_id}),
            builder_.CreateVector<int32_t>({ei_internal_id}),
            BuiltinOptions_ReshapeOptions,
            CreateReshapeOptions(builder_,
                                 builder_.CreateVector<int32_t>(target_shape))
                .Union()));
        break;
      }
      case EiViewKind::kMultiHopOneInputReshapeAndNegExpandDims: {
        // 1D [B * N] -> 1-input RESHAPE({-1, N}) via TfLiteReshapeParams ->
        // EXPAND_DIMS(axis = -1) -> [B, N, 1].
        const int ei_2d_id =
            AddTensor<int32_t>({TensorType_INT32, {B, N}}, {}, false);
        const std::vector<int32_t> reshape_2d = {-1, N};
        operators_.push_back(CreateOperator(
            builder_, /*opcode_index=*/3,
            builder_.CreateVector<int32_t>({ei_id_}),
            builder_.CreateVector<int32_t>({ei_2d_id}),
            BuiltinOptions_ReshapeOptions,
            CreateReshapeOptions(builder_,
                                 builder_.CreateVector<int32_t>(reshape_2d))
                .Union()));
        const int neg_axis_id = AddConstInput(TensorData{TensorType_INT32, {1}},
                                              std::vector<int32_t>{-1});
        operators_.push_back(CreateOperator(
            builder_, /*opcode_index=*/2,
            builder_.CreateVector<int32_t>({ei_2d_id, neg_axis_id}),
            builder_.CreateVector<int32_t>({ei_internal_id}),
            BuiltinOptions_ExpandDimsOptions,
            CreateExpandDimsOptions(builder_).Union()));
        break;
      }
    }

    std::vector<uint8_t> empty_attrs;
    flatbuffers::Offset<StableHLOCompositeOptions> comp_opts =
        CreateStableHLOCompositeOptionsDirect(
            builder_, "odml.moe_experts",
            /*decomposition_subgraph_index=*/1, &empty_attrs);
    operators_.push_back(CreateOperator(
        builder_, /*opcode_index=*/4,
        builder_.CreateVector<int32_t>({tokens_internal_id, rw_internal_id,
                                        ei_internal_id, w_gate_id, w_up_id,
                                        w_down_id, scale_id}),
        builder_.CreateVector<int32_t>({out_id_}), BuiltinOptions_NONE,
        /*builtin_options=*/0, /*custom_options=*/0,
        CustomOptionsFormat_FLEXBUFFERS, /*mutating_variable_inputs=*/0,
        /*intermediates=*/0, /*large_custom_options_offset=*/0,
        /*large_custom_options_size=*/0,
        BuiltinOptions2_StableHLOCompositeOptions, comp_opts.Union()));

    BuildInterpreter({tokens_shape, tokens_shape, rw_shape, ei_in_shape}, -1,
                     false, false, /*allocate_and_delegate=*/false);

    TfLiteYNNPackDelegateOptions delegate_options =
        TfLiteYNNPackDelegateOptionsDefault();
    delegate_options.num_threads = 1;
    delegate_options.static_shape = static_shape;
    SetDelegate(Interpreter::TfLiteDelegatePtr(
        TfLiteYNNPackDelegateCreate(&delegate_options),
        TfLiteYNNPackDelegateDelete));
    ApplyDelegate();
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
      fprintf(stderr, "Failed to allocate tensors\n");
    }
  }

  int x() const { return x_id_; }
  int residual() const { return residual_id_; }
  int logits() const { return logits_id_; }
  int ei() const { return ei_id_; }
  int out() const { return out_id_; }

  size_t NumNodesInExecutionPlan() const {
    return interpreter_->execution_plan().size();
  }

  TfLiteStatus ResizeInputTensor(int tensor_index,
                                 const std::vector<int>& new_shape) {
    return interpreter_->ResizeInputTensor(tensor_index, new_shape);
  }

  TfLiteStatus AllocateTensors() { return interpreter_->AllocateTensors(); }

  const std::vector<float>& w_gate_data() const { return w_gate_data_; }
  const std::vector<float>& w_up_data() const { return w_up_data_; }
  const std::vector<float>& w_down_data() const { return w_down_data_; }
  const std::vector<float>& scale_data() const { return scale_data_; }

 private:
  int x_id_;
  int residual_id_;
  int logits_id_;
  int ei_id_;
  int out_id_;
  std::vector<float> w_gate_data_;
  std::vector<float> w_up_data_;
  std::vector<float> w_down_data_;
  std::vector<float> scale_data_;
};

TEST(MoeTest, ResizeBatchWithAllInternalMoeInputs) {
  const int D_in = 16, D_mid = 32, E = 4;

  auto run_and_verify = [&](InternalInputsMoeModel& model, int B_cur, int N_cur,
                            int K_cur, EiViewKind view_kind) {
    ASSERT_EQ(model.ResizeInputTensor(model.x(), {B_cur, N_cur, D_in}),
              kTfLiteOk);
    ASSERT_EQ(model.ResizeInputTensor(model.residual(), {B_cur, N_cur, D_in}),
              kTfLiteOk);
    ASSERT_EQ(model.ResizeInputTensor(model.logits(), {B_cur, N_cur, K_cur}),
              kTfLiteOk);
    std::vector<int> ei_shape;
    switch (view_kind) {
      case EiViewKind::kExpandDimsPosAxis:
        ei_shape = {B_cur, N_cur};
        break;
      case EiViewKind::kReshapeBatchDynamic:
      case EiViewKind::kReshapeSeqDynamic:
        ei_shape = {B_cur * N_cur, K_cur};
        break;
      case EiViewKind::kMultiHopOneInputReshapeAndNegExpandDims:
        ei_shape = {B_cur * N_cur};
        break;
    }
    ASSERT_EQ(model.ResizeInputTensor(model.ei(), ei_shape), kTfLiteOk);
    ASSERT_EQ(model.AllocateTensors(), kTfLiteOk);
    ASSERT_EQ(model.GetTensorShape(model.out()),
              (std::vector<int>{B_cur, N_cur, D_in}));

    const int T = B_cur * N_cur;
    std::vector<float> x(T * D_in);
    std::vector<float> residual(T * D_in);
    std::vector<float> tokens(T * D_in);
    for (int i = 0; i < T * D_in; ++i) {
      x[i] = 0.3f * std::sin(static_cast<float>(i) + 0.2f);
      residual[i] = 0.2f * std::cos(static_cast<float>(i) + 0.7f);
      tokens[i] = x[i] + residual[i];
    }

    std::vector<float> logits(T * K_cur);
    std::vector<float> rw(T * K_cur);
    for (int t = 0; t < T; ++t) {
      float max_logit = -1e30f;
      for (int k = 0; k < K_cur; ++k) {
        logits[t * K_cur + k] =
            0.5f * std::sin(static_cast<float>(t * K_cur + k));
        max_logit = std::max(max_logit, logits[t * K_cur + k]);
      }
      float sum_exp = 0.0f;
      for (int k = 0; k < K_cur; ++k) {
        rw[t * K_cur + k] = std::exp(logits[t * K_cur + k] - max_logit);
        sum_exp += rw[t * K_cur + k];
      }
      for (int k = 0; k < K_cur; ++k) {
        rw[t * K_cur + k] /= sum_exp;
      }
    }

    std::vector<int32_t> ei(T * K_cur);
    for (int i = 0; i < T * K_cur; ++i) {
      ei[i] = (i * 3 + 1) % E;
    }

    model.PopulateTensor(model.x(), x);
    model.PopulateTensor(model.residual(), residual);
    model.PopulateTensor(model.logits(), logits);
    model.PopulateTensor(model.ei(), ei);
    ASSERT_EQ(model.Invoke(), kTfLiteOk);

    std::vector<float> expected_out(T * D_in);
    RunReferenceMoe(B_cur, N_cur, D_in, D_mid, E, K_cur, tokens.data(),
                    rw.data(), ei.data(), model.w_gate_data().data(),
                    model.w_up_data().data(), model.w_down_data().data(),
                    model.scale_data().data(), model.scale_data().size(),
                    expected_out.data());
    std::vector<float> out = model.ExtractVector<float>(model.out());
    for (size_t i = 0; i < expected_out.size(); ++i) {
      EXPECT_NEAR(out[i], expected_out[i], 1e-4f) << "Mismatch at " << i;
    }
  };

  for (bool static_shape : {false, true}) {
    SCOPED_TRACE(static_shape ? "static_shape" : "dynamic_shape");
    {
      // ADD + SOFTMAX + EXPAND_DIMS(axis=2) + MoE fused in one partition.
      InternalInputsMoeModel model(/*B=*/1, /*N=*/3, D_in, D_mid, E, /*K=*/1,
                                   static_shape,
                                   EiViewKind::kExpandDimsPosAxis);
      ASSERT_EQ(model.NumNodesInExecutionPlan(), 1u);
      run_and_verify(model, /*B_cur=*/2, /*N_cur=*/3, /*K_cur=*/1,
                     EiViewKind::kExpandDimsPosAxis);
      run_and_verify(model, /*B_cur=*/3, /*N_cur=*/3, /*K_cur=*/1,
                     EiViewKind::kExpandDimsPosAxis);
      run_and_verify(model, /*B_cur=*/2, /*N_cur=*/5, /*K_cur=*/1,
                     EiViewKind::kExpandDimsPosAxis);
      run_and_verify(model, /*B_cur=*/2, /*N_cur=*/1, /*K_cur=*/1,
                     EiViewKind::kExpandDimsPosAxis);
    }
    {
      // ADD + SOFTMAX + 2-input RESHAPE([-1, N, K]) + MoE fused in 1 partition.
      InternalInputsMoeModel model(/*B=*/1, /*N=*/3, D_in, D_mid, E, /*K=*/2,
                                   static_shape,
                                   EiViewKind::kReshapeBatchDynamic);
      ASSERT_EQ(model.NumNodesInExecutionPlan(), 1u);
      run_and_verify(model, /*B_cur=*/2, /*N_cur=*/3, /*K_cur=*/2,
                     EiViewKind::kReshapeBatchDynamic);
      run_and_verify(model, /*B_cur=*/3, /*N_cur=*/3, /*K_cur=*/2,
                     EiViewKind::kReshapeBatchDynamic);
      run_and_verify(model, /*B_cur=*/1, /*N_cur=*/3, /*K_cur=*/2,
                     EiViewKind::kReshapeBatchDynamic);
    }
    {
      // ADD + SOFTMAX + 2-input RESHAPE([1, -1, K]) + MoE resizing sequence N.
      InternalInputsMoeModel model(/*B=*/1, /*N=*/2, D_in, D_mid, E, /*K=*/2,
                                   static_shape,
                                   EiViewKind::kReshapeSeqDynamic);
      ASSERT_EQ(model.NumNodesInExecutionPlan(), 1u);
      run_and_verify(model, /*B_cur=*/1, /*N_cur=*/5, /*K_cur=*/2,
                     EiViewKind::kReshapeSeqDynamic);
      run_and_verify(model, /*B_cur=*/1, /*N_cur=*/1, /*K_cur=*/2,
                     EiViewKind::kReshapeSeqDynamic);
      run_and_verify(model, /*B_cur=*/1, /*N_cur=*/3, /*K_cur=*/2,
                     EiViewKind::kReshapeSeqDynamic);
    }
    {
      // Multi-hop: 1D [B * N] -> 1-input RESHAPE({-1, N}) via
      // TfLiteReshapeParams -> EXPAND_DIMS(axis=-1) -> [B, N, 1] + MoE.
      InternalInputsMoeModel model(
          /*B=*/1, /*N=*/3, D_in, D_mid, E, /*K=*/1, static_shape,
          EiViewKind::kMultiHopOneInputReshapeAndNegExpandDims);
      ASSERT_EQ(model.NumNodesInExecutionPlan(), 1u);
      run_and_verify(model, /*B_cur=*/2, /*N_cur=*/3, /*K_cur=*/1,
                     EiViewKind::kMultiHopOneInputReshapeAndNegExpandDims);
      run_and_verify(model, /*B_cur=*/3, /*N_cur=*/3, /*K_cur=*/1,
                     EiViewKind::kMultiHopOneInputReshapeAndNegExpandDims);
      run_and_verify(model, /*B_cur=*/1, /*N_cur=*/3, /*K_cur=*/1,
                     EiViewKind::kMultiHopOneInputReshapeAndNegExpandDims);
    }
  }
}

TEST(MoeTest, UnquantizedInt8WeightsFallback) {
  // 7-input `odml.moe_experts` with INT8 weights that have missing (`scale ==
  // 0`) or asymmetric (`zero_point != 0`) legacy quantization params must be
  // rejected cleanly and fall back to decomposition.
  const int B = 1, N = 2, D_in = 16, D_mid = 32, E = 4, K = 2;
  const std::vector<int32_t> tokens_shape = {B, N, D_in};
  const std::vector<int32_t> rw_shape = {B, N, K};
  const std::vector<int32_t> ei_shape = {B, N, K};
  const std::vector<int32_t> wg_shape = {D_mid, E, 1, D_in};
  const std::vector<int32_t> wd_shape = {D_in, E, 1, D_mid};
  const std::vector<int32_t> s_shape = {1};

  class CompositeFallbackResolver : public OpResolver {
   public:
    const TfLiteRegistration* FindOp(BuiltinOperator op, int) const override {
      if (op == BuiltinOperator_STABLEHLO_COMPOSITE) {
        static TfLiteRegistration reg = []() {
          TfLiteRegistration r{};
          r.builtin_code = kTfLiteBuiltinStablehloComposite;
          return r;
        }();
        return &reg;
      }
      return nullptr;
    }
    const TfLiteRegistration* FindOp(const char* op, int) const override {
      if (op != nullptr && std::strcmp(op, "moe_fallback") == 0) {
        static TfLiteRegistration reg = []() {
          TfLiteRegistration r{};
          r.builtin_code = kTfLiteBuiltinCustom;
          r.custom_name = "moe_fallback";
          return r;
        }();
        return &reg;
      }
      return nullptr;
    }
  };

  for (auto [legacy_scale, legacy_zp] : {std::pair<float, int64_t>{0.0f, 0},
                                         std::pair<float, int64_t>{0.1f, 3}}) {
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<Tensor>> tensors;
    std::vector<flatbuffers::Offset<Buffer>> buffers;
    buffers.push_back(CreateBufferDirect(builder));
    auto add_buffer = [&](const void* data, size_t bytes) -> int {
      int id = buffers.size();
      buffers.push_back(CreateBuffer(
          builder,
          builder.CreateVector(reinterpret_cast<const uint8_t*>(data), bytes)));
      return id;
    };

    tensors.push_back(CreateTensorDirect(builder, &tokens_shape,
                                         TensorType_FLOAT32, 0, "tokens"));
    tensors.push_back(
        CreateTensorDirect(builder, &rw_shape, TensorType_FLOAT32, 0, "rw"));
    tensors.push_back(
        CreateTensorDirect(builder, &ei_shape, TensorType_INT32, 0, "ei"));

    const std::vector<int8_t> wg(D_mid * E * D_in, 1);
    const std::vector<int8_t> wd(D_in * E * D_mid, 1);
    const std::vector<float> s = {1.0f};
    auto make_q = [&]() {
      if (legacy_scale == 0.0f) {
        return flatbuffers::Offset<QuantizationParameters>();
      }
      std::vector<float> scales = {legacy_scale};
      std::vector<int64_t> zps = {legacy_zp};
      return CreateQuantizationParametersDirect(builder, /*min=*/nullptr,
                                                /*max=*/nullptr, &scales, &zps);
    };
    tensors.push_back(CreateTensorDirect(builder, &wg_shape, TensorType_INT8,
                                         add_buffer(wg.data(), wg.size()), "wg",
                                         make_q()));
    tensors.push_back(CreateTensorDirect(builder, &wg_shape, TensorType_INT8,
                                         add_buffer(wg.data(), wg.size()), "wu",
                                         make_q()));
    tensors.push_back(CreateTensorDirect(builder, &wd_shape, TensorType_INT8,
                                         add_buffer(wd.data(), wd.size()), "wd",
                                         make_q()));
    tensors.push_back(CreateTensorDirect(
        builder, &s_shape, TensorType_FLOAT32,
        add_buffer(s.data(), s.size() * sizeof(float)), "scale"));
    tensors.push_back(CreateTensorDirect(builder, &tokens_shape,
                                         TensorType_FLOAT32, 0, "out"));

    std::vector<uint8_t> empty_attrs;
    auto comp_opts = CreateStableHLOCompositeOptionsDirect(
        builder, "odml.moe_experts", 1, &empty_attrs);
    std::vector<flatbuffers::Offset<OperatorCode>> opcodes = {
        CreateOperatorCode(
            builder,
            static_cast<int8_t>(
                BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES),
            0, 1, BuiltinOperator_STABLEHLO_COMPOSITE),
        CreateOperatorCodeDirect(builder, BuiltinOperator_CUSTOM,
                                 "moe_fallback")};
    std::vector<int32_t> m_in = {0, 1, 2, 3, 4, 5, 6};
    std::vector<int32_t> m_out = {7};
    std::vector<int32_t> sg_ins = {0, 1, 2};
    std::vector<flatbuffers::Offset<Operator>> ops = {CreateOperator(
        builder, 0, builder.CreateVector(m_in), builder.CreateVector(m_out),
        BuiltinOptions_NONE, 0, 0, CustomOptionsFormat_FLEXBUFFERS, 0, 0, 0, 0,
        BuiltinOptions2_StableHLOCompositeOptions, comp_opts.Union())};

    std::vector<flatbuffers::Offset<Tensor>> decomp_tensors = {
        CreateTensorDirect(builder, &tokens_shape, TensorType_FLOAT32, 0, "t0"),
        CreateTensorDirect(builder, &rw_shape, TensorType_FLOAT32, 0, "t1"),
        CreateTensorDirect(builder, &ei_shape, TensorType_INT32, 0, "t2"),
        CreateTensorDirect(builder, &wg_shape, TensorType_INT8, 0, "t3"),
        CreateTensorDirect(builder, &wg_shape, TensorType_INT8, 0, "t4"),
        CreateTensorDirect(builder, &wd_shape, TensorType_INT8, 0, "t5"),
        CreateTensorDirect(builder, &s_shape, TensorType_FLOAT32, 0, "t6"),
        CreateTensorDirect(builder, &tokens_shape, TensorType_FLOAT32, 0,
                           "t7")};
    std::vector<flatbuffers::Offset<Operator>> decomp_ops = {CreateOperator(
        builder, 1, builder.CreateVector(m_in), builder.CreateVector(m_out))};
    std::vector<flatbuffers::Offset<SubGraph>> subgraphs = {
        CreateSubGraphDirect(builder, &tensors, &sg_ins, &m_out, &ops),
        CreateSubGraphDirect(builder, &decomp_tensors, &m_in, &m_out,
                             &decomp_ops)};
    FinishModelBuffer(builder, CreateModelDirect(builder, 3, &opcodes,
                                                 &subgraphs, "m", &buffers));

    std::vector<uint8_t> model_buf(
        builder.GetBufferPointer(),
        builder.GetBufferPointer() + builder.GetSize());
    auto fb = FlatBufferModel::BuildFromBuffer(
        reinterpret_cast<const char*>(model_buf.data()), model_buf.size());
    ASSERT_NE(fb, nullptr);
    CompositeFallbackResolver resolver;
    TfLiteYNNPackDelegateOptions opts = TfLiteYNNPackDelegateOptionsDefault();
    opts.num_threads = 1;
    Interpreter::TfLiteDelegatePtr delegate(TfLiteYNNPackDelegateCreate(&opts),
                                            TfLiteYNNPackDelegateDelete);
    std::unique_ptr<Interpreter> interpreter;
    ASSERT_EQ(InterpreterBuilder(*fb, resolver)(&interpreter), kTfLiteOk);
    ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate.get()), kTfLiteOk);
    ASSERT_EQ(interpreter->execution_plan().size(), 1u);
    const auto* n0 =
        interpreter->node_and_registration(interpreter->execution_plan()[0]);
    ASSERT_NE(n0, nullptr);
    EXPECT_STREQ(n0->second.custom_name, "moe_fallback");
  }
}

TEST(MoeTest, ResolveZeroPointTensorIndexValidationAndTensorZero) {
  const int B = 1, N = 2, D_in = 32, D_mid = 64, E = 4, K = 2, block_size = 16;
  const std::vector<int32_t> tokens_shape = {B, N, D_in};
  const std::vector<int32_t> rw_shape = {B, N, K};
  const std::vector<int32_t> ei_shape = {B, N, K};
  const std::vector<int32_t> wg_shape = {D_mid, E, 1, D_in};
  const std::vector<int32_t> wd_shape = {D_in, E, 1, D_mid};
  const std::vector<int32_t> sg_shape = {D_mid, E, 1, D_in / block_size};
  const std::vector<int32_t> sd_shape = {D_in, E, 1, D_mid / block_size};
  const std::vector<int32_t> s_shape = {1, 1, 1, E};

  enum class ZpCase {
    kOutOfBoundsZpIdx,
    kZpEqualsScaleIdx,
    kZpEqualsWeightIdx,
    kFloatZpTensor,
    kMismatchedZpShape,
    kNonConstantScaleTensor,
    kNonConstantZpTensor,
    kValidZpAtTensorZero,
  };

  const std::vector<int8_t> wg_data =
      MakeQuantizedWeights(D_mid * E * D_in, /*seed=*/1, TensorType_INT8);
  const std::vector<int8_t> wu_data =
      MakeQuantizedWeights(D_mid * E * D_in, /*seed=*/2, TensorType_INT8);
  const std::vector<int8_t> wd_data =
      MakeQuantizedWeights(D_in * E * D_mid, /*seed=*/3, TensorType_INT8);
  const std::vector<float> g_scales(D_mid * E * (D_in / block_size), 0.02f);
  const std::vector<float> d_scales(D_in * E * (D_mid / block_size), 0.02f);
  std::vector<int32_t> g_zp(g_scales.size());
  std::vector<int32_t> d_zp(d_scales.size());
  for (size_t i = 0; i < g_zp.size(); ++i) {
    g_zp[i] = static_cast<int32_t>((i % 5) - 2);
  }
  for (size_t i = 0; i < d_zp.size(); ++i) {
    d_zp[i] = static_cast<int32_t>(((i + 2) % 5) - 2);
  }
  const std::vector<float> out_scales(E, 1.0f);

  auto run_case = [&](ZpCase zp_case) {
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<Tensor>> tensors;
    std::vector<flatbuffers::Offset<Buffer>> buffers;
    buffers.push_back(CreateBufferDirect(builder));
    auto add_buffer = [&](const void* data, size_t bytes) -> int {
      int id = buffers.size();
      buffers.push_back(CreateBuffer(
          builder,
          builder.CreateVector(reinterpret_cast<const uint8_t*>(data), bytes)));
      return id;
    };

    // Place `g_zp` at tensor index 0 when testing `zp_idx == 0` binding.
    int zp0_id = -1;
    if (zp_case == ZpCase::kValidZpAtTensorZero) {
      zp0_id = tensors.size();
      tensors.push_back(CreateTensorDirect(
          builder, &sg_shape, TensorType_INT32,
          add_buffer(g_zp.data(), g_zp.size() * sizeof(int32_t)), "zp0"));
    }

    const int tokens_id = tensors.size();
    tensors.push_back(CreateTensorDirect(builder, &tokens_shape,
                                         TensorType_FLOAT32, 0, "tokens"));
    const int rw_id = tensors.size();
    tensors.push_back(
        CreateTensorDirect(builder, &rw_shape, TensorType_FLOAT32, 0, "rw"));
    const int ei_id = tensors.size();
    tensors.push_back(
        CreateTensorDirect(builder, &ei_shape, TensorType_INT32, 0, "ei"));

    auto add_gate_or_up = [&](const std::vector<int8_t>& w_bytes,
                              const char* name, bool apply_case) -> int {
      const int w_id = tensors.size();
      const int scale_id = w_id + 1;
      int zp_id = w_id + 2;
      if (apply_case) {
        switch (zp_case) {
          case ZpCase::kOutOfBoundsZpIdx:
            zp_id = 999;
            break;
          case ZpCase::kZpEqualsScaleIdx:
            zp_id = scale_id;
            break;
          case ZpCase::kZpEqualsWeightIdx:
            zp_id = w_id;
            break;
          case ZpCase::kValidZpAtTensorZero:
            zp_id = zp0_id;
            break;
          default:
            break;
        }
      }
      auto bq = CreateBlockwiseQuantizationDirect(builder, scale_id, zp_id,
                                                  block_size);
      auto q = CreateQuantizationParameters(
          builder, 0, 0, 0, 0, QuantizationDetails_BlockwiseQuantization,
          bq.Union(), 3);
      tensors.push_back(CreateTensorDirect(
          builder, &wg_shape, TensorType_INT8,
          add_buffer(w_bytes.data(), w_bytes.size()), name, q));

      const int s_buf =
          (apply_case && zp_case == ZpCase::kNonConstantScaleTensor)
              ? 0
              : add_buffer(g_scales.data(), g_scales.size() * sizeof(float));
      tensors.push_back(
          CreateTensorDirect(builder, &sg_shape, TensorType_FLOAT32, s_buf));

      if (!apply_case || (zp_case != ZpCase::kOutOfBoundsZpIdx &&
                          zp_case != ZpCase::kZpEqualsScaleIdx &&
                          zp_case != ZpCase::kZpEqualsWeightIdx &&
                          zp_case != ZpCase::kValidZpAtTensorZero)) {
        if (apply_case && zp_case == ZpCase::kFloatZpTensor) {
          const std::vector<float> bad_zp(g_zp.size(), 0.0f);
          tensors.push_back(CreateTensorDirect(
              builder, &sg_shape, TensorType_FLOAT32,
              add_buffer(bad_zp.data(), bad_zp.size() * sizeof(float))));
        } else if (apply_case && zp_case == ZpCase::kMismatchedZpShape) {
          const std::vector<int32_t> bad_shape = {D_mid, E, 1, 1};
          const std::vector<int32_t> bad_zp(D_mid * E, 0);
          tensors.push_back(CreateTensorDirect(
              builder, &bad_shape, TensorType_INT32,
              add_buffer(bad_zp.data(), bad_zp.size() * sizeof(int32_t))));
        } else {
          const int z_buf =
              (apply_case && zp_case == ZpCase::kNonConstantZpTensor)
                  ? 0
                  : add_buffer(g_zp.data(), g_zp.size() * sizeof(int32_t));
          tensors.push_back(
              CreateTensorDirect(builder, &sg_shape, TensorType_INT32, z_buf));
        }
      }
      return w_id;
    };

    const int wg_id = add_gate_or_up(wg_data, "wg", /*apply_case=*/true);
    const int wu_id =
        add_gate_or_up(wu_data, "wu",
                       /*apply_case=*/zp_case == ZpCase::kValidZpAtTensorZero);

    const int wd_id = tensors.size();
    const int sd_id = wd_id + 1;
    const int zd_id = wd_id + 2;
    auto bq_d =
        CreateBlockwiseQuantizationDirect(builder, sd_id, zd_id, block_size);
    auto q_d = CreateQuantizationParameters(
        builder, 0, 0, 0, 0, QuantizationDetails_BlockwiseQuantization,
        bq_d.Union(), 3);
    tensors.push_back(CreateTensorDirect(
        builder, &wd_shape, TensorType_INT8,
        add_buffer(wd_data.data(), wd_data.size()), "wd", q_d));
    tensors.push_back(CreateTensorDirect(
        builder, &sd_shape, TensorType_FLOAT32,
        add_buffer(d_scales.data(), d_scales.size() * sizeof(float))));
    tensors.push_back(CreateTensorDirect(
        builder, &sd_shape, TensorType_INT32,
        add_buffer(d_zp.data(), d_zp.size() * sizeof(int32_t))));

    const int scale_id = tensors.size();
    tensors.push_back(CreateTensorDirect(
        builder, &s_shape, TensorType_FLOAT32,
        add_buffer(out_scales.data(), out_scales.size() * sizeof(float))));
    const int out_id = tensors.size();
    tensors.push_back(CreateTensorDirect(builder, &tokens_shape,
                                         TensorType_FLOAT32, 0, "out"));

    std::vector<uint8_t> empty_attrs;
    auto comp_opts = CreateStableHLOCompositeOptionsDirect(
        builder, "odml.moe_experts", 1, &empty_attrs);
    std::vector<flatbuffers::Offset<OperatorCode>> opcodes = {
        CreateOperatorCode(
            builder,
            static_cast<int8_t>(
                BuiltinOperator_PLACEHOLDER_FOR_GREATER_OP_CODES),
            0, 1, BuiltinOperator_STABLEHLO_COMPOSITE)};
    std::vector<int32_t> op_in = {tokens_id, rw_id, ei_id,   wg_id,
                                  wu_id,     wd_id, scale_id};
    std::vector<int32_t> op_out = {out_id};
    std::vector<int32_t> sg_in = {tokens_id, rw_id, ei_id};
    std::vector<flatbuffers::Offset<Operator>> ops = {CreateOperator(
        builder, 0, builder.CreateVector(op_in), builder.CreateVector(op_out),
        BuiltinOptions_NONE, 0, 0, CustomOptionsFormat_FLEXBUFFERS, 0, 0, 0, 0,
        BuiltinOptions2_StableHLOCompositeOptions, comp_opts.Union())};
    std::vector<flatbuffers::Offset<SubGraph>> subgraphs = {
        CreateSubGraphDirect(builder, &tensors, &sg_in, &op_out, &ops)};
    FinishModelBuffer(builder, CreateModelDirect(builder, 3, &opcodes,
                                                 &subgraphs, "m", &buffers));

    std::vector<uint8_t> model_buf(
        builder.GetBufferPointer(),
        builder.GetBufferPointer() + builder.GetSize());
    auto fb = FlatBufferModel::BuildFromBuffer(
        reinterpret_cast<const char*>(model_buf.data()), model_buf.size());
    ASSERT_NE(fb, nullptr);
    TfLiteRegistration dummy_reg{};
    SingleOpResolver resolver(BuiltinOperator_STABLEHLO_COMPOSITE, &dummy_reg);
    TfLiteYNNPackDelegateOptions opts = TfLiteYNNPackDelegateOptionsDefault();
    opts.num_threads = 1;
    opts.enable_blockwise_quantization = true;
    Interpreter::TfLiteDelegatePtr delegate(TfLiteYNNPackDelegateCreate(&opts),
                                            TfLiteYNNPackDelegateDelete);
    std::unique_ptr<Interpreter> interpreter;
    ASSERT_EQ(InterpreterBuilder(*fb, resolver)(&interpreter), kTfLiteOk);

    if (zp_case != ZpCase::kValidZpAtTensorZero) {
      // `ModifyGraphWithDelegate` rejects the unsupported MoE composite and
      // attempts to inline decomposition subgraph 1 (absent here), failing.
      EXPECT_NE(interpreter->ModifyGraphWithDelegate(delegate.get()),
                kTfLiteOk);
      return;
    }

    ASSERT_EQ(interpreter->ModifyGraphWithDelegate(delegate.get()), kTfLiteOk);
    ASSERT_EQ(interpreter->AllocateTensors(), kTfLiteOk);

    const std::vector<float> tokens = MakeTokens(B * N * D_in);
    const std::vector<float> rw(B * N * K, 0.5f);
    const std::vector<int32_t> ei = {0, 2, 1, 3};
    std::memcpy(interpreter->tensor(tokens_id)->data.raw, tokens.data(),
                tokens.size() * sizeof(float));
    std::memcpy(interpreter->tensor(rw_id)->data.raw, rw.data(),
                rw.size() * sizeof(float));
    std::memcpy(interpreter->tensor(ei_id)->data.raw, ei.data(),
                ei.size() * sizeof(int32_t));
    ASSERT_EQ(interpreter->Invoke(), kTfLiteOk);

    const std::vector<float> wg_ref = DequantizeBlockwise(
        wg_data, g_scales, g_zp, D_mid, E, D_in, block_size);
    const std::vector<float> wu_ref = DequantizeBlockwise(
        wu_data, g_scales, g_zp, D_mid, E, D_in, block_size);
    const std::vector<float> wd_ref = DequantizeBlockwise(
        wd_data, d_scales, d_zp, D_in, E, D_mid, block_size);
    std::vector<float> expected_out(B * N * D_in);
    RunReferenceMoe(B, N, D_in, D_mid, E, K, tokens.data(), rw.data(),
                    ei.data(), wg_ref.data(), wu_ref.data(), wd_ref.data(),
                    out_scales.data(), out_scales.size(), expected_out.data());
    float max_abs = 0.0f;
    for (float v : expected_out) max_abs = std::max(max_abs, std::abs(v));
    const float* actual = interpreter->typed_tensor<float>(out_id);
    for (int i = 0; i < B * N * D_in; ++i) {
      EXPECT_NEAR(actual[i], expected_out[i], 0.02f * max_abs) << "at " << i;
    }
  };

  for (ZpCase zp_case :
       {ZpCase::kOutOfBoundsZpIdx, ZpCase::kZpEqualsScaleIdx,
        ZpCase::kZpEqualsWeightIdx, ZpCase::kFloatZpTensor,
        ZpCase::kMismatchedZpShape, ZpCase::kNonConstantScaleTensor,
        ZpCase::kNonConstantZpTensor, ZpCase::kValidZpAtTensorZero}) {
    SCOPED_TRACE(static_cast<int>(zp_case));
    run_case(zp_case);
  }
}

}  // namespace
}  // namespace ynnpack
}  // namespace tflite
