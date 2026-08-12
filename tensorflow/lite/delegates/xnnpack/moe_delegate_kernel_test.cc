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

#include "tensorflow/lite/delegates/xnnpack/moe_delegate_kernel.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"  // from @XNNPACK
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace xnnpack {
namespace {

void DummyReportError(TfLiteContext*, const char*, ...) {}

TfLiteIntArray* CreateDims(const std::vector<int>& dims) {
  TfLiteIntArray* arr = TfLiteIntArrayCreate(dims.size());
  for (size_t i = 0; i < dims.size(); ++i) arr->data[i] = dims[i];
  return arr;
}

struct NodeAndReg {
  TfLiteNode* node = nullptr;
  TfLiteRegistration* reg = nullptr;
};

class MoeKernelAccuracyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(xnn_initialize(/*allocator=*/nullptr), xnn_status_success);
  }
};

TEST(MoeExpertsDelegateKernelTest, SupportsInt8WeightType) {
  TfLiteContext context = {};
  context.ReportError = DummyReportError;
  context.tensors_size = 11;
  std::vector<TfLiteTensor> tensors(11);
  context.tensors = tensors.data();

  // Define 10 input tensors for int8 mode + 1 output tensor.
  tensors[0].type = kTfLiteFloat32;
  tensors[1].type = kTfLiteFloat32;
  tensors[2].type = kTfLiteInt32;

  // gate/ff1 weights (64 elements: [8, 2, 1, 4])
  tensors[3].type = kTfLiteInt8;
  tensors[3].allocation_type = kTfLiteMmapRo;
  tensors[3].dims = CreateDims({8, 2, 1, 4});

  tensors[5].type = kTfLiteInt8;
  tensors[5].allocation_type = kTfLiteMmapRo;
  tensors[5].dims = CreateDims({8, 2, 1, 4});

  // linear weight (64 elements: [4, 2, 1, 8])
  tensors[7].type = kTfLiteInt8;
  tensors[7].allocation_type = kTfLiteMmapRo;
  tensors[7].dims = CreateDims({4, 2, 1, 8});

  // gate/ff1 scales (16 elements: [8, 2])
  tensors[4].type = kTfLiteFloat32;
  tensors[4].allocation_type = kTfLiteMmapRo;
  tensors[4].dims = CreateDims({8, 2});

  tensors[6].type = kTfLiteFloat32;
  tensors[6].allocation_type = kTfLiteMmapRo;
  tensors[6].dims = CreateDims({8, 2});

  // linear scale (8 elements: [4, 2])
  tensors[8].type = kTfLiteFloat32;
  tensors[8].allocation_type = kTfLiteMmapRo;
  tensors[8].dims = CreateDims({4, 2});

  // per_expert_scale (2 elements: [2])
  tensors[9].type = kTfLiteFloat32;
  tensors[9].allocation_type = kTfLiteMmapRo;
  tensors[9].dims = CreateDims({2});

  // output (fp32)
  tensors[10].type = kTfLiteFloat32;

  std::vector<int> inputs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> outputs = {10};
  TfLiteIntArray* input_array = CreateDims(inputs);
  TfLiteIntArray* output_array = CreateDims(outputs);

  TfLiteNode node = {};
  node.inputs = input_array;
  node.outputs = output_array;

  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("num_experts", 2);
    fbb.Int("num_active_experts", 1);
    fbb.Int("model_dim", 4);
    fbb.Int("hidden_dim", 8);
    fbb.String("weight_type", "int8");
  });
  fbb.Finish();
  node.custom_initial_data = fbb.GetBuffer().data();
  node.custom_initial_data_size = fbb.GetBuffer().size();

  TfLiteRegistration registration = {};
  registration.builtin_code = kTfLiteBuiltinCustom;
  registration.custom_name = "moe";

  EXPECT_EQ(
      MoeExpertsDelegateKernel::IsSupported(&context, &node, &registration, 0),
      kTfLiteOk);

  for (int idx : {3, 4, 5, 6, 7, 8, 9}) {
    TfLiteIntArrayFree(tensors[idx].dims);
  }
  TfLiteIntArrayFree(input_array);
  TfLiteIntArrayFree(output_array);
}

TEST_F(MoeKernelAccuracyTest, AccuracyFp32Execution) {
  const int num_experts = 2;
  const int num_active_experts = 1;
  const int model_dim = 2;
  const int hidden_dim = 2;

  std::vector<float> src_data = {1.0f, 2.0f};
  std::vector<float> top_weights_data = {1.0f};
  std::vector<int32_t> top_indices_data = {0};

  // Expert 0:
  // gate row 0 (row_idx=0): [1.0, 2.0] -> gate[0] = 1*1 + 2*2 = 5.0
  // gate row 1 (row_idx=2): [-1.0, 0.5] -> gate[1] = 1*(-1) + 2*0.5 = 0.0
  std::vector<float> gate_data = {
      1.0f,  2.0f,  // row 0 (out=0, exp=0)
      0.0f,  0.0f,  // row 1 (out=0, exp=1)
      -1.0f, 0.5f,  // row 2 (out=1, exp=0)
      0.0f,  0.0f   // row 3 (out=1, exp=1)
  };

  // ff1 row 0: [1.0, 0.0] -> ff1[0] = 1.0
  // ff1 row 1: [0.0, 1.0] -> ff1[1] = 2.0
  std::vector<float> ff1_data = {
      1.0f, 0.0f,  // row 0 (out=0, exp=0)
      0.0f, 0.0f,  // row 1 (out=0, exp=1)
      0.0f, 1.0f,  // row 2 (out=1, exp=0)
      0.0f, 0.0f   // row 3 (out=1, exp=1)
  };

  // hidden[0] = GeluTanh(5.0) * 1.0 = 5.0
  // hidden[1] = GeluTanh(0.0) * 2.0 = 0.0
  // linear row 0: [1.0, 0.0] -> down[0] = 5.0 * 1.0 = 5.0
  // linear row 1: [0.0, 1.0] -> down[1] = 0.0 * 1.0 = 0.0
  std::vector<float> linear_data = {
      1.0f, 0.0f,  // row 0 (out=0, exp=0)
      0.0f, 0.0f,  // row 1 (out=0, exp=1)
      0.0f, 1.0f,  // row 2 (out=1, exp=0)
      0.0f, 0.0f   // row 3 (out=1, exp=1)
  };

  std::vector<float> per_expert_scale_data = {1.0f, 1.0f};
  std::vector<float> output_data = {0.0f, 0.0f};

  std::vector<TfLiteTensor> tensors(8);
  tensors[0].type = kTfLiteFloat32;
  tensors[0].data.f = src_data.data();
  tensors[0].dims = CreateDims({1, model_dim});

  tensors[1].type = kTfLiteFloat32;
  tensors[1].data.f = top_weights_data.data();
  tensors[1].dims = CreateDims({1, num_active_experts});

  tensors[2].type = kTfLiteInt32;
  tensors[2].data.i32 = top_indices_data.data();
  tensors[2].dims = CreateDims({1, num_active_experts});

  tensors[3].type = kTfLiteFloat32;
  tensors[3].allocation_type = kTfLiteMmapRo;
  tensors[3].data.f = gate_data.data();
  tensors[3].dims = CreateDims({hidden_dim, num_experts, 1, model_dim});

  tensors[4].type = kTfLiteFloat32;
  tensors[4].allocation_type = kTfLiteMmapRo;
  tensors[4].data.f = ff1_data.data();
  tensors[4].dims = CreateDims({hidden_dim, num_experts, 1, model_dim});

  tensors[5].type = kTfLiteFloat32;
  tensors[5].allocation_type = kTfLiteMmapRo;
  tensors[5].data.f = linear_data.data();
  tensors[5].dims = CreateDims({model_dim, num_experts, 1, hidden_dim});

  tensors[6].type = kTfLiteFloat32;
  tensors[6].allocation_type = kTfLiteMmapRo;
  tensors[6].data.f = per_expert_scale_data.data();
  tensors[6].dims = CreateDims({num_experts});

  tensors[7].type = kTfLiteFloat32;
  tensors[7].data.f = output_data.data();
  tensors[7].dims = CreateDims({1, model_dim});

  std::vector<int> inputs = {0, 1, 2, 3, 4, 5, 6};
  std::vector<int> outputs = {7};
  TfLiteIntArray* input_array = CreateDims(inputs);
  TfLiteIntArray* output_array = CreateDims(outputs);

  TfLiteNode node = {};
  node.inputs = input_array;
  node.outputs = output_array;

  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("num_experts", num_experts);
    fbb.Int("num_active_experts", num_active_experts);
    fbb.Int("model_dim", model_dim);
    fbb.Int("hidden_dim", hidden_dim);
    fbb.String("weight_type", "fp32");
    fbb.String("activation", "gelu_tanh");
  });
  fbb.Finish();
  node.custom_initial_data = fbb.GetBuffer().data();
  node.custom_initial_data_size = fbb.GetBuffer().size();

  TfLiteRegistration registration = {};
  registration.builtin_code = kTfLiteBuiltinCustom;
  registration.custom_name = "moe";

  NodeAndReg node_and_reg = {&node, &registration};

  TfLiteContext context = {};
  context.ReportError = DummyReportError;
  context.tensors = tensors.data();
  context.tensors_size = tensors.size();
  context.GetNodeAndRegistration =
      [](TfLiteContext* ctx, int node_index, TfLiteNode** out_node,
         TfLiteRegistration** out_reg) -> TfLiteStatus {
    auto* nr = static_cast<NodeAndReg*>(ctx->impl_);
    *out_node = nr->node;
    *out_reg = nr->reg;
    return kTfLiteOk;
  };
  context.ResizeTensor = [](TfLiteContext* ctx, TfLiteTensor* tensor,
                            TfLiteIntArray* new_size) -> TfLiteStatus {
    TfLiteIntArrayFree(tensor->dims);
    tensor->dims = new_size;
    return kTfLiteOk;
  };
  context.impl_ = &node_and_reg;

  std::vector<int> nodes_to_replace_vec = {0};
  TfLiteDelegateParams params = {};
  params.nodes_to_replace = CreateDims(nodes_to_replace_vec);

  auto kernel = MoeExpertsDelegateKernel::Create(&context, &params,
                                                 /*threadpool=*/nullptr);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(kernel->Prepare(&context), kTfLiteOk);
  EXPECT_EQ(kernel->Invoke(&context), kTfLiteOk);

  EXPECT_NEAR(output_data[0], 5.0f, 1e-4f);
  EXPECT_NEAR(output_data[1], 0.0f, 1e-4f);

  for (size_t i = 0; i < tensors.size(); ++i) {
    TfLiteIntArrayFree(tensors[i].dims);
  }
  TfLiteIntArrayFree(input_array);
  TfLiteIntArrayFree(output_array);
  TfLiteIntArrayFree(params.nodes_to_replace);
}

TEST_F(MoeKernelAccuracyTest, AccuracyInt8Execution) {
  const int num_experts = 2;
  const int num_active_experts = 1;
  const int model_dim = 2;
  const int hidden_dim = 2;

  std::vector<float> src_data = {1.0f, 2.0f};
  std::vector<float> top_weights_data = {1.0f};
  std::vector<int32_t> top_indices_data = {0};

  // gate weight (int8) & scale:
  // Expert 0:
  // row 0: [2, 4] * 0.5 = [1.0, 2.0] -> gate[0] = 5.0
  // row 2: [-2, 1] * 0.5 = [-1.0, 0.5] -> gate[1] = 0.0
  std::vector<int8_t> gate_i8 = {2, 4, 0, 0, -2, 1, 0, 0};
  std::vector<float> gate_scales = {0.5f, 0.5f, 0.5f, 0.5f};

  // ff1 weight (int8) & scale:
  // row 0: [2, 0] * 0.5 = [1.0, 0.0] -> ff1[0] = 1.0
  // row 2: [0, 2] * 0.5 = [0.0, 1.0] -> ff1[1] = 2.0
  std::vector<int8_t> ff1_i8 = {2, 0, 0, 0, 0, 2, 0, 0};
  std::vector<float> ff1_scales = {0.5f, 0.5f, 0.5f, 0.5f};

  // linear weight (int8) & scale:
  // row 0: [2, 0] * 0.5 = [1.0, 0.0] -> down[0] = 5.0
  // row 2: [0, 2] * 0.5 = [0.0, 1.0] -> down[1] = 0.0
  std::vector<int8_t> linear_i8 = {2, 0, 0, 0, 0, 2, 0, 0};
  std::vector<float> linear_scales = {0.5f, 0.5f, 0.5f, 0.5f};

  std::vector<float> per_expert_scale_data = {1.0f, 1.0f};
  std::vector<float> output_data = {0.0f, 0.0f};

  std::vector<TfLiteTensor> tensors(11);
  tensors[0].type = kTfLiteFloat32;
  tensors[0].data.f = src_data.data();
  tensors[0].dims = CreateDims({1, model_dim});

  tensors[1].type = kTfLiteFloat32;
  tensors[1].data.f = top_weights_data.data();
  tensors[1].dims = CreateDims({1, num_active_experts});

  tensors[2].type = kTfLiteInt32;
  tensors[2].data.i32 = top_indices_data.data();
  tensors[2].dims = CreateDims({1, num_active_experts});

  // gate weight & scale
  tensors[3].type = kTfLiteInt8;
  tensors[3].allocation_type = kTfLiteMmapRo;
  tensors[3].data.raw = reinterpret_cast<char*>(gate_i8.data());
  tensors[3].dims = CreateDims({hidden_dim, num_experts, 1, model_dim});

  tensors[4].type = kTfLiteFloat32;
  tensors[4].allocation_type = kTfLiteMmapRo;
  tensors[4].data.f = gate_scales.data();
  tensors[4].dims = CreateDims({hidden_dim * num_experts});

  // ff1 weight & scale
  tensors[5].type = kTfLiteInt8;
  tensors[5].allocation_type = kTfLiteMmapRo;
  tensors[5].data.raw = reinterpret_cast<char*>(ff1_i8.data());
  tensors[5].dims = CreateDims({hidden_dim, num_experts, 1, model_dim});

  tensors[6].type = kTfLiteFloat32;
  tensors[6].allocation_type = kTfLiteMmapRo;
  tensors[6].data.f = ff1_scales.data();
  tensors[6].dims = CreateDims({hidden_dim * num_experts});

  // linear weight & scale
  tensors[7].type = kTfLiteInt8;
  tensors[7].allocation_type = kTfLiteMmapRo;
  tensors[7].data.raw = reinterpret_cast<char*>(linear_i8.data());
  tensors[7].dims = CreateDims({model_dim, num_experts, 1, hidden_dim});

  tensors[8].type = kTfLiteFloat32;
  tensors[8].allocation_type = kTfLiteMmapRo;
  tensors[8].data.f = linear_scales.data();
  tensors[8].dims = CreateDims({model_dim * num_experts});

  // per expert scale
  tensors[9].type = kTfLiteFloat32;
  tensors[9].allocation_type = kTfLiteMmapRo;
  tensors[9].data.f = per_expert_scale_data.data();
  tensors[9].dims = CreateDims({num_experts});

  // output
  tensors[10].type = kTfLiteFloat32;
  tensors[10].data.f = output_data.data();
  tensors[10].dims = CreateDims({1, model_dim});

  std::vector<int> inputs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<int> outputs = {10};
  TfLiteIntArray* input_array = CreateDims(inputs);
  TfLiteIntArray* output_array = CreateDims(outputs);

  TfLiteNode node = {};
  node.inputs = input_array;
  node.outputs = output_array;

  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("num_experts", num_experts);
    fbb.Int("num_active_experts", num_active_experts);
    fbb.Int("model_dim", model_dim);
    fbb.Int("hidden_dim", hidden_dim);
    fbb.String("weight_type", "int8");
    fbb.String("activation", "gelu_tanh");
  });
  fbb.Finish();
  node.custom_initial_data = fbb.GetBuffer().data();
  node.custom_initial_data_size = fbb.GetBuffer().size();

  TfLiteRegistration registration = {};
  registration.builtin_code = kTfLiteBuiltinCustom;
  registration.custom_name = "moe";

  NodeAndReg node_and_reg = {&node, &registration};

  TfLiteContext context = {};
  context.ReportError = DummyReportError;
  context.tensors = tensors.data();
  context.tensors_size = tensors.size();
  context.GetNodeAndRegistration =
      [](TfLiteContext* ctx, int node_index, TfLiteNode** out_node,
         TfLiteRegistration** out_reg) -> TfLiteStatus {
    auto* nr = static_cast<NodeAndReg*>(ctx->impl_);
    *out_node = nr->node;
    *out_reg = nr->reg;
    return kTfLiteOk;
  };
  context.ResizeTensor = [](TfLiteContext* ctx, TfLiteTensor* tensor,
                            TfLiteIntArray* new_size) -> TfLiteStatus {
    TfLiteIntArrayFree(tensor->dims);
    tensor->dims = new_size;
    return kTfLiteOk;
  };
  context.impl_ = &node_and_reg;

  std::vector<int> nodes_to_replace_vec = {0};
  TfLiteDelegateParams params = {};
  params.nodes_to_replace = CreateDims(nodes_to_replace_vec);

  auto kernel = MoeExpertsDelegateKernel::Create(&context, &params,
                                                 /*threadpool=*/nullptr);
  ASSERT_NE(kernel, nullptr);
  EXPECT_EQ(kernel->Prepare(&context), kTfLiteOk);
  EXPECT_EQ(kernel->Invoke(&context), kTfLiteOk);

  EXPECT_NEAR(output_data[0], 5.0f, 1e-4f);
  EXPECT_NEAR(output_data[1], 0.0f, 1e-4f);

  for (size_t i = 0; i < tensors.size(); ++i) {
    TfLiteIntArrayFree(tensors[i].dims);
  }
  TfLiteIntArrayFree(input_array);
  TfLiteIntArrayFree(output_array);
  TfLiteIntArrayFree(params.nodes_to_replace);
}

}  // namespace
}  // namespace xnnpack
}  // namespace tflite
