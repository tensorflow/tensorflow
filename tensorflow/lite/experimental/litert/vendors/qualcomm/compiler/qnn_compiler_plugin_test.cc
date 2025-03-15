// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstddef>
#include <cstdint>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"
#include "tensorflow/lite/experimental/litert/test/test_models.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/litert_compiler_plugin.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR/qnn_op.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/quantize_op_legalization.h"

namespace litert {
namespace {

using ::testing::Values;

// clang-format off
// TODO: Add support and uncomment these models.
const auto kSupportedOps =
                  Values(
                    "rms_norm_composite.tflite",
                    "simple_add_op.tflite",
                    "simple_div_op.tflite",
                    "simple_mul_op.tflite",
                    "simple_rsqrt_op.tflite",
                    "simple_slice_op.tflite",
                    "simple_sub_op.tflite",
                    "simple_sum_op.tflite",
                    "simple_tanh_op.tflite",
                    "simple_reshape_op.tflite",
                    "simple_batch_matmul_op.tflite",
                    "rms_norm.tflite",
                    "simple_concatenation_op.tflite",
                    "simple_softmax_op.tflite",
                    "simple_cast_op.tflite",
                    "simple_transpose_op.tflite",
                    "simple_sin_op.tflite",
                    "simple_cos_op.tflite",
                    "simple_select_op.tflite",
                    "simple_select_v2_op.tflite",
                    "simple_fully_connected_op.tflite",
                    "fully_connected_3d.tflite",
                    "simple_embedding_lookup_op.tflite",
                    "simple_logical_and_op.tflite",
                    "simple_less_op.tflite",
                    "simple_greater_op.tflite",
                    "simple_gelu_op.tflite",
                    "simple_dynamic_update_slice_op.tflite",
                    "simple_pack_op.tflite",
                    "simple_gather_op.tflite",
                    "simple_mean_op.tflite",
                    "simple_split_op.tflite",
                    "simple_average_poll_2d.tflite",
                    "simple_conv_2d_op.tflite",
                    "simple_depth_to_space_op.tflite",
                    "simple_depthwise_conv_2d_op.tflite",
                    "simple_hard_swish_op.tflite",
                    "simple_leaky_relu_op.tflite",
                    "simple_resize_bilinear_op.tflite",
                    "simple_space_to_depth_op.tflite",
                    kFeedForwardModel,
                    kKeyEinsumModel,
                    kQueryEinsumModel,
                    kValueEinsumModel,
                    kAttnVecEinsumModel,
                    kROPEModel,
                    kLookUpROPEModel,
                    kRMSNormModel,
                    kSDPAModel,
                    kAttentionModel,
                    kTransformerBlockModel
                    // kQSimpleMul16x16Model,
                    // kQMulAdd16x16Model,
                    // kQQueryEinsum16x8Model,
                    // kQKeyEinsum16x8Model,
                    // kQVauleEinsum16x8Model,
                    // kQAttnVecEinsum16x8Model
                    );

const auto kSupportedSocModels = Values(
    "V68",
    "V69",
    "V73",
    "V75",
    "V79"
);
// clang-format on

TEST(TestQnnPlugin, GetConfigInfo) {
  EXPECT_STREQ(LiteRtGetCompilerPluginSocManufacturer(), "Qualcomm");

  auto plugin = CreatePlugin();

  LiteRtParamIndex num_supported_soc_models;
  LITERT_ASSERT_OK(LiteRtGetNumCompilerPluginSupportedSocModels(
      plugin.get(), &num_supported_soc_models));
  ASSERT_EQ(num_supported_soc_models, 5);

  const char* config_id;
  LITERT_ASSERT_OK(
      LiteRtGetCompilerPluginSupportedSocModel(plugin.get(), 0, &config_id));
  EXPECT_STREQ(config_id, "V68");
}

TEST(TestQnnPlugin, PartitionMulOps) {
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel("one_mul.tflite");

  LiteRtOpListT selected_op_list;
  LITERT_ASSERT_OK(LiteRtCompilerPluginPartition(
      plugin.get(), /*soc_model=*/nullptr, model.Subgraph(0)->Get(),
      &selected_op_list));
  const auto selected_ops = selected_op_list.Values();

  ASSERT_EQ(selected_ops.size(), 1);
  EXPECT_EQ(selected_ops[0].first->OpCode(), kLiteRtOpCodeTflMul);
}

TEST(TestQnnPlugin, CompileMulSubgraph) {
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel("one_mul.tflite");

  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(
      LiteRtCompilerPluginCompile(plugin.get(), "V75", model.Get(), &compiled));

  const void* byte_code;
  size_t byte_code_size;

  LITERT_ASSERT_OK(LiteRtGetCompiledResultByteCode(
      compiled, /*byte_code_idx=*/0, &byte_code, &byte_code_size));

  absl::string_view byte_code_string(reinterpret_cast<const char*>(byte_code),
                                     byte_code_size);
  ASSERT_FALSE(byte_code_string.empty());

  const void* op_data;
  size_t op_data_size;
  LiteRtParamIndex byte_code_idx;

  LITERT_ASSERT_OK(LiteRtGetCompiledResultCallInfo(
      compiled, /*call_idx=*/0, &op_data, &op_data_size, &byte_code_idx));

  absl::string_view op_data_string(reinterpret_cast<const char*>(op_data),
                                   op_data_size);
  ASSERT_EQ("qnn_partition_0", op_data_string);

  LiteRtDestroyCompiledResult(compiled);
}

TEST(TestQnnPlugin, ShareContextBinary) {
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel("cst_multi_subgraph.tflite");

  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(
      LiteRtCompilerPluginCompile(plugin.get(), "V75", model.Get(), &compiled));
  uint64_t num_byte_code;
  LITERT_ASSERT_OK(
      LiteRtCompiledResultNumByteCodeModules(compiled, &num_byte_code));
  ASSERT_EQ(num_byte_code, 1);

  LiteRtDestroyCompiledResult(compiled);
}

TEST(TestQnnPlugin, NotShareContextBinary) {
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel("multi_subgraph.tflite");

  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(
      LiteRtCompilerPluginCompile(plugin.get(), "V75", model.Get(), &compiled));
  uint64_t num_byte_code;
  LITERT_ASSERT_OK(
      LiteRtCompiledResultNumByteCodeModules(compiled, &num_byte_code));
  ASSERT_EQ(num_byte_code, 3);

  LiteRtDestroyCompiledResult(compiled);
}

TEST(TestLegalization, QuantizeOpLegalizedToCastOp) {
  static constexpr absl::string_view kQnnOpName = "Cast";
  static constexpr int kSUFixed8OffsetDiff = 128;
  const auto input_quantization_params = MakePerTensorQuantization(
      /*scale=*/1.0f, /*zero_point=*/0);
  const auto output_quantization_params = MakePerTensorQuantization(
      /*scale=*/1.0f, /*zero_point=*/kSUFixed8OffsetDiff);
  LiteRtOpT quantize_op;
  LiteRtTensorT input_tensor;
  LiteRtTensorT output_tensor;
  // Set quantization params, tensor type for input and output tensors.
  input_tensor.SetQarams(input_quantization_params);
  TensorType input_tensor_type =
      MakeRankedTensorType(kLiteRtElementTypeInt8, {1, 1});
  input_tensor.SetType(input_tensor_type);
  output_tensor.SetQarams(output_quantization_params);
  TensorType output_tensor_type =
      MakeRankedTensorType(kLiteRtElementTypeUInt8, {1, 1});
  output_tensor.SetType(output_tensor_type);
  quantize_op.Inputs().push_back(&input_tensor);
  quantize_op.Outputs().push_back(&output_tensor);
  quantize_op.SetOpCode(kLiteRtOpCodeTflQuantize);

  qnn::QuantizeOpLegalization legalization;
  Qnn_OpConfig_t legalized_qnn_op = qnn::BuildDefaultOp();
  litert::Op litert_quantize_op(&quantize_op);
  LITERT_ASSERT_OK(
      legalization.ConfigureQnnOp(litert_quantize_op, legalized_qnn_op));
  absl::string_view qnn_op_name(legalized_qnn_op.v1.typeName);
  EXPECT_EQ(qnn_op_name, kQnnOpName);
}

TEST(TestLegalization, QuantizeOpLegalizedToConvertOp) {
  static constexpr absl::string_view kQnnOpName = "Convert";
  static constexpr int kSUFixed8OffsetDiff = 0;
  const auto input_quantization_params = MakePerTensorQuantization(
      /*scale=*/1.0f, /*zero_point=*/0);
  const auto output_quantization_params = MakePerTensorQuantization(
      /*scale=*/1.0f, /*zero_point=*/kSUFixed8OffsetDiff);
  LiteRtOpT quantize_op;
  LiteRtTensorT input_tensor;
  LiteRtTensorT output_tensor;
  // Set quantization params, tensor type for input and output tensors.
  input_tensor.SetQarams(input_quantization_params);
  TensorType input_tensor_type =
      MakeRankedTensorType(kLiteRtElementTypeInt8, {1, 1});
  input_tensor.SetType(input_tensor_type);
  output_tensor.SetQarams(output_quantization_params);
  TensorType output_tensor_type =
      MakeRankedTensorType(kLiteRtElementTypeUInt8, {1, 1});
  output_tensor.SetType(output_tensor_type);
  quantize_op.Inputs().push_back(&input_tensor);
  quantize_op.Outputs().push_back(&output_tensor);
  quantize_op.SetOpCode(kLiteRtOpCodeTflQuantize);

  qnn::QuantizeOpLegalization legalization;
  Qnn_OpConfig_t legalized_qnn_op = qnn::BuildDefaultOp();
  litert::Op litert_quantize_op(&quantize_op);
  LITERT_ASSERT_OK(
      legalization.ConfigureQnnOp(litert_quantize_op, legalized_qnn_op));
  absl::string_view qnn_op_name(legalized_qnn_op.v1.typeName);
  EXPECT_EQ(qnn_op_name, kQnnOpName);
}

TEST(TestLegalization, QuantizeOpLegalizedToQuantizeOp) {
  static constexpr absl::string_view kQnnOpName = "Quantize";
  const auto output_quantization_params = MakePerTensorQuantization(
      /*scale=*/1.0f, /*zero_point=*/0);
  LiteRtOpT quantize_op;
  LiteRtTensorT input_tensor;
  LiteRtTensorT output_tensor;
  // Set quantization params, tensor type for input and output tensors.
  TensorType input_tensor_type =
      MakeRankedTensorType(kLiteRtElementTypeFloat32, {1, 1});
  input_tensor.SetType(input_tensor_type);
  output_tensor.SetQarams(output_quantization_params);
  TensorType output_tensor_type =
      MakeRankedTensorType(kLiteRtElementTypeInt16, {1, 1});
  output_tensor.SetType(output_tensor_type);
  quantize_op.Inputs().push_back(&input_tensor);
  quantize_op.Outputs().push_back(&output_tensor);
  quantize_op.SetOpCode(kLiteRtOpCodeTflQuantize);

  qnn::QuantizeOpLegalization legalization;
  Qnn_OpConfig_t legalized_qnn_op = qnn::BuildDefaultOp();
  litert::Op litert_quantize_op(&quantize_op);
  LITERT_ASSERT_OK(
      legalization.ConfigureQnnOp(litert_quantize_op, legalized_qnn_op));
  absl::string_view qnn_op_name(legalized_qnn_op.v1.typeName);
  EXPECT_EQ(qnn_op_name, kQnnOpName);
}

class QnnPlyginSupportedSocCompilationTest
    : public ::testing::TestWithParam<std::string> {};

TEST_P(QnnPlyginSupportedSocCompilationTest, CompileMulSubgraph) {
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel("one_mul.tflite");
  auto soc_model = GetParam();

  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(LiteRtCompilerPluginCompile(plugin.get(), soc_model.c_str(),
                                               model.Get(), &compiled));

  const void* byte_code;
  size_t byte_code_size;

  LITERT_ASSERT_OK(LiteRtGetCompiledResultByteCode(
      compiled, /*byte_code_idx=*/0, &byte_code, &byte_code_size));

  absl::string_view byte_code_string(reinterpret_cast<const char*>(byte_code),
                                     byte_code_size);
  ASSERT_FALSE(byte_code_string.empty());

  const void* op_data;
  size_t op_data_size;
  LiteRtParamIndex byte_code_idx;

  LITERT_ASSERT_OK(LiteRtGetCompiledResultCallInfo(
      compiled, /*call_idx=*/0, &op_data, &op_data_size, &byte_code_idx));

  absl::string_view op_data_string(reinterpret_cast<const char*>(op_data),
                                   op_data_size);
  ASSERT_EQ("qnn_partition_0", op_data_string);

  LiteRtDestroyCompiledResult(compiled);
}

INSTANTIATE_TEST_SUITE_P(SupportedOpsTest, QnnPlyginSupportedSocCompilationTest,
                         kSupportedSocModels);

class QnnPluginOpValidationTest : public ::testing::TestWithParam<std::string> {
};

TEST_P(QnnPluginOpValidationTest, SupportedOpsTest) {
  LITERT_LOG(LITERT_INFO, "Validating TFLite model: %s", GetParam().c_str());
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel(GetParam());

  const auto subgraph = model.MainSubgraph();
  LiteRtSubgraph litert_subgraph = subgraph->Get();

  LiteRtOpListT selected_ops;
  LITERT_ASSERT_OK(LiteRtCompilerPluginPartition(
      plugin.get(), /*soc_model=*/nullptr, litert_subgraph, &selected_ops));

  EXPECT_EQ(selected_ops.Values().size(), litert_subgraph->Ops().size());
}

INSTANTIATE_TEST_SUITE_P(SupportedOpsTest, QnnPluginOpValidationTest,
                         kSupportedOps);

class QnnPluginOpCompatibilityTest
    : public ::testing::TestWithParam<std::string> {};

TEST_P(QnnPluginOpCompatibilityTest, SupportedOpsTest) {
  LITERT_LOG(LITERT_INFO, "Testing TFLite model: %s", GetParam().c_str());
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel(GetParam());

  LiteRtCompiledResult compiled;
  LITERT_ASSERT_OK(
      LiteRtCompilerPluginCompile(plugin.get(), "V75", model.Get(), &compiled));

  const void* byte_code;
  size_t byte_code_size;

  LITERT_ASSERT_OK(LiteRtGetCompiledResultByteCode(
      compiled, /*byte_code_idx=*/0, &byte_code, &byte_code_size));

  absl::string_view byte_code_string(reinterpret_cast<const char*>(byte_code),
                                     byte_code_size);
  ASSERT_FALSE(byte_code_string.empty());

  const void* op_data;
  size_t op_data_size;
  LiteRtParamIndex byte_code_idx;

  LITERT_ASSERT_OK(LiteRtGetCompiledResultCallInfo(
      compiled, /*call_idx=*/0, &op_data, &op_data_size, &byte_code_idx));

  absl::string_view op_data_string(reinterpret_cast<const char*>(op_data),
                                   op_data_size);
  ASSERT_EQ("qnn_partition_0", op_data_string);

  LiteRtDestroyCompiledResult(compiled);
}

INSTANTIATE_TEST_SUITE_P(SupportedOpsTest, QnnPluginOpCompatibilityTest,
                         kSupportedOps);

}  // namespace
}  // namespace litert
