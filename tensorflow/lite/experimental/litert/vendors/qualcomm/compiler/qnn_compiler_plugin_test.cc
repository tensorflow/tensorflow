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
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/test_macros.h"
#include "tensorflow/lite/experimental/litert/test/test_models.h"
#include "tensorflow/lite/experimental/litert/vendors/c/litert_compiler_plugin.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/litert_compiler_plugin.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/graph_mapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/qnn_manager.h"

namespace litert {
namespace {

using ::testing::Values;

// clang-format off
const auto kSupportedOps =
                  Values(
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
                    kTransformerBlockModel,
                    kQSimpleMul16x16Model,
                    kQMulAdd16x16Model,
                    kQQueryEinsum16x8Model,
                    kQKeyEinsum16x8Model,
                    kQVauleEinsum16x8Model,
                    kQAttnVecEinsum16x8Model
                    );
// clang-format on

TEST(TestQnnPlugin, GetConfigInfo) {
  EXPECT_STREQ(LiteRtGetCompilerPluginSocManufacturer(), "Qualcomm");

  auto plugin = CreatePlugin();

  LiteRtParamIndex num_supported_soc_models;
  LITERT_ASSERT_STATUS_OK(LiteRtGetNumCompilerPluginSupportedSocModels(
      plugin.get(), &num_supported_soc_models));
  ASSERT_EQ(num_supported_soc_models, 5);

  const char* config_id;
  LITERT_CHECK_STATUS_OK(
      LiteRtGetCompilerPluginSupportedSocModel(plugin.get(), 0, &config_id));
  EXPECT_STREQ(config_id, "V68");
}

TEST(TestQnnPlugin, PartitionMulOps) {
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel("one_mul.tflite");

  LiteRtOpListT selected_op_list;
  LITERT_ASSERT_STATUS_OK(LiteRtCompilerPluginPartition(
      plugin.get(), model.Subgraph(0)->Get(), &selected_op_list));
  const auto selected_ops = selected_op_list.Vec();

  ASSERT_EQ(selected_ops.size(), 1);
  EXPECT_EQ(selected_ops[0]->OpCode(), kLiteRtOpCodeTflMul);
}

TEST(TestQnnPlugin, CompileMulSubgraph) {
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel("one_mul.tflite");

  const auto subgraph = model.MainSubgraph();
  LiteRtSubgraph litert_subgraph = subgraph->Get();

  LiteRtCompiledResult compiled;
  LITERT_ASSERT_STATUS_OK(LiteRtCompilerPluginCompile(
      plugin.get(), "V75", &litert_subgraph, 1, &compiled));

  const void* byte_code;
  size_t byte_code_size;

  LITERT_ASSERT_STATUS_OK(
      LiteRtGetCompiledResultByteCode(compiled, &byte_code, &byte_code_size));

  absl::string_view byte_code_string(reinterpret_cast<const char*>(byte_code),
                                     byte_code_size);
  ASSERT_FALSE(byte_code_string.empty());

  const void* op_data;
  size_t op_data_size;

  LITERT_ASSERT_STATUS_OK(
      LiteRtGetCompiledResultCallInfo(compiled, 0, &op_data, &op_data_size));

  absl::string_view op_data_string(reinterpret_cast<const char*>(op_data),
                                   op_data_size);
  ASSERT_EQ("qnn_partition_0", op_data_string);

  LiteRtDestroyCompiledResult(compiled);
}

TEST(TestLargeQint16, CompileTest) {
  float dummy_scale = 1;
  int dummy_offset = 0;
  std::vector<uint32_t> in_0_dims = {512, 2304};
  std::vector<uint32_t> in_1_dims = {1};
  std::vector<uint32_t> out_0_dims = in_0_dims;

  LITERT_STACK_ARRAY(Qnn_Tensor_t, qnn_op_ins, 2, QNN_TENSOR_INIT);
  Qnn_QuantizeParams_t q_param = QNN_QUANTIZE_PARAMS_INIT;
  q_param.encodingDefinition = QNN_DEFINITION_DEFINED;
  q_param.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
  q_param.scaleOffsetEncoding.scale = dummy_scale;
  q_param.scaleOffsetEncoding.offset = dummy_offset;

  qnn_op_ins[0].v2.name = "0";
  qnn_op_ins[0].v2.type = QNN_TENSOR_TYPE_APP_WRITE;
  qnn_op_ins[0].v2.dataType = QNN_DATATYPE_SFIXED_POINT_16;
  qnn_op_ins[0].v2.rank = in_0_dims.size();
  qnn_op_ins[0].v2.dimensions = in_0_dims.data();
  qnn_op_ins[0].v2.quantizeParams = q_param;

  qnn_op_ins[1].v2.name = "1";
  qnn_op_ins[1].v2.type = QNN_TENSOR_TYPE_APP_WRITE;
  qnn_op_ins[1].v2.dataType = QNN_DATATYPE_SFIXED_POINT_16;
  qnn_op_ins[1].v2.rank = in_1_dims.size();
  qnn_op_ins[1].v2.dimensions = in_1_dims.data();
  qnn_op_ins[1].v2.quantizeParams = q_param;

  LITERT_STACK_ARRAY(Qnn_Tensor_t, qnn_op_outs, 1, QNN_TENSOR_INIT);

  qnn_op_outs[0].v2.name = "2";
  qnn_op_outs[0].v2.type = QNN_TENSOR_TYPE_APP_READ;
  qnn_op_outs[0].v2.dataType = QNN_DATATYPE_SFIXED_POINT_16;
  qnn_op_outs[0].v2.rank = out_0_dims.size();
  qnn_op_outs[0].v2.dimensions = out_0_dims.data();
  qnn_op_outs[0].v2.quantizeParams = q_param;

  Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
  op.v1.name = "mul";
  op.v1.packageName = "qti.aisw";
  op.v1.typeName = "ElementWiseMultiply";
  op.v1.numOfInputs = 2;
  op.v1.inputTensors = qnn_op_ins;
  op.v1.numOfOutputs = 1;
  op.v1.outputTensors = qnn_op_outs;

  LITERT_LOG(LITERT_INFO, "%s", "Creating QNN manager");
  auto backend_configs = qnn::QnnManager::DefaultBackendConfigs();
  auto qnn_manager = qnn::QnnManager::Create(
      backend_configs, /*shared_library_dir=*/std::nullopt,
      QNN_HTP_DEVICE_ARCH_V75);
  ASSERT_TRUE(qnn_manager);

  // Initialize context.

  LITERT_LOG(LITERT_INFO, "%s", "Creating context handle");
  auto context_configs = qnn::QnnManager::DefaultContextConfigs();
  auto context_handle = (*qnn_manager)->CreateContextHandle(context_configs);
  ASSERT_TRUE(context_handle);

  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel("one_mul.tflite");

  const auto subgraph = model.MainSubgraph();
  LiteRtSubgraph dummy_subgraph = subgraph->Get();
  qnn::GraphMapper graph_mapper(dummy_subgraph, (**qnn_manager),
                                context_handle->get());
  LITERT_ASSERT_STATUS_OK(graph_mapper.InitQnnGraph("test"));

  (*qnn_manager)
      ->Api()
      ->tensorCreateGraphTensor(graph_mapper.QnnGraph(), &qnn_op_ins[0]);
  (*qnn_manager)
      ->Api()
      ->tensorCreateGraphTensor(graph_mapper.QnnGraph(), &qnn_op_ins[1]);
  (*qnn_manager)
      ->Api()
      ->tensorCreateGraphTensor(graph_mapper.QnnGraph(), &qnn_op_outs[0]);
  (*qnn_manager)->Api()->graphAddNode(graph_mapper.QnnGraph(), op);

  LITERT_ASSERT_STATUS_OK(graph_mapper.Finalize());
}

class QnnPluginOpCompatibilityTest
    : public ::testing::TestWithParam<std::string> {};

TEST_P(QnnPluginOpCompatibilityTest, SupportedOpsTest) {
  LITERT_LOG(LITERT_INFO, "Testing TFLite model: %s", GetParam().c_str());
  auto plugin = CreatePlugin();
  auto model = testing::LoadTestFileModel(GetParam());

  const auto subgraph = model.MainSubgraph();
  LiteRtSubgraph litert_subgraph = subgraph->Get();

  LiteRtCompiledResult compiled;
  LITERT_ASSERT_STATUS_OK(LiteRtCompilerPluginCompile(
      plugin.get(), "V75", &litert_subgraph, 1, &compiled));

  const void* byte_code;
  size_t byte_code_size;

  LITERT_ASSERT_STATUS_OK(
      LiteRtGetCompiledResultByteCode(compiled, &byte_code, &byte_code_size));

  absl::string_view byte_code_string(reinterpret_cast<const char*>(byte_code),
                                     byte_code_size);
  ASSERT_FALSE(byte_code_string.empty());

  const void* op_data;
  size_t op_data_size;

  LITERT_ASSERT_STATUS_OK(
      LiteRtGetCompiledResultCallInfo(compiled, 0, &op_data, &op_data_size));

  absl::string_view op_data_string(reinterpret_cast<const char*>(op_data),
                                   op_data_size);
  ASSERT_EQ("qnn_partition_0", op_data_string);

  LiteRtDestroyCompiledResult(compiled);
}

INSTANTIATE_TEST_SUITE_P(SupportedOpsTest, QnnPluginOpCompatibilityTest,
                         kSupportedOps);

}  // namespace
}  // namespace litert
