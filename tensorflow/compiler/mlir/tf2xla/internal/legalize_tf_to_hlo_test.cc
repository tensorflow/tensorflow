/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tf2xla/internal/legalize_tf_to_hlo.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tf2xla/internal/test_matchers.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/client/client_library.h"
#include "xla/shape.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {

using ::tensorflow::monitoring::testing::CellReader;
using tpu::MlirToHloArgs;
using tpu::ShardingAndIndex;
using tpu::TPUCompileMetadataProto;

static constexpr char kMlirLegalizeCount[] =
    "/tensorflow/core/tf2xla/v1/mlir_failed_xla_legalize_tf_count";
static constexpr char kMlirLegalizeErrors[] =
    "/tensorflow/core/tf2xla/v1/mlir_failed_xla_legalize_tf_pass_count";
static constexpr char kBridgeStatusCounter[] =
    "/tensorflow/core/tf2xla/api/v2/phase2_compilation_status";
constexpr char kMlirCombinedMlirSuccess[] = "kMlirCombinedMlirSuccess";
constexpr char kMlirCombinedOldSuccess[] = "kMlirCombinedOldSuccess";
constexpr char kMlirCombinedOldFailure[] = "kMlirCombinedOldFailure";

static constexpr char kMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  func.func @main(%arg0 : tensor<1xf32>) -> tensor<1xf32> {
    %0 = "tf.Acos"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
   func.return %0 : tensor<1xf32>
  }
})";

static constexpr char kBadMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main() -> tensor<1xi32> {
      %0 = "tf.DoesntExist"() {value = dense<1000> : tensor<1xi32>} : () -> tensor<1xi32>
      func.return %0 : tensor<1xi32>
    }
  })";

absl::StatusOr<XlaCompiler::CompilationResult> CompileMlirModule(
    const char* module_str) {
  MlirToHloArgs mlir_to_hlo_args;
  mlir_to_hlo_args.rollout_state =
      ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_UNSPECIFIED;
  mlir_to_hlo_args.mlir_module = module_str;

  se::Platform* platform =
      se::PlatformManager::PlatformWithName("Host").value();
  auto client =
      xla::ClientLibrary::GetOrCreateCompileOnlyClient(platform).value();

  std::vector<TensorShape> arg_shapes = {{1}};
  TPUCompileMetadataProto metadata_proto;
  auto arg = metadata_proto.add_args();
  arg->set_dtype(DataType::DT_FLOAT);
  arg->set_kind(TPUCompileMetadataProto::Arg::PARAMETER);
  metadata_proto.add_retvals();
  bool use_tuple_args = true;
  std::vector<ShardingAndIndex> arg_core_mapping;
  std::vector<std::vector<xla::Shape>> per_core_arg_shapes;
  std::vector<std::unique_ptr<mlir::Pass>> custom_legalization_passes;
  auto compilation_result = std::make_unique<XlaCompilationResult>();

  return LegalizeTfToHlo(mlir_to_hlo_args, metadata_proto, use_tuple_args,
                         /*device_type=*/"XLA_TPU_JIT",
                         /*shape_determination_fns=*/{}, arg_shapes,
                         &arg_core_mapping, &per_core_arg_shapes,
                         custom_legalization_passes, client,
                         compilation_result.get());
}

TEST(LegalizeWithCombinedBridge, DoesNotUseMlirLowering) {
  CellReader<int64_t> mlir_bridge_legalize_count(kMlirLegalizeCount);
  CellReader<int64_t> counts(kBridgeStatusCounter);

  auto result = CompileMlirModule(kMlirModuleStr);

  ASSERT_THAT(result, IsOkOrFiltered());
  EXPECT_EQ(mlir_bridge_legalize_count.Delta("tf.Acos"), 0);
  EXPECT_THAT(result,
              IncrementedOrFiltered(counts.Delta(kMlirCombinedMlirSuccess), 1));
  EXPECT_THAT(result,
              IncrementedOrFiltered(counts.Delta(kMlirCombinedOldSuccess), 1));
}

TEST(LegalizeWithCombinedBridge,
     CorrectlyCountsMlirBridgePassingAndGraphBridgeFailing) {
  CellReader<int64_t> legalize_failure_count(kMlirLegalizeErrors);
  CellReader<int64_t> counts(kBridgeStatusCounter);

  auto result = CompileMlirModule(kBadMlirModuleStr);

  ASSERT_FALSE(result.ok());
  // Never failed to legalize because it was never attempted
  EXPECT_EQ(legalize_failure_count.Read("tf.DoesntExist", "Unknown"), 0);
  EXPECT_THAT(result,
              IncrementedOrFiltered(counts.Delta(kMlirCombinedMlirSuccess), 1));
  EXPECT_THAT(result,
              IncrementedOrFiltered(counts.Delta(kMlirCombinedOldFailure), 1));
}

TEST(LegalizeWithCombinedBridge, RecordsDynamicOps) {
  static constexpr char kDynamismFunctionCounterStreamzName[] =
      "/tensorflow/core/tf2xla/api/v2/dynamism_function_counter";
  constexpr char kNotDynamicFunctionName[] = "kNotDynamicFunction";
  CellReader<int64_t> dynamic_function_op_count(
      kDynamismFunctionCounterStreamzName);

  auto result = CompileMlirModule(kMlirModuleStr);

  ASSERT_TRUE(result.ok());
  EXPECT_EQ(dynamic_function_op_count.Delta(kNotDynamicFunctionName), 1);
}

};  // namespace internal
};  // namespace tf2xla
};  // namespace tensorflow
