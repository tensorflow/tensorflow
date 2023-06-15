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

#include "tensorflow/compiler/mlir/tf2xla/api/v1/legalize_tf.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/compiler/mlir/tf2xla/api/v1/device_type.pb.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tensorflow/tsl/lib/monitoring/test_utils.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace tensorflow {
namespace tf2xla {
namespace v1 {

using ::tensorflow::monitoring::testing::CellReader;
using tpu::FunctionToHloArgs;
using tpu::MlirToHloArgs;
using tpu::ShardingAndIndex;
using tpu::TPUCompileMetadataProto;
using ::tsl::monitoring::testing::Histogram;

static constexpr char kCompilationTimeStreamzName[] =
    "/tensorflow/core/tf2xla/api/v1/phase2_compilation_time";
static constexpr char kCompilationStatusStreamzName[] =
    "/tensorflow/core/tf2xla/api/v1/phase2_compilation_status";

static constexpr char kMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  func.func @main() -> () {
    func.return
  }
})";

tsl::StatusOr<XlaCompiler::CompilationResult> CompileMlirModule(
    ConfigProto::Experimental::MlirBridgeRollout rollout_state) {
  MlirToHloArgs mlir_to_hlo_args;
  mlir_to_hlo_args.rollout_state = rollout_state;
  mlir_to_hlo_args.mlir_module = kMlirModuleStr;

  se::Platform* platform =
      se::MultiPlatformManager::PlatformWithName("Host").value();
  auto client =
      xla::ClientLibrary::GetOrCreateCompileOnlyClient(platform).value();

  std::vector<TensorShape> arg_shapes;
  TPUCompileMetadataProto metadata_proto;
  bool use_tuple_args = true;
  std::vector<ShardingAndIndex> arg_core_mapping;
  std::vector<std::vector<xla::Shape>> per_core_arg_shapes;
  std::vector<std::unique_ptr<mlir::Pass>> custom_legalization_passes;

  return LegalizeMlirToHlo(mlir_to_hlo_args, metadata_proto, use_tuple_args,
                           /*device_type=*/"XLA_TPU_JIT",
                           custom_legalization_passes,
                           /*shape_determination_fns=*/{}, arg_shapes,
                           &arg_core_mapping, &per_core_arg_shapes, client);
}

TEST(LegalizeTFTest, RecordsStreamzForMlirBridge) {
  CellReader<Histogram> compilation_time(kCompilationTimeStreamzName);
  CellReader<int64_t> compilation_status(kCompilationStatusStreamzName);

  TF_ASSERT_OK_AND_ASSIGN(
      XlaCompiler::CompilationResult result,
      CompileMlirModule(
          ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_ENABLED));

  Histogram histogram =
      compilation_time.Delta("mlir_bridge_op_fallback_disabled");
  EXPECT_EQ(histogram.num(), 1);
  EXPECT_EQ(compilation_status.Delta("kMlirModeSuccess"), 1);
}

TEST(LegalizeTFTest, RecordsStreamzForMlirOpFallback) {
  CellReader<Histogram> compilation_time(kCompilationTimeStreamzName);

  TF_ASSERT_OK_AND_ASSIGN(
      XlaCompiler::CompilationResult result,
      CompileMlirModule(
          ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_UNSPECIFIED));

  Histogram histogram =
      compilation_time.Delta("mlir_bridge_op_fallback_enabled");
  EXPECT_EQ(histogram.num(), 1);
}

TEST(LegalizeTFTest, RecordsStreamzForNoMlirFallback) {
  FunctionDef my_func =
      tensorflow::FunctionDefHelper::Create("empty", {}, {}, {}, {}, {});

  tensorflow::FunctionDefLibrary fdef;
  *(fdef.add_function()) = my_func;
  tensorflow::FunctionLibraryDefinition flib_def(
      tensorflow::OpRegistry::Global(), fdef);

  OpInputList guaranteed_constants;
  NameAttrList function;
  FunctionToHloArgs function_to_hlo_args{&function,
                                         &flib_def,
                                         /*graph_def_version=*/0,
                                         {&guaranteed_constants}};

  se::Platform* cpu_platform =
      se::MultiPlatformManager::PlatformWithName("Host").value();
  auto client =
      xla::ClientLibrary::GetOrCreateCompileOnlyClient(cpu_platform).value();

  std::vector<TensorShape> arg_shapes;
  TPUCompileMetadataProto metadata_proto;
  bool use_tuple_args = true;
  std::vector<ShardingAndIndex> arg_core_mapping;
  std::vector<std::vector<xla::Shape>> per_core_arg_shapes;
  std::vector<std::unique_ptr<mlir::Pass>> custom_legalization_passes;

  // This doesn't actually compile correctly.
  tsl::StatusOr<XlaCompiler::CompilationResult> compile_result =
      LegalizeMlirToHlo(function_to_hlo_args, metadata_proto, use_tuple_args,
                        /*device_type=*/"XLA_CPU_JIT",
                        custom_legalization_passes,
                        /*shape_determination_fns=*/{}, arg_shapes,
                        &arg_core_mapping, &per_core_arg_shapes, client);

  EXPECT_FALSE(compile_result.ok());
}

}  // namespace v1
}  // namespace tf2xla
}  // namespace tensorflow
