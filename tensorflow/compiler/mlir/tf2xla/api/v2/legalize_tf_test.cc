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

#include "tensorflow/compiler/mlir/tf2xla/api/v2/legalize_tf.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tf2xla/api/v2/testing/compile_mlir.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/test_matchers.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "xla/client/client_library.h"
#include "xla/shape.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/lib/monitoring/test_utils.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"
#include "tensorflow/core/lib/monitoring/test_utils.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tensorflow/core/util/debug_data_dumper.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace tensorflow {
namespace tf2xla {
namespace v2 {

using ::tensorflow::monitoring::testing::CellReader;
using tensorflow::tf2xla::v2::testing::CompileMlirModule;
using ::testing::Not;
using ::testing::TestWithParam;
using tpu::FunctionToHloArgs;
using tpu::ShardingAndIndex;
using tpu::TPUCompileMetadataProto;
using tsl::testing::TmpDir;

static constexpr char kCompilationTimeStreamzName[] =
    "/tensorflow/core/tf2xla/api/v2/phase2_compilation_time";
static constexpr char kFullBridge[] = "full_bridge";
static constexpr char kCompilationStatusStreamzName[] =
    "/tensorflow/core/tf2xla/api/v2/phase2_compilation_status";
static const char kMlirWithFallbackModeSuccess[] =
    "kMlirWithFallbackModeSuccess";
static const char kMlirWithFallbackModeFailure[] =
    "kMlirWithFallbackModeFailure";
static const char kOldBridgeMlirFilteredFailure[] =
    "kOldBridgeMlirFilteredFailure";
static const char kOldBridgeWithFallbackModeFailure[] =
    "kOldBridgeWithFallbackModeFailure";
static const char kOldBridgeMlirFilteredSuccess[] =
    "kOldBridgeMlirFilteredSuccess";
static const char kOldBridgeWithFallbackModeSuccess[] =
    "kOldBridgeWithFallbackModeSuccess";
static const char kMlirCombinedMlirSuccess[] = "kMlirCombinedMlirSuccess";
static const char kMlirCombinedMlirFailure[] = "kMlirCombinedMlirFailure";
static const char kMlirCombinedOldSuccess[] = "kMlirCombinedOldSuccess";
static const char kMlirCombinedOldFailure[] = "kMlirCombinedOldFailure";

static constexpr char kMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  func.func @main() -> () {
    func.return
  }
})";

// MLIR which should not legalize at all
static constexpr char kBadMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  func.func @main() -> () {
    %0 = tf.Unknown() -> ()
    func.return %0
  }
})";

// MLIR which should be filtered by the MLIR bridge but fully legalize with the
// combined bridge.
static constexpr char kUnsupportedMlirBridgeModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  func.func @main() -> () {
    %cst0 = "tf.Const"(){ value = dense<0> : tensor<3x5xi1>} : () -> tensor<3x5xi1>
    %0 = "tf.Where"(%cst0) : (tensor<3x5xi1>) -> tensor<?x2xi64>
    func.return
  }
})";

TEST(LegalizeTFTest, RecordsStreamzForSuccessfulLegalizeWithMlirBridge) {
  CellReader<int64_t> compilation_status(kCompilationStatusStreamzName);

  TF_ASSERT_OK_AND_ASSIGN(
      XlaCompiler::CompilationResult result,
      CompileMlirModule(
          kMlirModuleStr,
          ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_UNSPECIFIED));

  // May have been filtered so check for lack of failure instead of success.
  EXPECT_EQ(compilation_status.Delta(kMlirWithFallbackModeFailure), 0);
}

TEST(LegalizeTFTest, MatMul) {
  static constexpr char kMatMulModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main() -> (tensor<5x11xf32>) {
      %arg0 = "tf.Const"() {value = dense<-3.0> : tensor<5x7xf32>} : () -> tensor<5x7xf32>
      %arg1 = "tf.Const"() {value = dense<-3.0> : tensor<11x7xf32>} : () -> tensor<11x7xf32>

      %1 = "tf.MatMul"(%arg0, %arg1) {transpose_a = false, transpose_b = true} : (tensor<5x7xf32>, tensor<11x7xf32>) -> tensor<5x11xf32>

      func.return %1 : tensor<5x11xf32>
    }
  })";
  TF_ASSERT_OK_AND_ASSIGN(
      XlaCompiler::CompilationResult result,
      CompileMlirModule(
          kMatMulModuleStr,
          ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_UNSPECIFIED));
}

struct MatMulTestCase {
  std::string mat_mul_method;
};

using BatchMatMulTest = TestWithParam<MatMulTestCase>;

TEST_P(BatchMatMulTest, BatchMatMul) {
  const MatMulTestCase& test_case = GetParam();
  static constexpr char kMatMulModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main() -> (tensor<1x4x4xf32>) {
      %%arg0 = "tf.Const"() {value = dense<-3.0> : tensor<1x4x2xf32>} : () -> tensor<1x4x2xf32>
      %%arg1 = "tf.Const"() {value = dense<-3.0> : tensor<1x2x4xf32>} : () -> tensor<1x2x4xf32>

      %%1 = "tf.%s"(%%arg0, %%arg1) {T = f32, adj_x = false, adj_y = false, grad_x = false, grad_y = false, device = ""} : (tensor<1x4x2xf32>, tensor<1x2x4xf32>) -> tensor<1x4x4xf32>

      func.return %%1 : tensor<1x4x4xf32>
    }
  })";
  std::string mat_mul_method =
      absl::StrFormat(kMatMulModuleStr, test_case.mat_mul_method);
  TF_ASSERT_OK_AND_ASSIGN(
      XlaCompiler::CompilationResult result,
      CompileMlirModule(
          mat_mul_method.c_str(),
          ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_UNSPECIFIED));
}

INSTANTIATE_TEST_SUITE_P(
    BatchMatMulTest, BatchMatMulTest,
    ::testing::ValuesIn<MatMulTestCase>({
        {"BatchMatMul"},
        {"BatchMatMulV2"},
        {"BatchMatMulV3"},
    }),
    [](const ::testing::TestParamInfo<BatchMatMulTest::ParamType>& info) {
      return info.param.mat_mul_method;
    });

TEST(LegalizeTFTest, DumpsProducedHLO) {
  Env* env = Env::Default();
  std::string test_dir = TmpDir();
  setenv("TF_DUMP_GRAPH_PREFIX", test_dir.c_str(), /*overwrite=*/1);
  setenv("TF_DUMP_GRAPH_NAME_FILTER", "*", 1);
  DEBUG_DATA_DUMPER()->LoadEnvvars();

  std::vector<std::string> files;
  TF_ASSERT_OK(env->GetChildren(test_dir, &files));
  int original_files_size = files.size();

  TF_ASSERT_OK_AND_ASSIGN(
      XlaCompiler::CompilationResult result,
      CompileMlirModule(
          kMlirModuleStr,
          ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_UNSPECIFIED));

  // Due to the shared test of this infrastructure, we just need to make sure
  // that the dumped file size is greater than what was originally inside
  // the test directory.
  TF_ASSERT_OK(env->GetChildren(test_dir, &files));
  EXPECT_THAT(files.size(), ::testing::Gt(original_files_size));
  setenv("TF_DUMP_GRAPH_PREFIX", test_dir.c_str(), /*overwrite=*/0);
}

TEST(LegalizeTFTest, RecordsStreamzForFailedLegalizeWithMlirBridge) {
  CellReader<int64_t> compilation_status(kCompilationStatusStreamzName);

  auto result = CompileMlirModule(
      kBadMlirModuleStr,
      ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_UNSPECIFIED);

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(compilation_status.Delta(kMlirCombinedMlirFailure), 1);
}

TEST(LegalizeTFTest, RecordsStreamzForSuccessWithCombinedBridge) {
  CellReader<int64_t> compilation_status(kCompilationStatusStreamzName);

  auto result = CompileMlirModule(
      kUnsupportedMlirBridgeModuleStr,
      ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_UNSPECIFIED);

  // MLIR Bridge will filter this unsupported MLIR, Combined will succeed.
  EXPECT_TRUE(result.ok());
  EXPECT_EQ(compilation_status.Delta(kMlirCombinedMlirSuccess), 1);
  EXPECT_EQ(compilation_status.Delta(kMlirCombinedMlirFailure), 0);
  EXPECT_EQ(compilation_status.Delta(kMlirCombinedOldSuccess), 1);
  EXPECT_EQ(compilation_status.Delta(kMlirCombinedOldFailure), 0);
  // Old bridge should never be called at all.
  EXPECT_EQ(compilation_status.Delta(kOldBridgeMlirFilteredFailure), 0);
  EXPECT_EQ(compilation_status.Delta(kOldBridgeWithFallbackModeFailure), 0);
  EXPECT_EQ(compilation_status.Delta(kOldBridgeMlirFilteredSuccess), 0);
  EXPECT_EQ(compilation_status.Delta(kOldBridgeWithFallbackModeSuccess), 0);
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
      se::PlatformManager::PlatformWithName("Host").value();
  auto client =
      xla::ClientLibrary::GetOrCreateCompileOnlyClient(cpu_platform).value();

  std::vector<TensorShape> arg_shapes;
  TPUCompileMetadataProto metadata_proto;
  bool use_tuple_args = true;
  std::vector<ShardingAndIndex> arg_core_mapping;
  std::vector<std::vector<xla::Shape>> per_core_arg_shapes;
  std::vector<std::unique_ptr<mlir::Pass>> custom_legalization_passes;

  // This doesn't actually compile correctly.
  absl::StatusOr<XlaCompiler::CompilationResult> compile_result =
      LegalizeMlirToHlo(function_to_hlo_args, metadata_proto, use_tuple_args,
                        /*device_type=*/"XLA_CPU_JIT",
                        custom_legalization_passes,
                        /*shape_determination_fns=*/{}, arg_shapes,
                        &arg_core_mapping, &per_core_arg_shapes, client);

  EXPECT_FALSE(compile_result.ok());
}

TEST(LegalizeTFTest, RecordsCompilationTimeForSuccessfulCompilation) {
  CellReader<monitoring::testing::Histogram> compilation_time(
      kCompilationTimeStreamzName);

  TF_ASSERT_OK_AND_ASSIGN(
      XlaCompiler::CompilationResult result,
      CompileMlirModule(
          kMlirModuleStr,
          ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_ENABLED));

  // Compilation time should have been updated.
  EXPECT_GT(compilation_time.Delta(kFullBridge).num(), 0);
}

TEST(LegalizeTFTest, SuccessfullyCompilesModulesWithReturnValues) {
  static constexpr char kHasReturnValuesAndNoMetadataRetvals[] = R"(
    module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
      func.func @main() -> (tensor<2xi32>) {
        %cst = "tf.Const"() {value = dense<[524170, 523952]> : tensor<2xi32>} : () -> tensor<2xi32>
        return %cst : tensor<2xi32>
    }
  })";

  auto compilation_result = CompileMlirModule(
      kHasReturnValuesAndNoMetadataRetvals,
      ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_UNSPECIFIED);
  EXPECT_TRUE(compilation_result.ok());

  // Ensure that the compilation result contains a constant.
  EXPECT_THAT(compilation_result,
              ComputationProtoContains("opcode:.*constant"));
}

TEST(LegalizeTFTest, SkipsTensorListSetItemIfDimensionsTooLarge) {
  static constexpr char kTensorListSetItemDimensionTooLarge[] = R"(
    module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
      func.func @main() -> tensor<!tf_type.variant<tensor<64x1xbf16>>> {
      // unknown rank
      %elem_shape = "tf.Const"() <{value = dense<-1> : tensor<i32>}> {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> tensor<i32>
      // zero reserved elements
      %num_elements = "tf.Const"() <{value = dense<0> : tensor<i32>}> {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> tensor<i32>

      %list = "tf.TensorListReserve"(%elem_shape, %num_elements) : (tensor<i32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<64x1xbf16>>>

      %index = "tf.Const"() <{value = dense<0> : tensor<i32>}> {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> tensor<i32>
      %element = "tf.Const"() <{value = dense<0.0> : tensor<64x1xbf16>}> {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : () -> tensor<64x1xbf16>
      // Results in a bad mismatch of shapes.
      %updated_list = "tf.TensorListSetItem"(%list, %index, %element) : (tensor<!tf_type.variant<tensor<64x1xbf16>>>, tensor<i32>, tensor<64x1xbf16>) -> tensor<!tf_type.variant<tensor<64x1xbf16>>>

      return %updated_list : tensor<!tf_type.variant<tensor<64x1xbf16>>>
    }
  })";

  auto compilation_result = CompileMlirModule(
      kTensorListSetItemDimensionTooLarge,
      ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_UNSPECIFIED);

  // Ensure that it compile
  ASSERT_TRUE(compilation_result.ok());
  // Assert that the tensor list operation is lowered to something.
  ASSERT_THAT(compilation_result,
              Not(ComputationProtoContains("%.*= \"tf.TensorListSetItem")));
  // Assert that the tensor list operation is lowered to something that doesn't
  // get stuck on a broken dynamic update slice.
  ASSERT_THAT(compilation_result,
              Not(ComputationProtoContains("%.*=.*DynamicUpdateSlice")));
}

TEST(LegalizeTFTest, LegalizesFunctionWithBoundedDynamicArg) {
  static constexpr char kMlirModuleWithBoundedDynamicArgStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  func.func @main(%arg0: tensor<?xi32, #mhlo.type_extensions<bounds = [3]>> ) -> (tensor<?xi32, #mhlo.type_extensions<bounds = [3]>>) {
    func.return %arg0 : tensor<?xi32, #mhlo.type_extensions<bounds = [3]>>
  }
})";

  auto compilation_result = CompileMlirModule(
      kMlirModuleWithBoundedDynamicArgStr,
      ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_UNSPECIFIED);

  ASSERT_TRUE(compilation_result.ok());
  EXPECT_THAT(compilation_result,
              ComputationProtoContains("element_type:.S32\n.*dimensions: 3"));
}

}  // namespace v2
}  // namespace tf2xla
}  // namespace tensorflow
