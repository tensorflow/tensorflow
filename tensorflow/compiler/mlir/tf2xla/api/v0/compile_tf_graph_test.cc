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

#include "tensorflow/compiler/mlir/tf2xla/api/v0/compile_tf_graph.h"

#include <variant>
#include <vector>

#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/lib/monitoring/test_utils.h"

namespace tensorflow {
namespace tf2xla {
namespace v0 {
namespace {

using ::tensorflow::monitoring::testing::CellReader;
using ::tensorflow::tpu::FunctionToHloArgs;
using ::tensorflow::tpu::MlirToHloArgs;
using ::tensorflow::tpu::ShardingAndIndex;
using ::tsl::monitoring::testing::Histogram;

static constexpr char kCompilationTimeStreamzName[] =
    "/tensorflow/core/tf2xla/api/v0/phase2_compilation_time";

static constexpr char kCompilationStatusStreamzName[] =
    "/tensorflow/core/tf2xla/api/v0/phase2_compilation_status";

MlirToHloArgs CreateTestMlirToHloArgs() {
  static constexpr char kMlirModuleStr[] = R"(
    module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main() -> () {
      func.return
    }
  })";

  MlirToHloArgs mlir_to_hlo_args;
  mlir_to_hlo_args.rollout_state =
      ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_DISABLED;
  mlir_to_hlo_args.mlir_module = kMlirModuleStr;

  return mlir_to_hlo_args;
}

class CompileTFGraphTest : public ::testing::Test {
 public:
  tsl::Status CompileWithComputation(
      const std::variant<tpu::MlirToHloArgs, tpu::FunctionToHloArgs>
          computation) {
    se::Platform* platform =
        se::MultiPlatformManager::PlatformWithName("Host").value();
    auto client =
        xla::ClientLibrary::GetOrCreateCompileOnlyClient(platform).value();

    std::vector<TensorShape> arg_shapes;
    bool use_tuple_args = true;
    std::vector<ShardingAndIndex> arg_core_mapping;
    std::vector<std::vector<xla::Shape>> per_core_arg_shapes;
    XlaCompiler::CompilationResult result;
    tpu::TPUCompileMetadataProto metadata_proto;
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_funcs;

    return CompileTensorflowGraphToHlo(
        computation, metadata_proto, use_tuple_args, shape_determination_funcs,
        arg_shapes, &arg_core_mapping, &per_core_arg_shapes, client, &result);
  }
};

TEST_F(CompileTFGraphTest, RecordsStreamzForMlirFallback) {
  CellReader<Histogram> compilation_time(kCompilationTimeStreamzName);

  MlirToHloArgs mlir_to_hlo_args = CreateTestMlirToHloArgs();

  TF_EXPECT_OK(CompileWithComputation(mlir_to_hlo_args));

  Histogram histogram = compilation_time.Delta("graph_old_bridge_has_mlir");

  EXPECT_EQ(histogram.num(), 1);
}

TEST_F(CompileTFGraphTest, RecordsStreamzForFunctionToHlo) {
  CellReader<Histogram> compilation_time(kCompilationTimeStreamzName);
  CellReader<int64_t> compilation_status(kCompilationStatusStreamzName);

  FunctionDef empty_function =
      tensorflow::FunctionDefHelper::Create("empty", {}, {}, {}, {}, {});

  tensorflow::FunctionDefLibrary fdef;
  *(fdef.add_function()) = empty_function;
  tensorflow::FunctionLibraryDefinition flib_def(
      tensorflow::OpRegistry::Global(), fdef);

  OpInputList guaranteed_constants;
  NameAttrList function;
  function.set_name("empty");

  FunctionToHloArgs function_to_hlo_args = {&function,
                                            &flib_def,
                                            /*graph_def_version=*/0,
                                            {&guaranteed_constants}};

  TF_EXPECT_OK(CompileWithComputation(function_to_hlo_args));

  Histogram histogram =
      compilation_time.Delta("graph_old_bridge_has_function_to_hlo");

  EXPECT_EQ(histogram.num(), 1);
  EXPECT_EQ(compilation_status.Delta("kOldBridgeNoMlirSuccess"), 1);
}

}  // namespace
}  // namespace v0
}  // namespace tf2xla
}  // namespace tensorflow
