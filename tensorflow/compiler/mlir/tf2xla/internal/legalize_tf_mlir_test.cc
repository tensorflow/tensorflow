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

#include "tensorflow/compiler/mlir/tf2xla/internal/legalize_tf_mlir.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tf2xla/internal/test_matchers.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/shape.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {
namespace {

using tpu::MlirToHloArgs;
using tpu::ShardingAndIndex;
using tpu::TPUCompileMetadataProto;

static constexpr char kMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main() -> tensor<1xi32> {
      %0 = "tf.Const"() {value = dense<1000> : tensor<1xi32>} : () -> tensor<1xi32>
      func.return %0 : tensor<1xi32>
    }
  })";

absl::StatusOr<std::string> CompileMlirModule(bool compile_to_xla_hlo,
                                              const char* module_str) {
  MlirToHloArgs mlir_to_hlo_args;
  mlir_to_hlo_args.mlir_module = module_str;

  std::vector<TensorShape> arg_shapes;
  TPUCompileMetadataProto metadata_proto;
  bool use_tuple_args = true;
  std::vector<ShardingAndIndex> arg_core_mapping;
  std::vector<std::vector<xla::Shape>> per_core_arg_shapes;
  std::vector<std::unique_ptr<mlir::Pass>> custom_legalization_passes;

  auto compilation_result = std::make_unique<XlaCompilationResult>();

  return CompileFromMlirToXlaHlo(
      compile_to_xla_hlo, mlir_to_hlo_args, metadata_proto,
      /*device_type=*/"XLA_TPU_JIT",
      /*shape_determination_fns=*/{}, use_tuple_args, compilation_result.get(),
      custom_legalization_passes, arg_shapes, &arg_core_mapping,
      &per_core_arg_shapes);
}

absl::StatusOr<XlaCompiler::CompilationResult> LegalizeMlirModule(
    const char* module_str) {
  MlirToHloArgs mlir_to_hlo_args;
  mlir_to_hlo_args.mlir_module = module_str;

  std::vector<TensorShape> arg_shapes;
  TPUCompileMetadataProto metadata_proto;
  bool use_tuple_args = true;
  std::vector<ShardingAndIndex> arg_core_mapping;
  std::vector<std::vector<xla::Shape>> per_core_arg_shapes;
  std::vector<std::unique_ptr<mlir::Pass>> custom_legalization_passes;

  auto compilation_result = std::make_unique<XlaCompilationResult>();

  return LegalizeWithMlirBridge(
      mlir_to_hlo_args, metadata_proto, use_tuple_args,
      /*device_type=*/"XLA_TPU_JIT",
      /*shape_determination_fns=*/{}, arg_shapes, &arg_core_mapping,
      &per_core_arg_shapes, custom_legalization_passes,
      compilation_result.get());
}

TEST(LegalizeWithMlirBridge, LegalizesToMhloProto) {
  auto result = LegalizeMlirModule(kMlirModuleStr);

  ASSERT_THAT(result, IsOkOrFiltered());
  EXPECT_THAT(result, ComputationProtoContains("opcode.*constant"));
}

TEST(CompileFromMlir, ReturnsModuleAsString) {
  auto result = CompileMlirModule(true, kMlirModuleStr);

  ASSERT_THAT(result, IsOkOrFiltered());
  // TODO(b/288289388) Update test once module is actually returned
  EXPECT_THAT(result, HasMlirModuleWith("mhlo.constant"));
}

}  // namespace

}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow
