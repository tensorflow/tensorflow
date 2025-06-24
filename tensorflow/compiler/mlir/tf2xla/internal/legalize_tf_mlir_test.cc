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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/compiler/mlir/tf2xla/internal/test_matchers.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/mlir_hlo/mhlo/IR/register.h"
#include "xla/shape.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {
namespace {

using tpu::ShardingAndIndex;
using tpu::TPUCompileMetadataProto;

static constexpr char kMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main() -> tensor<1xi32> {
      %0 = "tf.Const"() {value = dense<1000> : tensor<1xi32>} : () -> tensor<1xi32>
      func.return %0 : tensor<1xi32>
    }
  })";

absl::StatusOr<XlaCompilationResult> CompileMlirModule(bool compile_to_xla_hlo,
                                                       const char* module_str) {
  XlaCompilationResult compilation_result;
  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module;
  TF_RETURN_IF_ERROR(
      tensorflow::DeserializeMlirModule(module_str, &context, &mlir_module));

  std::vector<TensorShape> arg_shapes;
  TPUCompileMetadataProto metadata_proto;
  bool use_tuple_args = true;
  std::vector<ShardingAndIndex> arg_core_mapping;
  std::vector<std::vector<xla::Shape>> per_core_arg_shapes;
  std::vector<std::unique_ptr<mlir::Pass>> custom_legalization_passes;

  TF_RETURN_IF_ERROR(CompileFromMlirToXlaHlo(
      compile_to_xla_hlo, mlir_module.get(), metadata_proto,
      /*device_type=*/"XLA_TPU_JIT",
      /*shape_determination_fns=*/{}, use_tuple_args, &compilation_result,
      custom_legalization_passes, arg_shapes, &arg_core_mapping,
      &per_core_arg_shapes));
  return compilation_result;
}

TEST(CompileFromMlir, ReturnsLoweredModule) {
  auto result = CompileMlirModule(/*compile_to_xla_hlo=*/true, kMlirModuleStr);

  ASSERT_THAT(result, IsOkOrFiltered());
  EXPECT_THAT(result, ComputationProtoContains("opcode: \"constant\""));
}

}  // namespace

}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow
