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

#include "tensorflow/compiler/mlir/tf2xla/api/v0/compile_mlir_util.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace {

using ::mlir::OpPassManager;
using ::testing::HasSubstr;

static constexpr char kMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main() -> tensor<1xi32> {
      %0 = "tf.Const"() {value = dense<1000> : tensor<1xi32>} : () -> tensor<1xi32>
      func.return %0 : tensor<1xi32>
    }
  })";

TEST(LegalizeMlirTest, LegalizesModule) {
  mlir::DialectRegistry mlir_registry;
  RegisterAllTensorFlowDialects(mlir_registry);

  std::vector<tensorflow::TensorShape> arg_shapes;
  XlaCompilationResult compilation_result;
  Status status = CompileSerializedMlirToXlaHlo(
      kMlirModuleStr, arg_shapes, /*device_type=*/"XLA_TPU_JIT",
      /*use_tuple_args=*/true, /*enable_op_fallback=*/false,
      /*shape_determination_fns=*/{}, &compilation_result);

  EXPECT_TRUE(status.ok());
}

TEST(LegalizeMlirTest, FailsLegalizesModule) {
  constexpr char failed_legalization[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main() -> tensor<1xi32> {
      %0 = "tf.DoesntExist"() : () -> tensor<1xi32>
      func.return %0 : tensor<1xi32>
    }
  })";

  std::vector<tensorflow::TensorShape> arg_shapes;
  XlaCompilationResult compilation_result;
  Status status = CompileSerializedMlirToXlaHlo(
      failed_legalization, arg_shapes, /*device_type=*/"XLA_TPU_JIT",
      /*use_tuple_args=*/true, /*enable_op_fallback=*/false,
      /*shape_determination_fns=*/{}, &compilation_result);

  EXPECT_FALSE(status.ok());
}

TEST(CompileMlirUtil, CreatesPipeline) {
  OpPassManager pass_manager;
  llvm::StringRef device_type = "XLA_CPU_JIT";

  CreateConvertMlirToXlaHloPipeline(pass_manager, device_type,
                                    /*enable_op_fallback=*/false,
                                    /*custom_legalization_passes*/ {});

  EXPECT_FALSE(pass_manager.getPasses().empty());
}

TEST(CompileMlirUtil, HasLegalizationPass) {
  OpPassManager pass_manager;
  llvm::StringRef device_type = "XLA_CPU_JIT";
  absl::string_view kLegalizeTfPass =
      "xla-legalize-tf{allow-partial-conversion=false device-type=XLA_CPU_JIT "
      "legalize-chlo=true prefer-tf2xla=true use-tf2xla-fallback=true "
      "use-tf2xla-hlo-importer=false})";

  CreateConvertMlirToXlaHloPipeline(pass_manager, device_type,
                                    /*enable_op_fallback=*/true,
                                    /*custom_legalization_passes*/ {});

  std::string pass_description;
  llvm::raw_string_ostream raw_stream(pass_description);
  pass_manager.printAsTextualPipeline(raw_stream);

  EXPECT_THAT(pass_description, HasSubstr(kLegalizeTfPass));
}

TEST(CompileMlirUtil, CanonicalizationIsExplicitDuringInlining) {
  OpPassManager pass_manager;
  llvm::StringRef device_type = "XLA_CPU_JIT";
  absl::string_view kInlinePass =
      "inline{default-pipeline=canonicalize max-iterations=4 }";

  CreateConvertMlirToXlaHloPipeline(pass_manager, device_type,
                                    /*enable_op_fallback=*/true,
                                    /*custom_legalization_passes*/ {});

  std::string pass_description;
  llvm::raw_string_ostream raw_stream(pass_description);
  pass_manager.printAsTextualPipeline(raw_stream);

  EXPECT_THAT(pass_description, HasSubstr(kInlinePass));
}

TEST(LegalizeMlirTest, LegalizesModuleWithDynamicShape) {
  constexpr char legalization[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main(%arg0: tensor<?xi32, #mhlo.type_extensions<bounds = [1]>>) -> tensor<?xi32, #mhlo.type_extensions<bounds = [1]>> {
      %0 = "tf.Identity"(%arg0) : (tensor<?xi32, #mhlo.type_extensions<bounds = [1]>>) -> tensor<?xi32, #mhlo.type_extensions<bounds = [1]>>
      func.return %0 : tensor<?xi32, #mhlo.type_extensions<bounds = [1]>>
    }
  })";

  std::vector<tensorflow::TensorShape> arg_shapes = {{1}};
  XlaCompilationResult compilation_result;
  Status status = CompileSerializedMlirToXlaHlo(
      legalization, arg_shapes, /*device_type=*/"XLA_TPU_JIT",
      /*use_tuple_args=*/true, /*enable_op_fallback=*/false,
      /*shape_determination_fns=*/{}, &compilation_result);

  EXPECT_TRUE(status.ok());
}

}  // namespace
}  // namespace tensorflow
