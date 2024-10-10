/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_types.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/tf2hlo.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

// These tests are in a separate file because they require a separate build.

TEST(Tf2HloTest, Gpu) {
  // Create test input module
  constexpr absl::string_view kDataDirectory =
      "tensorflow/compiler/mlir/tfrt/transforms/ifrt/testdata";
  std::string mlir_module_path = tensorflow::GetDataDependencyFilepath(
      absl::StrCat(kDataDirectory, "/tf2hlo_gpu.mlir"));

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::RegisterAllTensorFlowDialects(registry);

  mlir::MLIRContext context(registry);

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, &context);

  ASSERT_TRUE(mlir_module);
  ASSERT_TRUE(mlir_module.get() != nullptr);

  xla::ifrt::MockClient mock_client;
  ON_CALL(mock_client, GetDefaultDeviceAssignment)
      .WillByDefault([]() -> absl::StatusOr<xla::DeviceAssignment> {
        return xla::DeviceAssignment(1, 1);
      });
  ON_CALL(mock_client, platform_name).WillByDefault([]() -> absl::string_view {
    return xla::CudaName();
  });

  std::vector<DtypeAndShape> dtype_and_shapes;
  dtype_and_shapes.push_back(DtypeAndShape{DT_FLOAT, {}});

  TF_ASSERT_OK_AND_ASSIGN(
      tensorflow::tpu::TPUCompileMetadataProto compile_metadata,
      GetCompileMetadata(mlir_module.get(), mock_client));
  TF_ASSERT_OK(UpdateCompileMetadata(compile_metadata, dtype_and_shapes));

  auto result = CompileTfToHlo(mlir_module.get(), dtype_and_shapes, "main",
                               mock_client, compile_metadata,
                               tensorflow::IdentityShapeRepresentationFn());

  TF_ASSERT_OK(result.status());
}

}  // namespace
}  // namespace ifrt_serving
}  // namespace tensorflow
