/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_util.h"

#include <string>

#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.h"
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_sync.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"
#include "tfrt/init_tfrt_dialects.h"  // from @tf_runtime

namespace tfrt {
namespace fallback_async {
namespace {

TEST(SavedModelTest, MapFallbackArgs) {
  std::string saved_model_mlir_path = tensorflow::GetDataDependencyFilepath(
      "tensorflow/compiler/mlir/tfrt/tests/ir/testdata/test.mlir");

  mlir::DialectRegistry registry;
  RegisterTFRTDialects(registry);
  registry.insert<tfrt::fallback_async::FallbackAsyncDialect>();
  registry.insert<tfrt::fallback_sync::FallbackSyncDialect>();

  mlir::MLIRContext context(registry);
  auto module =
      mlir::parseSourceFile<mlir::ModuleOp>(saved_model_mlir_path, &context);
  ASSERT_TRUE(module);

  std::vector<std::pair<std::string, int>> func_and_index;
  ForEachArgConsumedByFallback(
      module.get(),
      [&func_and_index](llvm::StringRef func_name, int arg_index) {
        func_and_index.push_back({func_name.str(), arg_index});
      });

  ASSERT_EQ(func_and_index.size(), 1);
  EXPECT_EQ(func_and_index[0].first, "test");
  EXPECT_EQ(func_and_index[0].second, 2);
}

}  // namespace
}  // namespace fallback_async
}  // namespace tfrt
