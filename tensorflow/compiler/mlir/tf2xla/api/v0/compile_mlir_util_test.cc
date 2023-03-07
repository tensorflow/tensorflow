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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Pass/PassManager.h"  // from @llvm-project

namespace tensorflow {
namespace {

using ::mlir::OpPassManager;
using ::testing::HasSubstr;

TEST(CompileMlirUtil, CreatesPipeline) {
  OpPassManager pass_manager;
  llvm::StringRef device_type = "XLA_CPU_JIT";

  CreateConvertMlirToXlaHloPipeline(pass_manager, device_type,
                                    /*enable_op_fallback=*/true,
                                    /*custom_legalization_passes*/ {});

  EXPECT_FALSE(pass_manager.getPasses().empty());
}

TEST(CompileMlirUtil, HasLegalizationPass) {
  OpPassManager pass_manager;
  llvm::StringRef device_type = "XLA_CPU_JIT";
  absl::string_view kLegalizeTfPass =
      "xla-legalize-tf{allow-partial-conversion=false device-type=XLA_CPU_JIT "
      "legalize-chlo=true prefer-tf2xla=true use-tf2xla-fallback=true})";

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

}  // namespace
}  // namespace tensorflow
