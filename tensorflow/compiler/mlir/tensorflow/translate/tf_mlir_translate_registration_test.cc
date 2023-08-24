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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "tensorflow/core/platform/test.h"

namespace mlir {
namespace {

class MlirTranslationTest : public ::testing::Test {
 private:
  static constexpr char kMlirToGraphFlag[] = "-mlir-to-graph";

 public:
  MlirTranslationTest() : translation_(RegisterTranslation()) {
    // Create fake command line args so that the parser gets chosen.
    std::vector<const char*> argv = {""};
    argv.push_back(kMlirToGraphFlag);
    llvm::cl::ParseCommandLineOptions(argv.size(), &argv[0],
                                      "TF MLIR translation test\n");
  }

  LogicalResult Translate(StringRef source, std::string& sink) {
    auto source_manager = std::make_shared<llvm::SourceMgr>();
    auto source_buffer = llvm::MemoryBuffer::getMemBuffer(source);
    source_manager->AddNewSourceBuffer(std::move(source_buffer), llvm::SMLoc());
    mlir::MLIRContext context;
    llvm::raw_string_ostream os(sink);

    return (**translation_)(source_manager, os, &context);
  }

 private:
  llvm::cl::opt<const mlir::Translation*, false, mlir::TranslationParser>*
  RegisterTranslation() {
    // Can only register once per process.
    static const auto requested_translation =
        new llvm::cl::opt<const mlir::Translation*, false,
                          mlir::TranslationParser>(
            llvm::cl::desc("Translation to perform"));
    return requested_translation;
  }
  llvm::cl::opt<const mlir::Translation*, false, mlir::TranslationParser>*
      translation_;
};

TEST_F(MlirTranslationTest, TranslatesMlirToGraph) {
  static constexpr char kMlirSource[] = R"(
func.func @main() -> (tensor<1x2xf16>, tensor<2xf16>) {
  %graph:2 = tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Const"() {device = "", dtype = "tfdtype$DT_HALF", value = dense<1.0> : tensor<1x2xf16>} : () -> tensor<1x2xf16> loc("const1")
    %1:2 = tf_executor.island wraps "tf.Const"() {device = "", dtype = "tfdtype$DT_HALF", value = dense<[1.0, 2.0]> : tensor<2xf16>} : () -> tensor<2xf16> loc("const2")
    tf_executor.fetch %0#0, %1#0 : tensor<1x2xf16>, tensor<2xf16>
  }
  func.return %graph#0, %graph#1 : tensor<1x2xf16>, tensor<2xf16>
})";
  std::string result;

  auto status = Translate(kMlirSource, result);

  ASSERT_TRUE(status.succeeded());
  EXPECT_TRUE(absl::StrContains(result, "node {"));
}

}  // namespace
}  // namespace mlir
