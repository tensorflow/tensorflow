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

#include "tensorflow/compiler/mlir/tf2xla/internal/mlir_pass_instrumentation.h"

#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/compiler/mlir/tf2xla/api/v1/compile_mlir_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace mlir {
namespace {
static const char* kTestInstrumentationName = "test-intrumentatron";
static const char* kTestInstrumentationSearch = "tf.Identity";

struct StringStream : public llvm::raw_ostream {
  StringStream() { SetUnbuffered(); }
  ~StringStream() override = default;
  uint64_t current_pos() const override { return 0; }

  void write_impl(const char* ptr, size_t size) override {
    ss.write(ptr, size);
  }
  std::stringstream ss;
};

class TestPassInstrumentation : public ::testing::Test {
 public:
  void SetPassThatChangedIdentity(absl::string_view pass_name) {
    pass_that_changed_identity_ = pass_name;
  }
  absl::string_view GetPassThatChangedIdentity() {
    return pass_that_changed_identity_;
  }

 private:
  std::string pass_that_changed_identity_;
  friend class TestInstrumentor;
};

class TestInstrumentor : public PassInstrumentation {
 public:
  explicit TestInstrumentor(TestPassInstrumentation* test) : test_(test) {}

 private:
  void runBeforePass(Pass* pass, Operation* op) override {
    StringStream stream;
    op->print(stream, mlir::OpPrintingFlags().useLocalScope());
    ops_seen_by_pass_[pass] = stream.ss.str();
  }
  void runAfterPass(Pass* pass, Operation* op) override {
    StringStream stream;
    op->print(stream, mlir::OpPrintingFlags().useLocalScope());
    if (!absl::StrContains(stream.ss.str(), kTestInstrumentationSearch) &&
        absl::StrContains(ops_seen_by_pass_[pass],
                          kTestInstrumentationSearch)) {
      test_->SetPassThatChangedIdentity(pass->getName().str());
    }
  }

 private:
  TestPassInstrumentation* test_;
  std::unordered_map<mlir::Pass*, std::string> ops_seen_by_pass_;
};

TEST_F(TestPassInstrumentation, CreatedCalledAndSetsPassName) {
  RegisterPassInstrumentor(kTestInstrumentationName, [&]() {
    return std::make_unique<TestInstrumentor>(this);
  });
  constexpr char legalization[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @main(%arg0: tensor<?xi32, #mhlo.type_extensions<bounds = [1]>>) -> tensor<?xi32, #mhlo.type_extensions<bounds = [1]>> {
      %0 = "tf.Identity"(%arg0) : (tensor<?xi32, #mhlo.type_extensions<bounds = [1]>>) -> tensor<?xi32, #mhlo.type_extensions<bounds = [1]>>
      func.return %0 : tensor<?xi32, #mhlo.type_extensions<bounds = [1]>>
    }
  })";
  SetPassThatChangedIdentity("");
  std::vector<::tensorflow::TensorShape> arg_shapes = {{1}};
  auto compilation_result = tensorflow::XlaCompilationResult();

  TF_EXPECT_OK(tensorflow::CompileSerializedMlirToXlaHlo(
                   legalization, arg_shapes, /*device_type=*/"XLA_TPU_JIT",
                   /*use_tuple_args=*/true, /*enable_op_fallback=*/false,
                   /*shape_determination_fns=*/{}, &compilation_result)
                   .status());

  EXPECT_FALSE(GetPassThatChangedIdentity().empty());
}

}  // namespace
}  // namespace mlir
