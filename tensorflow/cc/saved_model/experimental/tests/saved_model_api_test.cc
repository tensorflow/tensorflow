/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/saved_model/experimental/public/saved_model_api.h"

#include <memory>
#include <string>
#include <unordered_set>

#include "tensorflow/cc/experimental/base/public/runtime.h"
#include "tensorflow/cc/experimental/base/public/runtime_builder.h"
#include "tensorflow/cc/experimental/base/public/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

constexpr char kTestData[] = "cc/saved_model/testdata";

std::string SavedModelPath(tensorflow::StringPiece saved_model_dir) {
  return tensorflow::io::JoinPath(tensorflow::testing::TensorFlowSrcRoot(),
                                  kTestData, saved_model_dir);
}

// This value parameterized test allows us to test both TFRT
// and non TFRT runtimes.
// https://github.com/google/googletest/blob/dcc92d0ab6c4ce022162a23566d44f673251eee4/googletest/docs/advanced.md#value-parameterized-tests
class CPPSavedModelAPITest : public ::testing::TestWithParam<bool> {};

TEST_P(CPPSavedModelAPITest, LoadsSavedModelWithTags) {
  cc::Status status;
  cc::RuntimeBuilder builder;
  bool use_tfrt = GetParam();
  if (use_tfrt) {
    GTEST_SKIP();  // TODO(chky) : Enable this once TFRT is open sourced.
  }

  builder.SetUseTFRT(use_tfrt);
  std::unique_ptr<cc::Runtime> runtime = builder.Build(&status);
  ASSERT_TRUE(status.ok()) << status.message();

  std::string model_dir = SavedModelPath("VarsAndArithmeticObjectGraph");
  std::unordered_set<std::string> tags = {"serve"};
  std::unique_ptr<cc::SavedModelAPI> model =
      cc::SavedModelAPI::Load(model_dir, *runtime, &status, &tags);

  // TODO(bmzhao): Change this to expect TF_OK when loading is implemented.
  // That unblocks writing other tests that require a TF_SavedModel*,
  // like loading a ConcreteFunction. This test at least checks that the
  // C API builds and can be minimally run.
  EXPECT_EQ(status.code(), TF_UNIMPLEMENTED);
}

TEST_P(CPPSavedModelAPITest, LoadsSavedModel) {
  cc::Status status;
  cc::RuntimeBuilder builder;
  bool use_tfrt = GetParam();
  if (use_tfrt) {
    GTEST_SKIP();  // TODO(chky) : Enable this once TFRT is open sourced.
  }

  builder.SetUseTFRT(use_tfrt);
  std::unique_ptr<cc::Runtime> runtime = builder.Build(&status);
  ASSERT_TRUE(status.ok()) << status.message();

  std::string model_dir = SavedModelPath("VarsAndArithmeticObjectGraph");
  std::unique_ptr<cc::SavedModelAPI> model =
      cc::SavedModelAPI::Load(model_dir, *runtime, &status);

  // TODO(bmzhao): Change this to expect TF_OK when loading is implemented.
  // That unblocks writing other tests that require a TF_SavedModel*,
  // like loading a ConcreteFunction. This test at least checks that the
  // C API builds and can be minimally run.
  EXPECT_EQ(status.code(), TF_UNIMPLEMENTED);
}

INSTANTIATE_TEST_SUITE_P(RuntimeAgnosticCPPSavedModelTests,
                         CPPSavedModelAPITest, ::testing::Bool());

}  // namespace

}  // namespace tensorflow
