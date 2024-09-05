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
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/tfrt/saved_model/saved_model_testutil.h"

namespace tensorflow {
namespace tfrt_stub {
namespace {

TEST(SavedModelTest, BasicError) {
  std::string saved_model_dir = tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/runtime_fallback/test/saved_model/basic_v1");

  TFRTSavedModelTest test(saved_model_dir);

  std::vector<tensorflow::Tensor> inputs;
  inputs.push_back(
      CreateTfTensor<int32_t>(/*shape=*/{1, 3}, /*data=*/{1, 1, 1}));

  std::vector<tensorflow::Tensor> outputs;
  EXPECT_FALSE(
      test.GetSavedModel()->Run({}, "serving_default", inputs, &outputs).ok());
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace tensorflow
