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
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/tfrt/saved_model/saved_model_testutil.h"

namespace tensorflow {
namespace tfrt_stub {
namespace {

TEST(SavedModelTest, MatmulGpu) {
  setenv("TF_DUMP_GRAPH_PREFIX", getenv("TEST_UNDECLARED_OUTPUTS_DIR"), 1);
  std::string saved_model_dir = tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/matmul_gpu");

  auto runtime = DefaultTfrtRuntime(/*num_threads=*/1);
  SavedModel::Options options(runtime.get());
  options.graph_execution_options.enable_tfrt_gpu = true;
  options.graph_execution_options.enable_grappler_function_optimizer = true;
  options.graph_execution_options.compile_options.enable_grappler = true;

  tensorflow::Status status;
  auto saved_model =
      SavedModelImpl::LoadSavedModel(options, saved_model_dir,
                                     /*tags=*/{"serve"}, &status);
  TF_CHECK_OK(status);

  std::vector<tensorflow::Tensor> inputs = {
      CreateTfTensor<float>({1, 3}, {1.0, 1.0, 1.0})};

  std::vector<tensorflow::Tensor> outputs;
  TF_ASSERT_OK(saved_model->Run(/*run_options=*/{}, "serving_default", inputs,
                                &outputs));

  ASSERT_EQ(outputs.size(), 1);
  EXPECT_THAT(GetTfTensorData<float>(outputs[0]),
              ::testing::ElementsAreArray({6.0}));
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace tensorflow

#endif
