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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_backend_compiler.h"
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/test_util.h"
#include "xla/tsl/framework/test_util/mock_serving_device_selector.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_model_context.h"
#include "tensorflow/core/tfrt/ifrt/ifrt_serving_core_selector.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/core/tfrt/saved_model/saved_model.h"
#include "tensorflow/core/tfrt/saved_model/saved_model_testutil.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {
namespace {

tsl::thread::ThreadPool& GetThreadPool() {
  constexpr int kMaxParallelism = 16;
  static tsl::thread::ThreadPool* thread_pool =
      new tsl::thread::ThreadPool(tsl::Env::Default(), tsl::ThreadOptions(),
                                  "IfrtSharding", kMaxParallelism);
  return *thread_pool;
}

TEST(SavedModelIfrt, Basic) {
  std::string saved_model_dir = tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/toy_v2");

  auto runtime =
      tensorflow::tfrt_stub::Runtime::Create(/*num_inter_op_threads=*/4);

  // Create contexts required for the compiler execution.
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  auto work_queue = tfrt::CreateMultiThreadedWorkQueue(
      /*num_threads=*/4, /*num_blocking_threads=*/4);

  tsl::test_util::MockServingDeviceSelector selector;
  ifrt_serving::IfrtServingCoreSelector core_selector(
      &selector, client->addressable_device_count());

  // Use IFRT compiler
  runtime->AddCreateRuntimeResourceFn(
      [&](tensorflow::tfrt_stub::ModelRuntimeContext& model_context) {
        model_context.resource_context()
            .CreateResource<tensorflow::ifrt_serving::IfrtModelContext>(
                "IfrtModelContext", client, &core_selector, &GetThreadPool(),
                /*compilation_environment_proto=*/nullptr);

        (*model_context.resource_context()
              .GetResource<tensorflow::ifrt_serving::IfrtModelContext>(
                  "IfrtModelContext"))
            ->set_checkpoint_loader_queue(work_queue.get());

        return absl::OkStatus();
      });
  tensorflow::ifrt_serving::IfrtBackendCompiler ifrt_compiler;

  auto options = DefaultSavedModelOptions(runtime.get());
  options.graph_execution_options.enable_mlrt = true;
  options.enable_lazy_loading = true;
  options.lazy_loading_use_graph_executor = true;
  options.graph_execution_options.compile_options.backend_compiler =
      &ifrt_compiler;

  TF_ASSERT_OK_AND_ASSIGN(
      auto saved_model, SavedModelImpl::LoadSavedModel(options, saved_model_dir,
                                                       /*tags=*/{"serve"}));

  // Set input 'x' to [[1, 1, 1]]
  std::vector<tensorflow::Tensor> inputs;
  inputs.push_back(
      CreateTfTensor<int32_t>(/*shape=*/{1, 3}, /*data=*/{1, 1, 1}));

  tfrt::SavedModel::RunOptions run_options;

  std::vector<tensorflow::Tensor> outputs;
  TF_ASSERT_OK(
      saved_model->Run(run_options, "serving_default", inputs, &outputs));
  ASSERT_EQ(outputs.size(), 1);

  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[0]),
              ::testing::ElementsAreArray({6}));
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace tensorflow
