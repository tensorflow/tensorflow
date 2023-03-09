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
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/tfrt/fallback/cost_recorder.h"
#include "tensorflow/core/tfrt/run_handler_thread_pool/run_handler_concurrent_work_queue.h"
#include "tensorflow/core/tfrt/saved_model/saved_model_mira_impl.h"
#include "tensorflow/core/tfrt/saved_model/saved_model_testutil.h"

namespace tensorflow {
namespace tfrt_stub {
namespace {

struct TestParams {
  bool enable_grappler = false;
  bool enable_lazy_loading = false;
  bool lazy_loading_use_graph_executor = false;
};

class SavedModelTest : public ::testing::TestWithParam<TestParams> {};

TEST_P(SavedModelTest, BasicV1) {
  // SavedModel toy contains a graph of a single 'tf.AddV2' op. It is generated
  // using the following python code:
  //  x = tf.placeholder(tf.int32, shape=(3))
  //  y = tf.compat.v1.get_variable(name='y', initializer=[1, 2, 3])
  //  r = tf.matmul(x, y)
  std::string saved_model_dir = tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/toy_v1");

  auto runtime = DefaultTfrtRuntime(/*num_threads=*/1);
  auto options = DefaultSavedModelOptions(runtime.get());
  options.enable_lazy_loading = GetParam().enable_lazy_loading;
  options.lazy_loading_use_graph_executor =
      GetParam().lazy_loading_use_graph_executor;
  options.graph_execution_options.compile_options.enable_grappler =
      GetParam().enable_grappler;

  auto saved_model = SavedModelImpl::LoadSavedModel(options, saved_model_dir,
                                                    /*tags=*/{"serve"});
  TF_CHECK_OK(saved_model.status());

  // Set input 'x' to [[1, 1, 1]]
  std::vector<tensorflow::Tensor> inputs;
  inputs.push_back(
      CreateTfTensor<int32_t>(/*shape=*/{1, 3}, /*data=*/{1, 1, 1}));

  tfrt::SavedModel::RunOptions run_options;

  std::vector<tensorflow::Tensor> outputs;
  TF_ASSERT_OK((*saved_model)->Run(run_options, "toy", inputs, &outputs));
  ASSERT_EQ(outputs.size(), 1);

  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[0]),
              ::testing::ElementsAreArray({6}));
}

// Tests all the value combinations of `TestParams`. For readability, use
// integers instead of booleans.
INSTANTIATE_TEST_SUITE_P(
    SavedModelLiteTest, SavedModelTest,
    ::testing::Values(
        // The values below are for:
        // enable_grappler, enable_lazy_loading, lazy_loading_use_graph_executor
        TestParams{0, 0, 0}, TestParams{1, 0, 0}, TestParams{0, 1, 0},
        TestParams{1, 1, 0}, TestParams{0, 1, 1}, TestParams{1, 1, 1}));

TEST(SavedModelTest, BasicV2) {
  // SavedModel toy contains a graph of a single 'tf.AddV2' op. It is generated
  // using the following python code:
  // self.w = tf.Variable(tf.ones((3)), name='w')
  // r = tf.matmul(x, self.w)
  std::string saved_model_dir = tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/toy_v2");

  TFRTSavedModelTest test(saved_model_dir);

  // Set input 'x' to [[1, 1, 1]]
  std::vector<tensorflow::Tensor> inputs;
  inputs.emplace_back(tensorflow::DT_INT32,
                      /*shape=*/tensorflow::TensorShape{1, 3});
  auto flat = inputs.back().flat<int32_t>();
  flat(0) = 1;
  flat(1) = 1;
  flat(2) = 1;

  std::vector<tensorflow::Tensor> outputs;
  TF_ASSERT_OK(
      test.GetSavedModel()->Run({}, "serving_default", inputs, &outputs));
  ASSERT_EQ(outputs.size(), 1);
  auto& output = outputs[0];

  ASSERT_EQ(output.NumElements(), 1);
  EXPECT_EQ(output.flat<int32_t>()(0), 6);
}

TEST(SavedModelTest, BasicInlineExecution) {
  // SavedModel toy contains a graph of a single 'tf.AddV2' op. It is generated
  // using the following python code:
  // self.w = tf.Variable(tf.ones((3)), name='w')
  // r = tf.matmul(x, self.w)
  std::string saved_model_dir = tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/toy_v2");

  auto runtime = DefaultTfrtRuntime(/*num_threads=*/1);

  runtime->SetCreateRequestQueueFn(
      [](int64_t) -> StatusOr<std::unique_ptr<WorkQueueInterface>> {
        return tensorflow::tfrt_stub::WrapDefaultWorkQueue(
            tfrt::CreateSingleThreadedWorkQueue());
      });

  auto options = DefaultSavedModelOptions(runtime.get());

  TF_ASSERT_OK_AND_ASSIGN(
      auto saved_model, SavedModelImpl::LoadSavedModel(options, saved_model_dir,
                                                       /*tags=*/{"serve"}));

  // Set input 'x' to [[1, 1, 1]]
  std::vector<tensorflow::Tensor> inputs;
  inputs.emplace_back(tensorflow::DT_INT32,
                      /*shape=*/tensorflow::TensorShape{1, 3});
  auto flat = inputs.back().flat<int32_t>();
  flat(0) = 1;
  flat(1) = 1;
  flat(2) = 1;

  std::vector<tensorflow::Tensor> outputs;
  TF_ASSERT_OK(saved_model->Run({}, "serving_default", inputs, &outputs));
  ASSERT_EQ(outputs.size(), 1);
  auto& output = outputs[0];

  ASSERT_EQ(output.NumElements(), 1);
  EXPECT_EQ(output.flat<int32_t>()(0), 6);
}

TEST(SavedModelTest, VariableOnTpu) {
  // A ReadVariableOp on 'TPU' would behave exactly the same as a ReadVariableOp
  // on 'CPU'. This is to be compatible with TF1 runtime.
  std::string saved_model_dir = tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/variable_on_tpu");

  auto runtime = DefaultTfrtRuntime(/*num_threads=*/1);
  auto options = DefaultSavedModelOptions(runtime.get());

  auto saved_model = SavedModelImpl::LoadSavedModel(options, saved_model_dir,
                                                    /*tags=*/{"serve"});
  TF_CHECK_OK(saved_model.status());

  // Set input 'x' to [[1, 1, 1]]
  std::vector<tensorflow::Tensor> inputs;
  inputs.emplace_back(tensorflow::DT_INT32,
                      /*shape=*/tensorflow::TensorShape{1, 3});
  auto flat = inputs.back().flat<int32_t>();
  flat(0) = 1;
  flat(1) = 1;
  flat(2) = 1;

  std::vector<tensorflow::Tensor> outputs;
  TF_ASSERT_OK((*saved_model)->Run({}, "serving_default", inputs, &outputs));
  ASSERT_EQ(outputs.size(), 1);
  auto& output = outputs[0];

  ASSERT_EQ(output.NumElements(), 1);
  EXPECT_EQ(output.flat<int32_t>()(0), 6);
}

std::vector<tensorflow::Tensor> CreateExpectedOutputs(
    const FunctionMetadata& function_metadata,
    const std::vector<std::pair<std::string, tensorflow::Tensor>>&
        named_outputs) {
  std::vector<tensorflow::Tensor> outputs;
  absl::flat_hash_map<std::string, tensorflow::Tensor> name_to_outputs;
  for (const auto& name_and_output : named_outputs) {
    name_to_outputs[name_and_output.first] = name_and_output.second;
  }

  for (const auto& name : function_metadata.GetOutputNames()) {
    outputs.push_back(name_to_outputs.at(name));
  }

  return outputs;
}

TEST(SavedModelTest, RunMultipleSignatures) {
  // SavedModel toy contains a graph of a single 'tf.AddV2' op. It is generated
  // using the following python code:
  //  x = tf.placeholder(tf.int32, shape=(3))
  //  y = tf.compat.v1.get_variable(name='y', initializer=[1, 2, 3])
  //  r = tf.matmul(x, y)
  std::string saved_model_dir = tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/toy_v1");

  auto runtime = DefaultTfrtRuntime(/*num_threads=*/1);
  auto options = DefaultSavedModelOptions(runtime.get());

  auto saved_model = SavedModelImpl::LoadSavedModel(options, saved_model_dir,
                                                    /*tags=*/{"serve"});
  TF_CHECK_OK(saved_model.status());

  std::vector<tensorflow::Tensor> toy_inputs;
  toy_inputs.push_back(CreateTfTensor<int32_t>(/*shape=*/{1, 3},
                                               /*data=*/{1, 1, 1}));
  std::vector<tensorflow::Tensor> another_toy_inputs;
  another_toy_inputs.push_back(CreateTfTensor<int32_t>(/*shape=*/{1, 3},
                                                       /*data=*/{2, 2, 2}));

  std::vector<tensorflow::Tensor> yet_another_toy_inputs;
  yet_another_toy_inputs.push_back(CreateTfTensor<int32_t>(/*shape=*/{1, 3},
                                                           /*data=*/{3, 3, 3}));

  // TODO(b/183220175): Construct `inputs` in place once `TenserHandle` is
  // copyable.
  std::vector<std::vector<tensorflow::Tensor>> inputs;
  inputs.push_back(std::move(toy_inputs));
  inputs.push_back(std::move(another_toy_inputs));
  inputs.push_back(std::move(yet_another_toy_inputs));

  std::vector<std::vector<tensorflow::Tensor>> outputs;
  std::vector<std::string> names = {"toy", "another_toy", "yet_another_toy"};
  TF_ASSERT_OK(
      (*saved_model)
          ->RunMultipleSignatures(/*run_options=*/{}, names, inputs, &outputs));

  ASSERT_EQ(outputs.size(), 3);

  {
    auto toy_metadata = (*saved_model)->GetFunctionMetadata("toy");
    ASSERT_TRUE(toy_metadata.has_value());
    std::vector<std::pair<std::string, tensorflow::Tensor>>
        expected_toy_named_outputs;
    expected_toy_named_outputs.push_back(
        {"r1", CreateTfTensor<int32_t>(/*shape=*/{1}, /*data=*/{6})});
    std::vector<tensorflow::Tensor> expected_toy_outputs =
        CreateExpectedOutputs(*toy_metadata, expected_toy_named_outputs);

    ASSERT_EQ(outputs[0].size(), 1);
    EXPECT_THAT(GetTfTensorData<int32_t>(outputs[0][0]),
                ::testing::ElementsAreArray(
                    GetTfTensorData<int32_t>(expected_toy_outputs[0])));
  }

  {
    auto another_toy_metadata =
        (*saved_model)->GetFunctionMetadata("another_toy");
    ASSERT_TRUE(another_toy_metadata.has_value());
    std::vector<std::pair<std::string, tensorflow::Tensor>>
        expected_another_toy_named_outputs;
    expected_another_toy_named_outputs.push_back(
        {"r21", CreateTfTensor<int32_t>(/*shape=*/{1}, /*data=*/{12})});
    expected_another_toy_named_outputs.push_back(
        {"r22", CreateTfTensor<int32_t>(/*shape=*/{1}, /*data=*/{18})});
    std::vector<tensorflow::Tensor> expected_another_toy_outputs =
        CreateExpectedOutputs(*another_toy_metadata,
                              expected_another_toy_named_outputs);

    ASSERT_EQ(outputs[1].size(), 2);
    EXPECT_THAT(GetTfTensorData<int32_t>(outputs[1][0]),
                ::testing::ElementsAreArray(
                    GetTfTensorData<int32_t>(expected_another_toy_outputs[0])));
    EXPECT_THAT(GetTfTensorData<int32_t>(outputs[1][1]),
                ::testing::ElementsAreArray(
                    GetTfTensorData<int32_t>(expected_another_toy_outputs[1])));
  }

  {
    auto yet_another_toy_metadata =
        (*saved_model)->GetFunctionMetadata("yet_another_toy");
    ASSERT_TRUE(yet_another_toy_metadata.has_value());
    std::vector<std::pair<std::string, tensorflow::Tensor>>
        expected_yet_another_toy_named_outputs;
    expected_yet_another_toy_named_outputs.push_back(
        {"r31", CreateTfTensor<int32_t>(/*shape=*/{1}, /*data=*/{18})});
    expected_yet_another_toy_named_outputs.push_back(
        {"r32", CreateTfTensor<int32_t>(/*shape=*/{1, 3},
                                        /*data=*/{21, 21, 21})});
    expected_yet_another_toy_named_outputs.push_back(
        {"r33", CreateTfTensor<int32_t>(/*shape=*/{1, 3},
                                        /*data=*/{24, 24, 24})});
    std::vector<tensorflow::Tensor> expected_yet_another_toy_outputs =
        CreateExpectedOutputs(*yet_another_toy_metadata,
                              expected_yet_another_toy_named_outputs);

    ASSERT_EQ(outputs[2].size(), 3);
    EXPECT_THAT(GetTfTensorData<int32_t>(outputs[2][0]),
                ::testing::ElementsAreArray(GetTfTensorData<int32_t>(
                    expected_yet_another_toy_outputs[0])));
    EXPECT_THAT(GetTfTensorData<int32_t>(outputs[2][1]),
                ::testing::ElementsAreArray(GetTfTensorData<int32_t>(
                    expected_yet_another_toy_outputs[1])));
    EXPECT_THAT(GetTfTensorData<int32_t>(outputs[2][2]),
                ::testing::ElementsAreArray(GetTfTensorData<int32_t>(
                    expected_yet_another_toy_outputs[2])));
  }
}

TEST(SavedModelTest, RunMultipleSignatures_OverlappingNodes) {
  // SavedModel toy contains a graph of a single 'tf.AddV2' op. It is generated
  // using the following python code:
  //  x = tf.placeholder(tf.int32, shape=(3))
  //  y = tf.compat.v1.get_variable(name='y', initializer=[1, 2, 3])
  //  r = tf.matmul(x, y)
  std::string saved_model_dir = tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/toy_v1");

  auto runtime = DefaultTfrtRuntime(/*num_threads=*/1);
  auto options = DefaultSavedModelOptions(runtime.get());

  auto saved_model = SavedModelImpl::LoadSavedModel(options, saved_model_dir,
                                                    /*tags=*/{"serve"});
  TF_CHECK_OK(saved_model.status());

  std::vector<std::vector<tensorflow::Tensor>> inputs = {
      {CreateTfTensor<int32_t>(/*shape=*/{1, 3},
                               /*data=*/{1, 1, 1})},
      {CreateTfTensor<int32_t>(/*shape=*/{1, 3},
                               /*data=*/{1, 1, 1})},
      {CreateTfTensor<int32_t>(/*shape=*/{1, 3},
                               /*data=*/{1, 1, 1})}};

  std::vector<std::vector<tensorflow::Tensor>> outputs;
  std::vector<std::string> names = {"toy", "another_toy", "toy"};
  TF_ASSERT_OK(
      (*saved_model)
          ->RunMultipleSignatures(/*run_options=*/{}, names, inputs, &outputs));
  ASSERT_EQ(outputs.size(), 3);

  ASSERT_EQ(outputs[0].size(), 1);
  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[0][0]),
              ::testing::ElementsAreArray({6}));

  {
    auto another_toy_metadata =
        (*saved_model)->GetFunctionMetadata("another_toy");
    ASSERT_TRUE(another_toy_metadata.has_value());
    std::vector<std::pair<std::string, tensorflow::Tensor>>
        expected_another_toy_named_outputs;
    expected_another_toy_named_outputs.push_back(
        {"r21", CreateTfTensor<int32_t>(/*shape=*/{1}, /*data=*/{6})});
    expected_another_toy_named_outputs.push_back(
        {"r22", CreateTfTensor<int32_t>(/*shape=*/{1}, /*data=*/{12})});
    std::vector<tensorflow::Tensor> expected_another_toy_outputs =
        CreateExpectedOutputs(*another_toy_metadata,
                              expected_another_toy_named_outputs);

    ASSERT_EQ(outputs[1].size(), 2);
    EXPECT_THAT(GetTfTensorData<int32_t>(outputs[1][0]),
                ::testing::ElementsAreArray(
                    GetTfTensorData<int32_t>(expected_another_toy_outputs[0])));
    EXPECT_THAT(GetTfTensorData<int32_t>(outputs[1][1]),
                ::testing::ElementsAreArray(
                    GetTfTensorData<int32_t>(expected_another_toy_outputs[1])));
  }

  ASSERT_EQ(outputs[2].size(), 1);
  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[2][0]),
              ::testing::ElementsAreArray({6}));
}

class SavedModelRunByTensorNamesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // SavedModel toy contains a graph of a single 'tf.AddV2' op. It is
    // generated using the following python code:
    //  x = tf.placeholder(tf.int32, shape=(3))
    //  y = tf.compat.v1.get_variable(name='y', initializer=[1, 2, 3])
    //  r = tf.matmul(x, y)
    auto saved_model_dir = tensorflow::GetDataDependencyFilepath(
        "tensorflow/core/tfrt/saved_model/tests/toy_v1");
    runtime_ = DefaultTfrtRuntime(/*num_threads=*/1);
    auto options = DefaultSavedModelOptions(runtime_.get());

    auto saved_model = SavedModelImpl::LoadSavedModel(options, saved_model_dir,
                                                      /*tags=*/{"serve"});
    TF_CHECK_OK(saved_model.status());
    saved_model_.reset(static_cast<SavedModelImpl*>(saved_model->release()));

    inputs_.push_back(
        std::make_pair("input1", CreateTfTensor<int32_t>(/*shape=*/{1, 3},
                                                         /*data=*/{1, 1, 1})));
    inputs_.push_back(
        std::make_pair("input2", CreateTfTensor<int32_t>(/*shape=*/{1, 3},
                                                         /*data=*/{2, 2, 2})));

    inputs_.push_back(
        std::make_pair("input3", CreateTfTensor<int32_t>(/*shape=*/{1, 3},
                                                         /*data=*/{3, 3, 3})));
  }

  std::unique_ptr<tensorflow::tfrt_stub::Runtime> runtime_;
  std::unique_ptr<SavedModelImpl> saved_model_;
  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs_;

  std::vector<std::string> output_tensor_names_{"result1", "result21",
                                                "result31"};
  std::vector<std::string> target_node_names_{"result22", "result32"};
};

TEST_F(SavedModelRunByTensorNamesTest, Basic) {
  std::vector<tensorflow::Tensor> outputs;

  TF_ASSERT_OK(saved_model_->RunByTensorNames(/*run_options=*/{}, inputs_,
                                              output_tensor_names_,
                                              target_node_names_, &outputs));

  ASSERT_EQ(outputs.size(), 3);

  // Check output "r1".
  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[0]), ::testing::ElementsAre(6));
  // Check output "r21".
  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[1]), ::testing::ElementsAre(12));
  // Check output "r31".
  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[2]), ::testing::ElementsAre(18));
}

TEST_F(SavedModelRunByTensorNamesTest, NoTargetNodes) {
  std::vector<tensorflow::Tensor> outputs;
  TF_ASSERT_OK(saved_model_->RunByTensorNames(
      /*run_options=*/{}, inputs_, output_tensor_names_,
      /*target_node_names=*/{}, &outputs));

  ASSERT_EQ(outputs.size(), 3);

  // Check output "r1".
  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[0]), ::testing::ElementsAre(6));
  // Check output "r21".
  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[1]), ::testing::ElementsAre(12));
  // Check output "r31".
  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[2]), ::testing::ElementsAre(18));
}

TEST_F(SavedModelRunByTensorNamesTest, NoOutputNodes) {
  std::vector<tensorflow::Tensor> outputs;
  outputs.emplace_back();  // Test outputs is first cleared.
  TF_ASSERT_OK(saved_model_->RunByTensorNames(
      /*run_options=*/{}, inputs_, /*output_tensor_names=*/{},
      target_node_names_, &outputs));
  ASSERT_EQ(outputs.size(), 0);
}

TEST_F(SavedModelRunByTensorNamesTest, ShuffleInputsAndOutputs) {
  std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
      {"input2", CreateTfTensor<int32_t>(/*shape=*/{1, 3}, /*data=*/{4, 4, 4})},
      {"input1", CreateTfTensor<int32_t>(/*shape=*/{1, 3}, /*data=*/{1, 1, 1})},
      {"input3", CreateTfTensor<int32_t>(/*shape=*/{1, 3}, /*data=*/{3, 3, 3})},
  };
  std::vector<std::string> output_tensor_names{"result22", "result1",
                                               "result31"};

  std::vector<tensorflow::Tensor> outputs;
  TF_ASSERT_OK(saved_model_->RunByTensorNames(
      /*run_options=*/{}, inputs, output_tensor_names, {}, &outputs));

  ASSERT_EQ(outputs.size(), 3);

  // Check output "r22".
  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[0]), ::testing::ElementsAre(30));
  // Check output "r1".
  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[1]), ::testing::ElementsAre(6));
  // Check output "r31".
  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[2]), ::testing::ElementsAre(18));
}

TEST(SavedModelTest, CustomWorkQueue) {
  // SavedModel toy contains a graph of a single 'tf.AddV2' op. It is generated
  // using the following python code:
  //  x = tf.placeholder(tf.int32, shape=(3))
  //  y = tf.compat.v1.get_variable(name='y', initializer=[1, 2, 3])
  //  r = tf.matmul(x, y)
  std::string saved_model_dir = tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/toy_v1");

  tfrt::tf::RunHandlerThreadWorkQueue::Options queue_options;
  queue_options.num_complementary_threads = 1;
  queue_options.num_main_threads = 1;
  queue_options.init_timeout_ms = 100;

  auto runtime = tensorflow::tfrt_stub::Runtime::Create(
      std::make_unique<tfrt::tf::RunHandlerThreadWorkQueue>(queue_options));

  auto options = DefaultSavedModelOptions(runtime.get());

  auto saved_model = SavedModelImpl::LoadSavedModel(options, saved_model_dir,
                                                    /*tags=*/{"serve"});
  TF_CHECK_OK(saved_model.status());

  // Set input 'x' to [[1, 1, 1]]
  std::vector<tensorflow::Tensor> inputs;
  inputs.push_back(
      CreateTfTensor<int32_t>(/*shape=*/{1, 3}, /*data=*/{1, 1, 1}));

  std::vector<tensorflow::Tensor> outputs;
  TF_ASSERT_OK((*saved_model)->Run({}, "toy", inputs, &outputs));
  ASSERT_EQ(outputs.size(), 1);

  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[0]),
              ::testing::ElementsAreArray({6}));

  // Run one more time to check per-request state is correct set up.
  outputs.clear();
  TF_ASSERT_OK((*saved_model)->Run({}, "toy", inputs, &outputs));
  ASSERT_EQ(outputs.size(), 1);

  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[0]),
              ::testing::ElementsAreArray({6}));
}

// Verifies the savedmodel runs correctly with work queues specified in
// RunOptions.
TEST(SavedModelTest, RunOptionsWorkQueue) {
  // SavedModel toy contains a graph of a single 'tf.AddV2' op. It is generated
  // using the following python code:
  //  x = tf.placeholder(tf.int32, shape=(3))
  //  y = tf.compat.v1.get_variable(name='y', initializer=[1, 2, 3])
  //  r = tf.matmul(x, y)
  std::string saved_model_dir = tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/toy_v1");

  auto runtime =
      tensorflow::tfrt_stub::Runtime::Create(/*num_inter_op_threads=*/4);

  auto options = DefaultSavedModelOptions(runtime.get());

  auto saved_model = SavedModelImpl::LoadSavedModel(options, saved_model_dir,
                                                    /*tags=*/{"serve"});
  TF_CHECK_OK(saved_model.status());

  // Set input 'x' to [[1, 1, 1]]
  std::vector<tensorflow::Tensor> inputs;
  inputs.push_back(
      CreateTfTensor<int32_t>(/*shape=*/{1, 3}, /*data=*/{1, 1, 1}));

  std::vector<tensorflow::Tensor> outputs;

  tfrt::tf::RunHandlerThreadWorkQueue::Options queue_options;
  queue_options.num_complementary_threads = 1;
  queue_options.num_main_threads = 1;
  queue_options.init_timeout_ms = 100;

  tfrt::tf::RunHandlerThreadWorkQueue run_handler_queue(queue_options);

  tfrt::SavedModel::RunOptions run_options;
  run_options.work_queue = &run_handler_queue;

  TF_ASSERT_OK((*saved_model)->Run(run_options, "toy", inputs, &outputs));
  ASSERT_EQ(outputs.size(), 1);

  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[0]),
              ::testing::ElementsAreArray({6}));

  // Run one more time to check per-request state is correct set up.
  outputs.clear();
  TF_ASSERT_OK((*saved_model)->Run(run_options, "toy", inputs, &outputs));
  ASSERT_EQ(outputs.size(), 1);

  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[0]),
              ::testing::ElementsAreArray({6}));
}

TEST(SavedModelTest, UseMira) {
  // SavedModel toy contains a graph of a single 'tf.AddV2' op. It is generated
  // using the following python code:
  //  x = tf.placeholder(tf.int32, shape=(3))
  //  y = tf.compat.v1.get_variable(name='y', initializer=[1, 2, 3])
  //  r = tf.matmul(x, y)
  std::string saved_model_dir = tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/toy_v1");

  auto runtime = DefaultTfrtRuntime(/*num_threads=*/1);
  auto options = DefaultSavedModelOptions(runtime.get());

  auto saved_model =
      SavedModelMiraImpl::LoadSavedModel(options, saved_model_dir,
                                         /*tags=*/{"serve"});
  TF_CHECK_OK(saved_model.status());

  // Set input 'x' to [[1, 1, 1]]
  std::vector<tensorflow::Tensor> inputs;
  inputs.push_back(
      CreateTfTensor<int32_t>(/*shape=*/{1, 3}, /*data=*/{1, 1, 1}));

  tfrt::SavedModel::RunOptions run_options;

  std::vector<tensorflow::Tensor> outputs;
  TF_ASSERT_OK((*saved_model)->Run(run_options, "toy", inputs, &outputs));
  ASSERT_EQ(outputs.size(), 1);

  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[0]),
              ::testing::ElementsAreArray({6}));
}

TEST(SavedModelTest, FunctionMetadata) {
  // SavedModel toy contains a graph of a single 'tf.AddV2' op. It is generated
  // using the following python code:
  //  x = tf.placeholder(tf.int32, shape=(3))
  //  y = tf.compat.v1.get_variable(name='y', initializer=[1, 2, 3])
  //  r = tf.matmul(x, y)
  std::string saved_model_dir = tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/toy_v1");

  TFRTSavedModelTest test(saved_model_dir);
  auto* saved_model = test.GetSavedModel();

  auto function_metadata = saved_model->GetFunctionMetadata("toy");
  ASSERT_TRUE(function_metadata.has_value());

  EXPECT_THAT(function_metadata->GetInputNames(),
              ::testing::ElementsAreArray({"x1"}));
  EXPECT_THAT(
      function_metadata->GetInputSpecs(),
      ::testing::ElementsAreArray({TensorSpec(tensorflow::DT_INT32, {1, 3})}));

  EXPECT_THAT(function_metadata->GetOutputNames(),
              ::testing::ElementsAreArray({"r1"}));
  EXPECT_THAT(function_metadata->GetOutputSpecs(),
              // Shape inference disabled, thus we only match dtype.
              ::testing::ElementsAreArray({::testing::Field(
                  &TensorSpec::dtype, tensorflow::DT_INT32)}));
}

TEST(SavedModelTest, WrongShape) {
  // SavedModel toy contains a graph of a single 'tf.AddV2' op. It is generated
  // using the following python code:
  // self.w = tf.Variable(tf.ones((3)), name='w')
  // r = tf.matmul(x, self.w)
  std::string saved_model_dir = tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/toy_v2");

  TFRTSavedModelTest test(saved_model_dir);

  // Set input 'x' to a wrong shape [[1, 1]]
  std::vector<tensorflow::Tensor> inputs;
  inputs.push_back(CreateTfTensor<int32_t>(/*shape=*/{1, 2}, /*data=*/{1, 1}));

  std::vector<tensorflow::Tensor> outputs;

  tfrt::SavedModel::RunOptions run_options;
  run_options.validate_input_specs = true;

  auto status = test.GetSavedModel()->Run(run_options, "serving_default",
                                          inputs, &outputs);
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(),
              ::testing::HasSubstr("input shape is wrong"));
}

TEST(SavedModelTest, RefTypeTensorInput) {
  // This test checks the loading does not fail for signatures with ref type
  // input/output.
  //
  // TODO(b/188580685): This is a short term workaround to skip signatures with
  // ref type input. We need to add correctness testing here for ref type inputs
  // once it is supported.
  std::string saved_model_dir = tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/ref_type_tensor_input");

  auto runtime = DefaultTfrtRuntime(/*num_threads=*/1);
  auto options = DefaultSavedModelOptions(runtime.get());
  options.graph_execution_options.compile_options.enable_grappler = true;

  auto saved_model = SavedModelImpl::LoadSavedModel(options, saved_model_dir,
                                                    /*tags=*/{"serve"});
  TF_ASSERT_OK(saved_model.status());
  EXPECT_THAT(
      (*saved_model)->GetFunctionNames(),
      ::testing::UnorderedElementsAre(
          "non_ref", "__tf_saved_model_session_initializer_save/restore_all"));
}

TEST(SavedModelTest, HashTableAssetV1) {
  std::string saved_model_dir = tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/"
      "hash_table_asset_v1");

  auto runtime = DefaultTfrtRuntime(/*num_threads=*/1);
  auto options = DefaultSavedModelOptions(runtime.get());
  options.graph_execution_options.compile_options.enable_grappler = true;
  options.graph_execution_options.compile_options.hoist_invariant_ops = true;

  auto saved_model = SavedModelImpl::LoadSavedModel(options, saved_model_dir,
                                                    /*tags=*/{"serve"});
  TF_CHECK_OK(saved_model.status());

  std::vector<tensorflow::Tensor> inputs;
  inputs.push_back(CreateTfStringTensor(/*shape=*/{}, /*data=*/{"cat"}));

  std::vector<tensorflow::Tensor> outputs;
  TF_ASSERT_OK((*saved_model)->Run({}, "serving_default", inputs, &outputs));
  ASSERT_EQ(outputs.size(), 1);

  EXPECT_THAT(GetTfTensorData<int64_t>(outputs[0]),
              ::testing::ElementsAreArray({0}));
}

TEST(ControlFlowTest, CtrlFlow) {
  std::string saved_model_dir = tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/if_v1");

  TFRTSavedModelTest test(saved_model_dir);

  std::vector<int32_t> x_data = {-1};
  std::vector<tensorflow::Tensor> inputs;
  inputs.push_back(CreateTfTensor<int32_t>(
      /*shape=*/{}, absl::MakeConstSpan(x_data)));

  std::vector<tensorflow::Tensor> outputs;
  TF_ASSERT_OK(
      test.GetSavedModel()->Run({}, "serving_default", inputs, &outputs));
  ASSERT_EQ(outputs.size(), 1);

  auto function_metadata =
      test.GetSavedModel()->GetFunctionMetadata("serving_default");
  ASSERT_TRUE(function_metadata.has_value());

  tensorflow::Tensor x(tensorflow::DT_INT32, tensorflow::TensorShape({}));
  std::copy(std::begin(x_data), std::end(x_data), x.flat<int32_t>().data());
  std::vector<tensorflow::Tensor> tf_inputs = {x};
  std::vector<tensorflow::Tensor> tf_outputs;
  ComputeCurrentTFResult(saved_model_dir, /*signature_name=*/"serving_default",
                         function_metadata->GetInputNames(), tf_inputs,
                         function_metadata->GetOutputNames(), &tf_outputs);
  ASSERT_EQ(tf_outputs.size(), 1);

  EXPECT_THAT(
      GetTfTensorData<int32_t>(outputs[0]),
      ::testing::ElementsAreArray(std::vector<int32_t>(
          tf_outputs[0].flat<int32_t>().data(),
          tf_outputs[0].flat<int32_t>().data() + tf_outputs[0].NumElements())));
}

TEST(SavedModelTest, ResourceGather) {
  std::string saved_model_dir = tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/resource_gather_v1");

  TFRTSavedModelTest test(saved_model_dir);

  std::vector<int32_t> x_data = {1};
  std::vector<tensorflow::Tensor> inputs;
  inputs.push_back(CreateTfTensor<int32_t>(
      /*shape=*/{}, absl::MakeConstSpan(x_data)));

  std::vector<tensorflow::Tensor> outputs;
  TF_ASSERT_OK(
      test.GetSavedModel()->Run({}, "serving_default", inputs, &outputs));
  ASSERT_EQ(outputs.size(), 1);

  auto function_metadata =
      test.GetSavedModel()->GetFunctionMetadata("serving_default");
  ASSERT_TRUE(function_metadata.has_value());

  tensorflow::Tensor x(tensorflow::DT_INT32, tensorflow::TensorShape({}));
  std::copy(std::begin(x_data), std::end(x_data), x.flat<int32_t>().data());
  std::vector<tensorflow::Tensor> tf_inputs = {x};
  std::vector<tensorflow::Tensor> tf_outputs;
  ComputeCurrentTFResult(saved_model_dir, /*signature_name=*/"serving_default",
                         function_metadata->GetInputNames(), tf_inputs,
                         function_metadata->GetOutputNames(), &tf_outputs);
  ASSERT_EQ(tf_outputs.size(), 1);

  EXPECT_THAT(
      GetTfTensorData<int32_t>(outputs[0]),
      ::testing::ElementsAreArray(std::vector<int32_t>(
          tf_outputs[0].flat<int32_t>().data(),
          tf_outputs[0].flat<int32_t>().data() + tf_outputs[0].NumElements())));
}

TEST(SavedModelTest, DTypeCoverage) {
  std::string saved_model_dir = tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/dtype_coverage_v1");

  TFRTSavedModelTest test(saved_model_dir);

  std::vector<tensorflow::Tensor> inputs;
  std::vector<tensorflow::Tensor> outputs;
  TF_ASSERT_OK(
      test.GetSavedModel()->Run({}, "serving_default", inputs, &outputs));
  ASSERT_EQ(outputs.size(), 16);

  auto function_metadata =
      test.GetSavedModel()->GetFunctionMetadata("serving_default");
  ASSERT_TRUE(function_metadata.has_value());

  std::vector<tensorflow::Tensor> tf_inputs;
  std::vector<tensorflow::Tensor> tf_outputs;
  ComputeCurrentTFResult(saved_model_dir, /*signature_name=*/"serving_default",
                         function_metadata->GetInputNames(), tf_inputs,
                         function_metadata->GetOutputNames(), &tf_outputs);
  ASSERT_EQ(tf_outputs.size(), 16);

  for (int i = 0; i < 16; ++i) {
    ExpectTensorEqual(outputs[i], tf_outputs[i]);
  }
}

TEST(SavedModelTest, Error) {
  std::string saved_model_dir = tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/error_v1");

  auto runtime = DefaultTfrtRuntime(/*num_threads=*/1);
  auto options = DefaultSavedModelOptions(runtime.get());

  auto saved_model = SavedModelImpl::LoadSavedModel(options, saved_model_dir,
                                                    /*tags=*/{"serve"});
  TF_ASSERT_OK(saved_model.status());

  std::vector<tensorflow::Tensor> outputs;
  auto status = (*saved_model)->Run({}, "serving_default", {}, &outputs);

  ASSERT_FALSE(status.ok());

  EXPECT_EQ(status.code(), tensorflow::error::INVALID_ARGUMENT);

  EXPECT_TRUE(absl::StrContains(
      status.error_message(), "You must feed a value for placeholder tensor"));
}

struct PowTestParam {
  std::string path;
  bool run_placer_grappler_on_functions;
};

class SavedModelPowTest : public ::testing::TestWithParam<PowTestParam> {};

TEST_P(SavedModelPowTest, Pow) {
  std::string saved_model_dir =
      tensorflow::GetDataDependencyFilepath(GetParam().path);

  auto runtime = DefaultTfrtRuntime(/*num_threads=*/1);
  auto options = DefaultSavedModelOptions(runtime.get());
  options.graph_execution_options.compile_options.enable_grappler = true;
  options.graph_execution_options.enable_grappler_function_optimizer = true;
  options.graph_execution_options.run_placer_grappler_on_functions =
      GetParam().run_placer_grappler_on_functions;

  auto saved_model = SavedModelImpl::LoadSavedModel(options, saved_model_dir,
                                                    /*tags=*/{"serve"});
  TF_CHECK_OK(saved_model.status());

  std::vector<int32_t> data = {2};
  std::vector<tensorflow::Tensor> inputs;
  inputs.push_back(
      CreateTfTensor<int32_t>(/*shape=*/{}, absl::MakeConstSpan(data)));

  std::vector<tensorflow::Tensor> outputs;
  TF_ASSERT_OK((*saved_model)->Run({}, "serving_default", inputs, &outputs));
  ASSERT_EQ(outputs.size(), 1);

  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[0]), ::testing::ElementsAre(8));
}

INSTANTIATE_TEST_SUITE_P(
    SavedModelPowTest, SavedModelPowTest,
    ::testing::Values(
        PowTestParam{"tensorflow/core/tfrt/saved_model/tests/pow", false},
        PowTestParam{"tensorflow/core/tfrt/saved_model/tests/pow_v2", false},
        PowTestParam{"tensorflow/core/tfrt/saved_model/tests/pow_v2", true}));

TEST(SavedModelPowTest, MapDataset) {
  std::string saved_model_dir = tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/data");

  auto runtime = DefaultTfrtRuntime(/*num_threads=*/1);
  auto options = DefaultSavedModelOptions(runtime.get());
  options.graph_execution_options.compile_options.enable_grappler = true;

  auto saved_model = SavedModelImpl::LoadSavedModel(options, saved_model_dir,
                                                    /*tags=*/{"serve"});
  TF_CHECK_OK(saved_model.status());

  std::vector<int32_t> data = {2};
  std::vector<tensorflow::Tensor> inputs;
  inputs.push_back(
      CreateTfTensor<int32_t>(/*shape=*/{}, absl::MakeConstSpan(data)));

  std::vector<tensorflow::Tensor> outputs;
  TF_ASSERT_OK((*saved_model)->Run({}, "serving_default", inputs, &outputs));
  ASSERT_EQ(outputs.size(), 1);

  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[0]), ::testing::ElementsAre(3));
}

TEST(SavedModelTest, ControlFlowV1) {
  // This test checks that loading a savedmodel with V1 control flows works
  // properly. The current workflow requires functionalization on V1 control
  // flows and may insert extra functions. This test is to guard on errors due
  // to handling V1 control flows (eg. adding different functions with name
  // conflicts).
  std::string saved_model_dir = tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/control_flow_v1");

  auto runtime = DefaultTfrtRuntime(/*num_threads=*/1);
  auto options = DefaultSavedModelOptions(runtime.get());
  options.graph_execution_options.compile_options.enable_grappler = true;

  auto saved_model = SavedModelImpl::LoadSavedModel(options, saved_model_dir,
                                                    /*tags=*/{"serve"});
  TF_ASSERT_OK(saved_model.status());
}

TEST(SavedModelTest, WhileLoopV1) {
  // This test checks that loading a savedmodel with V1 while works properly.
  // The current workflow applies functionalization which may change nodes in
  // the original graph. We insert additional nodes to prevent it from changing
  // fetch nodes.
  std::string saved_model_dir = tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/while_v1");

  auto runtime = DefaultTfrtRuntime(/*num_threads=*/1);
  auto options = DefaultSavedModelOptions(runtime.get());
  options.graph_execution_options.compile_options.enable_grappler = true;

  // TODO(chky): Implement while op in MLRT.
  if (options.graph_execution_options.enable_mlrt) return;

  auto saved_model = SavedModelImpl::LoadSavedModel(options, saved_model_dir,
                                                    /*tags=*/{"serve"});
  TF_ASSERT_OK(saved_model.status());

  std::vector<int32_t> data = {0};
  std::vector<tensorflow::Tensor> inputs;
  inputs.push_back(
      CreateTfTensor<int32_t>(/*shape=*/{}, absl::MakeConstSpan(data)));

  std::vector<tensorflow::Tensor> outputs;
  TF_ASSERT_OK((*saved_model)->Run({}, "serving_default", inputs, &outputs));
  ASSERT_EQ(outputs.size(), 1);

  EXPECT_THAT(GetTfTensorData<int32_t>(outputs[0]), ::testing::ElementsAre(10));
}

TEST(SavedModelTest, SparseTensorInput) {
  // This test checks the loading does not fail for signatures with sparse
  // input/output.
  //
  // TODO(b/184675681): This is a short term workaround to skip signatures with
  // sparse input. We need to add correctness testing here for sparse inputs
  // once it is supported.
  std::string saved_model_dir = tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/sparse_tensor_input");

  auto runtime = DefaultTfrtRuntime(/*num_threads=*/1);
  auto options = DefaultSavedModelOptions(runtime.get());
  options.graph_execution_options.compile_options.enable_grappler = true;

  auto saved_model = SavedModelImpl::LoadSavedModel(options, saved_model_dir,
                                                    /*tags=*/{"serve"});
  TF_ASSERT_OK(saved_model.status());
  EXPECT_THAT((*saved_model)->GetFunctionNames(),
              ::testing::ElementsAre("dense"));
}

TEST(SavedModelTest, DeadlineExceeded) {
  // SavedModel toy contains a graph of a single 'tf.AddV2' op. It is generated
  // using the following python code:
  //  x = tf.placeholder(tf.int32, shape=(3))
  //  y = tf.compat.v1.get_variable(name='y', initializer=[1, 2, 3])
  //  r = tf.matmul(x, y)
  std::string saved_model_dir = tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/toy_v1");

  auto runtime = DefaultTfrtRuntime(/*num_threads=*/1);
  auto options = DefaultSavedModelOptions(runtime.get());

  // TODO(chky): Implement cancellation in MLRT.
  if (options.graph_execution_options.enable_mlrt) return;

  auto saved_model = SavedModelImpl::LoadSavedModel(options, saved_model_dir,
                                                    /*tags=*/{"serve"});
  TF_CHECK_OK(saved_model.status());

  // Set input 'x' to [[1, 1, 1]]
  std::vector<tensorflow::Tensor> inputs;
  inputs.push_back(
      CreateTfTensor<int32_t>(/*shape=*/{1, 3}, /*data=*/{1, 1, 1}));

  std::vector<tensorflow::Tensor> outputs;

  tfrt::SavedModel::RunOptions run_options;
  run_options.deadline = absl::ToChronoTime(absl::Now());

  auto status = (*saved_model)->Run(run_options, "toy", inputs, &outputs);

  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(),
              ::testing::HasSubstr("Deadline exceeded"));
}

TEST(SavedModelTest, DisableCompilation) {
  // SavedModel toy contains a graph of a single 'tf.AddV2' op. It is generated
  // using the following python code:
  //  x = tf.placeholder(tf.int32, shape=(3))
  //  y = tf.compat.v1.get_variable(name='y', initializer=[1, 2, 3])
  //  r = tf.matmul(x, y)
  std::string saved_model_dir = tensorflow::GetDataDependencyFilepath(
      "tensorflow/core/tfrt/saved_model/tests/toy_v1");

  auto runtime = DefaultTfrtRuntime(/*num_threads=*/1);
  auto options = DefaultSavedModelOptions(runtime.get());
  options.enable_lazy_loading = true;

  auto saved_model = SavedModelImpl::LoadSavedModel(options, saved_model_dir,
                                                    /*tags=*/{"serve"});
  TF_CHECK_OK(saved_model.status());

  // Set input 'x' to [[1, 1, 1]]
  std::vector<tensorflow::Tensor> inputs;
  inputs.push_back(
      CreateTfTensor<int32_t>(/*shape=*/{1, 3}, /*data=*/{1, 1, 1}));

  std::vector<tensorflow::Tensor> outputs;

  tfrt::SavedModel::RunOptions run_options;
  run_options.disable_compilation = true;

  auto status = (*saved_model)->Run(run_options, "toy", inputs, &outputs);

  ASSERT_FALSE(status.ok());
  EXPECT_THAT(
      status.error_message(),
      ::testing::HasSubstr("GraphExecutor: compilation is disabled in "
                           "execution but the compiled graph is not found"));

  run_options.disable_compilation = false;
  TF_ASSERT_OK((*saved_model)->Run(run_options, "toy", inputs, &outputs));
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace tensorflow
