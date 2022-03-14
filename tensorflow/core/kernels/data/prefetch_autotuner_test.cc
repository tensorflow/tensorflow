/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/data/prefetch_autotuner.h"

#include <memory>
#include <vector>

#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {

using model::Model;

namespace {

class PrefetchAutotunerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Builds a model where a `ParallelMap` node feeds into a `Prefetch` node.
    std::shared_ptr<mutex> mutex1 = std::make_shared<mutex>();
    std::shared_ptr<condition_variable> cv1 =
        std::make_shared<condition_variable>();
    std::shared_ptr<model::Node> node1 = model::MakeAsyncKnownRatioNode(
        {/*id=*/1, /*name=*/"1", nullptr}, /*ratio=*/1,
        {model::MakeParameter("parallelism",
                              std::make_shared<model::SharedState>(
                                  /*value=*/1, mutex1, cv1),
                              /*min=*/1, /*max=*/5)});
    // Sets values s.t. ram usage for this node is 20
    node1->record_buffer_event(200, 10);

    std::shared_ptr<mutex> mutex2 = std::make_shared<mutex>();
    std::shared_ptr<condition_variable> cv2 =
        std::make_shared<condition_variable>();
    prefetch_buffer_ = std::make_shared<model::SharedState>(
        /*value=*/1, mutex2, cv2);
    prefetch_node_ = model::MakeAsyncKnownRatioNode(
        {/*id=*/2, /*name=*/"2", node1}, /*ratio=*/1,
        {model::MakeParameter("buffer_size", prefetch_buffer_,
                              /*min=*/0, /*max=*/10)});
    // Sets values s.t. ram usage for this node is 10
    prefetch_node_->record_buffer_event(100, 10);

    model_ =
        std::make_shared<model::Model>(model::Model::BudgetParams({100, 120}));
    model_->AddNode(
        [this](model::Node::Args args) { return this->prefetch_node_; }, "2",
        nullptr, &prefetch_node_);
    model_->AddNode([&node1](model::Node::Args args) { return node1; }, "1",
                    prefetch_node_, &node1);

    // Updates the current cached ram usage.
    model_->Optimize(model::AutotuneAlgorithm::HILL_CLIMB, 0,
                     &cancellation_manager_);
  }
  void TearDown() override { autotuner_.reset(); }

  void InitAutotuner(int64_t initial_buffer_size, int64_t buffer_size_min) {
    autotuner_ = std::make_unique<PrefetchAutotuner>(
        model_, prefetch_node_, initial_buffer_size, buffer_size_min);
  }

  void RecordConsumption(size_t current_buffer_size) {
    autotuner_->RecordConsumption(current_buffer_size);
    prefetch_buffer_->value = autotuner_->buffer_limit();
    // Updates the current cached ram usage.
    model_->Optimize(model::AutotuneAlgorithm::HILL_CLIMB, 0,
                     &cancellation_manager_);
  }

  int64_t GetBufferLimit() { return autotuner_->buffer_limit(); }

  std::unique_ptr<PrefetchAutotuner> autotuner_;
  std::shared_ptr<model::Node> prefetch_node_;
  std::shared_ptr<model::SharedState> prefetch_buffer_;
  CancellationManager cancellation_manager_;
  std::shared_ptr<model::Model> model_;
};

TEST_F(PrefetchAutotunerTest, Disabled) {
  InitAutotuner(2, 0);
  EXPECT_EQ(2, GetBufferLimit());
  RecordConsumption(0);
  RecordConsumption(2);
  RecordConsumption(0);
  RecordConsumption(2);
  EXPECT_EQ(2, GetBufferLimit());
}

TEST_F(PrefetchAutotunerTest, Enabled) {
  InitAutotuner(model::kAutotune, 0);
  EXPECT_EQ(1, GetBufferLimit());
  RecordConsumption(0);  // Expect buffer limit to stay the same.
  EXPECT_EQ(1, GetBufferLimit());
  RecordConsumption(1);
  EXPECT_EQ(1, GetBufferLimit());
  RecordConsumption(0);  // Expect buffer limit to increase.
  EXPECT_EQ(2, GetBufferLimit());
  RecordConsumption(2);
  EXPECT_EQ(2, GetBufferLimit());
  RecordConsumption(1);
  EXPECT_EQ(2, GetBufferLimit());
  RecordConsumption(0);  // Expect buffer limit to increase.
  EXPECT_EQ(4, GetBufferLimit());
  RecordConsumption(4);
  EXPECT_EQ(4, GetBufferLimit());
  RecordConsumption(0);  // Expect buffer limit to increase.
  EXPECT_EQ(8, GetBufferLimit());
  RecordConsumption(0);  // Expect buffer limit to stay the same!
  EXPECT_EQ(8, GetBufferLimit());
  RecordConsumption(0);  // Expect buffer limit to stay the same!
  EXPECT_EQ(8, GetBufferLimit());
}

TEST_F(PrefetchAutotunerTest, EnabledSteady) {
  InitAutotuner(model::kAutotune, 0);
  EXPECT_EQ(1, GetBufferLimit());
  RecordConsumption(0);  // Expect buffer limit to stay the same!
  EXPECT_EQ(1, GetBufferLimit());
  RecordConsumption(1);
  EXPECT_EQ(1, GetBufferLimit());
  RecordConsumption(0);  // Expect buffer limit to increase.
  EXPECT_EQ(2, GetBufferLimit());
  RecordConsumption(2);
  EXPECT_EQ(2, GetBufferLimit());
  RecordConsumption(0);  // Expect buffer limit to increase.
  EXPECT_EQ(4, GetBufferLimit());

  // Never reach zero again.
  std::vector<size_t> consumption_values = {2, 3, 1, 4, 1, 2, 3, 1};
  for (int i = 0; i < consumption_values.size(); ++i) {
    RecordConsumption(consumption_values[i]);
    EXPECT_EQ(4, GetBufferLimit())
        << "Failed at index " << i << " with value: " << consumption_values[i];
  }
}

TEST_F(PrefetchAutotunerTest, StartWithMin) {
  InitAutotuner(model::kAutotune, 2);
  EXPECT_EQ(2, GetBufferLimit());
  RecordConsumption(0);  // Expect buffer limit to stay the same!
  EXPECT_EQ(2, GetBufferLimit());
  RecordConsumption(2);  // Expect buffer limit to stay the same!
  EXPECT_EQ(2, GetBufferLimit());
  RecordConsumption(0);  // Expect buffer limit to increase.
  EXPECT_EQ(4, GetBufferLimit());
  RecordConsumption(4);  // Expect buffer limit to stay the same!
  EXPECT_EQ(4, GetBufferLimit());
  RecordConsumption(0);  // Expect buffer limit to increase.
  EXPECT_EQ(8, GetBufferLimit());

  // Never reach zero again.
  std::vector<size_t> consumption_values = {3, 5, 7, 1, 4, 6, 8, 3, 5, 1, 2, 4};
  for (int i = 0; i < consumption_values.size(); ++i) {
    RecordConsumption(consumption_values[i]);
    EXPECT_EQ(8, GetBufferLimit())
        << "Failed at index " << i << " with value: " << consumption_values[i];
  }
}

TEST_F(PrefetchAutotunerTest, MemoryConsumption) {
  InitAutotuner(model::kAutotune, 0);
  EXPECT_EQ(1, GetBufferLimit());
  RecordConsumption(1);
  EXPECT_EQ(1, GetBufferLimit());
  RecordConsumption(0);  // Expect buffer limit to increase.
  EXPECT_EQ(2, GetBufferLimit());
  RecordConsumption(2);
  EXPECT_EQ(2, GetBufferLimit());
  RecordConsumption(0);  // Expect buffer limit to increase.
  EXPECT_EQ(4, GetBufferLimit());
  RecordConsumption(4);
  EXPECT_EQ(4, GetBufferLimit());
  RecordConsumption(0);  // Expect buffer limit to increase.
  EXPECT_EQ(8, GetBufferLimit());
  RecordConsumption(8);
  EXPECT_EQ(8, GetBufferLimit());
  RecordConsumption(0);  // Expect buffer limit to stay the same! Fail
                         // memory consumption check.
  EXPECT_EQ(8, GetBufferLimit());
  RecordConsumption(8);
  EXPECT_EQ(8, GetBufferLimit());
  RecordConsumption(0);  // Expect buffer limit to stay the same! Fail
                         // memory consumption check.
  EXPECT_EQ(8, GetBufferLimit());
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
