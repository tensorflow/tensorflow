/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/model.h"
#include <memory>

#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace model {
namespace {

class AsyncInterleaveManyTest
    : public ::testing::TestWithParam<std::tuple<int64, int64>> {};

TEST_P(AsyncInterleaveManyTest, Model) {
  const int64 parallelism = std::get<0>(GetParam());
  const int64 input_time = std::get<1>(GetParam());
  std::shared_ptr<Node> async_interleave_many =
      model::MakeAsyncInterleaveManyNode(
          {0, "async_interleave_many", nullptr},
          {model::MakeParameter(
              "parallelism",
              std::make_shared<SharedState>(parallelism, nullptr, nullptr), 1,
              parallelism)});
  std::shared_ptr<Node> meta_source =
      model::MakeSourceNode({1, "meta_source", async_interleave_many});
  async_interleave_many->add_input(meta_source);
  auto cleanup_meta = gtl::MakeCleanup([async_interleave_many, meta_source]() {
    async_interleave_many->remove_input(meta_source);
  });
  std::shared_ptr<Node> source1 =
      model::MakeSourceNode({1, "source1", async_interleave_many});
  async_interleave_many->add_input(source1);
  auto cleanup1 = gtl::MakeCleanup([async_interleave_many, source1]() {
    async_interleave_many->remove_input(source1);
  });
  std::shared_ptr<Node> source2 =
      model::MakeSourceNode({2, "source2", async_interleave_many});
  async_interleave_many->add_input(source2);
  auto cleanup2 = gtl::MakeCleanup([async_interleave_many, source2]() {
    async_interleave_many->remove_input(source2);
  });
  std::vector<int64> input_times(1, input_time);
  async_interleave_many->add_processing_time(100);
  EXPECT_EQ(async_interleave_many->processing_time(), 100);
  EXPECT_EQ(async_interleave_many->ProcessingTime(), 0);
  EXPECT_EQ(async_interleave_many->OutputTime(&input_times), 0);
  async_interleave_many->record_element();
  EXPECT_EQ(async_interleave_many->num_elements(), 1);
  EXPECT_EQ(async_interleave_many->ProcessingTime(), 100);
  EXPECT_LE(async_interleave_many->OutputTime(&input_times), 100);
  EXPECT_GE(async_interleave_many->OutputTime(&input_times), 0);
  source1->add_processing_time(200);
  source2->add_processing_time(300);
  EXPECT_EQ(async_interleave_many->ProcessingTime(), 100);
  EXPECT_LE(async_interleave_many->OutputTime(&input_times), 100);
  EXPECT_GE(async_interleave_many->OutputTime(&input_times), 0);
  source1->record_element();
  source2->record_element();
  EXPECT_EQ(async_interleave_many->ProcessingTime(), 100 + 250);
  EXPECT_LE(async_interleave_many->OutputTime(&input_times),
            100 + 250 / parallelism);
  EXPECT_GE(async_interleave_many->OutputTime(&input_times), 0);
  async_interleave_many->record_element();
  EXPECT_EQ(async_interleave_many->ProcessingTime(), 50 + 250);
  EXPECT_LE(async_interleave_many->OutputTime(&input_times),
            50 + 250 / parallelism);
  EXPECT_GE(async_interleave_many->OutputTime(&input_times), 0);
}

INSTANTIATE_TEST_SUITE_P(Test, AsyncInterleaveManyTest,
                         ::testing::Combine(::testing::Values(1, 2),
                                            ::testing::Values(0, 50, 100,
                                                              200)));

class AsyncKnownRatioTest
    : public ::testing::TestWithParam<std::tuple<int64, int64, int64>> {};

TEST_P(AsyncKnownRatioTest, Model) {
  const int64 parallelism = std::get<0>(GetParam());
  const int64 input_time = std::get<1>(GetParam());
  const int64 num_inputs_per_output = std::get<2>(GetParam());
  std::shared_ptr<Node> async_known_many = model::MakeAsyncKnownRatioNode(
      {0, "async_known_many", nullptr}, num_inputs_per_output,
      {model::MakeParameter(
          "parallelism",
          std::make_shared<SharedState>(parallelism, nullptr, nullptr), 1,
          parallelism)});
  std::shared_ptr<Node> source1 =
      model::MakeSourceNode({1, "source1", async_known_many});
  async_known_many->add_input(source1);
  std::shared_ptr<Node> source2 =
      model::MakeSourceNode({2, "source2", async_known_many});
  async_known_many->add_input(source2);
  std::vector<int64> input_times(1, input_time);
  source1->add_processing_time(100);
  EXPECT_EQ(async_known_many->ProcessingTime(), 0);
  EXPECT_EQ(async_known_many->OutputTime(&input_times), 0);
  source2->add_processing_time(200);
  EXPECT_EQ(async_known_many->ProcessingTime(), 0);
  EXPECT_EQ(async_known_many->OutputTime(&input_times), 0);
  source1->record_element();
  EXPECT_EQ(async_known_many->ProcessingTime(), num_inputs_per_output * 100);
  EXPECT_LE(async_known_many->OutputTime(&input_times),
            num_inputs_per_output * 100);
  EXPECT_GE(async_known_many->OutputTime(&input_times), 0);
  source2->record_element();
  EXPECT_EQ(async_known_many->ProcessingTime(),
            num_inputs_per_output * (100 + 200));
  EXPECT_LE(async_known_many->OutputTime(&input_times),
            num_inputs_per_output * (100 + 200));
  EXPECT_GE(async_known_many->OutputTime(&input_times), 0);
  source1->record_element();
  EXPECT_EQ(async_known_many->ProcessingTime(),
            num_inputs_per_output * (50 + 200));
  EXPECT_LE(async_known_many->OutputTime(&input_times),
            num_inputs_per_output * (50 + 200));
  EXPECT_GE(async_known_many->OutputTime(&input_times), 0);
  source2->record_element();
  EXPECT_EQ(async_known_many->ProcessingTime(),
            num_inputs_per_output * (50 + 100));
  EXPECT_LE(async_known_many->OutputTime(&input_times),
            num_inputs_per_output * (50 + 100));
  EXPECT_GE(async_known_many->OutputTime(&input_times), 0);
  async_known_many->add_processing_time(128);
  EXPECT_EQ(async_known_many->ProcessingTime(),
            num_inputs_per_output * (50 + 100));
  EXPECT_LE(async_known_many->OutputTime(&input_times),
            num_inputs_per_output * (50 + 100));
  EXPECT_GE(async_known_many->OutputTime(&input_times), 0);
  async_known_many->record_element();
  EXPECT_EQ(async_known_many->ProcessingTime(),
            num_inputs_per_output * (50 + 100) + 128);
  EXPECT_LE(async_known_many->OutputTime(&input_times),
            num_inputs_per_output * (50 + 100) + 128 / parallelism);
  EXPECT_GE(async_known_many->OutputTime(&input_times), 0);
  async_known_many->record_element();
  EXPECT_EQ(async_known_many->ProcessingTime(),
            num_inputs_per_output * (50 + 100) + 64);
  EXPECT_LE(async_known_many->OutputTime(&input_times),
            num_inputs_per_output * (50 + 100) + 64 / parallelism);
  EXPECT_GE(async_known_many->OutputTime(&input_times), 0);
}

INSTANTIATE_TEST_SUITE_P(Test, AsyncKnownRatioTest,
                         ::testing::Combine(::testing::Values(1, 2, 4, 8),
                                            ::testing::Values(0, 50, 100, 200),
                                            ::testing::Values(0, 1, 2, 4)));

TEST(InterleaveManyTest, Model) {
  std::shared_ptr<Node> interleave_many =
      model::MakeInterleaveManyNode({0, "interleave_many", nullptr});
  std::shared_ptr<Node> meta_source =
      model::MakeSourceNode({1, "meta_source", interleave_many});
  interleave_many->add_input(meta_source);
  std::shared_ptr<Node> source1 =
      model::MakeSourceNode({1, "source1", interleave_many});
  interleave_many->add_input(source1);
  std::shared_ptr<Node> source2 =
      model::MakeSourceNode({2, "source2", interleave_many});
  interleave_many->add_input(source2);
  std::vector<int64> input_times(1, 0);
  interleave_many->add_processing_time(100);
  EXPECT_EQ(interleave_many->processing_time(), 100);
  EXPECT_EQ(interleave_many->ProcessingTime(), 0);
  EXPECT_EQ(interleave_many->OutputTime(&input_times), 0);
  interleave_many->record_element();
  EXPECT_EQ(interleave_many->num_elements(), 1);
  EXPECT_EQ(interleave_many->ProcessingTime(), 100);
  EXPECT_EQ(interleave_many->OutputTime(&input_times), 100);
  source1->add_processing_time(200);
  source2->add_processing_time(300);
  EXPECT_EQ(interleave_many->ProcessingTime(), 100);
  EXPECT_EQ(interleave_many->OutputTime(&input_times), 100);
  source1->record_element();
  source2->record_element();
  EXPECT_EQ(interleave_many->ProcessingTime(), 350);
  EXPECT_EQ(interleave_many->OutputTime(&input_times), 350);
  interleave_many->record_element();
  EXPECT_EQ(interleave_many->ProcessingTime(), 300);
  EXPECT_EQ(interleave_many->OutputTime(&input_times), 300);
}

class KnownRatioTest : public ::testing::TestWithParam<int64> {};

TEST_P(KnownRatioTest, Model) {
  const int64 num_inputs_per_output = GetParam();
  std::shared_ptr<Node> known_many = model::MakeKnownRatioNode(
      {0, "known_many", nullptr}, num_inputs_per_output);
  std::shared_ptr<Node> source1 =
      model::MakeSourceNode({1, "source1", known_many});
  known_many->add_input(source1);
  std::shared_ptr<Node> source2 =
      model::MakeSourceNode({2, "source2", known_many});
  known_many->add_input(source2);
  std::vector<int64> input_times(1, 0);
  source1->add_processing_time(100);
  EXPECT_EQ(known_many->ProcessingTime(), 0);
  EXPECT_EQ(known_many->OutputTime(&input_times), 0);
  source2->add_processing_time(200);
  EXPECT_EQ(known_many->ProcessingTime(), 0);
  EXPECT_EQ(known_many->OutputTime(&input_times), 0);
  source1->record_element();
  EXPECT_EQ(known_many->ProcessingTime(), num_inputs_per_output * 100);
  EXPECT_EQ(known_many->OutputTime(&input_times), num_inputs_per_output * 100);
  source2->record_element();
  EXPECT_EQ(known_many->ProcessingTime(), num_inputs_per_output * (100 + 200));
  EXPECT_EQ(known_many->OutputTime(&input_times),
            num_inputs_per_output * (100 + 200));
  source1->record_element();
  EXPECT_EQ(known_many->ProcessingTime(), num_inputs_per_output * (50 + 200));
  EXPECT_EQ(known_many->OutputTime(&input_times),
            num_inputs_per_output * (50 + 200));
  source2->record_element();
  EXPECT_EQ(known_many->ProcessingTime(), num_inputs_per_output * (50 + 100));
  EXPECT_EQ(known_many->OutputTime(&input_times),
            num_inputs_per_output * (50 + 100));
  known_many->add_processing_time(128);
  EXPECT_EQ(known_many->ProcessingTime(), num_inputs_per_output * (50 + 100));
  EXPECT_EQ(known_many->OutputTime(&input_times),
            num_inputs_per_output * (50 + 100));
  known_many->record_element();
  EXPECT_EQ(known_many->ProcessingTime(),
            num_inputs_per_output * (50 + 100) + 128);
  EXPECT_EQ(known_many->OutputTime(&input_times),
            num_inputs_per_output * (50 + 100) + 128);
  known_many->record_element();
  EXPECT_EQ(known_many->ProcessingTime(),
            num_inputs_per_output * (50 + 100) + 64);
  EXPECT_EQ(known_many->OutputTime(&input_times),
            num_inputs_per_output * (50 + 100) + 64);
}

INSTANTIATE_TEST_SUITE_P(Test, KnownRatioTest, ::testing::Values(0, 1, 2, 4));

TEST(SourceTest, Model) {
  std::shared_ptr<Node> source = model::MakeSourceNode({0, "source", nullptr});
  std::vector<int64> input_times(1, 0);
  source->add_processing_time(100);
  EXPECT_EQ(source->processing_time(), 100);
  EXPECT_EQ(source->ProcessingTime(), 0);
  EXPECT_EQ(source->OutputTime(&input_times), 0);
  source->record_element();
  EXPECT_EQ(source->num_elements(), 1);
  EXPECT_EQ(source->ProcessingTime(), 100);
  EXPECT_EQ(source->OutputTime(&input_times), 100);
  source->record_element();
  EXPECT_EQ(source->num_elements(), 2);
  EXPECT_EQ(source->ProcessingTime(), 50);
  EXPECT_EQ(source->OutputTime(&input_times), 50);
}

TEST(UnknownRatioTest, Model) {
  std::shared_ptr<Node> unknown_many =
      model::MakeUnknownRatioNode({0, "unknown_many", nullptr});
  std::shared_ptr<Node> source1 =
      model::MakeSourceNode({1, "source1", unknown_many});
  unknown_many->add_input(source1);
  std::shared_ptr<Node> source2 =
      model::MakeSourceNode({2, "source2", unknown_many});
  unknown_many->add_input(source2);
  std::vector<int64> input_times(1, 0);
  unknown_many->add_processing_time(100);
  EXPECT_EQ(unknown_many->processing_time(), 100);
  EXPECT_EQ(unknown_many->ProcessingTime(), 0);
  EXPECT_EQ(unknown_many->OutputTime(&input_times), 0);
  unknown_many->record_element();
  EXPECT_EQ(unknown_many->num_elements(), 1);
  EXPECT_EQ(unknown_many->ProcessingTime(), 100);
  EXPECT_EQ(unknown_many->OutputTime(&input_times), 100);
  source1->add_processing_time(100);
  source2->add_processing_time(200);
  EXPECT_EQ(unknown_many->ProcessingTime(), 100);
  EXPECT_EQ(unknown_many->OutputTime(&input_times), 100);
  source1->record_element();
  source2->record_element();
  EXPECT_EQ(unknown_many->ProcessingTime(), 400);
  EXPECT_EQ(unknown_many->OutputTime(&input_times), 400);
  unknown_many->record_element();
  EXPECT_EQ(unknown_many->ProcessingTime(), 200);
  EXPECT_EQ(unknown_many->OutputTime(&input_times), 200);
}

TEST(UnknownTest, Model) {
  std::shared_ptr<Node> unknown =
      model::MakeUnknownNode({0, "unknown", nullptr});
  std::shared_ptr<Node> source1 =
      model::MakeSourceNode({1, "source1", unknown});
  unknown->add_input(source1);
  std::shared_ptr<Node> source2 =
      model::MakeSourceNode({2, "source2", unknown});
  unknown->add_input(source2);
  std::vector<int64> input_times(1, 0);
  source1->add_processing_time(100);
  EXPECT_EQ(unknown->ProcessingTime(), 0);
  EXPECT_EQ(unknown->OutputTime(&input_times), 0);
  source2->add_processing_time(100);
  EXPECT_EQ(unknown->ProcessingTime(), 0);
  EXPECT_EQ(unknown->OutputTime(&input_times), 0);
  source1->record_element();
  EXPECT_EQ(unknown->ProcessingTime(), 100);
  EXPECT_EQ(unknown->OutputTime(&input_times), 100);
  source2->record_element();
  EXPECT_EQ(unknown->ProcessingTime(), 200);
  EXPECT_EQ(unknown->OutputTime(&input_times), 200);
  source1->record_element();
  EXPECT_EQ(unknown->ProcessingTime(), 150);
  EXPECT_EQ(unknown->OutputTime(&input_times), 150);
  source2->record_element();
  EXPECT_EQ(unknown->ProcessingTime(), 100);
  EXPECT_EQ(unknown->OutputTime(&input_times), 100);
  // Unknown node processing time should not affect its ProcessingTime() or
  // OutputTime().
  unknown->add_processing_time(100);
  EXPECT_EQ(unknown->processing_time(), 100);
  EXPECT_EQ(unknown->ProcessingTime(), 100);
  EXPECT_EQ(unknown->OutputTime(&input_times), 100);
  // Unknown node number of elements should not affect its ProcessingTime() or
  // OutputTime().
  unknown->record_element();
  EXPECT_EQ(unknown->num_elements(), 1);
  EXPECT_EQ(unknown->ProcessingTime(), 100);
  EXPECT_EQ(unknown->OutputTime(&input_times), 100);
}

class TestNode : public model::Node {
 public:
  using model::Node::Node;

  virtual ~TestNode() {}

 protected:
  std::shared_ptr<Node> Clone(std::shared_ptr<Node> output) const override
      SHARED_LOCKS_REQUIRED(mu_) {
    return nullptr;
  }

  int64 OutputTimeLocked(std::vector<int64>* input_times) const override
      SHARED_LOCKS_REQUIRED(mu_) {
    return 0;
  }

  int64 ProcessingTimeLocked() const override SHARED_LOCKS_REQUIRED(mu_) {
    return 0;
  }
};

TEST(SetterGetterTest, Node) {
  std::shared_ptr<TestNode> node =
      std::make_shared<TestNode>(model::Node::Args{-1, "TestNode", nullptr});
  EXPECT_EQ(node->id(), -1);
  EXPECT_EQ(node->name(), "TestNode");
  EXPECT_EQ(node->output(), nullptr);

  EXPECT_EQ(node->buffered_bytes(), 0);
  node->add_buffered_bytes(42);
  EXPECT_EQ(node->buffered_bytes(), 42);

  EXPECT_EQ(node->processing_time(), 0);
  node->record_start(1);
  EXPECT_EQ(node->processing_time(), 0);
  node->record_stop(41);
  EXPECT_EQ(node->processing_time(), 40);
  node->add_processing_time(2);
  EXPECT_EQ(node->processing_time(), 42);

  std::shared_ptr<TestNode> input =
      std::make_shared<TestNode>(model::Node::Args{-1, "TestInput", node});
  EXPECT_EQ(input->output(), node.get());
  EXPECT_EQ(node->inputs().size(), 0);
  node->add_input(input);
  EXPECT_EQ(node->inputs().size(), 1);
  EXPECT_EQ(node->inputs().front(), input);
  node->remove_input(input);
  EXPECT_EQ(node->inputs().size(), 0);

  EXPECT_EQ(node->num_elements(), 0);
  node->record_element();
  EXPECT_EQ(node->num_elements(), 1);
}

}  // namespace
}  // namespace model
}  // namespace data
}  // namespace tensorflow
