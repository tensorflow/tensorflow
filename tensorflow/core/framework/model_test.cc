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
  EXPECT_EQ(100, async_interleave_many->processing_time());
  EXPECT_EQ(0, async_interleave_many->ProcessingTime());
  EXPECT_EQ(0, async_interleave_many->OutputTime(&input_times));
  async_interleave_many->record_element();
  EXPECT_EQ(1, async_interleave_many->num_elements());
  EXPECT_EQ(100, async_interleave_many->ProcessingTime());
  EXPECT_EQ(std::max(0LL, 100 - input_time),
            async_interleave_many->OutputTime(&input_times));
  source1->add_processing_time(200);
  source2->add_processing_time(300);
  EXPECT_EQ(100, async_interleave_many->ProcessingTime());
  EXPECT_EQ(std::max(0LL, 100 - input_time),
            async_interleave_many->OutputTime(&input_times));
  source1->record_element();
  source2->record_element();
  EXPECT_EQ(100 + 250, async_interleave_many->ProcessingTime());
  EXPECT_EQ(std::max(0LL, 100 + 250 / parallelism - input_time),
            async_interleave_many->OutputTime(&input_times));
  async_interleave_many->record_element();
  EXPECT_EQ(50 + 250, async_interleave_many->ProcessingTime());
  EXPECT_EQ(std::max(0LL, 50 + 250 / parallelism - input_time),
            async_interleave_many->OutputTime(&input_times));
}

INSTANTIATE_TEST_CASE_P(Test, AsyncInterleaveManyTest,
                        ::testing::Combine(::testing::Values(1, 2),
                                           ::testing::Values(0, 50, 100, 200)));

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
  EXPECT_EQ(0, async_known_many->ProcessingTime());
  EXPECT_EQ(0, async_known_many->OutputTime(&input_times));
  source2->add_processing_time(200);
  EXPECT_EQ(0, async_known_many->ProcessingTime());
  EXPECT_EQ(0, async_known_many->OutputTime(&input_times));
  source1->record_element();
  EXPECT_EQ(num_inputs_per_output * 100, async_known_many->ProcessingTime());
  EXPECT_EQ(std::max(0LL, num_inputs_per_output * 100 - input_time),
            async_known_many->OutputTime(&input_times));
  source2->record_element();
  EXPECT_EQ(num_inputs_per_output * (100 + 200),
            async_known_many->ProcessingTime());
  EXPECT_EQ(std::max(0LL, num_inputs_per_output * (100 + 200) - input_time),
            async_known_many->OutputTime(&input_times));
  source1->record_element();
  EXPECT_EQ(num_inputs_per_output * (50 + 200),
            async_known_many->ProcessingTime());
  EXPECT_EQ(std::max(0LL, num_inputs_per_output * (50 + 200) - input_time),
            async_known_many->OutputTime(&input_times));
  source2->record_element();
  EXPECT_EQ(num_inputs_per_output * (50 + 100),
            async_known_many->ProcessingTime());
  EXPECT_EQ(std::max(0LL, num_inputs_per_output * (50 + 100) - input_time),
            async_known_many->OutputTime(&input_times));
  async_known_many->add_processing_time(128);
  EXPECT_EQ(num_inputs_per_output * (50 + 100),
            async_known_many->ProcessingTime());
  EXPECT_EQ(std::max(0LL, num_inputs_per_output * (50 + 100) - input_time),
            async_known_many->OutputTime(&input_times));
  async_known_many->record_element();
  EXPECT_EQ(num_inputs_per_output * (50 + 100) + 128,
            async_known_many->ProcessingTime());
  EXPECT_EQ(std::max(0LL, num_inputs_per_output * (50 + 100) +
                              128 / parallelism - input_time),
            async_known_many->OutputTime(&input_times));
  async_known_many->record_element();
  EXPECT_EQ(num_inputs_per_output * (50 + 100) + 64,
            async_known_many->ProcessingTime());
  EXPECT_EQ(std::max(0LL, num_inputs_per_output * (50 + 100) +
                              64 / parallelism - input_time),
            async_known_many->OutputTime(&input_times));
}

INSTANTIATE_TEST_CASE_P(Test, AsyncKnownRatioTest,
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
  EXPECT_EQ(100, interleave_many->processing_time());
  EXPECT_EQ(0, interleave_many->ProcessingTime());
  EXPECT_EQ(0, interleave_many->OutputTime(&input_times));
  interleave_many->record_element();
  EXPECT_EQ(1, interleave_many->num_elements());
  EXPECT_EQ(100, interleave_many->ProcessingTime());
  EXPECT_EQ(100, interleave_many->OutputTime(&input_times));
  source1->add_processing_time(200);
  source2->add_processing_time(300);
  EXPECT_EQ(100, interleave_many->ProcessingTime());
  EXPECT_EQ(100, interleave_many->OutputTime(&input_times));
  source1->record_element();
  source2->record_element();
  EXPECT_EQ(350, interleave_many->ProcessingTime());
  EXPECT_EQ(350, interleave_many->OutputTime(&input_times));
  interleave_many->record_element();
  EXPECT_EQ(300, interleave_many->ProcessingTime());
  EXPECT_EQ(300, interleave_many->OutputTime(&input_times));
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
  EXPECT_EQ(0, known_many->ProcessingTime());
  EXPECT_EQ(0, known_many->OutputTime(&input_times));
  source2->add_processing_time(200);
  EXPECT_EQ(0, known_many->ProcessingTime());
  EXPECT_EQ(0, known_many->OutputTime(&input_times));
  source1->record_element();
  EXPECT_EQ(num_inputs_per_output * 100, known_many->ProcessingTime());
  EXPECT_EQ(num_inputs_per_output * 100, known_many->OutputTime(&input_times));
  source2->record_element();
  EXPECT_EQ(num_inputs_per_output * (100 + 200), known_many->ProcessingTime());
  EXPECT_EQ(num_inputs_per_output * (100 + 200),
            known_many->OutputTime(&input_times));
  source1->record_element();
  EXPECT_EQ(num_inputs_per_output * (50 + 200), known_many->ProcessingTime());
  EXPECT_EQ(num_inputs_per_output * (50 + 200),
            known_many->OutputTime(&input_times));
  source2->record_element();
  EXPECT_EQ(num_inputs_per_output * (50 + 100), known_many->ProcessingTime());
  EXPECT_EQ(num_inputs_per_output * (50 + 100),
            known_many->OutputTime(&input_times));
  known_many->add_processing_time(128);
  EXPECT_EQ(num_inputs_per_output * (50 + 100), known_many->ProcessingTime());
  EXPECT_EQ(num_inputs_per_output * (50 + 100),
            known_many->OutputTime(&input_times));
  known_many->record_element();
  EXPECT_EQ(num_inputs_per_output * (50 + 100) + 128,
            known_many->ProcessingTime());
  EXPECT_EQ(num_inputs_per_output * (50 + 100) + 128,
            known_many->OutputTime(&input_times));
  known_many->record_element();
  EXPECT_EQ(num_inputs_per_output * (50 + 100) + 64,
            known_many->ProcessingTime());
  EXPECT_EQ(num_inputs_per_output * (50 + 100) + 64,
            known_many->OutputTime(&input_times));
}

INSTANTIATE_TEST_CASE_P(Test, KnownRatioTest, ::testing::Values(0, 1, 2, 4));

TEST(SourceTest, Model) {
  std::shared_ptr<Node> source = model::MakeSourceNode({0, "source", nullptr});
  std::vector<int64> input_times(1, 0);
  source->add_processing_time(100);
  EXPECT_EQ(100, source->processing_time());
  EXPECT_EQ(0, source->ProcessingTime());
  EXPECT_EQ(0, source->OutputTime(&input_times));
  source->record_element();
  EXPECT_EQ(1, source->num_elements());
  EXPECT_EQ(100, source->ProcessingTime());
  EXPECT_EQ(100, source->OutputTime(&input_times));
  source->record_element();
  EXPECT_EQ(2, source->num_elements());
  EXPECT_EQ(50, source->ProcessingTime());
  EXPECT_EQ(50, source->OutputTime(&input_times));
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
  EXPECT_EQ(100, unknown_many->processing_time());
  EXPECT_EQ(0, unknown_many->ProcessingTime());
  EXPECT_EQ(0, unknown_many->OutputTime(&input_times));
  unknown_many->record_element();
  EXPECT_EQ(1, unknown_many->num_elements());
  EXPECT_EQ(100, unknown_many->ProcessingTime());
  EXPECT_EQ(100, unknown_many->OutputTime(&input_times));
  source1->add_processing_time(100);
  source2->add_processing_time(200);
  EXPECT_EQ(100, unknown_many->ProcessingTime());
  EXPECT_EQ(100, unknown_many->OutputTime(&input_times));
  source1->record_element();
  source2->record_element();
  EXPECT_EQ(400, unknown_many->ProcessingTime());
  EXPECT_EQ(400, unknown_many->OutputTime(&input_times));
  unknown_many->record_element();
  EXPECT_EQ(200, unknown_many->ProcessingTime());
  EXPECT_EQ(200, unknown_many->OutputTime(&input_times));
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
  EXPECT_EQ(0, unknown->ProcessingTime());
  EXPECT_EQ(0, unknown->OutputTime(&input_times));
  source2->add_processing_time(100);
  EXPECT_EQ(0, unknown->ProcessingTime());
  EXPECT_EQ(0, unknown->OutputTime(&input_times));
  source1->record_element();
  EXPECT_EQ(100, unknown->ProcessingTime());
  EXPECT_EQ(100, unknown->OutputTime(&input_times));
  source2->record_element();
  EXPECT_EQ(200, unknown->ProcessingTime());
  EXPECT_EQ(200, unknown->OutputTime(&input_times));
  source1->record_element();
  EXPECT_EQ(150, unknown->ProcessingTime());
  EXPECT_EQ(150, unknown->OutputTime(&input_times));
  source2->record_element();
  EXPECT_EQ(100, unknown->ProcessingTime());
  EXPECT_EQ(100, unknown->OutputTime(&input_times));
  // Unknown node processing time should not affect its ProcessingTime() or
  // OutputTime().
  unknown->add_processing_time(100);
  EXPECT_EQ(100, unknown->processing_time());
  EXPECT_EQ(100, unknown->ProcessingTime());
  EXPECT_EQ(100, unknown->OutputTime(&input_times));
  // Unknown node number of elements should not affect its ProcessingTime() or
  // OutputTime().
  unknown->record_element();
  EXPECT_EQ(1, unknown->num_elements());
  EXPECT_EQ(100, unknown->ProcessingTime());
  EXPECT_EQ(100, unknown->OutputTime(&input_times));
}

}  // namespace
}  // namespace model
}  // namespace data
}  // namespace tensorflow
