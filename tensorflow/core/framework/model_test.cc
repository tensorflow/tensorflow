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

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace model {
namespace {

using ::tensorflow::monitoring::testing::CellReader;
using ::testing::AllOf;
using ::testing::HasSubstr;

int64_t CountParametersOnNode(const string& node_name,
                              const Model::ModelParameters& parameters) {
  int64_t cnt = 0;
  for (const auto& pair : parameters) {
    if (pair.first == node_name) {
      cnt++;
    }
  }
  return cnt;
}

class AsyncInterleaveManyTest
    : public ::testing::TestWithParam<std::tuple<int64_t, double>> {};

TEST_P(AsyncInterleaveManyTest, Model) {
  const int64_t parallelism = std::get<0>(GetParam());
  const double input_time = std::get<1>(GetParam());
  std::shared_ptr<Node> async_interleave_many =
      model::MakeAsyncInterleaveManyNode(
          {0, "async_interleave_many", nullptr},
          {model::MakeParameter("parallelism",
                                std::make_shared<SharedState>(
                                    /*value=*/parallelism, nullptr, nullptr),
                                /*min=*/1,
                                /*max=*/8),
           model::MakeParameter(kCycleLength, nullptr,
                                /*min=*/1,
                                /*max=*/1)});
  std::shared_ptr<Node> meta_source =
      model::MakeSourceNode({1, "meta_source", async_interleave_many});
  async_interleave_many->add_input(meta_source);
  auto cleanup_meta = gtl::MakeCleanup([async_interleave_many, meta_source]() {
    async_interleave_many->remove_input(meta_source);
  });
  std::shared_ptr<Node> source1 =
      model::MakeSourceNode({2, "source1", async_interleave_many});
  async_interleave_many->add_input(source1);
  auto cleanup1 = gtl::MakeCleanup([async_interleave_many, source1]() {
    async_interleave_many->remove_input(source1);
  });
  std::shared_ptr<Node> source2 =
      model::MakeSourceNode({3, "source2", async_interleave_many});
  async_interleave_many->add_input(source2);
  auto cleanup2 = gtl::MakeCleanup([async_interleave_many, source2]() {
    async_interleave_many->remove_input(source2);
  });
  Model::NodeValues input_times;
  input_times[kModelInputTimeKey] = input_time;
  EXPECT_EQ(async_interleave_many->TotalBufferedBytes(), 0);
  EXPECT_EQ(async_interleave_many->TotalMaximumBufferedBytes(), 0);
  async_interleave_many->record_buffer_event(110, 10);
  EXPECT_EQ(async_interleave_many->TotalBufferedBytes(), 110);
  EXPECT_EQ(async_interleave_many->TotalMaximumBufferedBytes(),
            110 * parallelism / 10);
  async_interleave_many->add_processing_time(100);
  EXPECT_EQ(async_interleave_many->processing_time(), 100);
  EXPECT_EQ(
      async_interleave_many->TotalProcessingTime(/*processing_times=*/nullptr),
      0);
  EXPECT_EQ(async_interleave_many->OutputTime(&input_times, nullptr), 0);
  async_interleave_many->record_element();
  EXPECT_EQ(async_interleave_many->num_elements(), 1);
  EXPECT_EQ(
      async_interleave_many->TotalProcessingTime(/*processing_times=*/nullptr),
      100);
  EXPECT_LE(async_interleave_many->OutputTime(&input_times, nullptr), 100);
  EXPECT_GE(async_interleave_many->OutputTime(&input_times, nullptr), 0);
  source1->add_processing_time(200);
  source2->add_processing_time(300);
  EXPECT_EQ(
      async_interleave_many->TotalProcessingTime(/*processing_times=*/nullptr),
      100);
  EXPECT_LE(async_interleave_many->OutputTime(&input_times, nullptr), 100);
  EXPECT_GE(async_interleave_many->OutputTime(&input_times, nullptr), 0);
  source1->record_element();
  source2->record_element();
  EXPECT_EQ(
      async_interleave_many->TotalProcessingTime(/*processing_times=*/nullptr),
      100 + 250);
  EXPECT_LE(async_interleave_many->OutputTime(&input_times, nullptr),
            100 + 250 / parallelism);
  EXPECT_GE(async_interleave_many->OutputTime(&input_times, nullptr), 0);
  async_interleave_many->record_element();
  EXPECT_EQ(
      async_interleave_many->TotalProcessingTime(/*processing_times=*/nullptr),
      50 + 250);
  EXPECT_LE(async_interleave_many->OutputTime(&input_times, nullptr),
            50 + 250 / parallelism);
  EXPECT_GE(async_interleave_many->OutputTime(&input_times, nullptr), 0);
}

INSTANTIATE_TEST_SUITE_P(Test, AsyncInterleaveManyTest,
                         ::testing::Combine(::testing::Values(1, 2),
                                            ::testing::Values(0, 50, 100,
                                                              200)));

class AsyncKnownRatioTest
    : public ::testing::TestWithParam<std::tuple<int64_t, double, int64_t>> {};

TEST_P(AsyncKnownRatioTest, Model) {
  const int64_t parallelism = std::get<0>(GetParam());
  const double input_time = std::get<1>(GetParam());
  const int64_t num_inputs_per_output = std::get<2>(GetParam());
  std::shared_ptr<Node> async_known_many = model::MakeAsyncKnownRatioNode(
      {0, "async_known_many", nullptr}, num_inputs_per_output,
      {model::MakeParameter("parallelism",
                            std::make_shared<SharedState>(/*value=*/parallelism,
                                                          nullptr, nullptr),
                            /*min=*/1,
                            /*max=*/16)});
  std::shared_ptr<Node> source1 =
      model::MakeSourceNode({1, "source1", async_known_many});
  async_known_many->add_input(source1);
  std::shared_ptr<Node> source2 =
      model::MakeSourceNode({2, "source2", async_known_many});
  async_known_many->add_input(source2);
  Model::NodeValues input_times;
  input_times[kModelInputTimeKey] = input_time;
  EXPECT_EQ(async_known_many->TotalBufferedBytes(), 0);
  EXPECT_EQ(async_known_many->TotalMaximumBufferedBytes(), 0);
  async_known_many->record_buffer_event(110, 10);
  EXPECT_EQ(async_known_many->TotalBufferedBytes(), 110);
  EXPECT_EQ(async_known_many->TotalMaximumBufferedBytes(),
            num_inputs_per_output == 0
                ? 110.0 * parallelism / 10
                : 110.0 * parallelism / 10 / num_inputs_per_output);
  source1->add_processing_time(100);
  EXPECT_EQ(async_known_many->TotalProcessingTime(/*processing_times=*/nullptr),
            0);
  EXPECT_EQ(async_known_many->OutputTime(&input_times, nullptr), 0);
  source2->add_processing_time(200);
  EXPECT_EQ(async_known_many->TotalProcessingTime(/*processing_times=*/nullptr),
            0);
  EXPECT_EQ(async_known_many->OutputTime(&input_times, nullptr), 0);
  source1->record_element();
  EXPECT_EQ(async_known_many->TotalProcessingTime(/*processing_times=*/nullptr),
            num_inputs_per_output * 100);
  EXPECT_LE(async_known_many->OutputTime(&input_times, nullptr),
            num_inputs_per_output * 100);
  EXPECT_GE(async_known_many->OutputTime(&input_times, nullptr), 0);
  source2->record_element();
  EXPECT_EQ(async_known_many->TotalProcessingTime(/*processing_times=*/nullptr),
            num_inputs_per_output * (100 + 200));
  EXPECT_LE(async_known_many->OutputTime(&input_times, nullptr),
            num_inputs_per_output * (100 + 200));
  EXPECT_GE(async_known_many->OutputTime(&input_times, nullptr), 0);
  source1->record_element();
  EXPECT_EQ(async_known_many->TotalProcessingTime(/*processing_times=*/nullptr),
            num_inputs_per_output * (50 + 200));
  EXPECT_LE(async_known_many->OutputTime(&input_times, nullptr),
            num_inputs_per_output * (50 + 200));
  EXPECT_GE(async_known_many->OutputTime(&input_times, nullptr), 0);
  source2->record_element();
  EXPECT_EQ(async_known_many->TotalProcessingTime(/*processing_times=*/nullptr),
            num_inputs_per_output * (50 + 100));
  EXPECT_LE(async_known_many->OutputTime(&input_times, nullptr),
            num_inputs_per_output * (50 + 100));
  EXPECT_GE(async_known_many->OutputTime(&input_times, nullptr), 0);
  async_known_many->add_processing_time(128);
  EXPECT_EQ(async_known_many->TotalProcessingTime(/*processing_times=*/nullptr),
            num_inputs_per_output * (50 + 100));
  EXPECT_LE(async_known_many->OutputTime(&input_times, nullptr),
            num_inputs_per_output * (50 + 100));
  EXPECT_GE(async_known_many->OutputTime(&input_times, nullptr), 0);
  async_known_many->record_element();
  EXPECT_EQ(async_known_many->TotalProcessingTime(/*processing_times=*/nullptr),
            num_inputs_per_output * (50 + 100) + 128);
  EXPECT_LE(async_known_many->OutputTime(&input_times, nullptr),
            num_inputs_per_output * (50 + 100) + 128 / parallelism);
  EXPECT_GE(async_known_many->OutputTime(&input_times, nullptr), 0);
  async_known_many->record_element();
  EXPECT_EQ(async_known_many->TotalProcessingTime(/*processing_times=*/nullptr),
            num_inputs_per_output * (50 + 100) + 64);
  EXPECT_LE(async_known_many->OutputTime(&input_times, nullptr),
            num_inputs_per_output * (50 + 100) + 64 / parallelism);
  EXPECT_GE(async_known_many->OutputTime(&input_times, nullptr), 0);
}

INSTANTIATE_TEST_SUITE_P(Test, AsyncKnownRatioTest,
                         ::testing::Combine(::testing::Values(1, 2, 4, 8),
                                            ::testing::Values(0, 50, 100, 200),
                                            ::testing::Values(0, 1, 2, 4)));

TEST(InterleaveManyTest, Model) {
  auto parameter =
      model::MakeParameter("cycle_length", nullptr, /*min=*/1, /*max=*/1);
  std::shared_ptr<Node> interleave_many = model::MakeInterleaveManyNode(
      {0, "interleave_many", nullptr},
      {model::MakeParameter("cycle_length", nullptr, /*min=*/1, /*max=*/1)});
  std::shared_ptr<Node> meta_source =
      model::MakeSourceNode({1, "meta_source", interleave_many});
  interleave_many->add_input(meta_source);
  std::shared_ptr<Node> source1 =
      model::MakeSourceNode({2, "source1", interleave_many});
  interleave_many->add_input(source1);
  std::shared_ptr<Node> source2 =
      model::MakeSourceNode({3, "source2", interleave_many});
  interleave_many->add_input(source2);
  Model::NodeValues input_times;
  input_times[kModelInputTimeKey] = 0.0;
  interleave_many->add_processing_time(100);
  EXPECT_EQ(interleave_many->processing_time(), 100);
  EXPECT_EQ(interleave_many->TotalProcessingTime(/*processing_times=*/nullptr),
            0);
  EXPECT_EQ(interleave_many->OutputTime(&input_times, nullptr), 0);
  interleave_many->record_element();
  EXPECT_EQ(interleave_many->num_elements(), 1);
  EXPECT_EQ(interleave_many->TotalProcessingTime(/*processing_times=*/nullptr),
            100);
  EXPECT_EQ(interleave_many->OutputTime(&input_times, nullptr), 100);
  source1->add_processing_time(200);
  source2->add_processing_time(300);
  EXPECT_EQ(interleave_many->TotalProcessingTime(/*processing_times=*/nullptr),
            100);
  EXPECT_EQ(interleave_many->OutputTime(&input_times, nullptr), 100);
  source1->record_element();
  source2->record_element();
  EXPECT_EQ(interleave_many->TotalProcessingTime(/*processing_times=*/nullptr),
            350);
  EXPECT_EQ(interleave_many->OutputTime(&input_times, nullptr), 350);
  interleave_many->record_element();
  EXPECT_EQ(interleave_many->TotalProcessingTime(/*processing_times=*/nullptr),
            300);
  EXPECT_EQ(interleave_many->OutputTime(&input_times, nullptr), 300);
}

class KnownRatioTest : public ::testing::TestWithParam<int64_t> {};

TEST_P(KnownRatioTest, Model) {
  const int64_t num_inputs_per_output = GetParam();
  std::shared_ptr<Node> known_many = model::MakeKnownRatioNode(
      {0, "known_many", nullptr}, num_inputs_per_output);
  std::shared_ptr<Node> source1 =
      model::MakeSourceNode({1, "source1", known_many});
  known_many->add_input(source1);
  std::shared_ptr<Node> source2 =
      model::MakeSourceNode({2, "source2", known_many});
  known_many->add_input(source2);
  Model::NodeValues input_times;
  input_times[kModelInputTimeKey] = 0.0;
  source1->add_processing_time(100);
  EXPECT_EQ(known_many->TotalProcessingTime(/*processing_times=*/nullptr), 0);
  EXPECT_EQ(known_many->OutputTime(&input_times, nullptr), 0);
  source2->add_processing_time(200);
  EXPECT_EQ(known_many->TotalProcessingTime(/*processing_times=*/nullptr), 0);
  EXPECT_EQ(known_many->OutputTime(&input_times, nullptr), 0);
  source1->record_element();
  EXPECT_EQ(known_many->TotalProcessingTime(/*processing_times=*/nullptr),
            num_inputs_per_output * 100);
  EXPECT_EQ(known_many->OutputTime(&input_times, nullptr),
            num_inputs_per_output * 100);
  source2->record_element();
  EXPECT_EQ(known_many->TotalProcessingTime(/*processing_times=*/nullptr),
            num_inputs_per_output * (100 + 200));
  EXPECT_EQ(known_many->OutputTime(&input_times, nullptr),
            num_inputs_per_output * (100 + 200));
  source1->record_element();
  EXPECT_EQ(known_many->TotalProcessingTime(/*processing_times=*/nullptr),
            num_inputs_per_output * (50 + 200));
  EXPECT_EQ(known_many->OutputTime(&input_times, nullptr),
            num_inputs_per_output * (50 + 200));
  source2->record_element();
  EXPECT_EQ(known_many->TotalProcessingTime(/*processing_times=*/nullptr),
            num_inputs_per_output * (50 + 100));
  EXPECT_EQ(known_many->OutputTime(&input_times, nullptr),
            num_inputs_per_output * (50 + 100));
  known_many->add_processing_time(128);
  EXPECT_EQ(known_many->TotalProcessingTime(/*processing_times=*/nullptr),
            num_inputs_per_output * (50 + 100));
  EXPECT_EQ(known_many->OutputTime(&input_times, nullptr),
            num_inputs_per_output * (50 + 100));
  known_many->record_element();
  EXPECT_EQ(known_many->TotalProcessingTime(/*processing_times=*/nullptr),
            num_inputs_per_output * (50 + 100) + 128);
  EXPECT_EQ(known_many->OutputTime(&input_times, nullptr),
            num_inputs_per_output * (50 + 100) + 128);
  known_many->record_element();
  EXPECT_EQ(known_many->TotalProcessingTime(/*processing_times=*/nullptr),
            num_inputs_per_output * (50 + 100) + 64);
  EXPECT_EQ(known_many->OutputTime(&input_times, nullptr),
            num_inputs_per_output * (50 + 100) + 64);
}

INSTANTIATE_TEST_SUITE_P(Test, KnownRatioTest, ::testing::Values(0, 1, 2, 4));

TEST(SourceTest, Model) {
  std::shared_ptr<Node> source = model::MakeSourceNode({0, "source", nullptr});
  Model::NodeValues input_times;
  input_times[kModelInputTimeKey] = 0.0;
  source->add_processing_time(100);
  EXPECT_EQ(source->processing_time(), 100);
  EXPECT_EQ(source->TotalProcessingTime(/*processing_times=*/nullptr), 0);
  EXPECT_EQ(source->OutputTime(&input_times, nullptr), 0);
  source->record_element();
  EXPECT_EQ(source->num_elements(), 1);
  EXPECT_EQ(source->TotalProcessingTime(/*processing_times=*/nullptr), 100);
  EXPECT_EQ(source->OutputTime(&input_times, nullptr), 100);
  source->record_element();
  EXPECT_EQ(source->num_elements(), 2);
  EXPECT_EQ(source->TotalProcessingTime(/*processing_times=*/nullptr), 50);
  EXPECT_EQ(source->OutputTime(&input_times, nullptr), 50);
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
  Model::NodeValues input_times;
  input_times[kModelInputTimeKey] = 0.0;
  unknown_many->add_processing_time(100);
  EXPECT_EQ(unknown_many->processing_time(), 100);
  EXPECT_EQ(unknown_many->TotalProcessingTime(/*processing_times=*/nullptr), 0);
  EXPECT_EQ(unknown_many->OutputTime(&input_times, nullptr), 0);
  unknown_many->record_element();
  EXPECT_EQ(unknown_many->num_elements(), 1);
  EXPECT_EQ(unknown_many->TotalProcessingTime(/*processing_times=*/nullptr),
            100);
  EXPECT_EQ(unknown_many->OutputTime(&input_times, nullptr), 100);
  source1->add_processing_time(100);
  source2->add_processing_time(200);
  EXPECT_EQ(unknown_many->TotalProcessingTime(/*processing_times=*/nullptr),
            100);
  EXPECT_EQ(unknown_many->OutputTime(&input_times, nullptr), 100);
  source1->record_element();
  source2->record_element();
  EXPECT_EQ(unknown_many->TotalProcessingTime(/*processing_times=*/nullptr),
            400);
  EXPECT_EQ(unknown_many->OutputTime(&input_times, nullptr), 400);
  unknown_many->record_element();
  EXPECT_EQ(unknown_many->TotalProcessingTime(/*processing_times=*/nullptr),
            200);
  EXPECT_EQ(unknown_many->OutputTime(&input_times, nullptr), 200);
}

class AsyncUnknownRatioTest
    : public ::testing::TestWithParam<std::tuple<int64_t, double>> {};

TEST_P(AsyncUnknownRatioTest, Model) {
  const int64_t parallelism = std::get<0>(GetParam());
  const double input_time = std::get<1>(GetParam());
  std::shared_ptr<Node> async_unknown_many = model::MakeAsyncUnknownRatioNode(
      {0, "async_unknown_many", nullptr},
      {model::MakeParameter("parallelism",
                            std::make_shared<SharedState>(/*value=*/parallelism,
                                                          nullptr, nullptr),
                            /*min=*/1,
                            /*max=*/16)});
  std::shared_ptr<Node> source1 =
      model::MakeSourceNode({1, "source1", async_unknown_many});
  async_unknown_many->add_input(source1);
  std::shared_ptr<Node> source2 =
      model::MakeSourceNode({2, "source2", async_unknown_many});
  async_unknown_many->add_input(source2);
  Model::NodeValues input_times;
  input_times[kModelInputTimeKey] = input_time;
  EXPECT_EQ(async_unknown_many->TotalBufferedBytes(), 0);
  EXPECT_EQ(async_unknown_many->TotalMaximumBufferedBytes(), 0);
  async_unknown_many->record_buffer_event(110, 10);
  EXPECT_EQ(async_unknown_many->TotalBufferedBytes(), 110);
  EXPECT_EQ(async_unknown_many->TotalMaximumBufferedBytes(),
            110.0 * parallelism / 10);
  source1->add_processing_time(100);
  EXPECT_EQ(
      async_unknown_many->TotalProcessingTime(/*processing_times=*/nullptr), 0);
  EXPECT_EQ(async_unknown_many->OutputTime(&input_times, nullptr), 0);
  source2->add_processing_time(200);
  EXPECT_EQ(
      async_unknown_many->TotalProcessingTime(/*processing_times=*/nullptr), 0);
  EXPECT_EQ(async_unknown_many->OutputTime(&input_times, nullptr), 0);
  source1->record_element();
  EXPECT_EQ(
      async_unknown_many->TotalProcessingTime(/*processing_times=*/nullptr), 0);
  EXPECT_EQ(async_unknown_many->OutputTime(&input_times, nullptr), 0);
  async_unknown_many->record_element();
  // Estimated ratio is 1.
  double ratio = 1.0;
  EXPECT_EQ(
      async_unknown_many->TotalProcessingTime(/*processing_times=*/nullptr),
      ratio * 100);
  EXPECT_LE(async_unknown_many->OutputTime(&input_times, nullptr), 100);
  EXPECT_GE(async_unknown_many->OutputTime(&input_times, nullptr), 0);
  source2->record_element();
  EXPECT_EQ(
      async_unknown_many->TotalProcessingTime(/*processing_times=*/nullptr),
      ratio * (100 + 200));
  EXPECT_LE(async_unknown_many->OutputTime(&input_times, nullptr),
            ratio * (100 + 200));
  EXPECT_GE(async_unknown_many->OutputTime(&input_times, nullptr), 0);
  source2->record_element();
  EXPECT_EQ(
      async_unknown_many->TotalProcessingTime(/*processing_times=*/nullptr),
      ratio * (100 + 100));
  EXPECT_LE(async_unknown_many->OutputTime(&input_times, nullptr),
            ratio * (100 + 100));
  EXPECT_GE(async_unknown_many->OutputTime(&input_times, nullptr), 0);
  source1->record_element();
  // Estimated ratio is 2
  ratio = 2.0;
  EXPECT_EQ(
      async_unknown_many->TotalProcessingTime(/*processing_times=*/nullptr),
      ratio * (50 + 100));
  EXPECT_LE(async_unknown_many->OutputTime(&input_times, nullptr),
            ratio * (50 + 100));
  EXPECT_GE(async_unknown_many->OutputTime(&input_times, nullptr), 0);
  source2->record_element();
  source2->record_element();
  EXPECT_EQ(
      async_unknown_many->TotalProcessingTime(/*processing_times=*/nullptr),
      ratio * (50 + 50));
  EXPECT_LE(async_unknown_many->OutputTime(&input_times, nullptr),
            ratio * (50 + 50));
  EXPECT_GE(async_unknown_many->OutputTime(&input_times, nullptr), 0);
  async_unknown_many->add_processing_time(128);
  EXPECT_EQ(
      async_unknown_many->TotalProcessingTime(/*processing_times=*/nullptr),
      ratio * (50 + 50) + 128);
  EXPECT_LE(async_unknown_many->OutputTime(&input_times, nullptr),
            ratio * (50 + 50) + 128 / parallelism);
  EXPECT_GE(async_unknown_many->OutputTime(&input_times, nullptr),
            128 / parallelism);
  async_unknown_many->record_element();
  // Estimated ratio is 1.0
  ratio = 1.0;
  EXPECT_EQ(
      async_unknown_many->TotalProcessingTime(/*processing_times=*/nullptr),
      ratio * (50 + 50) + 128 / 2);
  EXPECT_LE(async_unknown_many->OutputTime(&input_times, nullptr),
            ratio * (50 + 50) + 128 / 2 / parallelism);
  EXPECT_GE(async_unknown_many->OutputTime(&input_times, nullptr),
            128 / 2 / parallelism);
  async_unknown_many->record_element();
  // Estimated ratio is 2/3
  ratio = 2.0 / 3.0;
  EXPECT_FLOAT_EQ(
      async_unknown_many->TotalProcessingTime(/*processing_times=*/nullptr),
      ratio * (50 + 50) + 128 / 3.0);
  EXPECT_LE(async_unknown_many->OutputTime(&input_times, nullptr),
            ratio * (50 + 50) + 128 / 3.0 / parallelism);
  EXPECT_GE(async_unknown_many->OutputTime(&input_times, nullptr),
            128 / 3.0 / parallelism);
}

INSTANTIATE_TEST_SUITE_P(Test, AsyncUnknownRatioTest,
                         ::testing::Combine(::testing::Values(1, 2, 4, 8),
                                            ::testing::Values(0, 50, 100,
                                                              200)));

TEST(UnknownTest, Model) {
  std::shared_ptr<Node> unknown =
      model::MakeUnknownNode({0, "unknown", nullptr});
  std::shared_ptr<Node> source1 =
      model::MakeSourceNode({1, "source1", unknown});
  unknown->add_input(source1);
  std::shared_ptr<Node> source2 =
      model::MakeSourceNode({2, "source2", unknown});
  unknown->add_input(source2);
  Model::NodeValues input_times;
  input_times[kModelInputTimeKey] = 0.0;
  source1->add_processing_time(100);
  EXPECT_EQ(unknown->TotalProcessingTime(/*processing_times=*/nullptr), 0);
  EXPECT_EQ(unknown->OutputTime(&input_times, nullptr), 0);
  source2->add_processing_time(100);
  EXPECT_EQ(unknown->TotalProcessingTime(/*processing_times=*/nullptr), 0);
  EXPECT_EQ(unknown->OutputTime(&input_times, nullptr), 0);
  source1->record_element();
  EXPECT_EQ(unknown->TotalProcessingTime(/*processing_times=*/nullptr), 100);
  EXPECT_EQ(unknown->OutputTime(&input_times, nullptr), 100);
  source2->record_element();
  EXPECT_EQ(unknown->TotalProcessingTime(/*processing_times=*/nullptr), 200);
  EXPECT_EQ(unknown->OutputTime(&input_times, nullptr), 200);
  source1->record_element();
  EXPECT_EQ(unknown->TotalProcessingTime(/*processing_times=*/nullptr), 150);
  EXPECT_EQ(unknown->OutputTime(&input_times, nullptr), 150);
  source2->record_element();
  EXPECT_EQ(unknown->TotalProcessingTime(/*processing_times=*/nullptr), 100);
  EXPECT_EQ(unknown->OutputTime(&input_times, nullptr), 100);
  // Unknown node processing time should not affect its TotalProcessingTime() or
  // OutputTime().
  unknown->add_processing_time(100);
  EXPECT_EQ(unknown->processing_time(), 100);
  EXPECT_EQ(unknown->TotalProcessingTime(/*processing_times=*/nullptr), 100);
  EXPECT_EQ(unknown->OutputTime(&input_times, nullptr), 100);
  // Unknown node number of elements should not affect its TotalProcessingTime()
  // or OutputTime().
  unknown->record_element();
  EXPECT_EQ(unknown->num_elements(), 1);
  EXPECT_EQ(unknown->TotalProcessingTime(/*processing_times=*/nullptr), 100);
  EXPECT_EQ(unknown->OutputTime(&input_times, nullptr), 100);
}

TEST(BufferedBytesTest, Node) {
  std::shared_ptr<Node> node = model::MakeAsyncInterleaveManyNode(
      {-1, "TestNode", nullptr},
      {model::MakeParameter(
           "parallelism",
           std::make_shared<SharedState>(/*value=*/3, nullptr, nullptr),
           /*min=*/1, /*max=*/7),
       model::MakeParameter(kCycleLength, nullptr,
                            /*min=*/1,
                            /*max=*/1)});
  EXPECT_EQ(node->id(), -1);
  EXPECT_EQ(node->name(), "TestNode");
  EXPECT_EQ(node->output(), nullptr);

  EXPECT_EQ(node->buffered_bytes(), 0);
  EXPECT_EQ(node->buffered_elements(), 0);
  EXPECT_EQ(node->TotalBufferedBytes(), 0);
  EXPECT_EQ(node->TotalMaximumBufferedBytes(), 0);

  node->record_buffer_event(20, 1);
  EXPECT_EQ(node->buffered_bytes(), 20);
  EXPECT_EQ(node->buffered_elements(), 1);
  EXPECT_EQ(node->TotalBufferedBytes(), 20);
  EXPECT_EQ(node->TotalMaximumBufferedBytes(), 60);

  node->record_buffer_event(10, 1);
  EXPECT_EQ(node->buffered_bytes(), 30);
  EXPECT_EQ(node->buffered_elements(), 2);
  EXPECT_EQ(node->TotalBufferedBytes(), 30);
  EXPECT_EQ(node->TotalMaximumBufferedBytes(), 45);

  node->record_buffer_event(18, 1);
  EXPECT_EQ(node->buffered_bytes(), 48);
  EXPECT_EQ(node->buffered_elements(), 3);
  EXPECT_EQ(node->bytes_produced(), 0);
  EXPECT_EQ(node->num_elements(), 0);
  EXPECT_EQ(node->TotalBufferedBytes(), 48);
  EXPECT_EQ(node->TotalMaximumBufferedBytes(), 48);

  node->record_buffer_event(-20, -1);
  node->record_element();
  node->record_bytes_produced(20);
  EXPECT_EQ(node->buffered_bytes(), 28);
  EXPECT_EQ(node->buffered_elements(), 2);
  EXPECT_EQ(node->bytes_produced(), 20);
  EXPECT_EQ(node->num_elements(), 1);
  EXPECT_EQ(node->TotalBufferedBytes(), 28);
  EXPECT_EQ(node->TotalMaximumBufferedBytes(), 51);

  node->record_buffer_event(-10, -1);
  node->record_element();
  node->record_bytes_produced(10);
  EXPECT_EQ(node->buffered_bytes(), 18);
  EXPECT_EQ(node->buffered_elements(), 1);
  EXPECT_EQ(node->bytes_produced(), 30);
  EXPECT_EQ(node->num_elements(), 2);
  EXPECT_EQ(node->TotalBufferedBytes(), 18);
  EXPECT_EQ(node->TotalMaximumBufferedBytes(), 49.5);

  EXPECT_EQ(node->processing_time(), 0);
  node->record_start(1);
  EXPECT_EQ(node->processing_time(), 0);
  node->record_stop(41);
  EXPECT_EQ(node->processing_time(), 40);
  node->add_processing_time(2);
  EXPECT_EQ(node->processing_time(), 42);

  std::shared_ptr<Node> input = model::MakeAsyncKnownRatioNode(
      {0, "TestInput", node}, 2,
      {model::MakeParameter("parallelism",
                            std::make_shared<SharedState>(5, nullptr, nullptr),
                            0, 6)});
  EXPECT_EQ(input->output(), node.get());
  EXPECT_EQ(node->inputs().size(), 0);
  node->add_input(input);
  EXPECT_EQ(node->inputs().size(), 1);
  EXPECT_EQ(node->inputs().front(), input);

  input->record_buffer_event(28, 1);
  EXPECT_EQ(node->bytes_consumed(), 0);
  EXPECT_EQ(node->TotalBufferedBytes(), 46);
  EXPECT_EQ(node->TotalMaximumBufferedBytes(), 119.5);

  input->record_buffer_event(-28, -1);
  input->record_element();
  input->record_bytes_produced(28);
  node->record_bytes_consumed(28);
  EXPECT_EQ(node->bytes_consumed(), 28);
  EXPECT_EQ(node->TotalBufferedBytes(), 18);
  EXPECT_EQ(node->TotalMaximumBufferedBytes(), 119.5);

  node->remove_input(input);
  EXPECT_EQ(node->inputs().size(), 0);
}

// Returns a weighted sum of a prior and the actual processing time.
double weighted_processing_time(int64_t num_elements, double processing_time,
                                double prior) {
  if (num_elements < 30) {
    double prior_weight = 1.0L / static_cast<double>(2 << num_elements);
    return prior_weight * prior + (1.0L - prior_weight) * processing_time;
  } else {
    return processing_time;
  }
}

TEST(TestManyElements, Model) {
  std::shared_ptr<Node> interleave_many = model::MakeInterleaveManyNode(
      {0, "interleave_many", nullptr},
      {model::MakeParameter("cycle_length", nullptr, /*min=*/1, /*max=*/1)});
  std::shared_ptr<Node> source1 =
      model::MakeSourceNode({1, "source1", interleave_many});
  interleave_many->add_input(source1);
  interleave_many->add_processing_time(100);
  interleave_many->record_element();
  source1->add_processing_time(200);
  for (int i = 0; i < 100; i++) {
    source1->record_element();
  }
  EXPECT_LE(interleave_many->TotalProcessingTime(/*processing_times=*/nullptr),
            (weighted_processing_time(100, 2, 0)) + 100);
  EXPECT_GE(interleave_many->TotalProcessingTime(/*processing_times=*/nullptr),
            0);
}

TEST(CollectAutotuneParametersWithElementsTest, Model) {
  std::shared_ptr<Node> unknown =
      model::MakeUnknownNode({0, "unknown", nullptr});
  std::shared_ptr<Node> async_known_ratio = model::MakeAsyncKnownRatioNode(
      {1, "source", unknown}, 2,
      {model::MakeParameter("parallelism",
                            std::make_shared<SharedState>(
                                /*value=*/model::kAutotune, nullptr, nullptr),
                            /*min=*/1,
                            /*max=*/5)});
  async_known_ratio->record_element();
  unknown->add_input(async_known_ratio);

  Model::ModelParameters parameters = unknown->CollectTunableParameters();

  EXPECT_EQ(CountParametersOnNode(unknown->long_name(), parameters), 0);
  EXPECT_EQ(CountParametersOnNode(async_known_ratio->long_name(), parameters),
            1);
  EXPECT_EQ(parameters.size(), 1);
}

TEST(DontCollectNonAutotuneParametersTest, Model) {
  std::shared_ptr<Node> unknown =
      model::MakeUnknownNode({0, "unknown", nullptr});
  std::shared_ptr<Node> async_known_ratio = model::MakeAsyncKnownRatioNode(
      {1, "source", unknown}, 2,
      {model::MakeParameter(
          "parallelism",
          std::make_shared<SharedState>(/*value=*/3, nullptr, nullptr),
          /*min=*/1, /*max=*/5)});
  async_known_ratio->record_element();
  unknown->add_input(async_known_ratio);
  Model::ModelParameters parameters = unknown->CollectTunableParameters();

  EXPECT_EQ(parameters.size(), 0);
}

TEST(DontCollectAutotuneDisabledParametersTest, Model) {
  std::shared_ptr<Node> unknown =
      model::MakeUnknownNode({0, "unknown", nullptr});
  std::shared_ptr<Node> async_known_ratio = model::MakeAsyncKnownRatioNode(
      {1, "source", unknown}, 2,
      {model::MakeParameter("parallelism",
                            std::make_shared<SharedState>(
                                /*value=*/model::kAutotune, nullptr, nullptr),
                            /*min=*/1,
                            /*max=*/5)});
  async_known_ratio->record_element();
  async_known_ratio->set_autotune(false);
  unknown->add_input(async_known_ratio);
  Model::ModelParameters parameters = unknown->CollectTunableParameters();

  EXPECT_EQ(parameters.size(), 0);
}

TEST(DontCollectParametersWithoutElementsTest, Model) {
  std::shared_ptr<Node> unknown =
      model::MakeUnknownNode({0, "unknown", nullptr});
  std::shared_ptr<Node> async_known_ratio = model::MakeAsyncKnownRatioNode(
      {1, "source", unknown}, 2,
      {model::MakeParameter("parallelism",
                            std::make_shared<SharedState>(
                                /*value=*/model::kAutotune, nullptr, nullptr),
                            /*min=*/1,
                            /*max=*/5)});
  unknown->add_input(async_known_ratio);
  Model::ModelParameters parameters = unknown->CollectTunableParameters();

  EXPECT_EQ(parameters.size(), 0);
}

// Precision for comparison of the gradient and a relative output time change.
constexpr double kComparisonPrecision = 1e-1;

// Parameter step for a relative output time change.
constexpr double kParameterStep = 1e-5;

TEST(AsyncInterleaveManyGradientTest, Model) {
  const double input_time = 100;
  std::shared_ptr<Parameter> interleave_parameter =
      model::MakeParameter("parallelism",
                           std::make_shared<SharedState>(
                               /*value=*/model::kAutotune, nullptr, nullptr),
                           /*min=*/1, /*max=*/5);
  std::shared_ptr<Node> async_interleave_many =
      model::MakeAsyncInterleaveManyNode(
          {0, "async_interleave_many", nullptr},
          {interleave_parameter, model::MakeParameter("cycle_length", nullptr,
                                                      /*min=*/1, /*max=*/1)});
  std::shared_ptr<Node> meta_source =
      model::MakeSourceNode({1, "meta_source", async_interleave_many});
  async_interleave_many->add_input(meta_source);
  auto cleanup_meta = gtl::MakeCleanup([async_interleave_many, meta_source]() {
    async_interleave_many->remove_input(meta_source);
  });
  std::shared_ptr<Parameter> source1_parameter =
      model::MakeParameter("parallelism",
                           std::make_shared<SharedState>(
                               /*value=*/model::kAutotune, nullptr, nullptr),
                           /*min=*/1,
                           /*max=*/7);
  std::shared_ptr<Node> source1 = model::MakeAsyncInterleaveManyNode(
      {2, "async_interleave_many", async_interleave_many},
      {source1_parameter,
       model::MakeParameter("cycle_length", nullptr, /*min=*/1, /*max=*/1)});
  async_interleave_many->add_input(source1);
  auto cleanup1 = gtl::MakeCleanup([async_interleave_many, source1]() {
    async_interleave_many->remove_input(source1);
  });
  std::shared_ptr<Node> source2 =
      model::MakeSourceNode({3, "source2", async_interleave_many});
  async_interleave_many->add_input(source2);
  auto cleanup2 = gtl::MakeCleanup([async_interleave_many, source2]() {
    async_interleave_many->remove_input(source2);
  });
  Model::NodeValues input_times;
  input_times[kModelInputTimeKey] = input_time;
  async_interleave_many->record_element();
  async_interleave_many->add_processing_time(100);
  source1->record_element();
  source1->add_processing_time(100);
  source2->record_element();
  source2->add_processing_time(300);

  interleave_parameter->value = 1;
  source1_parameter->value = 1;

  // Test gradient of own parameters.
  Model::ParameterGradients gradients;
  double output_time =
      async_interleave_many->OutputTime(&input_times, &gradients);
  interleave_parameter->value += kParameterStep;
  double new_output_time =
      async_interleave_many->OutputTime(&input_times, nullptr);
  EXPECT_NEAR(gradients[std::make_pair(async_interleave_many->long_name(),
                                       interleave_parameter->name)],
              (new_output_time - output_time) / kParameterStep,
              kComparisonPrecision);

  // Test propagation of input's gradient.
  interleave_parameter->value -= kParameterStep;
  source1_parameter->value += kParameterStep;
  new_output_time = async_interleave_many->OutputTime(&input_times, nullptr);
  EXPECT_NEAR(
      gradients[std::make_pair(source1->long_name(), source1_parameter->name)],
      (new_output_time - output_time) / kParameterStep, kComparisonPrecision);
}

class AsyncKnownRatioGradientTest : public ::testing::TestWithParam<string> {};

TEST_P(AsyncKnownRatioGradientTest, Model) {
  const string parameter_name = GetParam();
  const double input_time = 100;
  const int64_t num_inputs_per_output = 2;

  std::shared_ptr<Parameter> known_parameter =
      model::MakeParameter(parameter_name,
                           std::make_shared<SharedState>(
                               /*value=*/model::kAutotune, nullptr, nullptr),
                           /*min=*/1,
                           /*max=*/5);
  std::shared_ptr<Node> async_known_many =
      model::MakeAsyncKnownRatioNode({0, "async_known_many", nullptr},
                                     num_inputs_per_output, {known_parameter});
  std::shared_ptr<Parameter> source1_parameter =
      model::MakeParameter(parameter_name,
                           std::make_shared<SharedState>(
                               /*value=*/model::kAutotune, nullptr, nullptr),
                           /*min=*/1,
                           /*max=*/7);
  std::shared_ptr<Node> source1 = model::MakeAsyncKnownRatioNode(
      {1, "source1", async_known_many}, num_inputs_per_output,
      {source1_parameter});
  async_known_many->add_input(source1);
  std::shared_ptr<Node> source2 =
      model::MakeSourceNode({2, "source2", async_known_many});
  Model::NodeValues input_times;
  input_times[kModelInputTimeKey] = input_time;
  async_known_many->add_input(source2);
  source1->record_element();
  source1->add_processing_time(100);
  source2->record_element();
  source2->add_processing_time(100);
  async_known_many->record_element();
  async_known_many->add_processing_time(300);

  // Test gradient of own parameters.
  Model::ParameterGradients gradients;
  known_parameter->value = 1;
  source1_parameter->value = 1;
  double output_time = async_known_many->OutputTime(&input_times, &gradients);
  known_parameter->value += kParameterStep;
  double new_output_time = async_known_many->OutputTime(&input_times, nullptr);
  EXPECT_NEAR(gradients[std::make_pair(async_known_many->long_name(),
                                       known_parameter->name)],
              (new_output_time - output_time) / kParameterStep,
              kComparisonPrecision);

  // Test propagation of input's gradient.
  known_parameter->value -= kParameterStep;
  source1_parameter->value += kParameterStep;
  new_output_time = async_known_many->OutputTime(&input_times, nullptr);
  EXPECT_NEAR(
      gradients[std::make_pair(source1->long_name(), source1_parameter->name)],
      (new_output_time - output_time) / kParameterStep, kComparisonPrecision);
}

INSTANTIATE_TEST_SUITE_P(Test, AsyncKnownRatioGradientTest,
                         ::testing::Values("parallelism", "buffer_size"));

TEST(InterleaveManyGradientTest, Model) {
  const double input_time = 100;
  const int64_t num_inputs_per_output = 2;
  std::shared_ptr<Node> interleave_many = model::MakeInterleaveManyNode(
      {0, "interleave_many", nullptr},
      {model::MakeParameter("cycle_length", nullptr, /*min=*/1, /*max=*/1)});
  std::shared_ptr<Parameter> known_parameter =
      model::MakeParameter("parallelism",
                           std::make_shared<SharedState>(
                               /*value=*/model::kAutotune, nullptr, nullptr),
                           /*min=*/1,
                           /*max=*/5);
  std::shared_ptr<Node> async_known_many =
      model::MakeAsyncKnownRatioNode({1, "async_known_many", interleave_many},
                                     num_inputs_per_output, {known_parameter});
  std::shared_ptr<Node> source1 =
      model::MakeSourceNode({2, "source1", interleave_many});
  interleave_many->record_element();
  interleave_many->add_processing_time(100);
  interleave_many->add_input(source1);
  interleave_many->add_input(async_known_many);
  async_known_many->record_element();
  async_known_many->add_processing_time(300);
  Model::NodeValues input_times;
  input_times[kModelInputTimeKey] = input_time;
  Model::ParameterGradients gradients;
  known_parameter->value = 1;
  double output_time = interleave_many->OutputTime(&input_times, &gradients);
  known_parameter->value += kParameterStep;
  double new_output_time = interleave_many->OutputTime(&input_times, nullptr);
  EXPECT_NEAR(gradients[std::make_pair(async_known_many->long_name(),
                                       known_parameter->name)],
              (new_output_time - output_time) / kParameterStep,
              kComparisonPrecision);
}

TEST(KnownRatioGradientTest, Model) {
  const double input_time = 100;
  const int64_t num_inputs_per_output = 2;
  std::shared_ptr<Node> known_many = model::MakeKnownRatioNode(
      {0, "known_many", nullptr}, num_inputs_per_output);
  std::shared_ptr<Parameter> known_parameter =
      model::MakeParameter("parallelism",
                           std::make_shared<SharedState>(
                               /*value=*/model::kAutotune, nullptr, nullptr),
                           /*min=*/1,
                           /*max=*/5);
  std::shared_ptr<Node> async_known_many =
      model::MakeAsyncKnownRatioNode({1, "async_known_many", known_many},
                                     num_inputs_per_output, {known_parameter});
  known_many->record_element();
  known_many->add_processing_time(100);
  known_many->add_input(async_known_many);
  async_known_many->record_element();
  async_known_many->add_processing_time(300);
  Model::NodeValues input_times;
  input_times[kModelInputTimeKey] = input_time;
  Model::ParameterGradients gradients;
  known_parameter->value = 1;
  double output_time = known_many->OutputTime(&input_times, &gradients);
  known_parameter->value += kParameterStep;
  double new_output_time = known_many->OutputTime(&input_times, nullptr);
  EXPECT_NEAR(gradients[std::make_pair(async_known_many->long_name(),
                                       known_parameter->name)],
              (new_output_time - output_time) / kParameterStep,
              kComparisonPrecision);
}

TEST(UnknownRatioGradientTest, Model) {
  const double input_time = 100;
  const int64_t num_inputs_per_output = 2;
  std::shared_ptr<Node> unknown_many =
      model::MakeUnknownRatioNode({0, "unknown_many", nullptr});
  std::shared_ptr<Parameter> known_parameter =
      model::MakeParameter("parallelism",
                           std::make_shared<SharedState>(
                               /*value=*/model::kAutotune, nullptr, nullptr),
                           /*min=*/1,
                           /*max=*/5);
  std::shared_ptr<Node> async_known_many =
      model::MakeAsyncKnownRatioNode({1, "async_known_many", unknown_many},
                                     num_inputs_per_output, {known_parameter});
  unknown_many->record_element();
  unknown_many->add_processing_time(100);
  unknown_many->add_input(async_known_many);
  async_known_many->record_element();
  async_known_many->add_processing_time(300);
  Model::NodeValues input_times;
  input_times[kModelInputTimeKey] = input_time;
  Model::ParameterGradients gradients;
  known_parameter->value = 1;
  double output_time = unknown_many->OutputTime(&input_times, &gradients);
  known_parameter->value += kParameterStep;
  double new_output_time = unknown_many->OutputTime(&input_times, nullptr);
  EXPECT_NEAR(gradients[std::make_pair(async_known_many->long_name(),
                                       known_parameter->name)],
              (new_output_time - output_time) / kParameterStep,
              kComparisonPrecision);
}

TEST(UnknownGradientTest, Model) {
  const double input_time = 100;
  const int64_t num_inputs_per_output = 2;
  std::shared_ptr<Node> unknown =
      model::MakeUnknownNode({0, "unknown", nullptr});
  std::shared_ptr<Parameter> known_parameter =
      model::MakeParameter("parallelism",
                           std::make_shared<SharedState>(
                               /*value=*/model::kAutotune, nullptr, nullptr),
                           /*min=*/1,
                           /*max=*/5);
  std::shared_ptr<Node> async_known_many =
      model::MakeAsyncKnownRatioNode({1, "async_known_many", unknown},
                                     num_inputs_per_output, {known_parameter});
  unknown->record_element();
  unknown->add_processing_time(100);
  unknown->add_input(async_known_many);
  async_known_many->record_element();
  async_known_many->add_processing_time(300);
  Model::NodeValues input_times;
  input_times[kModelInputTimeKey] = input_time;
  Model::ParameterGradients gradients;
  known_parameter->value = 1;
  double output_time = unknown->OutputTime(&input_times, &gradients);
  known_parameter->value += kParameterStep;
  double new_output_time = unknown->OutputTime(&input_times, nullptr);
  EXPECT_NEAR(gradients[std::make_pair(async_known_many->long_name(),
                                       known_parameter->name)],
              (new_output_time - output_time) / kParameterStep,
              kComparisonPrecision);
}

TEST(SnapshotTest, Model) {
  std::shared_ptr<Node> root =
      model::MakeUnknownNode({0, std::to_string(0), nullptr});
  std::shared_ptr<Node> current = root;

  int64_t num_nodes = 20;
  for (int64_t i = 1; i < num_nodes; i++) {
    std::shared_ptr<Node> input =
        model::MakeUnknownNode({i, std::to_string(i), current});
    input->set_autotune(std::rand() % 2 == 1);
    current->add_input(input);
    current = input;
  }

  std::shared_ptr<Node> cloned_root = root->Snapshot();
  current = root;
  std::shared_ptr<Node> cloned_current = cloned_root;

  for (int64_t i = 0; i < num_nodes; i++) {
    EXPECT_EQ(current->id(), cloned_current->id());
    EXPECT_EQ(current->name(), cloned_current->name());
    EXPECT_EQ(current->autotune(), cloned_current->autotune());
    EXPECT_NE(current.get(), cloned_current.get());

    if (i > 0) {
      EXPECT_EQ(current->output()->long_name(),
                cloned_current->output()->long_name());
      EXPECT_EQ(current->output()->autotune(),
                cloned_current->output()->autotune());
      EXPECT_NE(current->output(), cloned_current->output());
    } else {
      EXPECT_EQ(current->output(), nullptr);
      EXPECT_EQ(cloned_current->output(), nullptr);
    }

    if (i < num_nodes - 1) {
      current = current->inputs().front();
      cloned_current = cloned_current->inputs().front();
    }
  }
}

TEST(SaveModelTest, Model) {
  model::Model model;
  std::shared_ptr<Node> root = model::MakeUnknownNode({0, "unknown0", nullptr});
  model.AddNode([&root](model::Node::Args args) { return root; }, root->name(),
                nullptr, &root);
  std::shared_ptr<Node> current = root;

  int64_t num_nodes = 20;
  for (int64_t i = 1; i < num_nodes; i++) {
    std::shared_ptr<Node> input;
    switch (i % 6) {
      case 0:
        input = model::MakeInterleaveManyNode(
            {i, "interleave_many" + std::to_string(i), current},
            {model::MakeParameter("cycle_length", nullptr, /*min=*/1,
                                  /*max=*/1)});
        break;
      case 1:
        input = model::MakeAsyncInterleaveManyNode(
            {i, "async_interleave_many", current},
            {model::MakeParameter(
                 "parallelism",
                 std::make_shared<SharedState>(
                     /*value=*/model::kAutotune, nullptr, nullptr),
                 /*min=*/1,
                 /*max=*/7),
             model::MakeParameter("cycle_length", nullptr, /*min=*/1,
                                  /*max=*/1)});
        break;
      case 2:
        input = model::MakeKnownRatioNode(
            {i, "known_many" + std::to_string(i), current}, 3);
        break;
      case 3:
        input = model::MakeAsyncKnownRatioNode(
            {i, "async_known_many", current}, 4,
            {model::MakeParameter(
                 "parallelism",
                 std::make_shared<SharedState>(
                     /*value=*/model::kAutotune, nullptr, nullptr),
                 /*min=*/1,
                 /*max=*/5),
             model::MakeParameter("cycle_length", nullptr, /*min=*/1,
                                  /*max=*/1)});
        break;
      case 4:
        input = model::MakeUnknownRatioNode(
            {i, "unknown_many" + std::to_string(i), current});
        break;
      default:
        input =
            model::MakeUnknownNode({i, "unknown" + std::to_string(i), current});
    }
    input->record_element();
    input->add_processing_time(i * 50);
    input->record_buffer_event(i * 33, i * 5);
    input->set_autotune(true);
    model.AddNode([&input](model::Node::Args args) { return input; },
                  input->name(), current, &input);
    current = input;
  }

  // Make Save->Load roundtrip.
  Env* env = Env::Default();
  string tmpFile;
  EXPECT_TRUE(env->LocalTempFilename(&tmpFile));
  tmpFile += "_autotune_model_test";

  ModelProto::OptimizationParams optimization_params;
  optimization_params.set_algorithm(AutotuneAlgorithm::GRADIENT_DESCENT);
  optimization_params.set_cpu_budget(64);
  optimization_params.set_ram_budget(1024);
  optimization_params.set_model_input_time(43653.34534);
  TF_ASSERT_OK(
      model.Save(tmpFile, model.output()->Snapshot(), optimization_params));

  std::unique_ptr<model::Model> restored_model;
  ModelProto::OptimizationParams restored_optimization_params;
  TF_ASSERT_OK(
      model.Load(tmpFile, &restored_model, &restored_optimization_params));
  TF_ASSERT_OK(env->DeleteFile(tmpFile));

  // Check optimization parameters.
  EXPECT_EQ(optimization_params.algorithm(),
            restored_optimization_params.algorithm());
  EXPECT_EQ(optimization_params.cpu_budget(),
            restored_optimization_params.cpu_budget());
  EXPECT_EQ(optimization_params.ram_budget(),
            restored_optimization_params.ram_budget());
  EXPECT_EQ(optimization_params.model_input_time(),
            restored_optimization_params.model_input_time());

  std::shared_ptr<Node> restored_root = restored_model->output();
  std::shared_ptr<Node> restored_current = restored_root;
  current = root;
  EXPECT_EQ(current->output(), nullptr);
  EXPECT_EQ(restored_current->output(), nullptr);
  while (!current->inputs().empty() && !restored_current->inputs().empty()) {
    EXPECT_EQ(current->id(), restored_current->id());
    EXPECT_EQ(current->name(), restored_current->name());
    EXPECT_EQ(current->autotune(), restored_current->autotune());
    Model::NodeValues input_times_actual, input_times_expected;
    input_times_actual.clear();
    input_times_expected.clear();
    EXPECT_EQ(current->OutputTime(&input_times_actual, nullptr),
              restored_current->OutputTime(&input_times_expected, nullptr));
    EXPECT_EQ(current->TotalBufferedBytes(),
              restored_current->TotalBufferedBytes());
    EXPECT_EQ(current->TotalMaximumBufferedBytes(),
              restored_current->TotalMaximumBufferedBytes());
    EXPECT_EQ(current->Ratio(), restored_current->Ratio());
    EXPECT_NE(current.get(), restored_current.get());

    current = current->inputs().front();
    restored_current = restored_current->inputs().front();

    EXPECT_EQ(current->output()->long_name(), current->output()->long_name());
    EXPECT_EQ(current->output()->autotune(),
              restored_current->output()->autotune());
    EXPECT_NE(current->output(), restored_current->output());
  }
  EXPECT_TRUE(current->inputs().empty());
  EXPECT_TRUE(restored_current->inputs().empty());
}

class ComputeWaitTimeTest
    : public ::testing::TestWithParam<std::tuple<double, double, double>> {};

TEST_P(ComputeWaitTimeTest, Model) {
  const double producer_time = std::get<0>(GetParam());
  const double consumer_time = std::get<1>(GetParam());
  const double buffer_size = std::get<2>(GetParam());

  double producer_time_derivative = 0.0L;
  double consumer_time_derivative = 0.0L;
  double buffer_size_derivative = 0.0L;

  double wait_time = model::Node::ComputeWaitTime(
      producer_time, consumer_time, buffer_size, &producer_time_derivative,
      &consumer_time_derivative, &buffer_size_derivative);

  double new_wait_time = model::Node::ComputeWaitTime(
      producer_time + kParameterStep, consumer_time, buffer_size, nullptr,
      nullptr, nullptr);
  EXPECT_NEAR(producer_time_derivative,
              (new_wait_time - wait_time) / kParameterStep,
              kComparisonPrecision);

  if (producer_time >= kParameterStep) {
    new_wait_time = model::Node::ComputeWaitTime(producer_time - kParameterStep,
                                                 consumer_time, buffer_size,
                                                 nullptr, nullptr, nullptr);
    EXPECT_NEAR(producer_time_derivative,
                (wait_time - new_wait_time) / kParameterStep,
                kComparisonPrecision);
  }

  new_wait_time = model::Node::ComputeWaitTime(
      producer_time, consumer_time + kParameterStep, buffer_size, nullptr,
      nullptr, nullptr);
  EXPECT_NEAR(consumer_time_derivative,
              (new_wait_time - wait_time) / kParameterStep,
              kComparisonPrecision);

  if (consumer_time >= kParameterStep) {
    new_wait_time = model::Node::ComputeWaitTime(
        producer_time, consumer_time - kParameterStep, buffer_size, nullptr,
        nullptr, nullptr);
    EXPECT_NEAR(consumer_time_derivative,
                (wait_time - new_wait_time) / kParameterStep,
                kComparisonPrecision);
  }

  new_wait_time = model::Node::ComputeWaitTime(producer_time, consumer_time,
                                               buffer_size + kParameterStep,
                                               nullptr, nullptr, nullptr);
  EXPECT_NEAR(buffer_size_derivative,
              (new_wait_time - wait_time) / kParameterStep,
              kComparisonPrecision);

  if (buffer_size >= kParameterStep) {
    new_wait_time = model::Node::ComputeWaitTime(producer_time, consumer_time,
                                                 buffer_size - kParameterStep,
                                                 nullptr, nullptr, nullptr);
    EXPECT_NEAR(buffer_size_derivative,
                (wait_time - new_wait_time) / kParameterStep,
                kComparisonPrecision);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Test, ComputeWaitTimeTest,
    ::testing::Combine(::testing::Values(0, 20, 40, 80, 100),
                       ::testing::Values(0, 20, 40, 80, 100),
                       ::testing::Values(0, 1, 2, 4, 10, 20, 40)));

class SelfProcessingTimeTest : public ::testing::TestWithParam<int64_t> {};

TEST_P(SelfProcessingTimeTest, Model) {
  const int64_t add_times = GetParam();
  std::shared_ptr<Node> source = model::MakeSourceNode({0, "source", nullptr});
  for (int i = 0; i < add_times; i++) {
    source->add_processing_time(i);
    source->record_element();
  }
  double self_processing_time =
      (add_times == 0 ? 0.0 : (static_cast<double>(add_times) - 1.0) / 2.0);
  EXPECT_EQ(source->SelfProcessingTime(), self_processing_time);
}

INSTANTIATE_TEST_SUITE_P(Test, SelfProcessingTimeTest,
                         ::testing::Values(0, 1, 2, 5, 10, 20, 40));

class OptimizeZeroRamBudgetTest
    : public ::testing::TestWithParam<model::AutotuneAlgorithm> {};

TEST_P(OptimizeZeroRamBudgetTest, Model) {
  const model::AutotuneAlgorithm algorithm = GetParam();

  std::shared_ptr<mutex> mutex1 = std::make_shared<mutex>();
  std::shared_ptr<condition_variable> cv1 =
      std::make_shared<condition_variable>();
  std::shared_ptr<Node> node1 = model::MakeAsyncKnownRatioNode(
      {1, "1", nullptr}, 2,
      {model::MakeParameter("parallelism",
                            std::make_shared<SharedState>(
                                /*value=*/model::kAutotune, mutex1, cv1),
                            /*min=*/1, /*max=*/5)});
  node1->record_buffer_event(1, 1);
  node1->record_element();

  std::shared_ptr<mutex> mutex2 = std::make_shared<mutex>();
  std::shared_ptr<condition_variable> cv2 =
      std::make_shared<condition_variable>();
  std::shared_ptr<Node> node2 = model::MakeAsyncKnownRatioNode(
      {2, "2", node1}, 5,
      {model::MakeParameter("buffer_size",
                            std::make_shared<SharedState>(
                                /*value=*/model::kAutotune, mutex2, cv2),
                            /*min=*/0, /*max=*/6)});
  node2->record_buffer_event(1, 1);
  node2->record_element();

  std::shared_ptr<mutex> mutex3 = std::make_shared<mutex>();
  std::shared_ptr<condition_variable> cv3 =
      std::make_shared<condition_variable>();
  std::shared_ptr<Node> node3 = model::MakeAsyncInterleaveManyNode(
      {3, "3", node2},
      {model::MakeParameter("parallelism",
                            std::make_shared<SharedState>(
                                /*value=*/model::kAutotune, mutex3, cv3),
                            /*min=*/1, /*max=*/7),
       model::MakeParameter(kCycleLength, nullptr,
                            /*min=*/1,
                            /*max=*/1)});
  node3->record_buffer_event(1, 1);
  node3->record_element();

  EXPECT_EQ(node1->parameter_value("parallelism"), model::kAutotune);
  EXPECT_EQ(node2->parameter_value("buffer_size"), model::kAutotune);
  EXPECT_EQ(node3->parameter_value("parallelism"), model::kAutotune);

  model::Model model;
  model.AddNode([&node1](model::Node::Args args) { return node1; }, "1",
                nullptr, &node1);
  model.AddNode([&node2](model::Node::Args args) { return node2; }, "2", node1,
                &node2);
  model.AddNode([&node3](model::Node::Args args) { return node3; }, "3", node2,
                &node3);

  CancellationManager cancellation_manager;
  model.Optimize(algorithm, 40, 0, 0, &cancellation_manager);
  EXPECT_EQ(node1->parameter_value("parallelism"), 1);
  EXPECT_EQ(node2->parameter_value("buffer_size"), 0);
  EXPECT_EQ(node3->parameter_value("parallelism"), 1);
}

INSTANTIATE_TEST_SUITE_P(Test, OptimizeZeroRamBudgetTest,
                         ::testing::Values(0, 1, 2, 3));

TEST(RecordTimeTest, RecordTimeTest) {
  std::shared_ptr<Node> source = model::MakeSourceNode({});
  EXPECT_FALSE(source->is_recording());
  source->record_start(100);
  EXPECT_TRUE(source->is_recording());
  source->record_stop(200);
  EXPECT_FALSE(source->is_recording());
}

TEST(ModelTest, ModelMetrics) {
  CellReader<std::string> cell_reader("/tensorflow/data/model");
  model::Model model;
  std::shared_ptr<Node> root = model::MakeUnknownNode({0, "unknown0", nullptr});
  model.AddNode([&root](model::Node::Args args) { return root; }, root->name(),
                nullptr, &root);
  std::string model_id = strings::StrCat(reinterpret_cast<uintptr_t>(&model));
  EXPECT_THAT(cell_reader.Read(model_id),
              AllOf(HasSubstr("key: 0"), HasSubstr("name: \"unknown0\""),
                    HasSubstr("autotune: true")));
}

TEST(ModelTest, ModelCollectAndDestroyRaceCondition) {
  CellReader<std::string> cell_reader("/tensorflow/data/model");
  auto* model = new model::Model();
  std::string model_id = strings::StrCat(reinterpret_cast<uintptr_t>(model));
  thread::ThreadPool threads(Env::Default(), "collect_and_destroy_race", 2);
  threads.Schedule([&]() { cell_reader.Read(model_id); });
  threads.Schedule([&]() { delete model; });
}

class ModelTimingTest : public ::testing::Test {
 public:
  // Builds a Model from its text proto.
  void BuildModelFromProto(const std::string& model_pbtxt) {
    ModelProto model_proto;
    protobuf::TextFormat::ParseFromString(model_pbtxt, &model_proto);
    TF_CHECK_OK(Model::FromProto(model_proto, &model_));
    auto nodes = model_->output()->CollectNodes(
        TraversalOrder::BFS, [](const std::shared_ptr<Node>) { return true; });
    node_map_.clear();
    node_map_[model_->output()->id()] = model_->output().get();
    for (const auto& node : nodes) {
      node_map_[node->id()] = node.get();
    }
  }

  // Computes the timing given a Model text proto.
  void ComputeModelTiming(const std::string& model_pbtxt) {
    BuildModelFromProto(model_pbtxt);
    model_timing_ = std::make_unique<ModelTiming>(model_->output());
  }

  // Gets the timing information of a node given its id.
  const ModelTiming::NodeTiming* GetNodeTiming(int64_t node_id) const {
    return model_timing_->GetTiming(node_map_.at(node_id));
  }

  // Gets the node given its id.
  const Node* GetNode(int64_t node_id) const { return node_map_.at(node_id); }
  Node* MutableGetNode(int64_t node_id) const { return node_map_.at(node_id); }

 protected:
  std::unique_ptr<Model> model_;
  std::unique_ptr<ModelTiming> model_timing_;
  absl::flat_hash_map<int64_t, Node*> node_map_;
};

TEST_F(ModelTimingTest, Interleave) {
  ComputeModelTiming(R"pb(
    nodes: {
      key: 1
      value: {
        id: 1
        name: "Batch"
        autotune: true
        num_elements: 100
        processing_time: 1000
        node_class: KNOWN_RATIO
        ratio: 1
        inputs: 2
      }
    }
    nodes: {
      key: 2
      value: {
        id: 2
        name: "Interleave"
        autotune: true
        num_elements: 100
        processing_time: 1000
        node_class: INTERLEAVE_MANY
        inputs: 3
        inputs: 4
        parameters: { name: "cycle_length" value: 2 tunable: false }
      }
    }
    nodes: {
      key: 3
      value: {
        id: 3
        name: "Batch"
        autotune: true
        num_elements: 60
        processing_time: 1200
        node_class: KNOWN_RATIO
        ratio: 1
      }
    }
    nodes: {
      key: 4
      value: {
        id: 4
        name: "Batch"
        autotune: true
        num_elements: 40
        processing_time: 800
        node_class: KNOWN_RATIO
        ratio: 1
      }
    }
    output: 1
  )pb");

  EXPECT_DOUBLE_EQ(1, GetNodeTiming(/*node_id=*/1)->pipeline_ratio);
  EXPECT_DOUBLE_EQ(1, GetNodeTiming(/*node_id=*/2)->pipeline_ratio);
  EXPECT_DOUBLE_EQ(0.5, GetNodeTiming(/*node_id=*/3)->pipeline_ratio);
  EXPECT_DOUBLE_EQ(0.5, GetNodeTiming(/*node_id=*/4)->pipeline_ratio);

  EXPECT_DOUBLE_EQ(10, GetNodeTiming(/*node_id=*/1)->self_time_nsec);
  EXPECT_DOUBLE_EQ(10, GetNodeTiming(/*node_id=*/2)->self_time_nsec);
  EXPECT_DOUBLE_EQ(20, GetNodeTiming(/*node_id=*/3)->self_time_nsec);
  EXPECT_DOUBLE_EQ(20, GetNodeTiming(/*node_id=*/4)->self_time_nsec);

  EXPECT_DOUBLE_EQ(40, GetNodeTiming(/*node_id=*/1)->total_time_nsec);
  EXPECT_DOUBLE_EQ(30, GetNodeTiming(/*node_id=*/2)->total_time_nsec);
  EXPECT_DOUBLE_EQ(20, GetNodeTiming(/*node_id=*/3)->total_time_nsec);
  EXPECT_DOUBLE_EQ(20, GetNodeTiming(/*node_id=*/4)->total_time_nsec);
}

// When the `parallelism` parameter of a `ParallelInterleave` is not present in
// the model proto, its default value should be 1.
TEST_F(ModelTimingTest, TestDefaultParallelismInParallelInterleave) {
  const int32_t parallelism = 1;
  const int32_t deterministic = 1;
  const int32_t cycle_length = 3;
  ComputeModelTiming(strings::Printf(
      R"pb(
        nodes: {
          key: 1
          value: {
            id: 1
            name: "Batch"
            autotune: true
            num_elements: 100
            processing_time: 1000
            node_class: KNOWN_RATIO
            ratio: 1
            inputs: 2
          }
        }
        nodes: {
          key: 2
          value: {
            id: 2
            name: "ParallelInterleaveV4"
            autotune: true
            num_elements: 100
            processing_time: 2000
            node_class: ASYNC_INTERLEAVE_MANY
            inputs: 3
            inputs: 4
            inputs: 5
            inputs: 6
            inputs: 7
            parameters: { name: "deterministic" value: %d tunable: false }
            parameters: { name: "cycle_length" value: %d tunable: false }
          }
        }
        nodes: {
          key: 3
          value: {
            id: 3
            name: "Batch"
            autotune: true
            num_elements: 60
            processing_time: 60
            node_class: KNOWN_RATIO
            ratio: 1
          }
        }
        nodes: {
          key: 4
          value: {
            id: 4
            name: "Batch"
            autotune: true
            num_elements: 60
            processing_time: 1200
            node_class: KNOWN_RATIO
            ratio: 1
          }
        }
        nodes: {
          key: 5
          value: {
            id: 5
            name: "Batch"
            autotune: true
            num_elements: 40
            processing_time: 1200
            node_class: KNOWN_RATIO
            ratio: 1
          }
        }
        nodes: {
          key: 6
          value: {
            id: 6
            name: "Batch"
            autotune: true
            num_elements: 60
            processing_time: 2400
            node_class: KNOWN_RATIO
            ratio: 1
          }
        }
        nodes: {
          key: 7
          value: {
            id: 7
            name: "Batch"
            autotune: false  # Marked as an inactive input
            num_elements: 40
            processing_time: 2000
            node_class: KNOWN_RATIO
            ratio: 1
          }
        }
        output: 1
      )pb",
      deterministic, cycle_length));

  EXPECT_DOUBLE_EQ(1, GetNodeTiming(/*node_id=*/1)->pipeline_ratio);
  EXPECT_DOUBLE_EQ(1, GetNodeTiming(/*node_id=*/2)->pipeline_ratio);
  EXPECT_DOUBLE_EQ(1.0 / cycle_length,
                   GetNodeTiming(/*node_id=*/3)->pipeline_ratio);
  EXPECT_DOUBLE_EQ(1.0 / cycle_length,
                   GetNodeTiming(/*node_id=*/4)->pipeline_ratio);
  EXPECT_DOUBLE_EQ(1.0 / cycle_length,
                   GetNodeTiming(/*node_id=*/5)->pipeline_ratio);
  EXPECT_DOUBLE_EQ(1.0 / cycle_length,
                   GetNodeTiming(/*node_id=*/6)->pipeline_ratio);
  EXPECT_DOUBLE_EQ(0, GetNodeTiming(/*node_id=*/7)->pipeline_ratio);

  const double expected_self_time_1 = 1000.0 / 100.0;
  const double expected_self_time_2 = 2000.0 / 100.0 / parallelism;
  const double expected_self_time_3 = 60.0 / 60.0;
  const double expected_self_time_4 = 1200.0 / 60.0;
  const double expected_self_time_5 = 1200.0 / 40.0;
  const double expected_self_time_6 = 2400.0 / 60.0;
  const double expected_self_time_7 = 2000.0 / 40.0;

  EXPECT_DOUBLE_EQ(expected_self_time_1,
                   GetNodeTiming(/*node_id=*/1)->self_time_nsec);
  EXPECT_DOUBLE_EQ(expected_self_time_2,
                   GetNodeTiming(/*node_id=*/2)->self_time_nsec);
  EXPECT_DOUBLE_EQ(expected_self_time_3,
                   GetNodeTiming(/*node_id=*/3)->self_time_nsec);
  EXPECT_DOUBLE_EQ(expected_self_time_4,
                   GetNodeTiming(/*node_id=*/4)->self_time_nsec);
  EXPECT_DOUBLE_EQ(expected_self_time_5,
                   GetNodeTiming(/*node_id=*/5)->self_time_nsec);
  EXPECT_DOUBLE_EQ(expected_self_time_6,
                   GetNodeTiming(/*node_id=*/6)->self_time_nsec);
  EXPECT_DOUBLE_EQ(expected_self_time_7,
                   GetNodeTiming(/*node_id=*/7)->self_time_nsec);

  EXPECT_DOUBLE_EQ(expected_self_time_1,
                   GetNodeTiming(/*node_id=*/1)->total_time_nsec);
  EXPECT_DOUBLE_EQ(expected_self_time_3,
                   GetNodeTiming(/*node_id=*/3)->total_time_nsec);
  EXPECT_DOUBLE_EQ(expected_self_time_4,
                   GetNodeTiming(/*node_id=*/4)->total_time_nsec);
  EXPECT_DOUBLE_EQ(expected_self_time_5,
                   GetNodeTiming(/*node_id=*/5)->total_time_nsec);
  EXPECT_DOUBLE_EQ(expected_self_time_6,
                   GetNodeTiming(/*node_id=*/6)->total_time_nsec);
  EXPECT_DOUBLE_EQ(0, GetNodeTiming(/*node_id=*/7)->total_time_nsec);

  const double max_input_time = expected_self_time_6;
  double input_throughput = 1.0 / expected_self_time_4 +
                            1.0 / expected_self_time_5 +
                            1.0 / expected_self_time_6;
  const double active_inputs = 3.0;
  double expected_input_time;
  if (deterministic == 1) {
    expected_input_time = max_input_time / std::min(parallelism, cycle_length);
  } else {
    if (std::min(parallelism, cycle_length) < active_inputs) {
      input_throughput *= std::min(parallelism, cycle_length) / active_inputs;
    }
    expected_input_time = 1.0 / input_throughput;
  }
  EXPECT_DOUBLE_EQ(expected_input_time + expected_self_time_2,
                   GetNodeTiming(/*node_id=*/2)->total_time_nsec);
}

class ParallelInterleaveTimingTest
    : public ModelTimingTest,
      public ::testing::WithParamInterface<
          std::tuple<int32_t, int32_t, int32_t>> {};

TEST_P(ParallelInterleaveTimingTest, ScenarioTest) {
  const int32_t parallelism = std::get<0>(GetParam());
  const int32_t deterministic = std::get<1>(GetParam());
  const int32_t cycle_length = std::get<2>(GetParam());
  ComputeModelTiming(strings::Printf(
      R"pb(
        nodes: {
          key: 1
          value: {
            id: 1
            name: "Batch"
            autotune: true
            num_elements: 100
            processing_time: 1000
            node_class: KNOWN_RATIO
            ratio: 1
            inputs: 2
          }
        }
        nodes: {
          key: 2
          value: {
            id: 2
            name: "ParallelInterleaveV4"
            autotune: true
            num_elements: 100
            processing_time: 2000
            node_class: ASYNC_INTERLEAVE_MANY
            inputs: 3
            inputs: 4
            inputs: 5
            inputs: 6
            inputs: 7
            parameters: {
              name: "parallelism"
              value: %d
              min: 1
              max: 10
              tunable: true
            }
            parameters: { name: "deterministic" value: %d tunable: false }
            parameters: { name: "cycle_length" value: %d tunable: false }
          }
        }
        nodes: {
          key: 3
          value: {
            id: 3
            name: "Batch"
            autotune: true
            num_elements: 60
            processing_time: 60
            node_class: KNOWN_RATIO
            ratio: 1
          }
        }
        nodes: {
          key: 4
          value: {
            id: 4
            name: "Batch"
            autotune: true
            num_elements: 60
            processing_time: 1200
            node_class: KNOWN_RATIO
            ratio: 1
          }
        }
        nodes: {
          key: 5
          value: {
            id: 5
            name: "Batch"
            autotune: true
            num_elements: 40
            processing_time: 1200
            node_class: KNOWN_RATIO
            ratio: 1
          }
        }
        nodes: {
          key: 6
          value: {
            id: 6
            name: "Batch"
            autotune: true
            num_elements: 60
            processing_time: 2400
            node_class: KNOWN_RATIO
            ratio: 1
          }
        }
        nodes: {
          key: 7
          value: {
            id: 7
            name: "Batch"
            autotune: false  # Marked as an inactive input
            num_elements: 40
            processing_time: 2000
            node_class: KNOWN_RATIO
            ratio: 1
          }
        }
        output: 1
      )pb",
      parallelism, deterministic, cycle_length));

  EXPECT_DOUBLE_EQ(1, GetNodeTiming(/*node_id=*/1)->pipeline_ratio);
  EXPECT_DOUBLE_EQ(1, GetNodeTiming(/*node_id=*/2)->pipeline_ratio);
  EXPECT_DOUBLE_EQ(1.0 / cycle_length,
                   GetNodeTiming(/*node_id=*/3)->pipeline_ratio);
  EXPECT_DOUBLE_EQ(1.0 / cycle_length,
                   GetNodeTiming(/*node_id=*/4)->pipeline_ratio);
  EXPECT_DOUBLE_EQ(1.0 / cycle_length,
                   GetNodeTiming(/*node_id=*/5)->pipeline_ratio);
  EXPECT_DOUBLE_EQ(1.0 / cycle_length,
                   GetNodeTiming(/*node_id=*/6)->pipeline_ratio);
  EXPECT_DOUBLE_EQ(0, GetNodeTiming(/*node_id=*/7)->pipeline_ratio);

  const double expected_self_time_1 = 1000.0 / 100.0;
  const double expected_self_time_2 = 2000.0 / 100.0 / parallelism;
  const double expected_self_time_3 = 60.0 / 60.0;
  const double expected_self_time_4 = 1200.0 / 60.0;
  const double expected_self_time_5 = 1200.0 / 40.0;
  const double expected_self_time_6 = 2400.0 / 60.0;
  const double expected_self_time_7 = 2000.0 / 40.0;

  EXPECT_DOUBLE_EQ(expected_self_time_1,
                   GetNodeTiming(/*node_id=*/1)->self_time_nsec);
  EXPECT_DOUBLE_EQ(expected_self_time_2,
                   GetNodeTiming(/*node_id=*/2)->self_time_nsec);
  EXPECT_DOUBLE_EQ(expected_self_time_3,
                   GetNodeTiming(/*node_id=*/3)->self_time_nsec);
  EXPECT_DOUBLE_EQ(expected_self_time_4,
                   GetNodeTiming(/*node_id=*/4)->self_time_nsec);
  EXPECT_DOUBLE_EQ(expected_self_time_5,
                   GetNodeTiming(/*node_id=*/5)->self_time_nsec);
  EXPECT_DOUBLE_EQ(expected_self_time_6,
                   GetNodeTiming(/*node_id=*/6)->self_time_nsec);
  EXPECT_DOUBLE_EQ(expected_self_time_7,
                   GetNodeTiming(/*node_id=*/7)->self_time_nsec);

  EXPECT_DOUBLE_EQ(expected_self_time_1,
                   GetNodeTiming(/*node_id=*/1)->total_time_nsec);
  EXPECT_DOUBLE_EQ(expected_self_time_3,
                   GetNodeTiming(/*node_id=*/3)->total_time_nsec);
  EXPECT_DOUBLE_EQ(expected_self_time_4,
                   GetNodeTiming(/*node_id=*/4)->total_time_nsec);
  EXPECT_DOUBLE_EQ(expected_self_time_5,
                   GetNodeTiming(/*node_id=*/5)->total_time_nsec);
  EXPECT_DOUBLE_EQ(expected_self_time_6,
                   GetNodeTiming(/*node_id=*/6)->total_time_nsec);
  EXPECT_DOUBLE_EQ(0, GetNodeTiming(/*node_id=*/7)->total_time_nsec);

  const double max_input_time = expected_self_time_6;
  double input_throughput = 1.0 / expected_self_time_4 +
                            1.0 / expected_self_time_5 +
                            1.0 / expected_self_time_6;
  const double active_inputs = 3.0;
  double expected_input_time;
  if (deterministic == 1) {
    expected_input_time = max_input_time / std::min(parallelism, cycle_length);
  } else {
    if (std::min(parallelism, cycle_length) < active_inputs) {
      input_throughput *= std::min(parallelism, cycle_length) / active_inputs;
    }
    expected_input_time = 1.0 / input_throughput;
  }
  EXPECT_DOUBLE_EQ(expected_input_time + expected_self_time_2,
                   GetNodeTiming(/*node_id=*/2)->total_time_nsec);
}

INSTANTIATE_TEST_SUITE_P(ParallelInterleaveTimingTest,
                         ParallelInterleaveTimingTest,
                         ::testing::Combine(::testing::Values(1, 2, 3),
                                            ::testing::Values(0, 1),
                                            ::testing::Values(1, 2, 3)));

TEST_F(ModelTimingTest, ParallelInterleave_Batch_ParallelMap) {
  ComputeModelTiming(R"pb(
    nodes: {
      key: 1
      value: {
        id: 1
        name: "Batch"
        autotune: true
        num_elements: 100
        processing_time: 1000
        node_class: KNOWN_RATIO
        ratio: 1
        inputs: 2
      }
    }
    nodes: {
      key: 2
      value: {
        id: 2
        name: "ParallelInterleaveV4"
        autotune: true
        num_elements: 100
        processing_time: 2000
        node_class: ASYNC_INTERLEAVE_MANY
        inputs: 3
        inputs: 4
        inputs: 5
        parameters: {
          name: "parallelism"
          value: 2
          min: 1
          max: 10
          tunable: true
        }
        parameters: { name: "cycle_length" value: 2 tunable: false }
        parameters: { name: "deterministic" value: 0 tunable: false }
      }
    }
    nodes: {
      key: 3
      value: {
        id: 3
        name: "Batch"
        autotune: true
        num_elements: 60
        processing_time: 60
        node_class: KNOWN_RATIO
        ratio: 1
      }
    }
    nodes: {
      key: 4
      value: {
        id: 4
        name: "Batch"
        autotune: true
        num_elements: 60
        processing_time: 1200
        node_class: KNOWN_RATIO
        ratio: 2
        inputs: 6
      }
    }
    nodes: {
      key: 5
      value: {
        id: 5
        name: "Batch"
        autotune: true
        num_elements: 40
        processing_time: 800
        node_class: KNOWN_RATIO
        ratio: 2
        inputs: 7
      }
    }
    nodes: {
      key: 6
      value: {
        id: 6
        name: "ParallelMapV2"
        autotune: true
        num_elements: 120
        processing_time: 2400
        node_class: ASYNC_KNOWN_RATIO
        ratio: 1
        parameters: {
          name: "parallelism"
          value: 2
          min: 1
          max: 16
          tunable: true
        }
      }
    }
    nodes: {
      key: 7
      value: {
        id: 7
        name: "ParallelMapV2"
        autotune: true
        num_elements: 120
        processing_time: 2400
        node_class: ASYNC_KNOWN_RATIO
        ratio: 1
        parameters: {
          name: "parallelism"
          value: 2
          min: 1
          max: 16
          tunable: true
        }
      }
    }
    output: 1
  )pb");

  EXPECT_DOUBLE_EQ(1, GetNodeTiming(/*node_id=*/1)->pipeline_ratio);
  EXPECT_DOUBLE_EQ(1, GetNodeTiming(/*node_id=*/2)->pipeline_ratio);
  EXPECT_DOUBLE_EQ(0.5, GetNodeTiming(/*node_id=*/3)->pipeline_ratio);
  EXPECT_DOUBLE_EQ(0.5, GetNodeTiming(/*node_id=*/4)->pipeline_ratio);
  EXPECT_DOUBLE_EQ(0.5, GetNodeTiming(/*node_id=*/5)->pipeline_ratio);
  EXPECT_DOUBLE_EQ(1, GetNodeTiming(/*node_id=*/6)->pipeline_ratio);
  EXPECT_DOUBLE_EQ(1, GetNodeTiming(/*node_id=*/7)->pipeline_ratio);

  EXPECT_DOUBLE_EQ(10, GetNodeTiming(/*node_id=*/1)->self_time_nsec);
  EXPECT_DOUBLE_EQ(10, GetNodeTiming(/*node_id=*/2)->self_time_nsec);
  EXPECT_DOUBLE_EQ(1, GetNodeTiming(/*node_id=*/3)->self_time_nsec);
  EXPECT_DOUBLE_EQ(20, GetNodeTiming(/*node_id=*/4)->self_time_nsec);
  EXPECT_DOUBLE_EQ(20, GetNodeTiming(/*node_id=*/5)->self_time_nsec);
  EXPECT_DOUBLE_EQ(10, GetNodeTiming(/*node_id=*/6)->self_time_nsec);
  EXPECT_DOUBLE_EQ(10, GetNodeTiming(/*node_id=*/7)->self_time_nsec);

  EXPECT_DOUBLE_EQ(10, GetNodeTiming(/*node_id=*/1)->total_time_nsec);
  EXPECT_DOUBLE_EQ(20, GetNodeTiming(/*node_id=*/2)->total_time_nsec);
  EXPECT_DOUBLE_EQ(1, GetNodeTiming(/*node_id=*/3)->total_time_nsec);
  EXPECT_DOUBLE_EQ(20, GetNodeTiming(/*node_id=*/4)->total_time_nsec);
  EXPECT_DOUBLE_EQ(20, GetNodeTiming(/*node_id=*/5)->total_time_nsec);
  EXPECT_DOUBLE_EQ(10, GetNodeTiming(/*node_id=*/6)->total_time_nsec);
  EXPECT_DOUBLE_EQ(10, GetNodeTiming(/*node_id=*/7)->total_time_nsec);
}

class BufferSizeTest : public ::testing::Test {
 public:
  void ReadModel(const std::string& model_pbtxt) {
    ModelProto model_proto;
    protobuf::TextFormat::ParseFromString(model_pbtxt, &model_proto);
    TF_CHECK_OK(Model::FromProto(model_proto, &model_));
    auto nodes = model_->output()->CollectNodes(
        TraversalOrder::BFS, [](const std::shared_ptr<Node>) { return true; });
    node_map_.clear();
    node_map_[model_->output()->id()] = model_->output();
    for (const auto& node : nodes) {
      node_map_[node->id()] = node;
    }
  }

  // Returns a node given its node id. If node id does not exist, it will fail.
  std::shared_ptr<Node> GetNode(int64_t node_id) const {
    return node_map_.at(node_id);
  }

 protected:
  std::unique_ptr<Model> model_;
  absl::flat_hash_map<int64_t, std::shared_ptr<Node>> node_map_;
};

TEST_F(BufferSizeTest, OptimizeBuffers_PlentyOfMemory) {
  ReadModel(R"pb(
    nodes: {
      key: 1
      value: {
        id: 1
        name: "Prefetch"
        autotune: true
        bytes_produced: 10000
        num_elements: 100
        processing_time: 2000
        node_class: ASYNC_KNOWN_RATIO
        inputs: 2
        ratio: 1
        parameters: {
          name: "buffer_size"
          value: 3
          state_value: 3
          min: 1
          max: 10
          tunable: true
        }
      }
    }
    nodes: {
      key: 2
      value: {
        id: 2
        name: "Prefetch"
        autotune: true
        bytes_produced: 10000
        num_elements: 100
        processing_time: 2000
        node_class: ASYNC_KNOWN_RATIO
        inputs: 3
        ratio: 1
        parameters: {
          name: "buffer_size"
          value: 5
          state_value: 5
          min: 1
          max: 10
          tunable: true
        }
      }
    }
    nodes: {
      key: 3
      value: {
        id: 3
        name: "Prefetch"
        autotune: true
        bytes_produced: 10000
        num_elements: 100
        processing_time: 2000
        node_class: ASYNC_KNOWN_RATIO
        inputs: 4
        ratio: 1
        parameters: {
          name: "buffer_size"
          value: 5
          state_value: 5
          min: 1
          max: 10
          tunable: true
        }
      }
    }
    nodes: {
      key: 4
      value: {
        id: 4
        name: "Prefetch"
        autotune: true
        bytes_produced: 10000
        num_elements: 100
        processing_time: 2000
        node_class: ASYNC_KNOWN_RATIO
        inputs: 5
        ratio: 1
        parameters: {
          name: "buffer_size"
          value: 5
          state_value: 5
          min: 1
          max: 8
          tunable: true
        }
      }
    }
    nodes: {
      key: 5
      value: {
        id: 5
        name: "Prefetch"
        autotune: true
        bytes_produced: 10000
        num_elements: 100
        processing_time: 2000
        node_class: ASYNC_KNOWN_RATIO
        ratio: 1
        parameters: {
          name: "buffer_size"
          value: 8
          state_value: 8
          min: 1
          max: 8
          tunable: true
        }
      }
    }
    output: 1
  )pb");

  std::shared_ptr<Node> node_1 = GetNode(1);
  std::shared_ptr<Node> node_2 = GetNode(2);
  std::shared_ptr<Node> node_3 = GetNode(3);
  std::shared_ptr<Node> node_4 = GetNode(4);
  std::shared_ptr<Node> node_5 = GetNode(5);
  // Set node 1 low watermark to 1 and high watermark to 2. Expect that it is
  // downsized to 2.
  node_1->record_buffer_event(100, 1);
  node_1->record_buffer_event(100, 1);
  EXPECT_EQ(1, node_1->buffered_elements_low());
  EXPECT_EQ(2, node_1->buffered_elements_high());
  // Set node 2 low watermark to 1 and high watermark to 5. Expect that it is
  // not changed.
  node_2->record_buffer_event(100, 1);
  node_2->record_buffer_event(400, 4);
  node_2->record_buffer_event(-100, -1);
  EXPECT_EQ(1, node_2->buffered_elements_low());
  EXPECT_EQ(5, node_2->buffered_elements_high());
  // Set node 3 low watermark to 0 and high watermark to 5. Expect that it is
  // upsized to 10.
  node_3->record_buffer_event(100, 1);
  node_3->record_buffer_event(-100, -1);
  node_3->record_buffer_event(500, 5);
  node_3->record_buffer_event(-100, -1);
  EXPECT_EQ(0, node_3->buffered_elements_low());
  EXPECT_EQ(5, node_3->buffered_elements_high());
  // Set node 4 low watermark to 0 and high watermark to 5. Its max buffer size
  // is set to 8. Expect that it is upsized to 8.
  node_4->record_buffer_event(100, 1);
  node_4->record_buffer_event(-100, -1);
  node_4->record_buffer_event(500, 5);
  node_4->record_buffer_event(-100, -1);
  EXPECT_EQ(0, node_4->buffered_elements_low());
  EXPECT_EQ(5, node_4->buffered_elements_high());
  // Set node 5 low watermark to 1 and high watermark to 2. Its current buffer
  // size is set to 8. Expect that it is downsized to 8/2 rather than (2 - 1 + 1
  // = 3) because downsize is capped to half its size.
  node_5->record_buffer_event(100, 1);
  node_5->record_buffer_event(-100, 1);
  EXPECT_EQ(1, node_5->buffered_elements_low());
  EXPECT_EQ(2, node_5->buffered_elements_high());

  model_->OptimizeBuffers(node_1->Snapshot(), 10000);

  EXPECT_EQ(2, node_1->parameter_value(kBufferSize));
  EXPECT_EQ(5, node_2->parameter_value(kBufferSize));
  EXPECT_EQ(10, node_3->parameter_value(kBufferSize));
  EXPECT_EQ(8, node_4->parameter_value(kBufferSize));
  EXPECT_EQ(6, node_5->parameter_value(kBufferSize));
  EXPECT_EQ(2, node_1->buffered_elements_low());
  EXPECT_EQ(2, node_1->buffered_elements_high());
  EXPECT_EQ(4, node_2->buffered_elements_low());
  EXPECT_EQ(4, node_2->buffered_elements_high());
  EXPECT_EQ(4, node_3->buffered_elements_low());
  EXPECT_EQ(4, node_3->buffered_elements_high());
  EXPECT_EQ(4, node_4->buffered_elements_low());
  EXPECT_EQ(4, node_4->buffered_elements_high());
  EXPECT_EQ(2, node_5->buffered_elements_low());
  EXPECT_EQ(2, node_5->buffered_elements_high());
}

TEST_F(BufferSizeTest, OptimizeBuffers_TightMemory) {
  ReadModel(R"pb(
    nodes: {
      key: 1
      value: {
        id: 1
        name: "Prefetch"
        autotune: true
        bytes_produced: 10000
        num_elements: 100
        processing_time: 2000
        node_class: ASYNC_KNOWN_RATIO
        inputs: 2
        ratio: 1
        parameters: {
          name: "buffer_size"
          value: 5
          state_value: 5
          min: 1
          max: 10
          tunable: true
        }
      }
    }
    nodes: {
      key: 2
      value: {
        id: 2
        name: "Prefetch"
        autotune: true
        bytes_produced: 10000
        num_elements: 100
        processing_time: 2000
        node_class: ASYNC_KNOWN_RATIO
        inputs: 3
        ratio: 1
        parameters: {
          name: "buffer_size"
          value: 5
          state_value: 5
          min: 1
          max: 10
          tunable: true
        }
      }
    }
    nodes: {
      key: 3
      value: {
        id: 3
        name: "Prefetch"
        autotune: true
        bytes_produced: 10000
        num_elements: 100
        processing_time: 2000
        node_class: ASYNC_KNOWN_RATIO
        inputs: 4
        ratio: 1
        parameters: {
          name: "buffer_size"
          value: 5
          state_value: 5
          min: 1
          max: 10
          tunable: true
        }
      }
    }
    nodes: {
      key: 4
      value: {
        id: 4
        name: "Prefetch"
        autotune: true
        bytes_produced: 10000
        num_elements: 100
        processing_time: 2000
        node_class: ASYNC_KNOWN_RATIO
        ratio: 1
        parameters: {
          name: "buffer_size"
          value: 5
          state_value: 5
          min: 1
          max: 8
          tunable: true
        }
      }
    }
    output: 1
  )pb");

  std::shared_ptr<Node> node_1 = GetNode(1);
  std::shared_ptr<Node> node_2 = GetNode(2);
  std::shared_ptr<Node> node_3 = GetNode(3);
  std::shared_ptr<Node> node_4 = GetNode(4);
  // Set low watermark to 0 and high watermark to 5 for all nodes.
  node_1->record_buffer_event(100, 1);
  node_1->record_buffer_event(-100, -1);
  node_1->record_buffer_event(500, 5);
  EXPECT_EQ(0, node_1->buffered_elements_low());
  EXPECT_EQ(5, node_1->buffered_elements_high());
  node_2->record_buffer_event(100, 1);
  node_2->record_buffer_event(100, 1);
  node_2->record_buffer_event(-100, -1);
  node_2->record_buffer_event(-100, -1);
  node_2->record_buffer_event(400, 4);
  node_2->record_buffer_event(100, 1);
  EXPECT_EQ(0, node_2->buffered_elements_low());
  EXPECT_EQ(5, node_2->buffered_elements_high());
  node_3->record_buffer_event(100, 1);
  node_3->record_buffer_event(-100, -1);
  node_3->record_buffer_event(500, 5);
  EXPECT_EQ(0, node_3->buffered_elements_low());
  EXPECT_EQ(5, node_3->buffered_elements_high());
  node_4->record_buffer_event(100, 1);
  node_4->record_buffer_event(-100, -1);
  node_4->record_buffer_event(100, 1);
  node_4->record_buffer_event(-100, -1);
  node_4->record_buffer_event(500, 5);
  node_4->record_buffer_event(-100, -1);
  EXPECT_EQ(0, node_4->buffered_elements_low());
  EXPECT_EQ(5, node_4->buffered_elements_high());

  model_->OptimizeBuffers(node_1->Snapshot(), 3000);

  EXPECT_DOUBLE_EQ(7.0, node_1->parameter_value(kBufferSize));
  EXPECT_DOUBLE_EQ(7.0, node_2->parameter_value(kBufferSize));
  EXPECT_DOUBLE_EQ(7.0, node_3->parameter_value(kBufferSize));
  EXPECT_DOUBLE_EQ(7.0, node_4->parameter_value(kBufferSize));
  EXPECT_EQ(5, node_1->buffered_elements_low());
  EXPECT_EQ(5, node_1->buffered_elements_high());
  EXPECT_EQ(5, node_2->buffered_elements_low());
  EXPECT_EQ(5, node_2->buffered_elements_high());
  EXPECT_EQ(5, node_3->buffered_elements_low());
  EXPECT_EQ(5, node_3->buffered_elements_high());
  EXPECT_EQ(4, node_4->buffered_elements_low());
  EXPECT_EQ(4, node_4->buffered_elements_high());
}

TEST_F(ModelTimingTest, OptimizeStageBased_OneStage) {
  BuildModelFromProto(R"pb(
    nodes: {
      key: 1
      value: {
        id: 1
        name: "ParallelMapV2"
        autotune: true
        num_elements: 100
        processing_time: 5000
        bytes_produced: 10000
        node_class: ASYNC_KNOWN_RATIO
        ratio: 1
        inputs: 2
        parameters: {
          name: "parallelism"
          value: 4
          min: 1
          max: 16
          tunable: true
        }
      }
    }
    nodes: {
      key: 2
      value: {
        id: 2
        name: "Map"
        autotune: true
        num_elements: 100
        processing_time: 3000
        node_class: KNOWN_RATIO
        ratio: 1
        inputs: 3
      }
    }
    nodes: {
      key: 3
      value: {
        id: 3
        name: "SSTable"
        autotune: true
        num_elements: 100
        processing_time: 1000
        node_class: KNOWN_RATIO
        ratio: 2
      }
    }
    output: 1
  )pb");

  CancellationManager cancellation_manager;
  model_->Optimize(AutotuneAlgorithm::STAGE_BASED, 20, 1000, 50,
                   &cancellation_manager);

  EXPECT_EQ(5, GetNode(/*node_id=*/1)->parameter_value("parallelism"));
}

TEST_F(ModelTimingTest, OptimizeStageBased_CappedByParameterMax) {
  BuildModelFromProto(R"pb(
    nodes: {
      key: 1
      value: {
        id: 1
        name: "ParallelMapV2"
        autotune: true
        num_elements: 100
        processing_time: 5000
        bytes_produced: 10000
        node_class: ASYNC_KNOWN_RATIO
        ratio: 1
        inputs: 2
        parameters: { name: "parallelism" value: 4 min: 1 max: 3 tunable: true }
      }
    }
    nodes: {
      key: 2
      value: {
        id: 2
        name: "Map"
        autotune: true
        num_elements: 100
        processing_time: 3000
        node_class: KNOWN_RATIO
        ratio: 1
        inputs: 3
      }
    }
    nodes: {
      key: 3
      value: {
        id: 3
        name: "SSTable"
        autotune: true
        num_elements: 100
        processing_time: 1000
        node_class: KNOWN_RATIO
        ratio: 2
      }
    }
    output: 1
  )pb");

  CancellationManager cancellation_manager;
  model_->Optimize(AutotuneAlgorithm::STAGE_BASED, 20, 1000, 50,
                   &cancellation_manager);

  // The max value is set to 3. Otherwise, the expected parallelism value is 5.
  EXPECT_EQ(3, GetNode(/*node_id=*/1)->parameter_value("parallelism"));
}

TEST_F(ModelTimingTest, OptimizeStageBased_TwoStages) {
  BuildModelFromProto(R"pb(
    nodes: {
      key: 1
      value: {
        id: 1
        name: "ParallelMapV2"
        autotune: true
        num_elements: 100
        processing_time: 25000
        bytes_produced: 10000
        node_class: ASYNC_KNOWN_RATIO
        ratio: 1
        inputs: 2
        parameters: {
          name: "parallelism"
          value: 4
          min: 1
          max: 16
          tunable: true
        }
      }
    }
    nodes: {
      key: 2
      value: {
        id: 2
        name: "ParallelMapV2"
        autotune: true
        num_elements: 100
        processing_time: 20000
        bytes_produced: 10000
        node_class: ASYNC_KNOWN_RATIO
        ratio: 1
        inputs: 3
        parameters: {
          name: "parallelism"
          value: 4
          min: 1
          max: 16
          tunable: true
        }
      }
    }
    nodes: {
      key: 3
      value: {
        id: 3
        name: "SSTable"
        autotune: true
        num_elements: 100
        processing_time: 1000
        node_class: KNOWN_RATIO
        ratio: 2
      }
    }
    output: 1
  )pb");

  CancellationManager cancellation_manager;
  model_->Optimize(AutotuneAlgorithm::STAGE_BASED, 5, 1000, 50,
                   &cancellation_manager);

  EXPECT_EQ(5, GetNode(/*node_id=*/1)->parameter_value("parallelism"));
  EXPECT_EQ(5, GetNode(/*node_id=*/2)->parameter_value("parallelism"));
}

TEST_F(ModelTimingTest, OptimizeStageBased_TwoStages_RamBudgetExceeded) {
  BuildModelFromProto(R"pb(
    nodes: {
      key: 1
      value: {
        id: 1
        name: "ParallelMapV2"
        autotune: true
        num_elements: 100
        processing_time: 62000
        bytes_produced: 10000
        node_class: ASYNC_KNOWN_RATIO
        ratio: 1
        inputs: 2
        parameters: {
          name: "parallelism"
          value: 4
          state_value: 4
          min: 1
          max: 16
          tunable: true
        }
      }
    }
    nodes: {
      key: 2
      value: {
        id: 2
        name: "ParallelMapV2"
        autotune: true
        num_elements: 100
        processing_time: 70000
        bytes_produced: 10000
        node_class: ASYNC_KNOWN_RATIO
        ratio: 1
        inputs: 3
        parameters: {
          name: "parallelism"
          value: 4
          state_value: 4
          min: 1
          max: 16
          tunable: true
        }
      }
    }
    nodes: {
      key: 3
      value: {
        id: 3
        name: "SSTable"
        autotune: true
        num_elements: 100
        processing_time: 1000
        node_class: KNOWN_RATIO
        ratio: 2
      }
    }
    output: 1
  )pb");

  CancellationManager cancellation_manager;
  // Not enough RAM, the original `parallelism` should not change.
  model_->Optimize(AutotuneAlgorithm::STAGE_BASED, 10, 100, 0,
                   &cancellation_manager);
  EXPECT_EQ(4, GetNode(/*node_id=*/1)->parameter_value("parallelism"));
  EXPECT_EQ(4, GetNode(/*node_id=*/2)->parameter_value("parallelism"));
  // Has enough RAM, the original `parallelism` should increase.
  model_->Optimize(AutotuneAlgorithm::STAGE_BASED, 10, 100000, 0,
                   &cancellation_manager);
  EXPECT_EQ(12, GetNode(/*node_id=*/1)->parameter_value("parallelism"));
  EXPECT_EQ(16, GetNode(/*node_id=*/2)->parameter_value("parallelism"));
  // Not enough RAM, the original `parallelism` should not change.
  model_->Optimize(AutotuneAlgorithm::STAGE_BASED, 10, 100, 0,
                   &cancellation_manager);
  EXPECT_EQ(12, GetNode(/*node_id=*/1)->parameter_value("parallelism"));
  EXPECT_EQ(16, GetNode(/*node_id=*/2)->parameter_value("parallelism"));
}

TEST_F(ModelTimingTest, OptimizeStageBased_PipelineRatio) {
  BuildModelFromProto(R"pb(
    nodes: {
      key: 1
      value: {
        id: 1
        name: "ParallelBatch"
        autotune: true
        num_elements: 100
        processing_time: 5000
        bytes_produced: 10000
        node_class: ASYNC_KNOWN_RATIO
        ratio: 2
        inputs: 2
        parameters: {
          name: "parallelism"
          value: 4
          min: 1
          max: 16
          tunable: true
        }
      }
    }
    nodes: {
      key: 2
      value: {
        id: 2
        name: "Map"
        autotune: true
        num_elements: 100
        processing_time: 3000
        node_class: KNOWN_RATIO
        ratio: 1
        inputs: 3
      }
    }
    nodes: {
      key: 3
      value: {
        id: 3
        name: "SSTable"
        autotune: true
        num_elements: 100
        processing_time: 1000
        node_class: KNOWN_RATIO
        ratio: 2
      }
    }
    output: 1
  )pb");

  CancellationManager cancellation_manager;
  model_->Optimize(AutotuneAlgorithm::STAGE_BASED, 20, 10000, 50,
                   &cancellation_manager);

  EXPECT_EQ(16, GetNode(/*node_id=*/1)->parameter_value("parallelism"));
}

TEST_F(ModelTimingTest, OptimizeStageBased_PipelineRatioLessThanOne) {
  BuildModelFromProto(R"pb(
    nodes: {
      key: 1
      value: {
        id: 1
        name: "ParallelBatch"
        autotune: true
        num_elements: 100
        processing_time: 40000
        bytes_produced: 10000
        node_class: ASYNC_KNOWN_RATIO
        ratio: 0.5
        inputs: 2
        parameters: {
          name: "parallelism"
          value: 4
          min: 1
          max: 16
          tunable: true
        }
      }
    }
    nodes: {
      key: 2
      value: {
        id: 2
        name: "Map"
        autotune: true
        num_elements: 100
        processing_time: 3000
        node_class: KNOWN_RATIO
        ratio: 1
        inputs: 3
      }
    }
    nodes: {
      key: 3
      value: {
        id: 3
        name: "SSTable"
        autotune: true
        num_elements: 100
        processing_time: 1000
        node_class: KNOWN_RATIO
        ratio: 2
      }
    }
    output: 1
  )pb");

  CancellationManager cancellation_manager;
  model_->Optimize(AutotuneAlgorithm::STAGE_BASED, 20, 10000, 50,
                   &cancellation_manager);

  EXPECT_EQ(14, GetNode(/*node_id=*/1)->parameter_value("parallelism"));
}

TEST_F(ModelTimingTest, ComputeTargetTime) {
  model_ = std::make_unique<Model>();

  model_->RecordIteratorGapTime(10);
  model_->RecordIteratorGapTime(10);
  model_->RecordIteratorGapTime(10);
  model_->RecordIteratorGapTime(10);
  model_->RecordIteratorGapTime(10);
  model_->RecordIteratorGapTime(1000);
  // Gap times that are >= 10 seconds are always dropped.
  model_->RecordIteratorGapTime(10000000);

  EXPECT_DOUBLE_EQ(10, model_->ComputeTargetTimeNsec() * 1e-3);
}

TEST_F(ModelTimingTest, ComputeTargetTime_NoOutlier) {
  model_ = std::make_unique<Model>();

  model_->RecordIteratorGapTime(10);
  model_->RecordIteratorGapTime(10);
  model_->RecordIteratorGapTime(10);
  model_->RecordIteratorGapTime(10);
  model_->RecordIteratorGapTime(20);
  model_->RecordIteratorGapTime(20);
  model_->RecordIteratorGapTime(20);
  model_->RecordIteratorGapTime(20);
  // Gap times that are >= 10 seconds are always dropped.
  model_->RecordIteratorGapTime(10000000);

  EXPECT_DOUBLE_EQ(15.0, model_->ComputeTargetTimeNsec() * 1e-3);
}

TEST_F(ModelTimingTest, ComputeTargetTime_TestWindow) {
  model_ = std::make_unique<Model>();

  // The window size is 100. Only the last 100 gap times are used to compute the
  // target time.
  for (int i = 0; i < 100; ++i) {
    model_->RecordIteratorGapTime(20);
  }
  for (int i = 0; i < 100; ++i) {
    model_->RecordIteratorGapTime(10);
  }

  EXPECT_DOUBLE_EQ(10.0, model_->ComputeTargetTimeNsec() * 1e-3);
}

TEST_F(ModelTimingTest, SelfTime) {
  BuildModelFromProto(R"pb(
    nodes: {
      key: 1
      value: {
        id: 1
        name: "ParallelMapV2"
        autotune: true
        num_elements: 100
        processing_time: 20000
        node_class: ASYNC_KNOWN_RATIO
        ratio: 1
        inputs: 2
        parameters: {
          name: "parallelism"
          value: 2
          min: 1
          max: 16
          tunable: true
        }
      }
    }
    nodes: {
      key: 2
      value: {
        id: 2
        name: "SSTable"
        autotune: true
        num_elements: 100
        processing_time: 100000
        node_class: KNOWN_RATIO
        ratio: 1
      }
    }
    output: 1
  )pb");

  auto node_1 = MutableGetNode(/*node_id=*/1);
  EXPECT_DOUBLE_EQ(100, node_1->ComputeSelfTime());
  node_1->add_processing_time(400);
  node_1->record_element();
  EXPECT_DOUBLE_EQ(110, node_1->ComputeSelfTime());
  auto node_2 = MutableGetNode(/*node_id=*/2);
  EXPECT_DOUBLE_EQ(1000, node_2->ComputeSelfTime());
  node_2->add_processing_time(100);
  node_2->record_element();
  EXPECT_DOUBLE_EQ(910, node_2->ComputeSelfTime());
}

}  // namespace
}  // namespace model
}  // namespace data
}  // namespace tensorflow
