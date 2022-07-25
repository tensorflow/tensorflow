/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/fallback/cost_recorder.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace tfrt_stub {
namespace {

constexpr absl::string_view kTestOpName = "test_op_name";
constexpr uint64_t kTestCost = 1234;
constexpr uint64_t kTestAvgCost = 1851;

TEST(CostRecorderTest, RecordCostTest) {
  CostRecorder recorder = CostRecorder(nullptr);

  recorder.RecordCost(kTestOpName, kTestCost);
  recorder.RecordCost(kTestOpName, kTestCost);

  EXPECT_EQ(recorder.size(), 1);
}

TEST(CostRecorderTest, WriteToFileTest) {
  CostRecorder recorder = CostRecorder(nullptr);
  ASSERT_EQ(recorder.size(), 0);

  std::string measured_cost_path;
  tensorflow::Env::Default()->LocalTempFilename(&measured_cost_path);
  TF_CHECK_OK(recorder.WriteToFile(measured_cost_path));

  OpCostMapProto op_cost_map_proto;
  TF_CHECK_OK(tensorflow::ReadTextProto(
      tensorflow::Env::Default(), measured_cost_path, &op_cost_map_proto));

  EXPECT_EQ(op_cost_map_proto.op_cost_map_size(), 0);
}

TEST(CostRecorderTest, ProtoRecordsTest) {
  CostRecorder recorder = CostRecorder(nullptr);

  // Records the cost of op.
  recorder.RecordCost(kTestOpName, kTestCost);
  recorder.RecordCost(kTestOpName, 2 * kTestCost);
  ASSERT_EQ(recorder.size(), 1);

  // Writes op's cost to the disk.
  std::string measured_cost_path;
  tensorflow::Env::Default()->LocalTempFilename(&measured_cost_path);
  TF_CHECK_OK(recorder.WriteToFile(measured_cost_path));

  // Reads op's cost from the disk.
  OpCostMapProto op_cost_map_proto;
  TF_CHECK_OK(tensorflow::ReadTextProto(
      tensorflow::Env::Default(), measured_cost_path, &op_cost_map_proto));

  EXPECT_EQ(
      op_cost_map_proto.op_cost_map().find(std::string(kTestOpName))->second,
      kTestAvgCost);
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace tensorflow
