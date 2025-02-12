/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <functional>
#include <memory>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/histogram/histogram.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

static void EXPECT_SummaryMatches(const Summary& actual,
                                  const string& expected_str) {
  Summary expected;
  CHECK(protobuf::TextFormat::ParseFromString(expected_str, &expected));
  EXPECT_EQ(expected.DebugString(), actual.DebugString());
}

// --------------------------------------------------------------------------
// SummaryHistoOp
// --------------------------------------------------------------------------
class SummaryHistoOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType dt) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "HistogramSummary")
                     .Input(FakeInput())
                     .Input(FakeInput(dt))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(SummaryHistoOpTest, SimpleFloat) {
  MakeOp(DT_FLOAT);

  // Feed and run
  AddInputFromArray<tstring>(TensorShape({}), {"taghisto"});
  AddInputFromArray<float>(TensorShape({3, 2}),
                           {0.1f, -0.7f, 4.1f, 4., 5.f, 4.f});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output size.
  Tensor* out_tensor = GetOutput(0);
  ASSERT_EQ(0, out_tensor->dims());
  Summary summary;
  ParseProtoUnlimited(&summary, out_tensor->scalar<tstring>()());
  ASSERT_EQ(summary.value_size(), 1);
  EXPECT_EQ(summary.value(0).tag(), "taghisto");
  histogram::Histogram histo;
  EXPECT_TRUE(histo.DecodeFromProto(summary.value(0).histo()));
  EXPECT_EQ(
      "Count: 6  Average: 2.7500  StdDev: 2.20\n"
      "Min: -0.7000  Median: 3.9593  Max: 5.0000\n"
      "------------------------------------------------------\n"
      "[      -0.76,      -0.69 )       1  16.667%  16.667% ###\n"
      "[      0.093,        0.1 )       1  16.667%  33.333% ###\n"
      "[        3.8,        4.2 )       3  50.000%  83.333% ##########\n"
      "[        4.6,        5.1 )       1  16.667% 100.000% ###\n",
      histo.ToString());
}

TEST_F(SummaryHistoOpTest, SimpleDouble) {
  MakeOp(DT_DOUBLE);

  // Feed and run
  AddInputFromArray<tstring>(TensorShape({}), {"taghisto"});
  AddInputFromArray<double>(TensorShape({3, 2}), {0.1, -0.7, 4.1, 4., 5., 4.});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output size.
  Tensor* out_tensor = GetOutput(0);
  ASSERT_EQ(0, out_tensor->dims());
  Summary summary;
  ParseProtoUnlimited(&summary, out_tensor->scalar<tstring>()());
  ASSERT_EQ(summary.value_size(), 1);
  EXPECT_EQ(summary.value(0).tag(), "taghisto");
  histogram::Histogram histo;
  EXPECT_TRUE(histo.DecodeFromProto(summary.value(0).histo()));
  EXPECT_EQ(
      "Count: 6  Average: 2.7500  StdDev: 2.20\n"
      "Min: -0.7000  Median: 3.9593  Max: 5.0000\n"
      "------------------------------------------------------\n"
      "[      -0.76,      -0.69 )       1  16.667%  16.667% ###\n"
      "[      0.093,        0.1 )       1  16.667%  33.333% ###\n"
      "[        3.8,        4.2 )       3  50.000%  83.333% ##########\n"
      "[        4.6,        5.1 )       1  16.667% 100.000% ###\n",
      histo.ToString());
}

TEST_F(SummaryHistoOpTest, SimpleHalf) {
  MakeOp(DT_HALF);

  // Feed and run
  AddInputFromList<tstring>(TensorShape({}), {"taghisto"});
  AddInputFromList<Eigen::half>(TensorShape({3, 2}),
                                {0.1, -0.7, 4.1, 4., 5., 4.});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output size.
  Tensor* out_tensor = GetOutput(0);
  ASSERT_EQ(0, out_tensor->dims());
  Summary summary;
  ParseProtoUnlimited(&summary, out_tensor->scalar<tstring>()());
  ASSERT_EQ(summary.value_size(), 1);
  EXPECT_EQ(summary.value(0).tag(), "taghisto");
  histogram::Histogram histo;
  EXPECT_TRUE(histo.DecodeFromProto(summary.value(0).histo()));
  EXPECT_EQ(
      "Count: 6  Average: 2.7502  StdDev: 2.20\n"
      "Min: -0.7002  Median: 3.9593  Max: 5.0000\n"
      "------------------------------------------------------\n"
      "[      -0.76,      -0.69 )       1  16.667%  16.667% ###\n"
      "[      0.093,        0.1 )       1  16.667%  33.333% ###\n"
      "[        3.8,        4.2 )       3  50.000%  83.333% ##########\n"
      "[        4.6,        5.1 )       1  16.667% 100.000% ###\n",
      histo.ToString());
}

TEST_F(SummaryHistoOpTest, Error_WrongDimsTags) {
  MakeOp(DT_FLOAT);

  // Feed and run
  AddInputFromArray<tstring>(TensorShape({2, 1}), {"tag1", "tag2"});
  AddInputFromArray<float>(TensorShape({2}), {1.0f, -0.73f});
  absl::Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(s.ToString(), "tags must be scalar")) << s;
}

TEST_F(SummaryHistoOpTest, Error_TooManyTagValues) {
  MakeOp(DT_FLOAT);

  // Feed and run
  AddInputFromArray<tstring>(TensorShape({2}), {"tag1", "tag2"});
  AddInputFromArray<float>(TensorShape({2, 1}), {1.0f, -0.73f});
  absl::Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(s.ToString(), "tags must be scalar")) << s;
}

// --------------------------------------------------------------------------
// SummaryMergeOp
// --------------------------------------------------------------------------
class SummaryMergeOpTest : public OpsTestBase {
 protected:
  void MakeOp(int num_inputs) {
    TF_ASSERT_OK(NodeDefBuilder("myop", "MergeSummary")
                     .Input(FakeInput(num_inputs))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(SummaryMergeOpTest, Simple) {
  MakeOp(1);

  // Feed and run
  Summary s1;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "value { tag: \"tag1\" simple_value: 1.0 } "
      "value { tag: \"tag2\" simple_value: -0.73 } ",
      &s1));
  Summary s2;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "value { tag: \"tag3\" simple_value: 10000.0 }", &s2));
  Summary s3;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "value { tag: \"tag4\" simple_value: 11.0 }", &s3));

  AddInputFromArray<tstring>(
      TensorShape({3}),
      {s1.SerializeAsString(), s2.SerializeAsString(), s3.SerializeAsString()});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output size.
  Tensor* out_tensor = GetOutput(0);
  ASSERT_EQ(0, out_tensor->dims());
  Summary summary;
  ParseProtoUnlimited(&summary, out_tensor->scalar<tstring>()());

  EXPECT_SummaryMatches(summary,
                        "value { tag: \"tag1\" simple_value: 1.0 } "
                        "value { tag: \"tag2\" simple_value: -0.73 } "
                        "value { tag: \"tag3\" simple_value: 10000.0 }"
                        "value { tag: \"tag4\" simple_value: 11.0 }");
}

TEST_F(SummaryMergeOpTest, Simple_MultipleInputs) {
  MakeOp(3);

  // Feed and run
  Summary s1;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "value { tag: \"tag1\" simple_value: 1.0 } "
      "value { tag: \"tag2\" simple_value: -0.73 } ",
      &s1));
  Summary s2;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "value { tag: \"tag3\" simple_value: 10000.0 }", &s2));
  Summary s3;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "value { tag: \"tag4\" simple_value: 11.0 }", &s3));

  AddInputFromArray<tstring>(TensorShape({}), {s1.SerializeAsString()});
  AddInputFromArray<tstring>(TensorShape({}), {s2.SerializeAsString()});
  AddInputFromArray<tstring>(TensorShape({}), {s3.SerializeAsString()});
  TF_ASSERT_OK(RunOpKernel());

  // Check the output size.
  Tensor* out_tensor = GetOutput(0);
  ASSERT_EQ(0, out_tensor->dims());
  Summary summary;
  ParseProtoUnlimited(&summary, out_tensor->scalar<tstring>()());

  EXPECT_SummaryMatches(summary,
                        "value { tag: \"tag1\" simple_value: 1.0 } "
                        "value { tag: \"tag2\" simple_value: -0.73 } "
                        "value { tag: \"tag3\" simple_value: 10000.0 }"
                        "value { tag: \"tag4\" simple_value: 11.0 }");
}

TEST_F(SummaryMergeOpTest, Error_MismatchedSize) {
  MakeOp(1);

  // Feed and run
  Summary s1;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "value { tag: \"tag1\" simple_value: 1.0 } "
      "value { tag: \"tagduplicate\" simple_value: -0.73 } ",
      &s1));
  Summary s2;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "value { tag: \"tagduplicate\" simple_value: 1.0 } ", &s2));
  AddInputFromArray<tstring>(TensorShape({2}),
                             {s1.SerializeAsString(), s2.SerializeAsString()});
  absl::Status s = RunOpKernel();
  EXPECT_TRUE(absl::StrContains(s.ToString(), "Duplicate tag")) << s;
}

}  // namespace
}  // namespace tensorflow
