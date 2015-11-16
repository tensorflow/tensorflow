#include <functional>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/histogram/histogram.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/public/env.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {
namespace {

static void EXPECT_SummaryMatches(const Summary& actual,
                                  const string& expected_str) {
  Summary expected;
  CHECK(protobuf::TextFormat::ParseFromString(expected_str, &expected));
  EXPECT_EQ(expected.DebugString(), actual.DebugString());
}

class SummaryScalarOpTest : public OpsTestBase {
 protected:
  void MakeOp(DataType dt) {
    RequireDefaultOps();
    ASSERT_OK(NodeDefBuilder("myop", "ScalarSummary")
                  .Input(FakeInput())
                  .Input(FakeInput(dt))
                  .Finalize(node_def()));
    ASSERT_OK(InitOp());
  }
};

TEST_F(SummaryScalarOpTest, SimpleFloat) {
  MakeOp(DT_FLOAT);

  // Feed and run
  AddInputFromArray<string>(TensorShape({3}), {"tag1", "tag2", "tag3"});
  AddInputFromArray<float>(TensorShape({3}), {1.0, -0.73, 10000.0});
  ASSERT_OK(RunOpKernel());

  // Check the output size.
  Tensor* out_tensor = GetOutput(0);
  ASSERT_EQ(0, out_tensor->dims());
  Summary summary;
  ParseProtoUnlimited(&summary, out_tensor->scalar<string>()());
  EXPECT_SummaryMatches(summary, R"(
      value { tag: 'tag1' simple_value: 1.0 }
      value { tag: 'tag2' simple_value: -0.73 }
      value { tag: 'tag3' simple_value: 10000.0 }
  )");
}

TEST_F(SummaryScalarOpTest, SimpleDouble) {
  MakeOp(DT_DOUBLE);

  // Feed and run
  AddInputFromArray<string>(TensorShape({3}), {"tag1", "tag2", "tag3"});
  AddInputFromArray<double>(TensorShape({3}), {1.0, -0.73, 10000.0});
  ASSERT_OK(RunOpKernel());

  // Check the output size.
  Tensor* out_tensor = GetOutput(0);
  ASSERT_EQ(0, out_tensor->dims());
  Summary summary;
  ParseProtoUnlimited(&summary, out_tensor->scalar<string>()());
  EXPECT_SummaryMatches(summary, R"(
      value { tag: 'tag1' simple_value: 1.0 }
      value { tag: 'tag2' simple_value: -0.73 }
      value { tag: 'tag3' simple_value: 10000.0 }
  )");
}

TEST_F(SummaryScalarOpTest, Error_MismatchedSize) {
  MakeOp(DT_FLOAT);

  // Feed and run
  AddInputFromArray<string>(TensorShape({2}), {"tag1", "tag2"});
  AddInputFromArray<float>(TensorShape({3}), {1.0, -0.73, 10000.0});
  Status s = RunOpKernel();
  EXPECT_TRUE(StringPiece(s.ToString()).contains("not the same shape")) << s;
}

TEST_F(SummaryScalarOpTest, Error_WrongDimsTags) {
  MakeOp(DT_FLOAT);

  // Feed and run
  AddInputFromArray<string>(TensorShape({2, 1}), {"tag1", "tag2"});
  AddInputFromArray<float>(TensorShape({2}), {1.0, -0.73});
  Status s = RunOpKernel();
  EXPECT_TRUE(
      StringPiece(s.ToString()).contains("tags and values not the same shape"))
      << s;
}

TEST_F(SummaryScalarOpTest, Error_WrongDimsValues) {
  MakeOp(DT_FLOAT);

  // Feed and run
  AddInputFromArray<string>(TensorShape({2}), {"tag1", "tag2"});
  AddInputFromArray<float>(TensorShape({2, 1}), {1.0, -0.73});
  Status s = RunOpKernel();
  EXPECT_TRUE(
      StringPiece(s.ToString()).contains("tags and values not the same shape"))
      << s;
}

// --------------------------------------------------------------------------
// SummaryHistoOp
// --------------------------------------------------------------------------
class SummaryHistoOpTest : public OpsTestBase {
 protected:
  void MakeOp() {
    ASSERT_OK(NodeDefBuilder("myop", "HistogramSummary")
                  .Input(FakeInput())
                  .Input(FakeInput())
                  .Finalize(node_def()));
    ASSERT_OK(InitOp());
  }
};

TEST_F(SummaryHistoOpTest, Simple) {
  MakeOp();

  // Feed and run
  AddInputFromArray<string>(TensorShape({}), {"taghisto"});
  AddInputFromArray<float>(TensorShape({3, 2}), {0.1, -0.7, 4.1, 4., 5., 4.});
  ASSERT_OK(RunOpKernel());

  // Check the output size.
  Tensor* out_tensor = GetOutput(0);
  ASSERT_EQ(0, out_tensor->dims());
  Summary summary;
  ParseProtoUnlimited(&summary, out_tensor->scalar<string>()());
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

TEST_F(SummaryHistoOpTest, Error_WrongDimsTags) {
  MakeOp();

  // Feed and run
  AddInputFromArray<string>(TensorShape({2, 1}), {"tag1", "tag2"});
  AddInputFromArray<float>(TensorShape({2}), {1.0, -0.73});
  Status s = RunOpKernel();
  EXPECT_TRUE(StringPiece(s.ToString()).contains("tags must be scalar")) << s;
}

TEST_F(SummaryHistoOpTest, Error_TooManyTagValues) {
  MakeOp();

  // Feed and run
  AddInputFromArray<string>(TensorShape({2}), {"tag1", "tag2"});
  AddInputFromArray<float>(TensorShape({2, 1}), {1.0, -0.73});
  Status s = RunOpKernel();
  EXPECT_TRUE(StringPiece(s.ToString()).contains("tags must be scalar")) << s;
}

// --------------------------------------------------------------------------
// SummaryMergeOp
// --------------------------------------------------------------------------
class SummaryMergeOpTest : public OpsTestBase {
 protected:
  void MakeOp(int num_inputs) {
    ASSERT_OK(NodeDefBuilder("myop", "MergeSummary")
                  .Input(FakeInput(num_inputs))
                  .Finalize(node_def()));
    ASSERT_OK(InitOp());
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

  AddInputFromArray<string>(
      TensorShape({3}),
      {s1.SerializeAsString(), s2.SerializeAsString(), s3.SerializeAsString()});
  ASSERT_OK(RunOpKernel());

  // Check the output size.
  Tensor* out_tensor = GetOutput(0);
  ASSERT_EQ(0, out_tensor->dims());
  Summary summary;
  ParseProtoUnlimited(&summary, out_tensor->scalar<string>()());

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

  AddInputFromArray<string>(TensorShape({}), {s1.SerializeAsString()});
  AddInputFromArray<string>(TensorShape({}), {s2.SerializeAsString()});
  AddInputFromArray<string>(TensorShape({}), {s3.SerializeAsString()});
  ASSERT_OK(RunOpKernel());

  // Check the output size.
  Tensor* out_tensor = GetOutput(0);
  ASSERT_EQ(0, out_tensor->dims());
  Summary summary;
  ParseProtoUnlimited(&summary, out_tensor->scalar<string>()());

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
  AddInputFromArray<string>(TensorShape({2}),
                            {s1.SerializeAsString(), s2.SerializeAsString()});
  Status s = RunOpKernel();
  EXPECT_TRUE(StringPiece(s.ToString()).contains("Duplicate tag")) << s;
}

}  // namespace
}  // namespace tensorflow
