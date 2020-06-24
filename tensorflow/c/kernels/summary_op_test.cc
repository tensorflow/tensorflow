/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/test.h"

#include "tensorflow/core/framework/summary.pb.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/c/tf_tensor_internal.h"

#include<stdio.h>
#include<sstream>
#include<iostream>
namespace tensorflow {
namespace {

class DummyDevice : public DeviceBase {
 public:
  explicit DummyDevice(Env* env) : DeviceBase(env) {}
  Allocator* GetAllocator(AllocatorAttributes /*attr*/) override {
    return cpu_allocator();
  }
};

// Helper for comparing ouput and expected output
static void EXPECT_SummaryMatches(const Summary& actual,
                                  const string& expected_str) {
  Summary expected;
  (protobuf::TextFormat::ParseFromString(expected_str, &expected));
  EXPECT_EQ(expected.DebugString(), actual.DebugString());
}


void TestScalarSummaryOp(Tensor* tags, Tensor* values, string expected_summary,
                         error::Code expected_code) {
  // initialize node used to fetch OpKernel 
  Status status;
  NodeDef def;
  def.set_op("SummaryScalar");
  def.set_device(DEVICE_CPU);
  AttrValue valuesTypeAttr;
  SetAttrValue(values->dtype(), &valuesTypeAttr);
  (*def.mutable_attr())["T"] = valuesTypeAttr;
  def.add_input(
      strings::StrCat("input1: ", DataTypeString(tags->dtype())));
  def.add_input(
    strings::StrCat("input2: ", DataTypeString(values->dtype())));

  std::unique_ptr<OpKernel> kernel =
      CreateOpKernel(DeviceType(DEVICE_CPU), nullptr, nullptr, def, 1, &status); 
  ASSERT_TRUE(status.ok()) << status.ToString();

  // initialize OpKernel parameters 
  OpKernelContext::Params params;
  DummyDevice dummy_device(nullptr);
  params.device = &dummy_device;
  params.op_kernel = kernel.get();
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.emplace_back(tags);
  inputs.emplace_back(values);
  params.inputs = &inputs;
  OpKernelContext ctx(&params, 1); 
  AllocatorAttributes alloc_attrs;
  std::vector<AllocatorAttributes> output_alloc_attrs({alloc_attrs});
  params.output_attr_array = output_alloc_attrs.data();
  kernel->Compute(&ctx);
  ASSERT_EQ(expected_code, ctx.status().code());
  if (expected_code == error::OK){ 
    Summary summary; 
    ParseProtoUnlimited(&summary, ctx.mutable_output(0)->scalar<tstring>()());
    EXPECT_SummaryMatches(summary, expected_summary);
  }
}

TEST(ScalarSummaryOpTest, SimpleFloat) {
  int vectorSize = 3; 
  Tensor tags(DT_STRING, {vectorSize}); 
  Tensor values(DT_FLOAT, {vectorSize}); 
  tags.vec<tstring>()(0) = "tag1";
  tags.vec<tstring>()(1) = "tag2";
  tags.vec<tstring>()(2) = "tag3";
  values.vec<float>()(0) = 1.0f; 
  values.vec<float>()(1) = -0.73f;
  values.vec<float>()(2) = 10000.0f;
  TestScalarSummaryOp(&tags, &values, R"(
                      value { tag: 'tag1' simple_value: 1.0 }
                      value { tag: 'tag2' simple_value: -0.73}
                      value { tag: 'tag3' simple_value: 10000.0})", error::OK); 
}

TEST(ScalarSummaryOpTest, SimpleDouble) {
  int vectorSize = 3; 
  Tensor tags(DT_STRING, {vectorSize}); 
  Tensor values(DT_DOUBLE, {vectorSize}); 
  tags.vec<tstring>()(0) = "tag1";
  tags.vec<tstring>()(1) = "tag2";
  tags.vec<tstring>()(2) = "tag3";
  values.vec<double>()(0) = 1.0; 
  values.vec<double>()(1) = -0.73;
  values.vec<double>()(2) = 10000.0;
  TestScalarSummaryOp(&tags, &values, R"(
                      value { tag: 'tag1' simple_value: 1.0 }
                      value { tag: 'tag2' simple_value: -0.73}
                      value { tag: 'tag3' simple_value: 10000.0})", error::OK); 
}

TEST(ScalarSummaryOpTest, SimpleHalf) {
  int vectorSize = 3; 
  Tensor tags(DT_STRING, {vectorSize}); 
  Tensor values(DT_HALF, {vectorSize}); 
  tags.vec<tstring>()(0) = "tag1";
  tags.vec<tstring>()(1) = "tag2";
  tags.vec<tstring>()(2) = "tag3";
  values.vec<Eigen::half>()(0) = static_cast<Eigen::half>(1.0); 
  values.vec<Eigen::half>()(1) = static_cast<Eigen::half>(-2.0);
  values.vec<Eigen::half>()(2) = static_cast<Eigen::half>(10000.0);
  TestScalarSummaryOp(&tags, &values, R"(
                      value { tag: 'tag1' simple_value: 1.0 }
                      value { tag: 'tag2' simple_value: -2.0}
                      value { tag: 'tag3' simple_value: 10000.0})", error::OK); 
}

TEST(ScalarSummaryOpTest, Error_WrongDimsTags) {
  int vectorSize = 3; 
  Tensor tags(DT_STRING, {2, 1}); 
  Tensor values(DT_FLOAT, {2}); 
  tags.matrix<tstring>()(0, 0) = "tag1";
  tags.matrix<tstring>()(1, 0) = "tag2";
  values.vec<float>()(0) = 1.0f; 
  values.vec<float>()(1) = -2.0f;
  TestScalarSummaryOp(&tags, &values, R"()", error::INVALID_ARGUMENT); 
}

TEST(ScalarSummaryOpTest, Error_WrongValuesTags) {
  Tensor tags(DT_STRING, {2}); 
  Tensor values(DT_FLOAT, {2, 1}); 
  tags.vec<tstring>()(0) = "tag1";
  tags.vec<tstring>()(1) = "tag2";
  values.matrix<float>()(0, 0) = 1.0f; 
  values.matrix<float>()(1, 0) = -2.0f;
  TestScalarSummaryOp(&tags, &values, R"()", error::INVALID_ARGUMENT); 
}

TEST(ScalarSummaryOpTest, IsRegistered){ 
  const OpRegistrationData* reg;
  TF_CHECK_OK(OpRegistry::Global()->LookUp("SummaryScalar", &reg));
}



PartialTensorShape S(std::initializer_list<int64> dims) {
  return PartialTensorShape(dims);
}



}  // namespace
}  // namespace tensorflow
