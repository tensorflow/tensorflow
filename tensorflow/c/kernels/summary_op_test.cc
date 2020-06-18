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

void TestScalarSummaryOp(Tensor* tags, Tensor* values, error::Code expected_code) {
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
  OpKernelContext::Params params;
  DummyDevice dummy_device(nullptr);
  params.device = &dummy_device;
  params.op_kernel = kernel.get();
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.emplace_back(tags);
  inputs.emplace_back(values);
  params.inputs = &inputs;
  OpKernelContext ctx(&params, 1); 
  kernel->Compute(&ctx);

  ASSERT_EQ(expected_code, ctx.status().code());
  if (expected_code == error::OK) {
    ASSERT_EQ(true, false)
        << ctx.mutable_output(0)->shape().DebugString();
  }
}

TEST(ScalarSummaryOpTest, Test) {
  int vectorSize = 2; 
  Tensor tags(DT_STRING, {vectorSize}); 
  Tensor values(DT_FLOAT, {vectorSize}); 
  for (int i = 0; i < vectorSize; ++i){  
    values.vec<float>()(i) = static_cast<uint8>(i); 
  }
  tags.vec<tstring>()(0) = "tag 1";
  tags.vec<tstring>()(1) = "tag 2";
  TestScalarSummaryOp(&tags, &values, error::INVALID_ARGUMENT); 
}


PartialTensorShape S(std::initializer_list<int64> dims) {
  return PartialTensorShape(dims);
}



}  // namespace
}  // namespace tensorflow
