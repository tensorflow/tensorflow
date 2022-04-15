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

#include "tensorflow/core/framework/resource_handle.h"

#include <memory>
#include <string>

#include "tensorflow/core/framework/resource_handle.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class MockResource : public ResourceBase {
 public:
  MockResource(bool* alive, int payload) : alive_(alive), payload_(payload) {
    if (alive_ != nullptr) {
      *alive_ = true;
    }
  }
  ~MockResource() override {
    if (alive_ != nullptr) {
      *alive_ = false;
    }
  }
  string DebugString() const override { return ""; }
  bool* alive_;
  int payload_;
};

class ResourceHandleTest : public ::testing::Test {};

TEST_F(ResourceHandleTest, RefCounting) {
  const int payload = -123;
  bool alive = false;
  auto resource = new MockResource(&alive, payload);
  EXPECT_TRUE(alive);
  {
    auto handle =
        ResourceHandle::MakeRefCountingHandle(resource, "cpu", {}, {});
    EXPECT_TRUE(alive);
    EXPECT_EQ(resource->RefCount(), 1);
    {
      auto handle_copy = handle;
      EXPECT_TRUE(alive);
      EXPECT_EQ(resource->RefCount(), 2);
    }
    EXPECT_TRUE(alive);
    EXPECT_EQ(resource->RefCount(), 1);
  }
  EXPECT_FALSE(alive);
}

TEST_F(ResourceHandleTest, SummarizeValue) {
  ResourceHandleProto proto;
  TensorShapeProto shape;
  shape.add_dim()->set_size(4);
  shape.add_dim()->set_size(8);
  proto.set_device("cpu:0");
  proto.set_container("test_container");
  proto.set_name("test_var");
  auto dtypes_and_shapes = proto.add_dtypes_and_shapes();
  dtypes_and_shapes->set_dtype(DT_INT32);
  dtypes_and_shapes->mutable_shape()->MergeFrom(shape);

  auto handle = std::make_unique<ResourceHandle>(proto);
  EXPECT_EQ(handle->SummarizeValue(),
            "ResourceHandle(name=\"test_var\", device=\"cpu:0\", "
            "container=\"test_container\", type=\"\", dtype and shapes : \"[ "
            "DType enum: 3, Shape: [4,8] ]\")");
}

}  // namespace
}  // namespace tensorflow
