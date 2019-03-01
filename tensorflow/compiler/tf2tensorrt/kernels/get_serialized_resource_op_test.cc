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

#include <dirent.h>
#include <string.h>
#include <fstream>
#include <vector>

#include "tensorflow/compiler/tf2tensorrt/utils/trt_resources.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"

#if GOOGLE_CUDA
#if GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {

class GetSerializedResourceOpTest : public OpsTestBase {};

TEST_F(GetSerializedResourceOpTest, Basic) {
  // Create the GPU device.
  std::unique_ptr<Device> device(
      DeviceFactory::NewDevice("GPU", {}, "/job:worker/replica:0/task:0"));

  // Create the resource.
  class MySerializableResource : public SerializableResourceBase {
   public:
    string DebugString() const override { return ""; }
    Status SerializeToString(string* serialized) override {
      *serialized = "my_serialized_str";
      return Status::OK();
    }
  };
  const string container = "mycontainer";
  const string resource_name = "myresource";
  SerializableResourceBase* resource = new MySerializableResource();
  ResourceMgr* rm = device->resource_manager();
  EXPECT_TRUE(rm->Create(container, resource_name, resource).ok());

  // Create the op.
  SetDevice(DEVICE_GPU, std::move(device));
  TF_ASSERT_OK(NodeDefBuilder("op", "GetSerializedResourceOp")
                   .Input(FakeInput(DT_STRING))
                   .Input(FakeInput(DT_STRING))
                   .Finalize(node_def()));
  TF_ASSERT_OK(InitOp());

  // Execute the op.
  AddInputFromArray<string>(TensorShape({}), {container});
  AddInputFromArray<string>(TensorShape({}), {resource_name});
  TF_ASSERT_OK(RunOpKernel());

  // Verify the result.
  // TODO(laigd): OpsTestBase::GetOutput() doesn't work.
  Tensor* output = context_->mutable_output(0);
  EXPECT_EQ("my_serialized_str", output->scalar<string>()());
}

}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_TENSORRT
#endif  // GOOGLE_CUDA
