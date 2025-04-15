/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/eager/eager_operation.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(EagerOperationTest, DeviceName) {
  StaticDeviceMgr device_mgr(DeviceFactory::NewDevice(
      "CPU", {}, "/job:localhost/replica:0/task:0/device:CPU:0"));
  auto ctx = new EagerContext(
      SessionOptions(),
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT, false,
      &device_mgr, false, nullptr, nullptr, nullptr,
      /*run_eager_op_as_function=*/true);

  auto op = new EagerOperation(ctx);

  TF_ASSERT_OK(op->SetDeviceName("/device:DONTHAVE"));
  EXPECT_EQ("/device:DONTHAVE:*", op->DeviceName());

  TF_ASSERT_OK(op->SetDeviceName(""));
  EXPECT_EQ("", op->DeviceName());

  TF_ASSERT_OK(op->SetDeviceName("/job:localhost"));
  EXPECT_EQ("/job:localhost", op->DeviceName());

  EXPECT_NE(absl::OkStatus(), op->SetDeviceName("/not/a/valid/name"));

  delete op;
  ctx->Unref();
}

TEST(EagerOperationTest, EagerFunctionParamsAndStepId) {
  StaticDeviceMgr device_mgr(DeviceFactory::NewDevice(
      "CPU", {}, "/job:localhost/replica:0/task:0/device:CPU:0"));
  auto ctx = new EagerContext(
      SessionOptions(),
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT, false,
      &device_mgr, false, nullptr, nullptr, nullptr,
      /*run_eager_op_as_function=*/true);

  tensorflow::FunctionDef function_def;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(
      "    signature {"
      "      name: 'DummyFunction'"
      "    }",
      &function_def));
  TF_ASSERT_OK(ctx->AddFunctionDef(function_def));

  auto op = new EagerOperation(ctx);
  EXPECT_FALSE(op->eager_func_params().has_value());

  string device_name = "/job:localhost/replica:0/task:0/device:CPU:0";
  TF_ASSERT_OK(op->SetDeviceName(device_name.c_str()));
  TF_ASSERT_OK(op->Reset("DummyFunction", device_name.c_str()));

  op->SetStepId(255);
  EXPECT_EQ(op->eager_func_params()->step_id.value(), 255);

  delete op;
  ctx->Unref();
}

}  // namespace
}  // namespace tensorflow
