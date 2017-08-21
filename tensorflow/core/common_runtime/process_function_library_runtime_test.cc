/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"

#include <vector>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

class ProcessFunctionLibraryRuntimeTest : public ::testing::Test {
 public:
  ProcessFunctionLibraryRuntimeTest() {
    SessionOptions options;
    auto* device_count = options.config.mutable_device_count();
    device_count->insert({"CPU", 2});
    TF_CHECK_OK(DeviceFactory::AddDevices(options, "/job:a/replica:0/task:0",
                                          &devices_));
    device_mgr_.reset(new DeviceMgr(devices_));
    FunctionDefLibrary proto;
    lib_def_.reset(new FunctionLibraryDefinition(OpRegistry::Global(), proto));
    OptimizerOptions opts;
    proc_flr_.reset(new ProcessFunctionLibraryRuntime(
        device_mgr_.get(), Env::Default(), TF_GRAPH_DEF_VERSION, lib_def_.get(),
        opts));
  }

 protected:
  std::vector<Device*> devices_;
  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<FunctionLibraryDefinition> lib_def_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> proc_flr_;
};

TEST_F(ProcessFunctionLibraryRuntimeTest, Basic) {
  FunctionLibraryRuntime* flr =
      proc_flr_->GetFLR("/job:a/replica:0/task:0/cpu:0");
  EXPECT_NE(flr, nullptr);
  EXPECT_EQ(flr->device(), devices_[0]);
  flr = proc_flr_->GetFLR("/job:a/replica:0/task:0/cpu:1");
  EXPECT_NE(flr, nullptr);
  EXPECT_EQ(flr->device(), devices_[1]);
  flr = proc_flr_->GetFLR("abc");
  EXPECT_EQ(flr, nullptr);
}

}  // anonymous namespace
}  // namespace tensorflow
