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
#include "tensorflow/core/common_runtime/function_testlib.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace {

class ProcessFunctionLibraryRuntimeTest : public ::testing::Test {
 protected:
  void Init(const std::vector<FunctionDef>& flib) {
    SessionOptions options;
    auto* device_count = options.config.mutable_device_count();
    device_count->insert({"CPU", 2});
    TF_CHECK_OK(DeviceFactory::AddDevices(options, "/job:a/replica:0/task:0",
                                          &devices_));
    device_mgr_.reset(new DeviceMgr(devices_));
    FunctionDefLibrary proto;
    for (const auto& fdef : flib) *(proto.add_function()) = fdef;
    lib_def_.reset(new FunctionLibraryDefinition(OpRegistry::Global(), proto));
    OptimizerOptions opts;
    proc_flr_.reset(new ProcessFunctionLibraryRuntime(
        device_mgr_.get(), Env::Default(), TF_GRAPH_DEF_VERSION, lib_def_.get(),
        opts));
  }

  Status Run(const string& name, test::function::Attrs attrs,
             const std::vector<Tensor>& args, std::vector<Tensor*> rets) {
    FunctionLibraryRuntime::Handle handle;
    Status status = proc_flr_->Instantiate(name, attrs, &handle);
    if (!status.ok()) {
      return status;
    }

    std::atomic<int32> call_count(0);
    std::function<void(std::function<void()>)> runner =
        [&call_count](std::function<void()> fn) {
          ++call_count;
          test::function::FunctionTestSchedClosure(fn);
        };

    Notification done;
    FunctionLibraryRuntime::Options opts;
    opts.runner = &runner;
    std::vector<Tensor> out;
    proc_flr_->Run(opts, handle, args, &out, [&status, &done](const Status& s) {
      status = s;
      done.Notify();
    });
    done.WaitForNotification();
    if (!status.ok()) {
      return status;
    }
    CHECK_EQ(rets.size(), out.size());
    for (size_t i = 0; i < rets.size(); ++i) {
      *rets[i] = out[i];
    }

    EXPECT_GE(call_count, 1);  // Test runner is used.

    return Status::OK();
  }

  std::vector<Device*> devices_;
  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<FunctionLibraryDefinition> lib_def_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> proc_flr_;
};

TEST_F(ProcessFunctionLibraryRuntimeTest, Basic) {
  Init({});
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

TEST_F(ProcessFunctionLibraryRuntimeTest, ObtainFunctionTarget) {
  AttrSlice empty_attrs;
  string target =
      ProcessFunctionLibraryRuntime::ObtainFunctionTarget(empty_attrs);
  EXPECT_EQ("", target);

  AttrValueMap attr_values;
  AttrValue v;
  v.set_s("/job:a/replica:0/task:0/cpu:1");
  AddAttr("_target", v, &attr_values);
  AttrSlice attrs(&attr_values);
  target = ProcessFunctionLibraryRuntime::ObtainFunctionTarget(attrs);
  EXPECT_EQ("/job:a/replica:0/task:0/cpu:1", target);
}

TEST_F(ProcessFunctionLibraryRuntimeTest, SingleCall) {
  Init({test::function::XTimesTwo()});
  auto x = test::AsTensor<float>({1, 2, 3, 4});
  Tensor y;
  TF_CHECK_OK(
      Run("XTimesTwo",
          {{"T", DT_FLOAT}, {"_target", "/job:a/replica:0/task:0/cpu:0"}}, {x},
          {&y}));
  test::ExpectTensorEqual<float>(y, test::AsTensor<float>({2, 4, 6, 8}));
}

TEST_F(ProcessFunctionLibraryRuntimeTest, SingleCallFindDevice) {
  Init({test::function::FindDevice()});
  Tensor y;
  TF_CHECK_OK(Run("FindDevice", {{"_target", "/job:a/replica:0/task:0/cpu:0"}},
                  {}, {&y}));
  test::ExpectTensorEqual<string>(
      y, test::AsTensor<string>({"/job:a/replica:0/task:0/cpu:0"},
                                TensorShape({})));
}

TEST_F(ProcessFunctionLibraryRuntimeTest, MultipleCallsSameDeviceXTimes) {
  Init({test::function::XTimesTwo(), test::function::XTimesFour()});
  auto x = test::AsTensor<float>({1, 2, 3, 4});
  Tensor y;
  TF_CHECK_OK(
      Run("XTimesTwo",
          {{"T", DT_FLOAT}, {"_target", "/job:a/replica:0/task:0/cpu:0"}}, {x},
          {&y}));
  test::ExpectTensorEqual<float>(y, test::AsTensor<float>({2, 4, 6, 8}));
  TF_CHECK_OK(
      Run("XTimesFour",
          {{"T", DT_FLOAT}, {"_target", "/job:a/replica:0/task:0/cpu:0"}}, {x},
          {&y}));
  test::ExpectTensorEqual<float>(y, test::AsTensor<float>({4, 8, 12, 16}));
}

TEST_F(ProcessFunctionLibraryRuntimeTest, MultipleCallsSameDeviceFindDevice) {
  Init({test::function::FindDevice()});
  Tensor y;
  TF_CHECK_OK(Run("FindDevice", {{"_target", "/job:a/replica:0/task:0/cpu:1"}},
                  {}, {&y}));
  test::ExpectTensorEqual<string>(
      y, test::AsTensor<string>({"/job:a/replica:0/task:0/cpu:1"},
                                TensorShape({})));
  TF_CHECK_OK(Run("FindDevice", {{"_target", "/job:a/replica:0/task:0/cpu:1"}},
                  {}, {&y}));
  test::ExpectTensorEqual<string>(
      y, test::AsTensor<string>({"/job:a/replica:0/task:0/cpu:1"},
                                TensorShape({})));
}

TEST_F(ProcessFunctionLibraryRuntimeTest, MultipleCallsDiffDeviceFindDevice) {
  Init({test::function::FindDevice()});
  Tensor y;
  TF_CHECK_OK(Run("FindDevice", {{"_target", "/job:a/replica:0/task:0/cpu:0"}},
                  {}, {&y}));
  test::ExpectTensorEqual<string>(
      y, test::AsTensor<string>({"/job:a/replica:0/task:0/cpu:0"},
                                TensorShape({})));
  TF_CHECK_OK(Run("FindDevice", {{"_target", "/job:a/replica:0/task:0/cpu:1"}},
                  {}, {&y}));
  test::ExpectTensorEqual<string>(
      y, test::AsTensor<string>({"/job:a/replica:0/task:0/cpu:1"},
                                TensorShape({})));
}

}  // anonymous namespace
}  // namespace tensorflow
