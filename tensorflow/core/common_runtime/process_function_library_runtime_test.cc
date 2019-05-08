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
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

#ifdef GOOGLE_CUDA
#include "cuda/include/cuda.h"
#include "cuda/include/cuda_runtime_api.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {
namespace {

class TestClusterFLR : public DistributedFunctionLibraryRuntime {
 public:
  TestClusterFLR() {}

  Status Instantiate(const string& function_name,
                     const FunctionLibraryDefinition& lib_def, AttrSlice attrs,
                     const FunctionLibraryRuntime::InstantiateOptions& options,
                     FunctionLibraryRuntime::LocalHandle* handle) override {
    mutex_lock l(mu_);
    *handle = next_handle_;
    next_handle_++;
    return Status::OK();
  }

  void Run(const FunctionLibraryRuntime::Options& opts,
           FunctionLibraryRuntime::LocalHandle handle,
           gtl::ArraySlice<Tensor> args, std::vector<Tensor>* rets,
           FunctionLibraryRuntime::DoneCallback done) override {}

 private:
  mutex mu_;
  int next_handle_ GUARDED_BY(mu_) = 0;
};

// TODO(b/128707168): Tests requiring a GPU device are currently always skipped
// because the check for whether a GPU device is present happens before the GPU
// device is set up.
class ProcessFunctionLibraryRuntimeTest : public ::testing::Test {
 public:
  ProcessFunctionLibraryRuntimeTest() {
    SessionOptions options;
    auto* device_count = options.config.mutable_device_count();
    device_count->insert({"CPU", 2});
    std::vector<std::unique_ptr<Device>> devices;
    TF_CHECK_OK(DeviceFactory::AddDevices(options, "/job:a/replica:0/task:0",
                                          &devices));
    device_mgr_.reset(new DeviceMgr(std::move(devices)));
    TF_CHECK_OK(device_mgr_->LookupDevice(
        "/job:a/replica:0/task:0/device:CPU:0", &device0_));
    TF_CHECK_OK(device_mgr_->LookupDevice(
        "/job:a/replica:0/task:0/device:CPU:1", &device1_));
    // If no GPU is available, gpu_device_ will remain nullptr.
    Status status = device_mgr_->LookupDevice(
        "/job:a/replica:0/task:0/device:GPU:0", &gpu_device_);
    if (!status.ok()) {
      CHECK_EQ(nullptr, gpu_device_);
    }
  }

  ~ProcessFunctionLibraryRuntimeTest() override {
    if (rendezvous_ != nullptr) {
      rendezvous_->Unref();
    }
  }

  void Init(const std::vector<FunctionDef>& flib) {
    FunctionDefLibrary proto;
    for (const auto& fdef : flib) *(proto.add_function()) = fdef;
    lib_def_.reset(new FunctionLibraryDefinition(OpRegistry::Global(), proto));
    OptimizerOptions opts;
    cluster_flr_.reset(new TestClusterFLR());
    proc_flr_.reset(new ProcessFunctionLibraryRuntime(
        device_mgr_.get(), Env::Default(), TF_GRAPH_DEF_VERSION, lib_def_.get(),
        opts, nullptr, cluster_flr_.get()));
    rendezvous_ = new IntraProcessRendezvous(device_mgr_.get());
  }

  Status Instantiate(
      const string& name, test::function::Attrs attrs,
      const FunctionLibraryRuntime::InstantiateOptions& instantiate_opts,
      FunctionLibraryRuntime::Handle* handle) {
    return proc_flr_->Instantiate(name, attrs, instantiate_opts, handle);
  }

  Tensor GPUToCPU(const Tensor& device_tensor) {
#ifdef GOOGLE_CUDA
    CHECK(gpu_device_);
    CHECK(gpu_device_->tensorflow_gpu_device_info() != nullptr);
    DeviceContext* device_context =
        gpu_device_->tensorflow_gpu_device_info()->default_context;

    Notification n;
    Status status;
    Tensor cpu_tensor(device_tensor.dtype(), device_tensor.shape());
    device_context->CopyDeviceTensorToCPU(&device_tensor, "", gpu_device_,
                                          &cpu_tensor,
                                          [&n, &status](const Status& s) {
                                            status = s;
                                            n.Notify();
                                          });
    n.WaitForNotification();
    CHECK(status.ok());
    return cpu_tensor;
#else
    CHECK(false);
#endif  // GOOGLE_CUDA
  }

  Tensor CPUToGPU(const Tensor& cpu_tensor) {
#ifdef GOOGLE_CUDA
    CHECK(gpu_device_);
    CHECK(gpu_device_->tensorflow_gpu_device_info() != nullptr);
    DeviceContext* device_context =
        gpu_device_->tensorflow_gpu_device_info()->default_context;

    Notification n;
    Status status;
    Tensor device_tensor(gpu_device_->GetAllocator({}), cpu_tensor.dtype(),
                         cpu_tensor.shape(), {});
    device_context->CopyCPUTensorToDevice(&cpu_tensor, gpu_device_,
                                          &device_tensor,
                                          [&n, &status](const Status& s) {
                                            status = s;
                                            n.Notify();
                                          });
    n.WaitForNotification();
    CHECK(status.ok());
    return device_tensor;
#else
    CHECK(false);
#endif  // GOOGLE_CUDA
  }

  Status Run(const string& name, FunctionLibraryRuntime::Options opts,
             test::function::Attrs attrs,
             const FunctionLibraryRuntime::InstantiateOptions& instantiate_opts,
             const std::vector<Tensor>& args, std::vector<Tensor*> rets) {
    FunctionLibraryRuntime::Handle handle;
    Status status =
        proc_flr_->Instantiate(name, attrs, instantiate_opts, &handle);
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

    // Release the handle and then try running the function. It shouldn't
    // succeed.
    status = proc_flr_->ReleaseHandle(handle);
    if (!status.ok()) {
      return status;
    }
    Notification done2;
    proc_flr_->Run(opts, handle, args, &out,
                   [&status, &done2](const Status& s) {
                     status = s;
                     done2.Notify();
                   });
    done2.WaitForNotification();
    EXPECT_TRUE(errors::IsNotFound(status)) << "Actual status: " << status;
    EXPECT_TRUE(str_util::StrContains(status.error_message(), "not found."));

    return Status::OK();
  }

  Status RunInstantiated(FunctionLibraryRuntime::Handle handle,
                         FunctionLibraryRuntime::Options opts,
                         const std::vector<Tensor>& args,
                         std::vector<Tensor*> rets) {
    std::atomic<int32> call_count(0);
    std::function<void(std::function<void()>)> runner =
        [&call_count](std::function<void()> fn) {
          ++call_count;
          test::function::FunctionTestSchedClosure(fn);
        };

    opts.rendezvous = rendezvous_;
    opts.runner = &runner;
    Status status;
    Notification done;
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

  std::unique_ptr<DeviceMgr> device_mgr_;
  Device* device0_ = nullptr;  // Not owned. (Owned by device_mgr_.)
  Device* device1_ = nullptr;  // Not owned. (Owned by device_mgr_.)
  // Remains as nullptr if no GPU is available.
  Device* gpu_device_ = nullptr;  // Not owned. (Owned by device_mgr_.)
  std::unique_ptr<FunctionLibraryDefinition> lib_def_;
  std::unique_ptr<TestClusterFLR> cluster_flr_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> proc_flr_;
  IntraProcessRendezvous* rendezvous_ = nullptr;
};

TEST_F(ProcessFunctionLibraryRuntimeTest, GetFLRNull) {
  FunctionDefLibrary proto;
  std::unique_ptr<FunctionLibraryDefinition> lib_def(
      new FunctionLibraryDefinition(OpRegistry::Global(), proto));
  OptimizerOptions opts;
  std::unique_ptr<ProcessFunctionLibraryRuntime> proc_flr(
      new ProcessFunctionLibraryRuntime(
          nullptr /* device_mgr */, Env::Default(), TF_GRAPH_DEF_VERSION,
          lib_def.get(), opts, nullptr, nullptr /* cluster_flr */));
  FunctionLibraryRuntime* flr =
      proc_flr->GetFLR(ProcessFunctionLibraryRuntime::kDefaultFLRDevice);
  EXPECT_NE(flr, nullptr);
}

TEST_F(ProcessFunctionLibraryRuntimeTest, Basic) {
  Init({});
  FunctionLibraryRuntime* flr =
      proc_flr_->GetFLR("/job:a/replica:0/task:0/cpu:0");
  EXPECT_NE(flr, nullptr);
  EXPECT_EQ(flr->device(), device0_);
  flr = proc_flr_->GetFLR("/job:a/replica:0/task:0/device:CPU:0");
  EXPECT_NE(flr, nullptr);
  EXPECT_EQ(flr->device(), device0_);
  flr = proc_flr_->GetFLR("/device:CPU:0");
  EXPECT_NE(flr, nullptr);
  EXPECT_EQ(flr->device(), device0_);
  flr = proc_flr_->GetFLR("/job:a/replica:0/task:0/cpu:1");
  EXPECT_NE(flr, nullptr);
  EXPECT_EQ(flr->device(), device1_);
  flr = proc_flr_->GetFLR("abc");
  EXPECT_EQ(flr, nullptr);
}

TEST_F(ProcessFunctionLibraryRuntimeTest, GetDeviceIncarnation) {
  Init({});
  int64 incarnation;
  TF_EXPECT_OK(proc_flr_->GetDeviceIncarnation("/job:a/replica:0/task:0/cpu:1",
                                               &incarnation));
  // Incarnation is a random number other than 0.
  EXPECT_NE(incarnation, 0);
  Status s = proc_flr_->GetDeviceIncarnation("/job:a/replica:0/task:0/cpu:2",
                                             &incarnation);
  EXPECT_EQ(s.code(), error::INVALID_ARGUMENT);
}

TEST_F(ProcessFunctionLibraryRuntimeTest, SingleCall) {
  Init({test::function::XTimesTwo()});
  FunctionLibraryRuntime::Options opts;
  opts.source_device = "/job:a/replica:0/task:0/cpu:0";
  opts.rendezvous = rendezvous_;
  opts.remote_execution = true;
  FunctionLibraryRuntime::InstantiateOptions instantiate_opts;
  instantiate_opts.target = "/job:a/replica:0/task:0/cpu:0";
  auto x = test::AsTensor<float>({1, 2, 3, 4});
  Tensor y;
  TF_CHECK_OK(
      Run("XTimesTwo", opts, {{"T", DT_FLOAT}}, instantiate_opts, {x}, {&y}));
  test::ExpectTensorEqual<float>(y, test::AsTensor<float>({2, 4, 6, 8}));
}

TEST_F(ProcessFunctionLibraryRuntimeTest, SingleCallFindDevice) {
  Init({test::function::FindDevice()});
  FunctionLibraryRuntime::Options opts;
  opts.source_device = "/job:a/replica:0/task:0/cpu:0";
  opts.rendezvous = rendezvous_;
  opts.remote_execution = true;
  FunctionLibraryRuntime::InstantiateOptions instantiate_opts;
  instantiate_opts.target = "/job:a/replica:0/task:0/cpu:0";
  Tensor y;
  TF_CHECK_OK(Run("FindDevice", opts, {}, instantiate_opts, {}, {&y}));
  test::ExpectTensorEqual<string>(
      y, test::AsTensor<string>({"/job:a/replica:0/task:0/device:CPU:0"},
                                TensorShape({})));
}

TEST_F(ProcessFunctionLibraryRuntimeTest, MultipleCallsSameDeviceXTimes) {
  Init({test::function::XTimesTwo(), test::function::XTimesFour()});
  auto x = test::AsTensor<float>({1, 2, 3, 4});
  FunctionLibraryRuntime::Options opts;
  opts.source_device = "/job:a/replica:0/task:0/cpu:0";
  opts.rendezvous = rendezvous_;
  opts.remote_execution = true;
  FunctionLibraryRuntime::InstantiateOptions instantiate_opts;
  instantiate_opts.target = "/job:a/replica:0/task:0/cpu:0";
  Tensor y;
  TF_CHECK_OK(
      Run("XTimesTwo", opts, {{"T", DT_FLOAT}}, instantiate_opts, {x}, {&y}));
  test::ExpectTensorEqual<float>(y, test::AsTensor<float>({2, 4, 6, 8}));
  TF_CHECK_OK(
      Run("XTimesFour", opts, {{"T", DT_FLOAT}}, instantiate_opts, {x}, {&y}));
  test::ExpectTensorEqual<float>(y, test::AsTensor<float>({4, 8, 12, 16}));
}

TEST_F(ProcessFunctionLibraryRuntimeTest, MultipleCallsSameDeviceFindDevice) {
  Init({test::function::FindDevice()});
  FunctionLibraryRuntime::Options opts;
  opts.source_device = "/job:a/replica:0/task:0/cpu:0";
  opts.rendezvous = rendezvous_;
  opts.remote_execution = true;
  FunctionLibraryRuntime::InstantiateOptions instantiate_opts;
  instantiate_opts.target = "/job:a/replica:0/task:0/cpu:1";
  Tensor y;
  TF_CHECK_OK(Run("FindDevice", opts, {}, instantiate_opts, {}, {&y}));
  test::ExpectTensorEqual<string>(
      y, test::AsTensor<string>({"/job:a/replica:0/task:0/device:CPU:1"},
                                TensorShape({})));
  TF_CHECK_OK(Run("FindDevice", opts, {}, instantiate_opts, {}, {&y}));
  test::ExpectTensorEqual<string>(
      y, test::AsTensor<string>({"/job:a/replica:0/task:0/device:CPU:1"},
                                TensorShape({})));
}

TEST_F(ProcessFunctionLibraryRuntimeTest, MultipleCallsDiffDeviceFindDevice) {
  Init({test::function::FindDevice()});
  FunctionLibraryRuntime::Options opts;
  opts.source_device = "/job:a/replica:0/task:0/cpu:0";
  opts.rendezvous = rendezvous_;
  opts.remote_execution = true;
  Tensor y;
  FunctionLibraryRuntime::InstantiateOptions instantiate_opts_0;
  instantiate_opts_0.target = "/job:a/replica:0/task:0/device:CPU:0";
  TF_CHECK_OK(Run("FindDevice", opts, {}, instantiate_opts_0, {}, {&y}));
  test::ExpectTensorEqual<string>(
      y, test::AsTensor<string>({"/job:a/replica:0/task:0/device:CPU:0"},
                                TensorShape({})));
  FunctionLibraryRuntime::InstantiateOptions instantiate_opts_1;
  instantiate_opts_1.target = "/job:a/replica:0/task:0/device:CPU:1";
  TF_CHECK_OK(Run("FindDevice", opts, {}, instantiate_opts_1, {}, {&y}));
  test::ExpectTensorEqual<string>(
      y, test::AsTensor<string>({"/job:a/replica:0/task:0/device:CPU:1"},
                                TensorShape({})));
}

TEST_F(ProcessFunctionLibraryRuntimeTest, ClusterFLRSerialTest) {
  Init({test::function::FindDevice()});
  FunctionLibraryRuntime::Options opts;
  opts.source_device = "/job:a/replica:0/task:0/cpu:0";
  opts.rendezvous = rendezvous_;
  opts.remote_execution = true;
  FunctionLibraryRuntime::InstantiateOptions instantiate_opts;
  instantiate_opts.target = "/job:b/replica:0/task:0/device:CPU:0";
  FunctionLibraryRuntime::Handle h;
  TF_CHECK_OK(Instantiate("FindDevice",
                          {{"_target", "/job:b/replica:0/task:0/device:CPU:0"}},
                          instantiate_opts, &h));
  EXPECT_EQ(0, proc_flr_->GetHandleOnDevice(
                   "/job:b/replica:0/task:0/device:CPU:0", h));
  TF_CHECK_OK(Instantiate("FindDevice",
                          {{"_target", "/job:b/replica:0/task:0/device:CPU:0"}},
                          instantiate_opts, &h));
  EXPECT_EQ(0, proc_flr_->GetHandleOnDevice(
                   "/job:b/replica:0/task:0/device:CPU:0", h));
  instantiate_opts.target = "/job:c/replica:0/task:0/device:CPU:0";
  TF_CHECK_OK(Instantiate("FindDevice",
                          {{"_target", "/job:c/replica:0/task:0/device:CPU:0"}},
                          instantiate_opts, &h));
  EXPECT_EQ(1, proc_flr_->GetHandleOnDevice(
                   "/job:c/replica:0/task:0/device:CPU:0", h));
}

TEST_F(ProcessFunctionLibraryRuntimeTest, ClusterFLRParallelTest) {
  Init({test::function::FindDevice()});
  FunctionLibraryRuntime::Options opts;
  opts.source_device = "/job:a/replica:0/task:0/cpu:0";
  opts.rendezvous = rendezvous_;
  opts.remote_execution = true;
  FunctionLibraryRuntime::InstantiateOptions instantiate_opts;
  instantiate_opts.target = "/job:b/replica:0/task:0/device:CPU:0";

  thread::ThreadPool* tp = new thread::ThreadPool(Env::Default(), "test", 4);
  auto fn = [this, &instantiate_opts]() {
    FunctionLibraryRuntime::Handle h;
    TF_CHECK_OK(Instantiate(
        "FindDevice", {{"_target", "/job:b/replica:0/task:0/device:CPU:0"}},
        instantiate_opts, &h));
    EXPECT_EQ(0, proc_flr_->GetHandleOnDevice(
                     "/job:b/replica:0/task:0/device:CPU:0", h));
  };

  for (int i = 0; i < 100; ++i) {
    tp->Schedule(fn);
  }
  delete tp;
}

bool IsCUDATensor(const Tensor& t) {
#ifdef GOOGLE_CUDA
  cudaPointerAttributes attributes;
  cudaError_t err =
      cudaPointerGetAttributes(&attributes, t.tensor_data().data());
  if (err == cudaErrorInvalidValue) return false;
  CHECK_EQ(cudaSuccess, err) << cudaGetErrorString(err);
  return (attributes.memoryType == cudaMemoryTypeDevice);
#else
  CHECK(false)
      << "IsCUDATensor should not be called when CUDA is not available";
#endif  // GOOGLE_CUDA
}

void TestTwoDeviceMult(
    ProcessFunctionLibraryRuntimeTest* fixture,
    const FunctionLibraryRuntime::InstantiateOptions& inst_opts,
    const string& error = "") {
  fixture->Init({test::function::TwoDeviceMult()});
  FunctionLibraryRuntime::Options opts;
  opts.rendezvous = fixture->rendezvous_;
  auto x = test::AsTensor<float>({1, 2, 3});
  Tensor y_cpu;
  Tensor y_gpu;
  Status status = fixture->Run("TwoDeviceMult", opts, {{"T", DT_FLOAT}},
                               inst_opts, {x}, {&y_cpu, &y_gpu});
  if (!error.empty()) {
    EXPECT_TRUE(errors::IsInvalidArgument(status))
        << "Actual status: " << status;
    EXPECT_TRUE(str_util::StrContains(status.error_message(), error))
        << "Actual error message: " << status.error_message();
    return;
  }

  EXPECT_TRUE(status.ok()) << "Actual status: " << status;
  EXPECT_FALSE(IsCUDATensor(y_cpu));
  test::ExpectTensorEqual<float>(y_cpu, test::AsTensor<float>({2, 4, 6}));

  EXPECT_TRUE(IsCUDATensor(y_gpu));
  Tensor y_gpu_on_cpu = fixture->GPUToCPU(y_gpu);
  test::ExpectTensorEqual<float>(y_gpu_on_cpu,
                                 test::AsTensor<float>({3, 6, 9}));
}

void TestTwoDeviceInputOutput(
    ProcessFunctionLibraryRuntimeTest* fixture,
    const FunctionLibraryRuntime::InstantiateOptions& inst_opts) {
  if (fixture->gpu_device_ == nullptr) {
    GTEST_SKIP() << "No GPUs available";
  }
  fixture->Init({test::function::TwoDeviceInputOutput()});

  FunctionLibraryRuntime::Options opts;
  opts.rendezvous = fixture->rendezvous_;
  Tensor x1 = test::AsTensor<float>({1, 2});
  if (str_util::StrContains(inst_opts.input_devices[0], "GPU")) {
    x1 = fixture->CPUToGPU(x1);
  }
  Tensor x2 = test::AsTensor<float>({10, 20});
  if (str_util::StrContains(inst_opts.input_devices[1], "GPU")) {
    x2 = fixture->CPUToGPU(x2);
  }
  Tensor y1;
  Tensor y2;
  TF_CHECK_OK(fixture->Run("TwoDeviceInputOutput", opts, {{"T", DT_FLOAT}},
                           inst_opts, {x1, x2}, {&y1, &y2}));

  if (str_util::StrContains(inst_opts.output_devices[0], "GPU")) {
    EXPECT_TRUE(IsCUDATensor(y1));
    y1 = fixture->GPUToCPU(y1);
  } else {
    EXPECT_FALSE(IsCUDATensor(y1));
  }
  test::ExpectTensorEqual<float>(y1, test::AsTensor<float>({2, 4}));

  if (str_util::StrContains(inst_opts.output_devices[1], "GPU")) {
    EXPECT_TRUE(IsCUDATensor(y2));
    y2 = fixture->GPUToCPU(y2);
  } else {
    EXPECT_FALSE(IsCUDATensor(y2));
  }
  test::ExpectTensorEqual<float>(y2, test::AsTensor<float>({30, 60}));
}

std::vector<string> CompleteDevices(const std::vector<string>& v) {
  std::vector<string> result;
  result.reserve(v.size());
  for (const string& s : v) {
    result.push_back(strings::StrCat("/job:a/replica:0/task:0/device:", s));
  }
  return result;
}

FunctionLibraryRuntime::InstantiateOptions MakeOptions(
    const string& target, const std::vector<string>& input_devices,
    const std::vector<string>& output_devices) {
  FunctionLibraryRuntime::InstantiateOptions inst_opts;
  inst_opts.target = target;
  inst_opts.input_devices = CompleteDevices(input_devices);
  inst_opts.output_devices = CompleteDevices(output_devices);
  inst_opts.is_multi_device_function = true;
  return inst_opts;
}

TEST_F(ProcessFunctionLibraryRuntimeTest, MultiDevice_ExplicitOutputDevice) {
  if (gpu_device_ == nullptr) {
    GTEST_SKIP() << "No GPUs available";
  }
  TestTwoDeviceMult(this, MakeOptions("CPU:0", {"CPU:0"}, {"CPU:0", "GPU:0"}));
}

TEST_F(ProcessFunctionLibraryRuntimeTest, MultiDevice_InferredOutputDevice) {
  if (gpu_device_ == nullptr) {
    GTEST_SKIP() << "No GPUs available";
  }
  TestTwoDeviceMult(this, MakeOptions("CPU:0", {"CPU:0"}, {}));
}

TEST_F(ProcessFunctionLibraryRuntimeTest, MultiDevice_ErrorWhenNoInputDevices) {
  if (gpu_device_ == nullptr) {
    GTEST_SKIP() << "No GPUs available";
  }
  TestTwoDeviceMult(this, MakeOptions("CPU:0", {}, {}),
                    "input_devices must have the same length");
}

TEST_F(ProcessFunctionLibraryRuntimeTest,
       MultiDevice_ErrorWhenTooManyInputDevices) {
  if (gpu_device_ == nullptr) {
    GTEST_SKIP() << "No GPUs available";
  }
  TestTwoDeviceMult(this, MakeOptions("CPU:0", {"CPU:0", "CPU:1"}, {}),
                    "input_devices must have the same length");
}

TEST_F(ProcessFunctionLibraryRuntimeTest,
       MultiDevice_ErrorWhenTooManyOutputDevices) {
  TestTwoDeviceMult(
      this, MakeOptions("CPU:0", {"CPU:0"}, {"CPU:0", "GPU:0", "CPU:1"}),
      "output_devices must either be empty or have the same length");
}

TEST_F(ProcessFunctionLibraryRuntimeTest,
       MultiDevice_ErrorWhenBadTargetDevice) {
  TestTwoDeviceMult(
      this, MakeOptions("GPU:11", {"CPU:0"}, {"CPU:0", "GPU:0"}),
      "Cannot instantiate multi-device function with target device GPU:11");
}

TEST_F(ProcessFunctionLibraryRuntimeTest, MultiDevice_ErrorWhenListInput) {
  const FunctionDef& def = test::function::FuncWithListInput();
  Init({def});
  FunctionLibraryRuntime::Handle handle;
  Status status = proc_flr_->Instantiate(
      "FuncWithListInput", test::function::Attrs({{"T", DT_FLOAT}, {"N", 1}}),
      MakeOptions("CPU:0", {"CPU:0"}, {}), &handle);
  ASSERT_TRUE(errors::IsInvalidArgument(status)) << "Actual status: " << status;
  ASSERT_TRUE(str_util::StrContains(
      status.error_message(),
      "FuncWithListInput has an input named \"x1\" that is a list of tensors"))
      << "Actual error message: " << status.error_message();
}

TEST_F(ProcessFunctionLibraryRuntimeTest, MultiDevice_ErrorWhenListOutput) {
  const FunctionDef& def = test::function::FuncWithListOutput();
  Init({def});
  FunctionLibraryRuntime::Handle handle;
  Status status = proc_flr_->Instantiate(
      "FuncWithListOutput", test::function::Attrs({{"T", DT_FLOAT}, {"N", 1}}),
      MakeOptions("CPU:0", {}, {"CPU:0"}), &handle);
  ASSERT_TRUE(errors::IsInvalidArgument(status)) << "Actual status: " << status;
  ASSERT_TRUE(str_util::StrContains(
      status.error_message(),
      "FuncWithListOutput has an output named \"y\" that is a list of tensors"))
      << "Actual error message: " << status.error_message();
}

TEST_F(ProcessFunctionLibraryRuntimeTest,
       MultiDevice_ExplicitMultiInputOutput) {
  TestTwoDeviceInputOutput(
      this, MakeOptions("CPU:0", {"CPU:0", "GPU:0"}, {"CPU:0", "GPU:0"}));
}

TEST_F(ProcessFunctionLibraryRuntimeTest, MultiDevice_FlipInputs) {
  TestTwoDeviceInputOutput(
      this, MakeOptions("CPU:0", {"GPU:0", "CPU:0"}, {"CPU:0", "GPU:0"}));
}

TEST_F(ProcessFunctionLibraryRuntimeTest, MultiDevice_FlipOutputs) {
  TestTwoDeviceInputOutput(
      this, MakeOptions("CPU:0", {"CPU:0", "GPU:0"}, {"GPU:0", "CPU:0"}));
}

TEST_F(ProcessFunctionLibraryRuntimeTest, MultiDevice_FlipBoth) {
  TestTwoDeviceInputOutput(
      this, MakeOptions("CPU:0", {"GPU:0", "CPU:0"}, {"GPU:0", "CPU:0"}));
}

TEST_F(ProcessFunctionLibraryRuntimeTest, MultiDevice_EmptyBodySwap) {
  if (gpu_device_ == nullptr) {
    GTEST_SKIP() << "No GPUs available";
  }
  FunctionLibraryRuntime::InstantiateOptions inst_opts =
      MakeOptions("CPU:0", {"GPU:0", "CPU:0"}, {"CPU:0", "GPU:0"});
  Init({test::function::EmptyBodySwap()});

  Tensor x1 = CPUToGPU(test::AsTensor<float>({1, 2}));
  Tensor x2 = test::AsTensor<float>({10, 20});
  Tensor y1;
  Tensor y2;
  TF_CHECK_OK(Run("EmptyBodySwap", {}, {{"T", DT_FLOAT}}, inst_opts, {x1, x2},
                  {&y1, &y2}));

  EXPECT_FALSE(IsCUDATensor(y1));
  test::ExpectTensorEqual<float>(y1, test::AsTensor<float>({10, 20}));

  EXPECT_TRUE(IsCUDATensor(y2));
  y2 = GPUToCPU(y2);
  test::ExpectTensorEqual<float>(y2, test::AsTensor<float>({1, 2}));
}

Tensor GetResourceHandle(const string& var_name, const string& container,
                         const string& device_name) {
  ResourceHandle handle;
  handle.set_device(device_name);
  handle.set_container(container);
  handle.set_name(var_name);
  handle.set_hash_code(MakeTypeIndex<Var>().hash_code());
  handle.set_maybe_type_name(MakeTypeIndex<Var>().name());
  Tensor tensor(DT_RESOURCE, TensorShape({}));
  tensor.scalar<ResourceHandle>()() = handle;
  return tensor;
}

TEST_F(ProcessFunctionLibraryRuntimeTest, MultiDevice_ResourceOutput_GPU) {
  if (gpu_device_ == nullptr) {
    GTEST_SKIP() << "No GPUs available";
  }
  FunctionLibraryRuntime::InstantiateOptions inst_opts =
      MakeOptions("CPU:0", {"GPU:0", "GPU:0"}, {"GPU:0", "GPU:0"});
  Init({test::function::ResourceOutput(),
        test::function::ReadResourceVariable()});

  // Make resource var
  Tensor resource_value = CPUToGPU(test::AsTensor<float>({10, 20}));
  Var* resource = new Var(DT_FLOAT);
  *resource->tensor() = resource_value;
  resource->is_initialized = true;
  ResourceMgr* mgr = gpu_device_->resource_manager();
  Status status = mgr->Create(mgr->default_container(), "my_gpu_var", resource);
  ASSERT_TRUE(status.ok()) << status.error_message();

  // Run the function taking a resource and outputing it
  FunctionLibraryRuntime::Options opts;
  opts.rendezvous = rendezvous_;
  Tensor x1 = CPUToGPU(test::AsTensor<float>({1, 2}));
  Tensor x2 = GetResourceHandle("my_gpu_var", mgr->default_container(),
                                "/job:a/replica:0/task:0/device:GPU:0");
  Tensor returned_handle;
  Tensor y2;
  TF_CHECK_OK(Run("ResourceOutput", opts, {{"T", DT_FLOAT}}, inst_opts,
                  {x1, x2}, {&returned_handle, &y2}));

  EXPECT_FALSE(IsCUDATensor(returned_handle));
  EXPECT_TRUE(IsCUDATensor(y2));
  y2 = GPUToCPU(y2);
  test::ExpectTensorEqual<float>(y2, test::AsTensor<float>({2, 4}));

  // Read the variable using the handle returned from previous function to
  // make sure the handle and read value is on the right device.
  inst_opts = MakeOptions("GPU:0", {"GPU:0"}, {"GPU:0"});
  Tensor read_resource;
  TF_CHECK_OK(Run("ReadResourceVariable", opts, {{"T", DT_FLOAT}}, inst_opts,
                  {returned_handle}, {&read_resource}));
  EXPECT_TRUE(IsCUDATensor(read_resource));
  read_resource = GPUToCPU(read_resource);
  test::ExpectTensorEqual<float>(read_resource,
                                 test::AsTensor<float>({10, 20}));
}

TEST_F(ProcessFunctionLibraryRuntimeTest, MultiDevice_PlacerError) {
  if (gpu_device_ == nullptr) {
    GTEST_SKIP() << "No GPUs available";
  }
  // ResourceOutput forwards second input to first output. Both are resources.
  // Placer should not be able to place this graph because we ask it to place
  // second input on GPU but first output to CPU.
  FunctionLibraryRuntime::InstantiateOptions inst_opts =
      MakeOptions("CPU:0", {"GPU:0", "GPU:0"}, {"CPU:0", "GPU:0"});
  Init({test::function::ResourceOutput(),
        test::function::ReadResourceVariable()});

  FunctionLibraryRuntime::Handle handle;
  Status status = proc_flr_->Instantiate(
      "ResourceOutput", test::function::Attrs({{"T", DT_FLOAT}}), inst_opts,
      &handle);
  ASSERT_TRUE(errors::IsInvalidArgument(status)) << "Actual status: " << status;
  ASSERT_TRUE(str_util::StrContains(status.error_message(), "Cannot place"));
}

REGISTER_OP("BrokenOp")
    .Input("in: T")
    .Output("out: T")
    .Attr("T: type")
    .SetShapeFn(shape_inference::UnknownShape);
class BrokenOp : public OpKernel {
 public:
  explicit BrokenOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    ctx->SetStatus(errors::Internal("I am broken"));
  }

  void Compute(OpKernelContext* ctx) override {
    ctx->SetStatus(errors::Internal("I am broken"));
  }
};
REGISTER_KERNEL_BUILDER(Name("BrokenOp").Device(DEVICE_CPU), BrokenOp);

TEST_F(ProcessFunctionLibraryRuntimeTest, MultiDevice_CreateKernelsEagerly) {
  auto T = DT_INT32;
  // The expected sequence of outputs from this function is [6, 4, 0, 1, ...].
  FunctionDef broken_func = FunctionDefHelper::Define(
      // Name
      "Broken",
      // Args
      {"x: int32"},
      // Return values
      {"y: int32"},
      // Attrs
      {},
      // Nodes
      {{{"y"}, "BrokenOp", {"x"}, {{"T", T}}}});
  Init({broken_func});

  FunctionLibraryRuntime::InstantiateOptions inst_opts =
      MakeOptions("CPU:0", {"CPU:0"}, {"CPU:0"});

  // Instantiating the broken function should work.
  FunctionLibraryRuntime::Handle handle;
  TF_CHECK_OK(Instantiate("Broken", {{"T", DT_INT32}}, inst_opts, &handle));
  TF_CHECK_OK(proc_flr_->ReleaseHandle(handle));

  // Instantiating the broken function while creating kernels eagerly should
  // fail.
  inst_opts.create_kernels_eagerly = true;
  Status status = Instantiate("Broken", {{"T", DT_INT32}}, inst_opts, &handle);
  EXPECT_TRUE(errors::IsInternal(status));
}

TEST_F(ProcessFunctionLibraryRuntimeTest, MultiDevice_StateHandle) {
  auto T = DT_INT32;
  // The expected sequence of outputs from this function is [6, 4, 0, 1, ...].
  FunctionDef stateful_func = FunctionDefHelper::Define(
      // Name
      "RandomUniformWrapper",
      // Args
      {"x: resource"},
      // Return values
      {"y: int32"},
      // Attrs
      {},
      // Nodes
      {FunctionDefHelper::Const<int32>("shape", gtl::ArraySlice<int32>({1})),
       FunctionDefHelper::Const<int32>("minval", 0),
       {{"maxval"}, "ReadVariableOp", {"x"}, {{"dtype", T}}, {}},
       // A stateful node.
       {{"y"},
        "RandomUniformInt",
        {"shape", "minval", "maxval"},
        {{"seed", 37}, {"seed2", 48}, {"Tout", T}, {"T", T}}}});
  Init({stateful_func});
  if (gpu_device_ == nullptr) {
    GTEST_SKIP() << "No GPUs available";
  }

  // Make resource variables.
  ResourceMgr* mgr = gpu_device_->resource_manager();
  Tensor resource_value = CPUToGPU(test::AsScalar<int>(10));
  Var* resource = new Var(T);
  *resource->tensor() = resource_value;
  resource->is_initialized = true;
  Status status = mgr->Create(mgr->default_container(), "my_gpu_var", resource);
  ASSERT_TRUE(status.ok()) << status.error_message();

  Tensor x = GetResourceHandle("my_gpu_var", mgr->default_container(),
                               "/job:a/replica:0/task:0/device:GPU:0");
  Tensor y;

  FunctionLibraryRuntime::InstantiateOptions inst_opts =
      MakeOptions("CPU:0", {"GPU:0"}, {"CPU:0"});

  // Instantiate the function with no state handle.
  FunctionLibraryRuntime::Handle handle;
  TF_CHECK_OK(Instantiate("RandomUniformWrapper", {{"T", DT_INT32}}, inst_opts,
                          &handle));
  for (auto expected : {6, 4}) {
    TF_CHECK_OK(RunInstantiated(handle, {}, {x}, {&y}));
    test::ExpectTensorEqual<int>(y, test::AsTensor<int>({expected}));
  }

  // Instantiating the function again with no state handle should result in the
  // same handle.
  FunctionLibraryRuntime::Handle other_handle;
  TF_CHECK_OK(Instantiate("RandomUniformWrapper", {{"T", DT_INT32}}, inst_opts,
                          &other_handle));
  EXPECT_EQ(handle, other_handle);
  // Running the function should yield continuation of the same sequence.
  for (auto expected : {0, 1}) {
    TF_CHECK_OK(RunInstantiated(other_handle, {}, {x}, {&y}));
    test::ExpectTensorEqual<int>(y, test::AsTensor<int>({expected}));
  }

  // Instantiating the function with a state handle should result in a different
  // handle.
  inst_opts.state_handle = "handle_1";
  TF_CHECK_OK(Instantiate("RandomUniformWrapper", {{"T", DT_INT32}}, inst_opts,
                          &other_handle));
  EXPECT_NE(handle, other_handle);
  // Running the function should yield the original sequeunce.
  for (auto expected : {6, 4, 0, 1}) {
    TF_CHECK_OK(RunInstantiated(other_handle, {}, {x}, {&y}));
    test::ExpectTensorEqual<int>(y, test::AsTensor<int>({expected}));
  }

  // Instantiating the function with a different state handle should result in a
  // different handle.
  inst_opts.state_handle = "handle_2";
  TF_CHECK_OK(Instantiate("RandomUniformWrapper", {{"T", DT_INT32}}, inst_opts,
                          &other_handle));
  EXPECT_NE(handle, other_handle);
  // Running the function should yield the original sequeunce.
  for (auto expected : {6, 4, 0, 1}) {
    TF_CHECK_OK(RunInstantiated(other_handle, {}, {x}, {&y}));
    test::ExpectTensorEqual<int>(y, test::AsTensor<int>({expected}));
  }

  // Repeatedly instantiating a function and releasing its handle will result in
  // repeating the original sequence.
  inst_opts.state_handle = "handle_3";
  for (int i = 0; i < 2; ++i) {
    TF_CHECK_OK(Instantiate("RandomUniformWrapper", {{"T", DT_INT32}},
                            inst_opts, &other_handle));
    EXPECT_NE(handle, other_handle);
    // Running the function should yield the original sequeunce.
    for (auto expected : {6, 4, 0, 1}) {
      TF_CHECK_OK(RunInstantiated(other_handle, {}, {x}, {&y}));
      test::ExpectTensorEqual<int>(y, test::AsTensor<int>({expected}));
    }
    TF_CHECK_OK(proc_flr_->ReleaseHandle(other_handle));
  }
}

}  // anonymous namespace
}  // namespace tensorflow
