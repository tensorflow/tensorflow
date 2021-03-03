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

#include <memory>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/composite_device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/function_testlib.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_var.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#elif TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_runtime.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {
namespace {

class TestClusterFLR : public DistributedFunctionLibraryRuntime {
 public:
  explicit TestClusterFLR(DeviceMgr* device_mgr) : device_mgr_(device_mgr) {}

  void Instantiate(const string& function_name,
                   const FunctionLibraryDefinition& lib_def, AttrSlice attrs,
                   const FunctionLibraryRuntime::InstantiateOptions& options,
                   FunctionLibraryRuntime::LocalHandle* handle,
                   FunctionLibraryRuntime::DoneCallback done) override {
    {
      mutex_lock l(mu_);
      *handle = next_handle_;
      next_handle_++;
    }
    done(Status::OK());
  }

  void Run(const FunctionLibraryRuntime::Options& opts,
           FunctionLibraryRuntime::LocalHandle handle,
           gtl::ArraySlice<Tensor> args, std::vector<Tensor>* rets,
           FunctionLibraryRuntime::DoneCallback done) override {}

  void Run(const FunctionLibraryRuntime::Options& opts,
           FunctionLibraryRuntime::LocalHandle handle,
           gtl::ArraySlice<FunctionArg> args, std::vector<FunctionRet>* rets,
           FunctionLibraryRuntime::DoneCallback done) override {}

  void CleanUp(uint64 step_id, FunctionLibraryRuntime::LocalHandle handle,
               FunctionLibraryRuntime::DoneCallback done) override {}

  DeviceMgr* remote_device_mgr() const override { return device_mgr_; }

 private:
  mutex mu_;
  int next_handle_ TF_GUARDED_BY(mu_) = 0;
  DeviceMgr* device_mgr_;
};

SessionMetadata GenerateSessionMetadata() {
  SessionMetadata session_metadata;
  session_metadata.set_name("name");
  session_metadata.set_version(42);
  return session_metadata;
}

// TODO(b/128707168): Tests requiring a GPU device are currently always skipped
// because the check for whether a GPU device is present happens before the GPU
// device is set up.
class ProcessFunctionLibraryRuntimeTest : public ::testing::Test {
 public:
  ProcessFunctionLibraryRuntimeTest() {
    SessionOptions options;
    auto* device_count = options.config.mutable_device_count();
    device_count->insert({"CPU", 3});
    std::vector<std::unique_ptr<Device>> created_devices;
    TF_CHECK_OK(DeviceFactory::AddDevices(options, "/job:a/replica:0/task:0",
                                          &created_devices));
    // Do not add CPU:2 to device manager. Used for removed device testing.
    device2_ = std::move(created_devices[2]);
    created_devices.erase(created_devices.begin() + 2);

    device_mgr_ = std::make_unique<DynamicDeviceMgr>();
    TF_CHECK_OK(device_mgr_->AddDevices(std::move(created_devices)));
    TF_CHECK_OK(device_mgr_->LookupDevice(
        "/job:a/replica:0/task:0/device:CPU:0", &device0_));
    TF_CHECK_OK(device_mgr_->LookupDevice(
        "/job:a/replica:0/task:0/device:CPU:1", &device1_));
    Device* device2_ptr = nullptr;
    EXPECT_NE(
        error::OK,
        device_mgr_
            ->LookupDevice("/job:a/replica:0/task:0/device:CPU:2", &device2_ptr)
            .code());
    // If no GPU is available, gpu_device_ will remain nullptr.
    Status status = device_mgr_->LookupDevice(
        "/job:a/replica:0/task:0/device:GPU:0", &gpu_device_);
    if (!status.ok()) {
      CHECK_EQ(nullptr, gpu_device_);
    }
  }

  void Init(const std::vector<FunctionDef>& flib,
            const SessionMetadata* session_metadata = nullptr) {
    FunctionDefLibrary proto;
    for (const auto& fdef : flib) *(proto.add_function()) = fdef;
    lib_def_.reset(new FunctionLibraryDefinition(OpRegistry::Global(), proto));
    OptimizerOptions opts;
    cluster_flr_.reset(new TestClusterFLR(device_mgr_.get()));
    proc_flr_.reset(new ProcessFunctionLibraryRuntime(
        device_mgr_.get(), Env::Default(), /*config=*/nullptr,
        TF_GRAPH_DEF_VERSION, lib_def_.get(), opts,
        /*thread_pool=*/nullptr, cluster_flr_.get(), session_metadata,
        Rendezvous::Factory{
            [this](const int64 step_id, const DeviceMgr* device_mgr,
                   Rendezvous** r) {
              *r = new IntraProcessRendezvous(device_mgr);
              if (rendezvous_ref_counts_.find(step_id) !=
                  rendezvous_ref_counts_.end()) {
                rendezvous_ref_counts_[step_id]++;
              } else {
                rendezvous_ref_counts_[step_id] = 1;
              }
              return Status::OK();
            },
            [this](const int64 step_id) {
              CHECK(rendezvous_ref_counts_.find(step_id) !=
                    rendezvous_ref_counts_.end());
              rendezvous_ref_counts_[step_id]--;
              return Status::OK();
            }}));
  }

  void AddCompositeDevice(CompositeDevice* d) {
    proc_flr_->AddCompositeDevice(d);
  }

  Status Instantiate(
      const string& name, test::function::Attrs attrs,
      const FunctionLibraryRuntime::InstantiateOptions& instantiate_opts,
      FunctionLibraryRuntime::Handle* handle) {
    return proc_flr_->Instantiate(name, attrs, instantiate_opts, handle);
  }

  Tensor GPUToCPU(const Tensor& device_tensor) {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    CHECK(gpu_device_);
    CHECK(gpu_device_->tensorflow_gpu_device_info() != nullptr);
    DeviceContext* device_context =
        gpu_device_->tensorflow_gpu_device_info()->default_context;

    Tensor cpu_tensor(device_tensor.dtype(), device_tensor.shape());
    CHECK(device_context
              ->CopyDeviceTensorToCPUSync(&device_tensor, "", gpu_device_,
                                          &cpu_tensor)
              .ok());
    return cpu_tensor;
#else
    CHECK(false);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  }

  Tensor CPUToGPU(const Tensor& cpu_tensor) {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    CHECK(gpu_device_);
    CHECK(gpu_device_->tensorflow_gpu_device_info() != nullptr);
    DeviceContext* device_context =
        gpu_device_->tensorflow_gpu_device_info()->default_context;

    Tensor device_tensor(gpu_device_->GetAllocator({}), cpu_tensor.dtype(),
                         cpu_tensor.shape(), {});
    CHECK(device_context
              ->CopyCPUTensorToDeviceSync(&cpu_tensor, gpu_device_,
                                          &device_tensor)
              .ok());
    return device_tensor;
#else
    CHECK(false);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  }

  template <typename T, typename K>
  Status RunWithRuntime(
      const string& name, FunctionLibraryRuntime::Options opts,
      test::function::Attrs attrs,
      const FunctionLibraryRuntime::InstantiateOptions& instantiate_opts,
      const T& args, std::vector<K*> rets,
      ProcessFunctionLibraryRuntime* pflr) {
    FunctionLibraryRuntime::Handle handle;
    Status status = pflr->Instantiate(name, attrs, instantiate_opts, &handle);
    if (!status.ok()) {
      return status;
    }
    bool is_cross_process = false;
    TF_CHECK_OK(pflr->IsCrossProcess(handle, &is_cross_process));
    EXPECT_FALSE(is_cross_process);

    std::atomic<int32> call_count(0);
    std::function<void(std::function<void()>)> runner =
        [&call_count](std::function<void()> fn) {
          ++call_count;
          test::function::FunctionTestSchedClosure(fn);
        };

    Notification done;
    opts.runner = &runner;
    std::vector<K> out;
    pflr->Run(opts, handle, args, &out, [&status, &done](const Status& s) {
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
    status = pflr->ReleaseHandle(handle);
    if (!status.ok()) {
      return status;
    }
    Notification done2;
    pflr->Run(opts, handle, args, &out, [&status, &done2](const Status& s) {
      status = s;
      done2.Notify();
    });
    done2.WaitForNotification();
    EXPECT_TRUE(errors::IsNotFound(status)) << "Actual status: " << status;
    EXPECT_TRUE(absl::StrContains(status.error_message(), "not found."));

    return Status::OK();
  }

  Status Run(const string& name, FunctionLibraryRuntime::Options opts,
             test::function::Attrs attrs,
             const FunctionLibraryRuntime::InstantiateOptions& instantiate_opts,
             const std::vector<Tensor>& args, std::vector<Tensor*> rets,
             ProcessFunctionLibraryRuntime* pflr = nullptr) {
    return RunWithRuntime<std::vector<Tensor>, Tensor>(
        name, opts, attrs, instantiate_opts, args, rets, proc_flr_.get());
  }

  Status RunWithPackedArgs(
      const string& name, FunctionLibraryRuntime::Options opts,
      test::function::Attrs attrs,
      const FunctionLibraryRuntime::InstantiateOptions& instantiate_opts,
      const FunctionArgsInterface& args, std::vector<FunctionRet*> rets,
      ProcessFunctionLibraryRuntime* pflr = nullptr) {
    return RunWithRuntime<FunctionArgsInterface, FunctionRet>(
        name, opts, attrs, instantiate_opts, args, rets, proc_flr_.get());
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

  std::unique_ptr<DynamicDeviceMgr> device_mgr_;
  Device* device0_ = nullptr;  // Not owned. (Owned by device_mgr_.)
  Device* device1_ = nullptr;  // Not owned. (Owned by device_mgr_.)
  std::unique_ptr<Device> device2_;
  // Remains as nullptr if no GPU is available.
  Device* gpu_device_ = nullptr;  // Not owned. (Owned by device_mgr_.)
  std::unique_ptr<FunctionLibraryDefinition> lib_def_;
  std::unique_ptr<TestClusterFLR> cluster_flr_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> proc_flr_;

  // To ensure that we are cleaning up the rendezvous properly.
  std::unordered_map<int64, int> rendezvous_ref_counts_;
};

TEST_F(ProcessFunctionLibraryRuntimeTest, GetFLRNull) {
  FunctionDefLibrary proto;
  std::unique_ptr<FunctionLibraryDefinition> lib_def(
      new FunctionLibraryDefinition(OpRegistry::Global(), proto));
  OptimizerOptions opts;
  std::unique_ptr<ProcessFunctionLibraryRuntime> proc_flr(
      new ProcessFunctionLibraryRuntime(
          nullptr /* device_mgr */, Env::Default(), /*config=*/nullptr,
          TF_GRAPH_DEF_VERSION, lib_def.get(), opts));
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
  opts.remote_execution = true;
  FunctionLibraryRuntime::InstantiateOptions instantiate_opts;
  instantiate_opts.target = "/job:a/replica:0/task:0/cpu:0";
  Tensor y;
  TF_CHECK_OK(Run("FindDevice", opts, {}, instantiate_opts, {}, {&y}));
  test::ExpectTensorEqual<tstring>(
      y, test::AsTensor<tstring>({"/job:a/replica:0/task:0/device:CPU:0"},
                                 TensorShape({})));
  EXPECT_EQ(1, rendezvous_ref_counts_.size());
  EXPECT_EQ(opts.step_id, rendezvous_ref_counts_.begin()->first);
  EXPECT_EQ(0, rendezvous_ref_counts_.begin()->second);
}

TEST_F(ProcessFunctionLibraryRuntimeTest, MultipleCallsSameDeviceXTimes) {
  Init({test::function::XTimesTwo(), test::function::XTimesFour()});
  auto x = test::AsTensor<float>({1, 2, 3, 4});
  FunctionLibraryRuntime::Options opts;
  opts.source_device = "/job:a/replica:0/task:0/cpu:0";
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
  opts.remote_execution = true;
  FunctionLibraryRuntime::InstantiateOptions instantiate_opts;
  instantiate_opts.target = "/job:a/replica:0/task:0/cpu:1";
  Tensor y;
  TF_CHECK_OK(Run("FindDevice", opts, {}, instantiate_opts, {}, {&y}));
  test::ExpectTensorEqual<tstring>(
      y, test::AsTensor<tstring>({"/job:a/replica:0/task:0/device:CPU:1"},
                                 TensorShape({})));
  TF_CHECK_OK(Run("FindDevice", opts, {}, instantiate_opts, {}, {&y}));
  test::ExpectTensorEqual<tstring>(
      y, test::AsTensor<tstring>({"/job:a/replica:0/task:0/device:CPU:1"},
                                 TensorShape({})));
}

TEST_F(ProcessFunctionLibraryRuntimeTest, MultipleCallsDiffDeviceFindDevice) {
  Init({test::function::FindDevice()});
  FunctionLibraryRuntime::Options opts;
  opts.source_device = "/job:a/replica:0/task:0/cpu:0";
  opts.remote_execution = true;
  Tensor y;
  FunctionLibraryRuntime::InstantiateOptions instantiate_opts_0;
  instantiate_opts_0.target = "/job:a/replica:0/task:0/device:CPU:0";
  TF_CHECK_OK(Run("FindDevice", opts, {}, instantiate_opts_0, {}, {&y}));
  test::ExpectTensorEqual<tstring>(
      y, test::AsTensor<tstring>({"/job:a/replica:0/task:0/device:CPU:0"},
                                 TensorShape({})));
  FunctionLibraryRuntime::InstantiateOptions instantiate_opts_1;
  instantiate_opts_1.target = "/job:a/replica:0/task:0/device:CPU:1";
  TF_CHECK_OK(Run("FindDevice", opts, {}, instantiate_opts_1, {}, {&y}));
  test::ExpectTensorEqual<tstring>(
      y, test::AsTensor<tstring>({"/job:a/replica:0/task:0/device:CPU:1"},
                                 TensorShape({})));
}

TEST_F(ProcessFunctionLibraryRuntimeTest, InstantiateFunctionOnRemovedDevice) {
  std::vector<std::unique_ptr<Device>> devices;
  Device* device2_ptr = device2_.get();
  devices.emplace_back(std::move(device2_));
  TF_CHECK_OK(device_mgr_->AddDevices(std::move(devices)));

  Init({test::function::FindDevice()});
  std::vector<Device*> remove_devices{device2_ptr};
  TF_CHECK_OK(device_mgr_->RemoveDevices(std::move(remove_devices)));

  // Since the process FLR device set is not updated yet, it still holds the
  // raw pointer to device2. Make sure that function instantion with device2
  // will not lead to segfault.
  FunctionLibraryRuntime::InstantiateOptions instantiate_opts;
  FunctionLibraryRuntime::Handle h;
  instantiate_opts.target = "/job:a/replica:0/task:0/device:CPU:1";
  instantiate_opts.is_multi_device_function = true;
  TF_CHECK_OK(Instantiate("FindDevice",
                          {{"_target", "/job:b/replica:0/task:0/device:CPU:2"}},
                          instantiate_opts, &h));
}

TEST_F(ProcessFunctionLibraryRuntimeTest, ClusterFLRSerialTest) {
  Init({test::function::FindDevice()});
  FunctionLibraryRuntime::Options opts;
  opts.source_device = "/job:a/replica:0/task:0/cpu:0";
  opts.remote_execution = true;
  FunctionLibraryRuntime::InstantiateOptions instantiate_opts;
  instantiate_opts.target = "/job:b/replica:0/task:0/device:CPU:0";
  FunctionLibraryRuntime::Handle h;
  TF_CHECK_OK(Instantiate("FindDevice",
                          {{"_target", "/job:b/replica:0/task:0/device:CPU:0"}},
                          instantiate_opts, &h));
  bool is_cross_process = false;
  TF_CHECK_OK(proc_flr_->IsCrossProcess(h, &is_cross_process));
  EXPECT_TRUE(is_cross_process);
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
#if GOOGLE_CUDA
  cudaPointerAttributes attributes;
  cudaError_t err =
      cudaPointerGetAttributes(&attributes, t.tensor_data().data());
  if (err == cudaErrorInvalidValue) return false;
  CHECK_EQ(cudaSuccess, err) << cudaGetErrorString(err);
  return (attributes.type == cudaMemoryTypeDevice);
#elif TENSORFLOW_USE_ROCM
  hipPointerAttribute_t attributes;
  hipError_t err = hipPointerGetAttributes(&attributes, t.tensor_data().data());
  if (err == hipErrorInvalidValue) return false;
  CHECK_EQ(hipSuccess, err) << hipGetErrorString(err);
  return (attributes.memoryType == hipMemoryTypeDevice);
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
  auto x = test::AsTensor<float>({1, 2, 3});
  Tensor y_cpu;
  Tensor y_gpu;
  Status status = fixture->Run("TwoDeviceMult", opts, {{"T", DT_FLOAT}},
                               inst_opts, {x}, {&y_cpu, &y_gpu});
  if (!error.empty()) {
    EXPECT_TRUE(errors::IsInvalidArgument(status))
        << "Actual status: " << status;
    EXPECT_TRUE(absl::StrContains(status.error_message(), error))
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
  Tensor x1 = test::AsTensor<float>({1, 2});
  if (absl::StrContains(inst_opts.input_devices[0], "GPU")) {
    x1 = fixture->CPUToGPU(x1);
  }
  Tensor x2 = test::AsTensor<float>({10, 20});
  if (absl::StrContains(inst_opts.input_devices[1], "GPU")) {
    x2 = fixture->CPUToGPU(x2);
  }
  Tensor y1;
  Tensor y2;
  TF_CHECK_OK(fixture->Run("TwoDeviceInputOutput", opts, {{"T", DT_FLOAT}},
                           inst_opts, {x1, x2}, {&y1, &y2}));

  if (absl::StrContains(inst_opts.output_devices[0], "GPU")) {
    EXPECT_TRUE(IsCUDATensor(y1));
    y1 = fixture->GPUToCPU(y1);
  } else {
    EXPECT_FALSE(IsCUDATensor(y1));
  }
  test::ExpectTensorEqual<float>(y1, test::AsTensor<float>({2, 4}));

  if (absl::StrContains(inst_opts.output_devices[1], "GPU")) {
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
  ASSERT_TRUE(absl::StrContains(
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
  ASSERT_TRUE(absl::StrContains(
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
  handle.set_hash_code(TypeIndex::Make<Var>().hash_code());
  handle.set_maybe_type_name(TypeIndex::Make<Var>().name());
  Tensor tensor(DT_RESOURCE, TensorShape({}));
  tensor.scalar<ResourceHandle>()() = handle;
  return tensor;
}

// Returns a function which adds two variables on different devices.
FunctionDef AddVarAcrossDevices() {
  return FunctionDefHelper::Create(
      // Name
      "AddVarAcrossDevices",
      // Args
      {"x: resource"},
      // Return values
      {"y: float"},
      // Attr def
      {},
      // Nodes
      {
          {{"read0"},
           "ReadVariableOp",
           {"x"},
           {{"dtype", DT_FLOAT}},
           {},
           "/device:CPU:0"},
          {{"read1"},
           "ReadVariableOp",
           {"x"},
           {{"dtype", DT_FLOAT}},
           {},
           "/device:CPU:1"},
          {{"add"},
           "Add",
           {"read0:value:0", "read1:value:0"},
           {{"T", DT_FLOAT}},
           {},
           "/device:CPU:0"},
      },
      {{"y", "add:z:0"}});
}

// An implementation of FunctionArgsInterface for packed inputs.
class TestFunctionPackedArgs : public FunctionArgsInterface {
 public:
  TestFunctionPackedArgs(const int index,
                         gtl::InlinedVector<TensorValue, 4>&& tensor_args) {
    packed_args_.emplace(index, std::move(tensor_args));
  }

  ~TestFunctionPackedArgs() override{};

  bool HasRemoteOrPackedInputs() const override { return true; };

  Status GetLocalArg(const FunctionArgIndex& index,
                     Tensor* val) const override {
    *val = *packed_args_.at(index.index).at(index.sub_index).tensor;
    return Status::OK();
  };

  std::vector<Tensor> GetLocalTensors() const override { return {}; }

 private:
  absl::flat_hash_map<int, gtl::InlinedVector<TensorValue, 4>> packed_args_;
};

TEST_F(ProcessFunctionLibraryRuntimeTest, MultiDevice_CompositeDevice) {
  Init({AddVarAcrossDevices()});
  // Create two variables on two devices.
  const Tensor initial_resource_value0 = test::AsTensor<float>({10, 20});
  Var* resource0 = new Var(DT_FLOAT);
  *resource0->tensor() = initial_resource_value0;
  resource0->is_initialized = true;
  const Tensor initial_resource_value1 = test::AsTensor<float>({30, 40});
  Var* resource1 = new Var(DT_FLOAT);
  *resource1->tensor() = initial_resource_value1;
  resource1->is_initialized = true;
  ResourceMgr* mgr0 = device0_->resource_manager();
  ResourceMgr* mgr1 = device1_->resource_manager();
  TF_ASSERT_OK(mgr0->Create(mgr0->default_container(), "var", resource0));
  TF_ASSERT_OK(mgr1->Create(mgr1->default_container(), "var", resource1));

  Tensor resource_handle0 =
      GetResourceHandle("var", mgr0->default_container(), device0_->name());
  Tensor resource_handle1 =
      GetResourceHandle("var", mgr1->default_container(), device1_->name());

  // Create a CompositeDevice
  Status s;
  std::unique_ptr<CompositeDevice> composite_device =
      CompositeDevice::MakeDevice({device0_->name(), device1_->name()},
                                  /*unique_device_id=*/0,
                                  device_mgr_->HostCPU()->parsed_name(), &s);
  TF_ASSERT_OK(s);
  AddCompositeDevice(composite_device.get());

  FunctionLibraryRuntime::Options opts;
  FunctionLibraryRuntime::InstantiateOptions inst_opts =
      MakeOptions("CPU:0", {"COMPOSITE:0"}, {"CPU:0"});
  inst_opts.composite_devices[composite_device->name()] =
      composite_device->underlying_devices();
  inst_opts.input_resource_dtypes_and_shapes[0] = {
      initial_resource_value0.dtype(), initial_resource_value0.shape()};

  // Packed TensorHandle
  {
    gtl::InlinedVector<TensorValue, 4> handles;
    handles.push_back(TensorValue(&resource_handle0));
    handles.push_back(TensorValue(&resource_handle1));
    TestFunctionPackedArgs args(0, std::move(handles));
    FunctionRet ret;
    TF_CHECK_OK(RunWithPackedArgs("AddVarAcrossDevices", opts,
                                  {{"T", DT_FLOAT}}, inst_opts, args, {&ret}));
    EXPECT_EQ(ret.index(), 0);
    test::ExpectTensorEqual<float>(absl::get<Tensor>(ret),
                                   test::AsTensor<float>({40, 60}));
  }

  // Packed Tensor
  {
    Tensor arg(DT_RESOURCE, TensorShape({2}));
    arg.flat<ResourceHandle>()(0) = resource_handle0.scalar<ResourceHandle>()();
    arg.flat<ResourceHandle>()(1) = resource_handle1.scalar<ResourceHandle>()();

    Tensor ret;
    TF_CHECK_OK(Run("AddVarAcrossDevices", opts, {{"T", DT_FLOAT}}, inst_opts,
                    {arg}, {&ret}));
    test::ExpectTensorEqual<float>(ret, test::AsTensor<float>({40, 60}));
  }
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
  ASSERT_TRUE(absl::StrContains(status.error_message(), "Cannot place"));
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

REGISTER_OP("SessionMetadataReader")
    .Input("x: int64")
    .Output("y: string")
    .SetIsStateful()
    .Doc(R"doc(SessionMetadataReader returns the session metadata.

x: int64
y: string
)doc");

class SessionMetadataReaderOp : public OpKernel {
 public:
  explicit SessionMetadataReaderOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    Tensor* out_tensor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("y", TensorShape({}), &out_tensor));
    if (ctx->session_metadata() != nullptr) {
      out_tensor->scalar<tstring>()() = ctx->session_metadata()->DebugString();
    } else {
      out_tensor->scalar<tstring>()() = "";
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("SessionMetadataReader").Device(DEVICE_CPU),
                        SessionMetadataReaderOp);

FunctionDef SessionMetadataReaderOpFn() {
  return FunctionDefHelper::Define(
      // Name
      "SessionMetadataReaderFn",
      // Args
      {"x: int64"},
      // Return values
      {"y: string"},
      // Attr def
      {},
      // Nodes
      {{{"y"}, "SessionMetadataReader", {"x"}, {}}});
}

TEST_F(ProcessFunctionLibraryRuntimeTest, SessionMetadataAbsent) {
  Init({SessionMetadataReaderOpFn()}, /*session_metadata=*/nullptr);
  FunctionLibraryRuntime::Options opts;
  opts.source_device = "/job:a/replica:0/task:0/cpu:0";
  opts.remote_execution = true;
  FunctionLibraryRuntime::InstantiateOptions instantiate_opts;
  instantiate_opts.target = "/job:a/replica:0/task:0/cpu:0";
  const auto x = test::AsTensor<int64>({17});
  Tensor y;
  TF_CHECK_OK(
      Run("SessionMetadataReaderFn", opts, {}, instantiate_opts, {x}, {&y}));
  EXPECT_EQ("", y.scalar<tstring>()());
}

TEST_F(ProcessFunctionLibraryRuntimeTest, SessionMetadataPresent) {
  const SessionMetadata session_metadata = GenerateSessionMetadata();
  Init({SessionMetadataReaderOpFn()}, &session_metadata);
  FunctionLibraryRuntime::Options opts;
  opts.source_device = "/job:a/replica:0/task:0/cpu:0";
  opts.remote_execution = true;
  FunctionLibraryRuntime::InstantiateOptions instantiate_opts;
  instantiate_opts.target = "/job:a/replica:0/task:0/cpu:0";
  const auto x = test::AsTensor<int64>({17});
  Tensor y;
  TF_CHECK_OK(
      Run("SessionMetadataReaderFn", opts, {}, instantiate_opts, {x}, {&y}));
  SessionMetadata read_metadata;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(y.scalar<tstring>()(),
                                                    &read_metadata));
  EXPECT_EQ(session_metadata.name(), read_metadata.name());
  EXPECT_EQ(session_metadata.version(), read_metadata.version());
}

TEST_F(ProcessFunctionLibraryRuntimeTest, CompositeDevicesAfterCloning) {
  Init({AddVarAcrossDevices()});

  Status s;
  std::unique_ptr<CompositeDevice> composite_device =
      CompositeDevice::MakeDevice({device0_->name(), device1_->name()},
                                  /*unique_device_id=*/0,
                                  device_mgr_->HostCPU()->parsed_name(), &s);
  TF_ASSERT_OK(s);
  AddCompositeDevice(composite_device.get());

  auto* flr = proc_flr_->GetFLR("/job:a/replica:0/task:0/cpu:0");
  ASSERT_NE(nullptr, flr);
  std::unique_ptr<FunctionLibraryDefinition> cloned_lib_def;
  std::unique_ptr<ProcessFunctionLibraryRuntime> cloned_proc_flr;
  FunctionLibraryRuntime* cloned_flr;
  TF_ASSERT_OK(flr->Clone(&cloned_lib_def, &cloned_proc_flr, &cloned_flr));
  EXPECT_EQ(
      cloned_proc_flr->device_set()->FindDeviceByName(composite_device->name()),
      composite_device.get());
}

TEST_F(ProcessFunctionLibraryRuntimeTest, SessionMetadataPresentAfterCloning) {
  const SessionMetadata session_metadata = GenerateSessionMetadata();
  Init({SessionMetadataReaderOpFn()}, &session_metadata);
  auto* flr = proc_flr_->GetFLR("/job:a/replica:0/task:0/cpu:0");
  ASSERT_NE(nullptr, flr);
  std::unique_ptr<FunctionLibraryDefinition> cloned_lib_def;
  std::unique_ptr<ProcessFunctionLibraryRuntime> cloned_proc_flr;
  FunctionLibraryRuntime* cloned_flr;
  TF_ASSERT_OK(flr->Clone(&cloned_lib_def, &cloned_proc_flr, &cloned_flr));
  FunctionLibraryRuntime::Options opts;
  opts.source_device = "/job:a/replica:0/task:0/cpu:0";
  opts.remote_execution = true;
  FunctionLibraryRuntime::InstantiateOptions instantiate_opts;
  instantiate_opts.target = "/job:a/replica:0/task:0/cpu:0";
  const auto x = test::AsTensor<int64>({17});
  Tensor y;
  Status s = RunWithRuntime<std::vector<Tensor>, Tensor>(
      "SessionMetadataReaderFn", opts, {}, instantiate_opts, {x}, {&y},
      cloned_proc_flr.get());
  TF_CHECK_OK(s);
  SessionMetadata read_metadata;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(y.scalar<tstring>()(),
                                                    &read_metadata));
  EXPECT_EQ(session_metadata.name(), read_metadata.name());
  EXPECT_EQ(session_metadata.version(), read_metadata.version());
}

}  // anonymous namespace
}  // namespace tensorflow
