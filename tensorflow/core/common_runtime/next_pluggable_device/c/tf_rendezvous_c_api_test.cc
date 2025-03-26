/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_rendezvous_c_api.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_rendezvous_c_api_helper.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_rendezvous_c_api_internal.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/status.h"

namespace tensorflow {

namespace {

Tensor CreateTestTensor() {
  Tensor t(DT_INT8, TensorShape({10, 20}));
  for (int64_t a = 0; a < t.shape().dim_size(0); a++) {
    for (int64_t b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<int8>()(a, b) = static_cast<int8>((a + 1) * (b + 1));
    }
  }
  return t;
}

class FakeAllocator : public Allocator {
 public:
  std::string Name() override { return "fake"; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    return port::AlignedMalloc(num_bytes, alignment);
  }
  void DeallocateRaw(void* ptr) override { return port::AlignedFree(ptr); }
};

class FakeDevice : public Device {
 public:
  explicit FakeDevice(const DeviceAttributes& attr) : Device(nullptr, attr) {}
  absl::Status Sync() override { return absl::OkStatus(); }
  Allocator* GetAllocator(AllocatorAttributes) override {
    return allocator_.get();
  }

  static std::unique_ptr<Device> Make(absl::string_view name,
                                      absl::string_view type) {
    DeviceAttributes device_attributes;
    device_attributes.set_name(std::string(name));
    device_attributes.set_device_type(std::string(type));
    return std::unique_ptr<Device>(new FakeDevice(device_attributes));
  }

 private:
  std::unique_ptr<FakeAllocator> allocator_ = std::make_unique<FakeAllocator>();
};

class FakeDeviceManager : public DeviceMgr {
 public:
  void ListDeviceAttributes(
      std::vector<DeviceAttributes>* devices) const override {
    devices->clear();
  }
  std::vector<Device*> ListDevices() const override {
    return std::vector<Device*>();
  }
  std::string DebugString() const override { return ""; }
  std::string DeviceMappingString() const override { return ""; }
  absl::Status LookupDevice(absl::string_view name,
                            Device** device) const override {
    *device = fake_device_.get();
    return absl::OkStatus();
  }
  bool ContainsDevice(int64_t device_incarnation) const override {
    return false;
  }
  void ClearContainers(absl::Span<const string> containers) const override {}
  int NumDeviceType(const string& type) const override { return 0; }
  int NumDevices() const override { return 0; }
  Device* HostCPU() const override { return nullptr; }

 private:
  std::unique_ptr<Device> fake_device_ = FakeDevice::Make("/cpu:0", "fake");
};

class TestDeviceContext : public DeviceContext {
 public:
  void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device,
                             Tensor* device_tensor, StatusCallback done,
                             bool sync_dst_compute) const override {
    Tensor test_tensor = CreateTestTensor();
    test::ExpectTensorEqual<int8>(test_tensor, *cpu_tensor);
    done(absl::OkStatus());
  }

  void CopyDeviceTensorToCPU(const Tensor* device_tensor,
                             absl::string_view tensor_name, Device* device,
                             Tensor* cpu_tensor, StatusCallback done) override {
    *cpu_tensor = CreateTestTensor();
    done(absl::OkStatus());
  }

  void CopyTensorInSameDevice(const Tensor* input_tensor, Device* device,
                              Tensor* output_tensor,
                              tsl::StatusCallback done) const override {
    done(absl::InternalError("TPU->TPU copy not implemented."));
  }
};

std::string CreateRendezvousKey(bool to_host) {
  const std::string task_prefix = "/job:worker/replica:0/task:0";
  const std::string src_device = to_host ? "/device:TPU:0" : "/device:CPU:0";
  const std::string dst_device = to_host ? "/device:CPU:0" : "/device:TPU:0";
  const std::string rendezvous_key_base = "rendezvous_key_base";

  return Rendezvous::CreateKey(absl::StrCat(task_prefix, src_device),
                               /*src_incarnation=*/1,
                               absl::StrCat(task_prefix, dst_device),
                               rendezvous_key_base, FrameAndIter(0, 0));
}

TEST(RendezvousCAPI, DeviceToHost) {
  auto device_manager = std::make_unique<FakeDeviceManager>();
  core::RefCountPtr<Rendezvous> rendezvous = core::RefCountPtr<Rendezvous>(
      new IntraProcessRendezvous(device_manager.get()));
  core::RefCountPtr<TestDeviceContext> device_context =
      core::RefCountPtr<TestDeviceContext>(new TestDeviceContext());

  std::string key = CreateRendezvousKey(/*to_host=*/true);
  Rendezvous::ParsedKey parsed_key;
  TF_ASSERT_OK(Rendezvous::ParseKey(key, &parsed_key));
  TF_RendezvousThunk* thunk = ToC(rendezvous.get());
  std::unique_ptr<tensorflow::RendezvousInterface> thunk_rendezvous =
      FromC(thunk);

  Rendezvous::Args send_args;
  send_args.device_context = device_context.get();
  TF_CHECK_OK(thunk_rendezvous->Send(parsed_key, send_args, Tensor(), false));

  Tensor result;
  absl::Notification callback_done;
  Rendezvous::Args recv_args;
  recv_args.device_context = device_context.get();
  recv_args.alloc_attrs.set_on_host(true);
  rendezvous->RecvAsync(parsed_key, recv_args,
                        [&](const absl::Status& status,
                            const RefCountedIntraProcessRendezvous::Args&,
                            const RefCountedIntraProcessRendezvous::Args&,
                            const Tensor& tensor, const bool) {
                          TF_ASSERT_OK(status);
                          result = tensor;
                          callback_done.Notify();
                        });
  callback_done.WaitForNotification();
  Tensor test_tensor = CreateTestTensor();
  test::ExpectTensorEqual<int8>(test_tensor, result);

  Destroy(thunk);
  delete thunk;
}

TEST(RendezvousCAPI, HostToDevice) {
  auto device_manager = std::make_unique<FakeDeviceManager>();
  core::RefCountPtr<Rendezvous> rendezvous = core::RefCountPtr<Rendezvous>(
      new IntraProcessRendezvous(device_manager.get()));
  core::RefCountPtr<TestDeviceContext> device_context =
      core::RefCountPtr<TestDeviceContext>(new TestDeviceContext());

  std::string key = CreateRendezvousKey(/*to_host=*/false);
  Rendezvous::ParsedKey parsed_key;
  TF_ASSERT_OK(Rendezvous::ParseKey(key, &parsed_key));
  TF_RendezvousThunk* thunk = ToC(rendezvous.get());
  std::unique_ptr<tensorflow::RendezvousInterface> thunk_rendezvous =
      FromC(thunk);

  Rendezvous::Args recv_args;
  recv_args.device_context = device_context.get();
  Tensor result;
  absl::Notification callback_done;
  thunk_rendezvous->RecvAsync(parsed_key, recv_args,
                              [&](const absl::Status& status,
                                  const RefCountedIntraProcessRendezvous::Args&,
                                  const RefCountedIntraProcessRendezvous::Args&,
                                  const Tensor& tensor, const bool) {
                                TF_ASSERT_OK(status);
                                result = tensor;
                                callback_done.Notify();
                              });

  Rendezvous::Args send_args;
  send_args.device_context = device_context.get();
  send_args.alloc_attrs.set_on_host(true);
  Tensor test_tensor = CreateTestTensor();
  TF_CHECK_OK(rendezvous->Send(parsed_key, send_args, test_tensor, false));
  callback_done.WaitForNotification();

  Destroy(thunk);
  delete thunk;
}

}  // namespace

}  // namespace tensorflow
