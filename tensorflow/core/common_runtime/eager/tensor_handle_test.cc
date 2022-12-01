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

#include "tensorflow/core/common_runtime/eager/tensor_handle.h"

#include "tensorflow/core/common_runtime/composite_device.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/random.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(TensorHandle_ShapeTest, AsyncShape) {
  Tensor t(DT_UINT16, TensorShape({2, 2}));
  EXPECT_TRUE(t.shape().IsSameSize(TensorShape({2, 2})));
  for (int64_t a = 0; a < t.shape().dim_size(0); a++) {
    for (int64_t b = 0; b < t.shape().dim_size(1); b++) {
      t.matrix<uint16>()(a, b) = uint16(a * b);
    }
  }

  StaticDeviceMgr device_mgr(DeviceFactory::NewDevice(
      "CPU", {}, "/job:localhost/replica:0/task:0/device:CPU:0"));
  auto ctx = new EagerContext(
      SessionOptions(),
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT, false,
      &device_mgr, false, nullptr, nullptr);
  TensorHandle* sync_th =
      TensorHandle::CreateLocalHandle(std::move(t), nullptr, nullptr, ctx);
  TensorHandle* async_th = TensorHandle::CreateEmptyLocalHandle(
      nullptr, nullptr, nullptr, DataType::DT_UINT16, ctx);

  EXPECT_TRUE(async_th->CopyInferenceShape(sync_th).ok());

  TensorShape sync_shape;
  TensorShape async_shape;
  EXPECT_TRUE(sync_th->Shape(&sync_shape).ok());
  EXPECT_TRUE(async_th->Shape(&async_shape).ok());
  EXPECT_EQ(sync_shape, async_shape);

  int num_dims = -1;
  EXPECT_TRUE(async_th->NumDims(&num_dims).ok());
  EXPECT_EQ(num_dims, 2);

  int64_t num_elements = -1;
  EXPECT_TRUE(async_th->NumElements(&num_elements).ok());
  EXPECT_EQ(num_elements, 4);

  sync_th->Unref();
  async_th->Unref();
  ctx->Unref();
}

static Device* CreateDevice(const char* type, const char* name,
                            bool is_local = true) {
  class FakeDevice : public Device {
   public:
    explicit FakeDevice(const DeviceAttributes& attr, bool is_local)
        : Device(nullptr, attr), is_local_(is_local) {}
    Status Sync() override { return OkStatus(); }
    Allocator* GetAllocator(AllocatorAttributes) override { return nullptr; }
    bool IsLocal() const override { return is_local_; }

   private:
    const bool is_local_;
  };
  DeviceAttributes attr;
  attr.set_name(name);
  attr.set_device_type(type);
  int64_t incarnation = random::New64();
  while (incarnation == 0) {
    incarnation = random::New64();
  }
  attr.set_incarnation(incarnation);
  return new FakeDevice(attr, is_local);
}

}  // namespace

class PackedTensorHandleTest : public ::testing::Test {
 public:
  PackedTensorHandleTest() {
    std::vector<std::unique_ptr<Device>> devices;
    for (const char* name : device_names_) {
      devices.emplace_back(CreateDevice("GPU", name));
    }
    devices.emplace_back(CreateDevice("CPU", host_name_));
    device_mgr_ = new StaticDeviceMgr(std::move(devices));

    context_ = new EagerContext(
        SessionOptions(),
        tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
        /* async= */ false, device_mgr_,
        /* device_mgr_owned= */ false, /* rendezvous= */ nullptr,
        /* cluster_flr= */ nullptr);
  }

  ~PackedTensorHandleTest() override {
    delete device_mgr_;
    context_->Unref();
  }

  EagerContext* context() { return context_; }

  std::vector<Device*> ListDevices() const {
    return device_mgr_->ListDevices();
  }

  bool IsReady(TensorHandle* handle) const { return handle->IsReady(); }

 private:
  const std::vector<const char*> device_names_ = {
      "/job:worker/replica:0/task:0/device:GPU:0",
      "/job:worker/replica:0/task:0/device:GPU:1",
      "/job:worker/replica:0/task:1/device:GPU:0",
      "/job:worker/replica:0/task:1/device:GPU:1"};

  const char* host_name_ = "/job:worker/replica:0/task:0/device:CPU:0";

  StaticDeviceMgr* device_mgr_;
  EagerContext* context_;
};

TEST_F(PackedTensorHandleTest, PackedHandle) {
  tensorflow::DataType dtype = DT_RESOURCE;
  TensorShape shape = {};
  DtypeAndPartialTensorShape dtype_and_shape = {DT_FLOAT, {2, 2}};

  // Create 2 local TensorHandles (ready)
  std::vector<TensorHandle*> handles;
  Tensor t0(dtype, shape);
  Device* d0 = ListDevices().at(0);
  TensorHandle* h0 =
      TensorHandle::CreateLocalHandle(std::move(t0), d0, d0, d0, context());
  h0->SetResourceHandleDtypeAndShape({dtype_and_shape});
  handles.push_back(h0);
  Tensor t1(dtype, shape);
  Device* d1 = ListDevices().at(1);
  TensorHandle* h1 =
      TensorHandle::CreateLocalHandle(std::move(t1), d1, d1, d1, context());
  h1->SetResourceHandleDtypeAndShape({dtype_and_shape});
  handles.push_back(h1);

  // Create 2 remote TensorHandles (not ready).
  const string remote_task = "/job:worker/replica:0/task:1";
  Device* d2 = ListDevices().at(2);
  TensorHandle* h2 = TensorHandle::CreateUnshapedRemoteHandle(
      /*op_id=*/0, /*output_num=*/0, remote_task, dtype, d2, context());
  handles.push_back(h2);
  Device* d3 = ListDevices().at(3);
  TensorHandle* h3 = TensorHandle::CreateUnshapedRemoteHandle(
      /*op_id=*/1, /*output_num=*/0, remote_task, dtype, d3, context());
  handles.push_back(h3);

  TensorHandle* packed_handle = nullptr;
  TF_EXPECT_OK(TensorHandle::CreatePackedHandle(std::move(handles), context(),
                                                &packed_handle));

  h0->Unref();
  h1->Unref();
  h2->Unref();
  h3->Unref();

  EXPECT_EQ(packed_handle->NumPackedHandles(), 4);
  EXPECT_EQ(packed_handle->Type(), TensorHandle::PACKED);
  EXPECT_EQ(packed_handle->dtype, dtype);
  TensorShape packed_shape;
  TF_ASSERT_OK(packed_handle->Shape(&packed_shape));
  EXPECT_EQ(packed_shape, shape);
  std::vector<DtypeAndPartialTensorShape> dtypes_and_shapes;
  TF_ASSERT_OK(
      packed_handle->GetResourceHandleDtypesAndShapes(&dtypes_and_shapes));
  EXPECT_EQ(dtypes_and_shapes.size(), 1);
  EXPECT_EQ(dtypes_and_shapes.at(0).dtype, DT_FLOAT);
  EXPECT_EQ(dtypes_and_shapes.at(0).shape.IsIdenticalTo({2, 2}), true);

  CompositeDevice* device =
      reinterpret_cast<CompositeDevice*>(packed_handle->device());
  EXPECT_EQ(device->name(), "/job:worker/replica:0/task:0/device:COMPOSITE:0");
  EXPECT_EQ(device->underlying_devices()->size(), 4);

  const std::vector<TensorHandle::HandleType> expected_handle_types = {
      TensorHandle::LOCAL, TensorHandle::LOCAL, TensorHandle::REMOTE,
      TensorHandle::REMOTE};
  for (int i = 0; i < packed_handle->NumPackedHandles(); ++i) {
    TensorHandle* h = nullptr;
    TF_ASSERT_OK(packed_handle->ExtractPackedHandle(i, &h));
    EXPECT_EQ(h->device(), ListDevices().at(i));
    EXPECT_EQ(h->Type(), expected_handle_types.at(i));
  }
  EXPECT_FALSE(IsReady(packed_handle));

  TF_ASSERT_OK(h2->SetRemoteShape(shape, ListDevices().at(2),
                                  context()->GetContextViewId()));
  EXPECT_FALSE(IsReady(packed_handle));
  TF_ASSERT_OK(h3->SetRemoteShape(shape, ListDevices().at(3),
                                  context()->GetContextViewId()));
  EXPECT_TRUE(IsReady(packed_handle));

  packed_handle->Unref();
}

TEST_F(PackedTensorHandleTest, PackedSingleHandle) {
  tensorflow::DataType dtype = DT_RESOURCE;
  TensorShape shape = {};

  Tensor t(dtype, shape);
  Device* d = ListDevices().at(0);
  TensorHandle* h =
      TensorHandle::CreateLocalHandle(std::move(t), d, d, d, context());
  std::vector<TensorHandle*> handles = {h};

  TensorHandle* packed_handle = nullptr;
  TF_EXPECT_OK(TensorHandle::CreatePackedHandle(std::move(handles), context(),
                                                &packed_handle));
  h->Unref();

  EXPECT_EQ(packed_handle->Type(), TensorHandle::PACKED);
  EXPECT_EQ(packed_handle->dtype, dtype);
  TensorShape packed_shape;
  TF_ASSERT_OK(packed_handle->Shape(&packed_shape));
  EXPECT_EQ(packed_shape, shape);

  CompositeDevice* device =
      reinterpret_cast<CompositeDevice*>(packed_handle->device());
  EXPECT_EQ(device->name(), "/job:worker/replica:0/task:0/device:COMPOSITE:0");
  EXPECT_EQ(device->underlying_devices()->size(), 1);
  EXPECT_EQ(packed_handle->NumPackedHandles(), 1);
  TensorHandle* h0 = nullptr;
  TF_ASSERT_OK(packed_handle->ExtractPackedHandle(0, &h0));
  EXPECT_EQ(h0->device(), d);
  EXPECT_TRUE(IsReady(packed_handle));
  packed_handle->Unref();
}

TEST(TensorHandle_ResourceDeviceTest, OnLocalDevice) {
  std::unique_ptr<Device> d0(
      CreateDevice("CPU", "/job:localhost/replica:0/task:0/device:CPU:0"));
  StaticDeviceMgr local_device_mgr(std::move(d0));
  auto ctx = new EagerContext(
      SessionOptions(),
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT, false,
      &local_device_mgr, false, nullptr, nullptr);

  tensorflow::DataType dtype = DT_RESOURCE;
  TensorShape shape = {2};
  Tensor t(dtype, shape);

  Device* d = local_device_mgr.ListDevices()[0];
  TensorHandle* th =
      TensorHandle::CreateLocalHandle(std::move(t), d, d, d, ctx);
  // Remote device incarnation for local resource should be 0 (invalid)
  EXPECT_EQ(0, th->resource_remote_device_incarnation());
  // Local device manager must contain the resource device.
  EXPECT_TRUE(local_device_mgr.ContainsDevice(
      th->resource_device()->attributes().incarnation()));

  std::unique_ptr<Device> d1(
      CreateDevice("CPU", "/job:localhost/replica:0/task:0/device:CPU:0"));
  StaticDeviceMgr new_device_mgr(std::move(d1));
  EXPECT_FALSE(new_device_mgr.ContainsDevice(
      th->resource_device()->attributes().incarnation()));

  th->Unref();
  ctx->Unref();
}

TEST(TensorHandle_ResourceDeviceTest, OnRemoteDevice) {
  std::unique_ptr<Device> d_local(
      CreateDevice("CPU", "/job:localhost/replica:0/task:0/device:CPU:0"));
  StaticDeviceMgr local_device_mgr(std::move(d_local));
  auto ctx = new EagerContext(
      SessionOptions(),
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT, false,
      &local_device_mgr, false, nullptr, nullptr);

  std::unique_ptr<Device> d0(
      CreateDevice("CPU", "/job:worker/task:0/device:CPU:0", false));
  Device* d0_ptr = d0.get();
  std::unique_ptr<Device> d1(
      CreateDevice("CPU", "/job:worker/task:1/device:CPU:0", false));
  Device* d1_ptr = d1.get();

  DynamicDeviceMgr remote_device_mgr;
  std::vector<std::unique_ptr<Device>> vector_d0;
  vector_d0.emplace_back(std::move(d0));
  TF_ASSERT_OK(remote_device_mgr.AddDevices(std::move(vector_d0)));

  TensorHandle* th0 = TensorHandle::CreateUnshapedRemoteHandle(
      0, 0, "", DT_RESOURCE, d0_ptr, ctx);
  EXPECT_TRUE(remote_device_mgr.ContainsDevice(
      th0->resource_remote_device_incarnation()));

  std::vector<std::unique_ptr<Device>> vector_d1;
  vector_d1.emplace_back(std::move(d1));
  TF_ASSERT_OK(remote_device_mgr.AddDevices(std::move(vector_d1)));
  EXPECT_TRUE(remote_device_mgr.ContainsDevice(
      th0->resource_remote_device_incarnation()));

  TensorHandle* th1 = TensorHandle::CreateUnshapedRemoteHandle(
      0, 0, "", DT_RESOURCE, d1_ptr, ctx);
  EXPECT_TRUE(remote_device_mgr.ContainsDevice(
      th1->resource_remote_device_incarnation()));

  std::vector<Device*> remove_d1{d1_ptr};
  TF_ASSERT_OK(remote_device_mgr.RemoveDevices(std::move(remove_d1)));
  EXPECT_FALSE(remote_device_mgr.ContainsDevice(
      th1->resource_remote_device_incarnation()));
  EXPECT_TRUE(remote_device_mgr.ContainsDevice(
      th0->resource_remote_device_incarnation()));

  th0->Unref();
  th1->Unref();
  ctx->Unref();
}

class RemoteTensorHandleTest : public ::testing::Test {
 public:
  RemoteTensorHandleTest() {
    std::vector<std::unique_ptr<Device>> devices;
    for (const char* name : device_names_) {
      devices.emplace_back(CreateDevice("CPU", name));
    }
    device_mgr_ = new StaticDeviceMgr(std::move(devices));

    context_ = new EagerContext(
        SessionOptions(),
        tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
        /* async= */ false, device_mgr_,
        /* device_mgr_owned= */ false, /* rendezvous= */ nullptr,
        /* cluster_flr= */ nullptr);
  }

  ~RemoteTensorHandleTest() override {
    delete device_mgr_;
    context_->Unref();
  }

  EagerContext* context() { return context_; }

  std::vector<Device*> ListDevices() const {
    return device_mgr_->ListDevices();
  }

 private:
  const std::vector<const char*> device_names_ = {
      "/job:worker/replica:0/task:0/device:CPU:0",
      "/job:worker/replica:0/task:1/device:CPU:0",
      "/job:worker/replica:0/task:2/device:CPU:0"};

  StaticDeviceMgr* device_mgr_;
  EagerContext* context_;
};

TEST_F(RemoteTensorHandleTest, UnknownRemoteDevice) {
  std::vector<std::unique_ptr<Device>> devices;
  devices.emplace_back(
      CreateDevice("CPU", "/job:worker/replica:0/task:0/device:CPU:0"));
  devices.emplace_back(
      CreateDevice("CPU", "/job:worker/replica:0/task:1/device:CPU:0"));
  devices.emplace_back(
      CreateDevice("CPU", "/job:worker/replica:0/task:2/device:CPU:0"));
  StaticDeviceMgr device_mgr(std::move(devices));

  EagerContext* context = new EagerContext(
      SessionOptions(),
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT,
      /* async= */ false, &device_mgr,
      /* device_mgr_owned= */ false, /* rendezvous= */ nullptr,
      /* cluster_flr= */ nullptr);

  tensorflow::DataType dtype = DT_FLOAT;
  TensorShape shape = {};

  const string remote_task = "/job:worker/replica:0/task:1";
  Device* d1 = device_mgr.ListDevices().at(1);
  TensorHandle* h = TensorHandle::CreateUnshapedRemoteHandle(
      /*op_id=*/0, /*output_num=*/0, remote_task, dtype, d1, context,
      /*unknown_device=*/true);
  EXPECT_EQ(h->device(), d1);

  Device* d2 = device_mgr.ListDevices().at(2);
  TF_ASSERT_OK(h->SetRemoteShapeAndDevice(
      shape, d1, context->GetContextViewId(), d2->name()));
  Status s;
  EXPECT_EQ(h->BackingDeviceName(&s), d2->name());
  TF_EXPECT_OK(s);
  EXPECT_EQ(h->device(), d2);
  h->Unref();
  context->Unref();
}

TEST(TensorHandle_DeviceNameTest, OnLocalDevice) {
  std::vector<std::unique_ptr<Device>> devices;
  devices.emplace_back(
      CreateDevice("CPU", "/job:localhost/replica:0/task:0/device:CPU:0"));
  devices.emplace_back(
      CreateDevice("GPU", "/job:localhost/replica:0/task:0/device:GPU:0"));
  StaticDeviceMgr local_device_mgr(std::move(devices));
  auto ctx = new EagerContext(
      SessionOptions(),
      tensorflow::ContextDevicePlacementPolicy::DEVICE_PLACEMENT_SILENT, false,
      &local_device_mgr, false, nullptr, nullptr);

  Device* dcpu = local_device_mgr.ListDevices()[0];
  Device* dgpu = local_device_mgr.ListDevices()[1];
  tensorflow::DataType dtype = DT_RESOURCE;
  TensorShape shape = {2};
  Tensor tcpu(dtype, shape);
  Tensor tgpu(dtype, shape);
  Status s;

  TensorHandle* th_cpu =
      TensorHandle::CreateLocalHandle(std::move(tcpu), dcpu, dcpu, dcpu, ctx);
  const char* device_name = th_cpu->DeviceName(&s);
  TF_EXPECT_OK(s);
  ASSERT_TRUE(absl::StrContains(device_name, "CPU")) << device_name;
  const char* backing_device_name = th_cpu->BackingDeviceName(&s);
  TF_EXPECT_OK(s);
  ASSERT_TRUE(absl::StrContains(backing_device_name, "CPU"))
      << backing_device_name;
  const char* device_type = th_cpu->DeviceType(&s);
  TF_EXPECT_OK(s);
  ASSERT_TRUE(absl::StrContains(device_type, "CPU")) << device_type;
  int device_id = th_cpu->DeviceId(&s);
  TF_EXPECT_OK(s);
  ASSERT_EQ(0, device_id) << device_id;

  TensorHandle* th_gpu =
      TensorHandle::CreateLocalHandle(std::move(tgpu), dgpu, dgpu, dgpu, ctx);
  device_name = th_gpu->DeviceName(&s);
  TF_EXPECT_OK(s);
  ASSERT_TRUE(absl::StrContains(device_name, "GPU")) << device_name;
  backing_device_name = th_gpu->BackingDeviceName(&s);
  TF_EXPECT_OK(s);
  std::cout << "backing_device_name for GPU: " << backing_device_name
            << std::endl;
  ASSERT_TRUE(absl::StrContains(backing_device_name, "GPU"))
      << backing_device_name;
  device_type = th_gpu->DeviceType(&s);
  TF_EXPECT_OK(s);
  ASSERT_TRUE(absl::StrContains(device_type, "GPU")) << device_type;
  device_id = th_gpu->DeviceId(&s);
  TF_EXPECT_OK(s);
  ASSERT_EQ(0, device_id) << device_id;

  th_cpu->Unref();
  th_gpu->Unref();
  ctx->Unref();
}

}  // namespace tensorflow
