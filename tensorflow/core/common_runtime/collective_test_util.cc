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
#include "tensorflow/core/common_runtime/collective_test_util.h"

#include <vector>

#include "absl/synchronization/notification.h"
#include "tensorflow/core/common_runtime/base_collective_executor.h"
#include "tensorflow/core/common_runtime/device_resolver_local.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

FailTestRMA::FailTestRMA(const DeviceMgr* dev_mgr,
                         DeviceResolverInterface* dev_resolver, int64_t step_id)
    : CollectiveRemoteAccessLocal(dev_mgr, dev_resolver, step_id),
      fail_after_(0) {}

bool FailTestRMA::MaybeFail(const StatusCallback& done) {
  bool fail_now = false;
  {
    mutex_lock l(mu_);
    if (fail_after_ > 0) {
      fail_now = (--fail_after_ == 0);
    }
  }
  if (fail_now) {
    auto error = errors::Internal("Deliberate failure");
    LOG(INFO) << "triggering failure " << error;
    buf_rendezvous()->StartAbort(error);
    // The current call hasn't reached BufRendezvous yet, so we need to call
    // its done separately.
    done(error);
    return true;
  }
  return false;
}

void FailTestRMA::RecvFromPeer(
    const string& peer_device, const string& peer_task, bool peer_is_local,
    const string& key, Device* to_device, DeviceContext* to_device_ctx,
    const AllocatorAttributes& to_alloc_attr, Tensor* to_tensor,
    const DeviceLocality& client_locality, int dev_to_dev_stream_index,
    CancellationManager* cancellation_manager, const StatusCallback& done) {
  if (MaybeFail(done)) return;
  CollectiveRemoteAccessLocal::RecvFromPeer(
      peer_device, peer_task, peer_is_local, key, to_device, to_device_ctx,
      to_alloc_attr, to_tensor, client_locality, dev_to_dev_stream_index,
      cancellation_manager, done);
}

void FailTestRMA::PostToPeer(const string& peer_device, const string& peer_task,
                             const string& key, Device* from_device,
                             DeviceContext* from_device_ctx,
                             const AllocatorAttributes& from_alloc_attr,
                             const Tensor* from_tensor,
                             const DeviceLocality& client_locality,
                             CancellationManager* cancellation_manager,
                             const StatusCallback& done) {
  if (MaybeFail(done)) return;
  CollectiveRemoteAccessLocal::PostToPeer(
      peer_device, peer_task, key, from_device, from_device_ctx,
      from_alloc_attr, from_tensor, client_locality, cancellation_manager,
      done);
}

namespace {

constexpr int kStepId = 0;

std::vector<std::unique_ptr<Device>> CreateCPUDevices(
    int num_workers, int num_devices_per_worker) {
  SessionOptions sess_opts;
  sess_opts.env = Env::Default();
  Bytes mem_limit(4 << 20);
  DeviceLocality dev_locality;
  std::vector<std::unique_ptr<Device>> devices;
  for (int wi = 0; wi < num_workers; ++wi) {
    for (int di = 0; di < num_devices_per_worker; ++di) {
      string dev_name = strings::StrCat("/job:worker/replica:0/task:", wi,
                                        "/device:CPU:", di);
      devices.push_back(absl::make_unique<ThreadPoolDevice>(
          sess_opts, dev_name, mem_limit, dev_locality, cpu_allocator()));
    }
  }
  return devices;
}

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
std::vector<std::unique_ptr<Device>> CreateGPUDevices() {
  // It's required to use the same virtual device configuration in one process,
  // so we configure kNumVirtualDevices which should be the maximum used in
  // tests.
  static constexpr int kNumVirtualDevices = 8;
  auto device_factory = DeviceFactory::GetFactory("GPU");
  CHECK(device_factory);
  SessionOptions options;
  std::vector<string> physical_devices;
  TF_CHECK_OK(device_factory->ListPhysicalDevices(&physical_devices));
  if (physical_devices.size() < kNumVirtualDevices) {
    int num_virtual_per_phsyical = static_cast<int>(std::ceil(
        static_cast<double>(kNumVirtualDevices) / physical_devices.size()));
    auto* virtual_devices = options.config.mutable_gpu_options()
                                ->mutable_experimental()
                                ->mutable_virtual_devices();
    for (int i = 0; i < physical_devices.size(); ++i) {
      auto* virtual_device = virtual_devices->Add();
      for (int j = 0; j < num_virtual_per_phsyical; ++j) {
        virtual_device->add_memory_limit_mb(1024);  // in MiB.
        virtual_device->add_priority(0);
      }
    }
  }
  std::vector<std::unique_ptr<Device>> devices;
  Status s = device_factory->CreateDevices(
      options, "/job:worker/replica:0/task:0", &devices);
  CHECK(s.ok());
  return devices;
}
#endif
}  // namespace

std::unique_ptr<CollectiveTestEnv> CreateCollectiveTestEnv(
    int num_workers, int num_devices_per_worker, DeviceType device_type) {
  auto test_env = absl::make_unique<CollectiveTestEnv>();
  test_env->param_resolver = absl::make_unique<TestParamResolver>();
  // We don't create CollecticeExecutor from the CollecticeExecutorMgr so we
  // don't need to pass rma.
  test_env->col_exec_mgr = absl::make_unique<TestCollectiveExecutorMgr>(
      test_env->param_resolver.get(), /*rma=*/nullptr);
  test_env->num_workers = num_workers;
  test_env->num_devices_per_worker = num_devices_per_worker;
  test_env->device_type = device_type;

  std::vector<std::unique_ptr<Device>> devices;
  if (device_type == DEVICE_CPU) {
    devices = CreateCPUDevices(num_workers, num_devices_per_worker);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  } else if (device_type == DEVICE_GPU) {
    CHECK(num_workers == 1) << "GPU only supports single worker tests";
    devices = CreateGPUDevices();
    if (devices.size() < num_devices_per_worker) {
      LOG(FATAL) << "The test is requesting more GPUs than available";
    }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  } else {
    LOG(FATAL) << "Unsupported device_type " << device_type;
  }
  test_env->device_mgr = absl::make_unique<StaticDeviceMgr>(std::move(devices));

  test_env->device_resolver =
      absl::make_unique<DeviceResolverLocal>(test_env->device_mgr.get());
  test_env->work_queue =
      std::make_shared<UnboundedWorkQueue>(Env::Default(), "test");
  // BaseCollectiveExecutor takes the ownership of remote_access.
  test_env->remote_access = new FailTestRMA(
      test_env->device_mgr.get(), test_env->device_resolver.get(), kStepId);
  test_env->col_exec.reset(new BaseCollectiveExecutor(
      test_env->col_exec_mgr.get(), test_env->remote_access, kStepId,
      test_env->device_mgr.get(), test_env->work_queue));

  return test_env;
}

core::RefCountPtr<CollectiveParams> CreateCollectiveParams(
    const CollectiveTestEnv& test_env, int rank, const string& collective_name,
    CollectiveType collective_type, DataType dtype, const TensorShape& shape,
    const std::vector<std::vector<int>> user_specified_rank) {
  static constexpr int kGroupKey = 5;
  static constexpr int kInstanceKey = 17;
  core::RefCountPtr<CollectiveParams> col_params(new CollectiveParams());
  col_params->name = "test_collective";
  col_params->default_rank = rank;

  // Set up a local device ring order that's not just 0,1,2...
  std::vector<int> local_ring_order;
  local_ring_order.reserve(test_env.num_devices_per_worker);
  for (int di = 0; di < test_env.num_devices_per_worker; ++di) {
    local_ring_order.push_back(di);
  }
  for (int di = 0; di < test_env.num_devices_per_worker; ++di) {
    bool is_odd = ((di % 2) == 1);
    int other = (di + (is_odd ? 7 : 3)) % test_env.num_devices_per_worker;
    if (di == other) continue;
    std::iter_swap(local_ring_order.begin() + di,
                   local_ring_order.begin() + other);
  }
  string lro_buf;
  for (auto d : local_ring_order) strings::StrAppend(&lro_buf, d, ", ");
  VLOG(1) << "local_ring_order " << lro_buf;

  // Set up group parameters.
  col_params->group.group_key = kGroupKey;
  col_params->group.group_size =
      test_env.num_workers * test_env.num_devices_per_worker;
  col_params->group.num_tasks = test_env.num_workers;
  col_params->group.device_type = test_env.device_type;
  for (int wi = 0; wi < test_env.num_workers; ++wi) {
    string task_name = strings::StrCat("/job:worker/replica:0/task:", wi);
    col_params->group.num_devices_per_task[task_name] =
        test_env.num_devices_per_worker;
    for (int di = 0; di < test_env.num_devices_per_worker; ++di) {
      CollGroupMember member;
      member.device.set_name(strings::StrCat(
          task_name, "/device:", test_env.device_type.type_string(), ":", di));
      member.task = task_name;
      // Normally each device would set is_local to its own perspective but
      // this test runs in a single process so is_local is always true.
      member.is_local = true;
      if (user_specified_rank.size() == test_env.num_workers &&
          user_specified_rank[wi].size() == test_env.num_devices_per_worker) {
        member.rank = user_specified_rank[wi][di];
      } else {
        member.rank = wi * test_env.num_workers + di;
      }

      col_params->group.members.push_back(member);
    }
  }

  // Set up instance parameters.
  col_params->instance.instance_key = kInstanceKey;
  col_params->instance.type = collective_type;
  col_params->instance.impl_details.collective_name = collective_name;
  col_params->instance.data_type = dtype;
  col_params->instance.shape = shape;
  col_params->instance.impl_details.subdiv_offsets.push_back(0);

  return col_params;
}

std::vector<int> GenerateEvenSubdivOffsets(int num_devices_per_worker,
                                           int num_subdivs) {
  std::vector<int> offsets;
  offsets.reserve(num_subdivs);
  int subdiv_stride = num_devices_per_worker / num_subdivs;
  for (int sdi = 0; sdi < num_subdivs; ++sdi) {
    offsets.push_back(sdi * subdiv_stride);
  }
  return offsets;
}

Tensor CopyTensorToDevice(Device* device, const Tensor& tensor) {
  if (device->device_type() == DEVICE_CPU) {
    return tensor;
  } else if (device->device_type() == DEVICE_GPU) {
    Tensor copied(device->GetAllocator(AllocatorAttributes()), tensor.dtype(),
                  tensor.shape());
    auto* dev_info = device->tensorflow_gpu_device_info();
    CHECK(dev_info);
    TF_CHECK_OK(dev_info->default_context->CopyCPUTensorToDeviceSync(
        &tensor, device, &copied));
    return copied;
  }
  LOG(FATAL) << "Unsupported device_type " << device->device_type();
}

Tensor CopyTensorToHost(Device* device, const Tensor& tensor) {
  if (device->device_type() == DEVICE_CPU) {
    return tensor;
  } else if (device->device_type() == DEVICE_GPU) {
    Tensor copied(tensor.dtype(), tensor.shape());
    auto* dev_info = device->tensorflow_gpu_device_info();
    CHECK(dev_info);
    TF_CHECK_OK(dev_info->default_context->CopyDeviceTensorToCPUSync(
        &tensor, "" /*tensor_name*/, device, &copied));
    return copied;
  }
  LOG(FATAL) << "Unsupported device_type " << device->device_type();
}

Status RunCollective(CollectiveTestEnv* test_env, CollectiveParams* col_params,
                     Device* device, Tensor* input, Tensor* output) {
  // Copy input and allocate output if on GPU.
  Tensor input_buffer;
  Tensor output_buffer;
  if (device->device_type() == DEVICE_CPU) {
    input_buffer = *input;
    output_buffer = *output;
  } else if (device->device_type() == DEVICE_GPU) {
    input_buffer = CopyTensorToDevice(device, *input);
    if (input == output) {
      // If the input is forwarded to the output, we keep the forwarding so that
      // we can test if the collective can run in-place.
      output_buffer = input_buffer;
    } else {
      output_buffer = Tensor(device->GetAllocator(AllocatorAttributes()),
                             output->dtype(), output->shape());
    }
  } else {
    LOG(FATAL) << "Unsupported device_type " << device->device_type();
  }
  // Requires the user the allocate output since we cannot infer the output
  // shape.
  CHECK(output->NumElements()) << "output must be allocated";

  // Prepare an OpKernelContext.
  OpKernelContext::Params op_params;
  CancellationManager cancellation_manager;
  op_params.step_id = kStepId;
  op_params.device = device;
  op_params.cancellation_manager = &cancellation_manager;
  gtl::InlinedVector<TensorValue, 4> inputs;
  inputs.push_back(TensorValue(&input_buffer));
  op_params.inputs = &inputs;
  gtl::InlinedVector<AllocatorAttributes, 4> input_aa({AllocatorAttributes()});
  op_params.input_alloc_attrs = &input_aa;
  DeviceContext* dev_ctx = nullptr;
  auto* dev_info = device->tensorflow_gpu_device_info();
  if (dev_info) {
    dev_ctx = dev_info->default_context;
    dev_ctx->Ref();
  } else {
    dev_ctx = new DeviceContext;
  }
  core::ScopedUnref unref_dev_ctx(dev_ctx);
  op_params.op_device_context = dev_ctx;
  int forward_from = 0;
  op_params.forward_from_array = &forward_from;
  AllocatorAttributes generic_alloc_attr;
  op_params.output_attr_array = &generic_alloc_attr;
  op_params.resource_manager = device->resource_manager();
  OpKernelContext ctx(&op_params, 1);

  // Prepare a collective instance.
  CollectiveImplementationInterface* collective_impl = nullptr;
  TF_CHECK_OK(CollectiveRegistry::Lookup(
      col_params->instance.impl_details.collective_name, &collective_impl));
  core::ScopedUnref unref_collective_impl(collective_impl);
  TF_RETURN_IF_ERROR(collective_impl->InitializeCollectiveParams(col_params));

  string exec_key = strings::StrCat(col_params->instance.instance_key, ":0:0");
  auto col_ctx = std::make_shared<CollectiveContext>(
      test_env->col_exec.get(), /*nccl_communicator*/ nullptr,
      test_env->device_mgr.get(), &ctx, &op_params, col_params, exec_key,
      kStepId, &input_buffer, &output_buffer);
  TF_RETURN_IF_ERROR(collective_impl->InitializeCollectiveContext(col_ctx));

  // Run the collective.
  Status status;
  Notification n;
  collective_impl->Run([&status, &n](Status s) {
    status = s;
    n.Notify();
  });
  n.WaitForNotification();
  if (status.ok()) {
    *output = CopyTensorToHost(device, output_buffer);
  }
  return status;
}

}  // namespace tensorflow
