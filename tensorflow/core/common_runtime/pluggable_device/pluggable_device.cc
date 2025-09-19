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

#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device.h"

#include <stdlib.h>

#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/synchronization/notification.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "tensorflow/core/common_runtime/device/device_event_mgr.h"
#include "tensorflow/core/common_runtime/device/device_id.h"
#include "tensorflow/core/common_runtime/device/device_id_manager.h"
#include "tensorflow/core/common_runtime/device_id_utils.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_context.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_init.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_process_state.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_util.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

// This factory helps ensure that different PluggableDevice objects that
// refer to the same physical device and stream group id use the same stream
// group object (and therefore the same device streams). This is necessary since
// there is a single memory allocator per device (see
// ProcessState::GetPluggableDeviceAllocator) and allocators must not be shared
// across streams.
// TODO(penpornk): Consider refactoring StreamGroupFactory to
// common_runtime/device.
class PluggableDevice::StreamGroupFactory {
 public:
  // Returns the unique stream group for use with the stream defined by
  // {tf_device_id, stream_group_within_device}, creating it if it does not yet
  // exist.
  // This function is thread safe.
  PluggableDevice::StreamGroup* GetOrCreate(const std::string& device_type,
                                            TfDeviceId tf_device_id,
                                            int stream_group_within_device,
                                            se::StreamExecutor* executor,
                                            const GPUOptions& options) {
    mutex_lock guard(lock_);
    StreamGroup* group = &streams_[key_type(device_type, tf_device_id.value(),
                                            stream_group_within_device)];
    if (!group->compute) {
      auto stream_or_status = executor->CreateStream();
      if (!stream_or_status.ok()) {
        LOG(ERROR) << "Failed to create stream for device "
                   << tf_device_id.value()
                   << " with status: " << stream_or_status.status();
        return group;
      }
      group->compute = stream_or_status->get();
      allocated_streams_.emplace_back(std::move(stream_or_status.value()));
      VLOG(2) << "Created stream[" << stream_group_within_device
              << "] = " << group->compute;

      stream_or_status = executor->CreateStream();
      if (!stream_or_status.ok()) {
        LOG(ERROR) << "Failed to create stream for device "
                   << tf_device_id.value()
                   << " with status: " << stream_or_status.status();
        return group;
      }
      group->host_to_device = stream_or_status->get();
      allocated_streams_.emplace_back(std::move(stream_or_status.value()));
      VLOG(2) << "Created host_to_device_stream[" << stream_group_within_device
              << "] = " << group->host_to_device;

      stream_or_status = executor->CreateStream();
      if (!stream_or_status.ok()) {
        LOG(ERROR) << "Failed to create stream for device "
                   << tf_device_id.value()
                   << " with status: " << stream_or_status.status();
        return group;
      }
      group->device_to_host = stream_or_status->get();
      allocated_streams_.emplace_back(std::move(stream_or_status.value()));
      VLOG(2) << "Created device_to_host_stream[" << stream_group_within_device
              << "] = " << group->device_to_host;

      int num_d2d_streams =
          options.experimental().num_dev_to_dev_copy_streams();
      if (num_d2d_streams == 0) num_d2d_streams = 1;
      if (num_d2d_streams < 1 || num_d2d_streams > 4) {
        LOG(ERROR)
            << "Illegal GPUOptions.experimental.num_dev_to_dev_copy_streams="
            << num_d2d_streams << " set to 1 instead.";
        num_d2d_streams = 1;
      }
      for (int i = 0; i < num_d2d_streams; ++i) {
        stream_or_status = executor->CreateStream();
        if (!stream_or_status.ok()) {
          LOG(ERROR) << "Failed to create stream for device "
                     << tf_device_id.value()
                     << " with status: " << stream_or_status.status();
          return group;
        }
        group->device_to_device.push_back(stream_or_status->get());
        allocated_streams_.emplace_back(std::move(stream_or_status.value()));
        VLOG(2) << "Created device_to_device_stream["
                << stream_group_within_device
                << "] = " << group->device_to_device.back();
      }
    }
    return group;
  }

  // Returns a reference to the StreamGroupFactory singleton. Note that this is
  // never destroyed, so the objects it owns are never deleted.
  static StreamGroupFactory& Global() {
    static StreamGroupFactory* instance = new StreamGroupFactory();
    return *instance;
  }

 private:
  mutex lock_;
  using key_type = std::tuple<std::string, int, int>;
  std::map<key_type, StreamGroup> streams_;
  std::vector<std::unique_ptr<se::Stream>> allocated_streams_;
  // StreamGroupFactory cannot be created directly; Call
  // StreamGroupFactory::Global to get the global instance.
  StreamGroupFactory() = default;
  StreamGroupFactory(const StreamGroupFactory&) = delete;
  void operator=(const StreamGroupFactory&) = delete;
};

PluggableDevice::PluggableDevice(
    const SessionOptions& options, const std::string& name,
    const std::string& device_type, const std::string& platform_name,
    Bytes memory_limit, const DeviceLocality& locality, TfDeviceId tf_device_id,
    const std::string& physical_device_desc, Allocator* device_allocator,
    Allocator* cpu_allocator, bool sync_every_op)
    : LocalDevice(options, Device::BuildDeviceAttributes(
                               name, device_type.c_str(), memory_limit,
                               locality, physical_device_desc)),
      device_allocator_(device_allocator),
      cpu_allocator_(cpu_allocator),
      tf_device_id_(tf_device_id),
      platform_name_(platform_name),
      sync_every_op_(sync_every_op) {
  if (options.config.has_pluggable_device_options()) {
    force_gpu_compatible_ =
        options.config.pluggable_device_options().force_gpu_compatible();
  }
  PluggableDeviceProcessState::singleton(device_type, platform_name)
      ->EnablePluggableDevice();
}

PluggableDevice::~PluggableDevice() {
  delete pluggable_device_info_;
  device_context_->Unref();
}

absl::Status PluggableDevice::Init(const SessionOptions& options) {
  se::Platform* platform = PluggableDeviceMachineManager(platform_name_);
  auto executor_status = DeviceIdUtil::ExecutorForTfDeviceId(
      DeviceType(device_type()), platform, tf_device_id_);
  if (!executor_status.status().ok()) {
    return errors::Internal("Failed to get StreamExecutor for device",
                            tf_device_id_.value());
  }
  executor_ = executor_status.value();

  em_ = EventMgrFactory::Singleton()->GetEventMgr(
      executor_, options.config.pluggable_device_options());

  stream_ = StreamGroupFactory::Global().GetOrCreate(
      device_type(), tf_device_id_, 0, executor_,
      options.config.pluggable_device_options());
  device_context_ = new PluggableDeviceContext(
      0, stream_->compute, stream_->host_to_device, stream_->device_to_host,
      stream_->device_to_device);
  pluggable_device_info_ = new DeviceBase::AcceleratorDeviceInfo;
  pluggable_device_info_->stream = stream_->compute;
  pluggable_device_info_->default_context = device_context_;
  pluggable_device_info_->event_mgr = em_;
  PlatformDeviceId platform_device_id;
  TF_RETURN_IF_ERROR(DeviceIdManager::TfToPlatformDeviceId(
      DeviceType(device_type()), tf_device_id_, &platform_device_id));
  pluggable_device_info_->gpu_id = platform_device_id.value();
  set_tensorflow_accelerator_device_info(pluggable_device_info_);

  // Whether and how the PluggableDevice uses its own threadpool.
  // This option is experimental. Once we confirm the best setting, we
  // may change the default behavior and completely remove this flag.
  // Default values might change in future releases.
  // Possible values:
  //   * global: PluggableDevice uses threads shared with CPU in the main
  //       compute thread-pool. This is currently the default.
  //   * gpu_private: PluggableDevice uses threads dedicated to this device.
  //   * gpu_shared: All PluggableDevices share a dedicated thread pool.

  // TODO(penpornk): Read the following configurations from a PluggableDevice
  // callback instead of GPU environment variables: TF_GPU_THREAD_MODE,
  // TF_GPU_THREAD_COUNT, TF_FORCE_GPU_ALLOC_GROWTH,
  // TF_ENABLE_GPU_GARBAGE_COLLECTION, and TF_GPU_HOST_MEM_LIMIT_IN_MB.
  string device_thread_mode;
  TF_RETURN_IF_ERROR(ReadStringFromEnvVar("TF_GPU_THREAD_MODE", "global",
                                          &device_thread_mode));
  device_thread_mode = absl::AsciiStrToLower(device_thread_mode);
  if (device_thread_mode != "global") {
    int64_t device_thread_count = -1;
    // Default to two threads. One for device compute and another for memory
    // copies.
    TF_RETURN_IF_ERROR(
        ReadInt64FromEnvVar("TF_GPU_THREAD_COUNT", 2, &device_thread_count));
    if (device_thread_mode == "gpu_private") {
      thread_pool_ = std::make_unique<thread::ThreadPool>(
          options.env, ThreadOptions(),
          absl::StrCat("gpu_private_", tf_device_id_.value()),
          static_cast<int32>(device_thread_count),
          !options.config.experimental().disable_thread_spinning(),
          /*allocator=*/nullptr);
      set_tensorflow_device_thread_pool(thread_pool_.get());
    } else if (device_thread_mode == "gpu_shared") {
      static thread::ThreadPool* thread_pool = new thread::ThreadPool(
          options.env, ThreadOptions(), "gpu_shared",
          static_cast<int32>(device_thread_count),
          !options.config.experimental().disable_thread_spinning(),
          /*allocator=*/nullptr);
      set_tensorflow_device_thread_pool(thread_pool);
    } else {
      string error_message =
          absl::StrCat("Invalid gpu_thread_mode: ", device_thread_mode);
      LOG(WARNING) << error_message;
      return errors::InvalidArgument(error_message);
    }
  }

  return absl::OkStatus();
}

Allocator* PluggableDevice::GetAllocator(AllocatorAttributes attr) {
  DCHECK(cpu_allocator_) << "CPU allocator must be set";
  if (attr.on_host()) {
    if (attr.gpu_compatible() || force_gpu_compatible_) {
      PluggableDeviceProcessState* ps =
          PluggableDeviceProcessState::singleton(device_type(), platform_name_);
      return ps->GetPluggableDeviceHostAllocator(0);
    } else {
      return cpu_allocator_;
    }
  } else {
    return device_allocator_;
  }
}

string PluggableDevice::ComputeOpKernelDebugString(const OpKernel& op_kernel,
                                                   const int stream_id) {
  return strings::StrCat(op_kernel.name(), " op ", op_kernel.type_string(),
                         " on ", platform_name_, tf_device_id_.value(),
                         " stream[", stream_id, "]");
}

void PluggableDevice::Compute(OpKernel* op_kernel, OpKernelContext* context) {
  PluggableDeviceContext* pluggable_device_context = device_context_;
  if (context->op_device_context() != nullptr) {
    pluggable_device_context =
        static_cast<PluggableDeviceContext*>(context->op_device_context());
  }
  const auto stream_id = pluggable_device_context->stream_id();

  const bool vlog_1 = VLOG_IS_ON(1);

  if (vlog_1) {
    VLOG(1) << "PluggableDevice::ComputeHelper "
            << ComputeOpKernelDebugString(*op_kernel, stream_id);
  }

  op_kernel->Compute(context);
  if (context->status().ok()) {
    if (sync_every_op_) {
      context->SetStatus(PluggableDeviceUtil::Sync(this));
      if (vlog_1) {
        VLOG(1) << "PluggableDevice::ComputeHelper finished"
                << ComputeOpKernelDebugString(*op_kernel, stream_id);
      }
    } else if (vlog_1) {
      VLOG(1) << "PluggableDevice::ComputeHelper scheduled"
              << ComputeOpKernelDebugString(*op_kernel, stream_id);
    }
  } else {
    if (vlog_1) {
      VLOG(1) << "PluggableDevice::ComputeHelper failed to schedule"
              << ComputeOpKernelDebugString(*op_kernel, stream_id);
    }
  }
}

// Based on the semantics of Device::Sync, this call should wait for
// all streams not just the current one.
absl::Status PluggableDevice::Sync() {
  return PluggableDeviceUtil::SyncAll(this);
}

void PluggableDevice::ComputeAsync(AsyncOpKernel* op_kernel,
                                   OpKernelContext* context,
                                   AsyncOpKernel::DoneCallback done) {
  PluggableDeviceContext* device_context = device_context_;
  if (context->op_device_context() != nullptr) {
    device_context =
        static_cast<PluggableDeviceContext*>(context->op_device_context());
  }
  const auto stream_id = device_context->stream_id();

  VLOG(1) << "PluggableDevice::ComputeAsync " << op_kernel->name() << " op "
          << op_kernel->type_string() << " on " << device_type()
          << tf_device_id_ << " stream[" << stream_id << "]";
  op_kernel->ComputeAsync(context, std::move(done));
}

absl::Status PluggableDevice::MaybeCopyTensorToPluggableDevice(
    const AllocatorAttributes& alloc_attrs, const Tensor& from, Tensor* to,
    StatusCallback done) {
  if (alloc_attrs.on_host()) {
    *to = from;
    done(absl::OkStatus());
    return absl::OkStatus();
  } else {
    if (!DMAHelper::CanUseDMA(&from)) {
      absl::Status err =
          errors::Internal("PluggableDevice copy from non-DMA ",
                           DataTypeString(from.dtype()), " tensor");
      done(err);
      return err;
    }
    AllocationAttributes allocation_attr;
    auto* copy = new Tensor(GetAllocator(alloc_attrs), from.dtype(),
                            from.shape(), allocation_attr);

    // If the tensor is not initialized, we likely ran out of memory.
    if (!copy->IsInitialized()) {
      delete copy;
      absl::Status err = errors::ResourceExhausted(
          "OOM when allocating tensor of shape ", from.shape().DebugString(),
          " and type ", DataTypeString(from.dtype()));
      done(err);
      return err;
    }

    auto wrapped_done = [to, copy,
                         done = std::move(done)](const absl::Status& s) {
      if (s.ok()) {
        *to = std::move(*copy);
      }
      delete copy;
      done(s);
    };

    device_context_->CopyCPUTensorToDevice(
        &from, this, copy, std::move(wrapped_done), false /*sync_dst_compute*/);
    return absl::OkStatus();
  }
}

absl::Status PluggableDevice::MakeTensorFromProto(
    const TensorProto& tensor_proto, const AllocatorAttributes alloc_attrs,
    Tensor* tensor) {
  AllocatorAttributes attr;
  attr.set_on_host(true);
  attr.set_gpu_compatible(true);
  Allocator* host_alloc = GetAllocator(attr);
  Tensor parsed(tensor_proto.dtype());
  if (!parsed.FromProto(host_alloc, tensor_proto)) {
    return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                   tensor_proto.DebugString());
  }

  if (parsed.dtype() == DT_VARIANT) {
    const Variant* from = parsed.flat<Variant>().data();
    int numa_node = attributes().locality().numa_node();
    Tensor copy(cpu_allocator(numa_node), DT_VARIANT, parsed.shape());
    Variant* copy_variant = copy.flat<Variant>().data();

    std::list<absl::Notification> notifications;
    absl::Status copy_status;
    auto copier = [this, &alloc_attrs, &notifications, &copy_status](
                      const Tensor& from, Tensor* to) {
      // Copier isn't run in a multithreaded environment, so we don't
      // have to worry about the notifications list being modified in parallel.
      notifications.emplace_back();
      absl::Notification& n = *notifications.rbegin();
      return MaybeCopyTensorToPluggableDevice(
          alloc_attrs, from, to, [&n, &copy_status](const absl::Status& s) {
            if (copy_status.ok()) {
              copy_status.Update(s);
            }
            n.Notify();
          });
    };
    absl::Status s;
    for (int64_t ix = 0; ix < parsed.NumElements(); ++ix) {
      s = VariantDeviceCopy(VariantDeviceCopyDirection::HOST_TO_DEVICE,
                            from[ix], &copy_variant[ix], copier);
      if (!s.ok()) {
        break;
      }
    }
    for (auto& n : notifications) {
      n.WaitForNotification();
    }
    if (!s.ok()) {
      return s;
    }
    *tensor = std::move(copy);
    return copy_status;
  } else {
    absl::Notification n;
    absl::Status status;
    TF_RETURN_IF_ERROR(MaybeCopyTensorToPluggableDevice(
        alloc_attrs, parsed, tensor, [&n, &status](const absl::Status& s) {
          status = s;
          n.Notify();
        }));
    n.WaitForNotification();
    return status;
  }
}

void PluggableDevice::CopyTensorInSameDevice(
    const Tensor* input_tensor, Tensor* output_tensor,
    const DeviceContext* device_context, StatusCallback done) {
  PluggableDeviceUtil::CopyPluggableDeviceTensorToSameDevice(
      static_cast<Device*>(this), device_context, input_tensor, output_tensor,
      std::move(done));
}

}  // namespace tensorflow
