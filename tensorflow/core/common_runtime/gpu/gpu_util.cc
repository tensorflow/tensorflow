#include "tensorflow/core/common_runtime/gpu/gpu_util.h"

//#include "base/commandlineflags.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/common_runtime/gpu/dma_helper.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/common_runtime/gpu/process_state.h"
#include "tensorflow/core/util/util.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor.h"

#include "tensorflow/core/platform/stream_executor_util.h"

#if defined(PLATFORM_GOOGLE)
DEFINE_int64(brain_gpu_util_debug_string_maxlen, 128,
             "When dumping gpu memory, prints up to this many bytes.");

DECLARE_bool(record_mem_types);
#else
tensorflow::int64 FLAGS_brain_gpu_util_debug_string_maxlen = 128;
bool FLAGS_EXPERIMENTAL_brain_gpu_multi_stream = false;
extern bool FLAGS_record_mem_types;
#endif

using perftools::gputools::DeviceMemoryBase;
using perftools::gputools::DeviceMemory;
using perftools::gputools::Stream;

namespace tensorflow {

namespace gpu = ::perftools::gputools;

/*static*/
void GPUUtil::SetProtoFromGPU(const Tensor& tensor, Device* dev,
                              const DeviceContext* device_context,
                              TensorProto* proto, bool is_dead,
                              StatusCallback done) {
  VLOG(1) << "SetProtoFromGPU device_context " << device_context;
  // Tensor values need to be copied from GPU to CPU ram so that
  // we can build the protobuf response for a RecvTensor RPC.
  // "device context" identifies the stream where the _Send op executed.
  CHECK(device_context);
  gpu::Stream* stream =
      static_cast<const GPUDeviceContext*>(device_context)->stream();

  if (!DMAHelper::CanUseDMA(&tensor)) {
    done(errors::Internal(strings::StrCat(
        "GPU copy from non-DMA ", DataTypeString(tensor.dtype()), "tensor")));
    return;
  }
  proto->set_dtype(tensor.dtype());
  tensor.shape().AsProto(proto->mutable_tensor_shape());
  // Prepare a Cord with the right data buf size, and DMA the
  // data over from the GPU buffer.  Note that 0-size tensors
  // do not have a backing buffer.
  const size_t num_bytes = is_dead ? 0 : tensor.TotalBytes();
  if (num_bytes > 0) {
    port::Tracing::ScopedAnnotation annotation("SetProtoFromGPU");
    Allocator* alloc = ProcessState::singleton()->GetCUDAHostAllocator(0);
    char* mb = alloc->Allocate<char>(num_bytes);
    const char* src_ptr =
        reinterpret_cast<const char*>(DMAHelper::base(&tensor));
    DeviceMemoryBase gpu_src_ptr(const_cast<char*>(src_ptr), num_bytes);
    stream->ThenMemcpy(mb, gpu_src_ptr, num_bytes);
    // Use of tensor may outlive stack scope, so keep a ref.
    Tensor* tensor_ref = new Tensor(tensor);
    dev->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
        stream, [stream, done, proto, mb, num_bytes, alloc, tensor_ref]() {
          if (!stream->ok()) {
            done(errors::Internal("SetProtoFromGPU: GPU Memcpy failed"));
            // TODO(pbar) We currently have no way to recover the
            // worker from a GPU stream in the error state.  Until
            // there is a way to reset the CUDA driver, it is
            // preferable to crash the process and restart.  Tracked
            // under b/23717097
            LOG(FATAL) << "SetProtoFromGPU: GPU Memcpy failed";
            return;
          }
          delete tensor_ref;
          port::CopyFromArray(proto->mutable_tensor_content(), mb, num_bytes);
          alloc->Deallocate<char>(mb);
          done(Status::OK());
        });
  } else {
    done(Status::OK());
  }
}

typedef ProcessState::MemDesc PMD;

/*static*/
void GPUUtil::CopyViaDMA(const string& edge_name,
                         DeviceContext* send_dev_context,
                         DeviceContext* recv_dev_context, Device* src,
                         Device* dst, AllocatorAttributes src_alloc_attr,
                         AllocatorAttributes dst_alloc_attr,
                         const Tensor* input, Tensor* output,
                         StatusCallback done) {
  port::Tracing::ScopedAnnotation annotation(edge_name);
  VLOG(1) << "CopyViaDMA " << edge_name;
  size_t total_bytes = input->TotalBytes();
  // Note that 0-size tensors have no backing buffer.
  if (total_bytes > 0) {
    const void* src_ptr = DMAHelper::base(input);
    void* dst_ptr = DMAHelper::base(output);
    VLOG(2) << "src_ptr " << src_ptr << " dst_ptr " << dst_ptr;
    if (FLAGS_record_mem_types) {
      ProcessState::MemDesc smd = ProcessState::singleton()->PtrType(src_ptr);
      ProcessState::MemDesc dmd = ProcessState::singleton()->PtrType(dst_ptr);
      VLOG(0) << "Src " << smd.DebugString() << " Dst " << dmd.DebugString();
      if (smd.loc == PMD::CPU && dmd.loc == PMD::GPU && (!smd.gpu_registered)) {
        LOG(WARNING) << "CPU -> GPU no reg for " << edge_name;
      }
      if (dmd.loc == PMD::CPU && smd.loc == PMD::GPU && (!dmd.gpu_registered)) {
        LOG(WARNING) << "GPU -> CPU no reg for " << edge_name;
      }
    }

    auto src_device_type = src->attributes().device_type();
    auto dst_device_type = dst->attributes().device_type();

    bool non_cpu_src = (!src_alloc_attr.on_host() &&
                        src_device_type != DeviceType(DEVICE_CPU).type());
    bool non_cpu_dst = (!dst_alloc_attr.on_host() &&
                        dst_device_type != DeviceType(DEVICE_CPU).type());
    if (non_cpu_src) {
      gpu::Stream* stream = send_dev_context->stream();
      if (stream == nullptr) {
        done(errors::Internal("Failed to find device stream"));
        return;
      }
      auto* src_dev_info = src->tensorflow_gpu_device_info();
      CHECK(src_dev_info);

      if (non_cpu_dst) {
        // Device to device copy
        DeviceMemoryBase gpu_dst_ptr(dst_ptr, total_bytes);
        stream->ThenMemcpy(
            &gpu_dst_ptr,
            DeviceMemoryBase{const_cast<void*>(src_ptr), total_bytes},
            total_bytes);
        if (dst_device_type == DeviceType(DEVICE_GPU).type()) {
          // Use of input may outlive stack scope, so keep a ref.
          Tensor* input_ref = new Tensor(*input);
          src_dev_info->event_mgr->ThenExecute(
              stream, [done, stream, input_ref]() {
                delete input_ref;
                if (!stream->ok()) {
                  done(errors::Internal("GPU->GPU Memcpy failed"));
                } else {
                  done(Status::OK());
                }
              });
        }
        send_dev_context->MaintainLifetimeOnStream(input, stream);
      } else {
        // Device to host copy.
        return send_dev_context->CopyDeviceTensorToCPU(input, edge_name, src,
                                                       output, done);
      }
    } else if (non_cpu_dst) {
      // Host to Device copy.
      // Note that this is already an async copy.
      recv_dev_context->CopyCPUTensorToDevice(input, dst, output, done);
    } else {
      memcpy(dst_ptr, src_ptr, total_bytes);
      done(Status::OK());
    }
  } else {
    // buffer is empty
    done(Status::OK());
  }
}

void GPUUtil::CopyGPUTensorToCPU(Device* gpu_device,
                                 const DeviceContext* device_context,
                                 const Tensor* gpu_tensor, Tensor* cpu_tensor,
                                 StatusCallback done) {
  VLOG(1) << "CopyGPUTensorToCPU";
  size_t total_bytes = gpu_tensor->TotalBytes();
  // Note that 0-size tensors have no backing buffer.
  if (total_bytes > 0) {
    const void* src_ptr = DMAHelper::base(gpu_tensor);
    void* dst_ptr = DMAHelper::base(cpu_tensor);
    CHECK(dst_ptr);
    auto* stream = gpu_device->tensorflow_gpu_device_info()->stream;
    if (device_context) {
      stream = static_cast<const GPUDeviceContext*>(device_context)->stream();
    }
    stream->ThenMemcpy(
        dst_ptr, DeviceMemoryBase{const_cast<void*>(src_ptr), total_bytes},
        total_bytes);
    stream->BlockHostUntilDone();
    if (!stream->ok()) {
      done(errors::Internal("CopyGPUTensorToCPU: GPU->CPU Memcpy failed"));
      return;
    }
  }

  done(Status::OK());
}

/*  static */
void GPUUtil::CopyCPUTensorToGPU(const Tensor* cpu_tensor,
                                 const DeviceContext* device_context,
                                 Device* gpu_device, Tensor* gpu_tensor,
                                 StatusCallback done) {
  VLOG(1) << "CopyCPUTensorToGPU";
  CHECK(DeviceType(gpu_device->attributes().device_type()) ==
        DeviceType(DEVICE_GPU));

  auto* dev_info = gpu_device->tensorflow_gpu_device_info();
  if (!dev_info) {
    done(errors::Internal("Failed to find dest device GPUDeviceInfo"));
    return;
  }
  if (cpu_tensor->TotalBytes() != gpu_tensor->TotalBytes()) {
    done(errors::Internal(
        strings::StrCat("Can't copy ", cpu_tensor->TotalBytes(),
                        " bytes of a tensor into another with ",
                        gpu_tensor->TotalBytes(), " bytes buffer.")));
    return;
  }
  const int64 total_bytes = cpu_tensor->TotalBytes();
  // Note that 0-size tensors have no backing buffer.
  if (total_bytes > 0) {
    const void* src_ptr = DMAHelper::base(cpu_tensor);
    void* dst_ptr = DMAHelper::base(gpu_tensor);
    DeviceMemoryBase gpu_dst_ptr(dst_ptr, total_bytes);

    CHECK(device_context);
    auto* stream =
        static_cast<const GPUDeviceContext*>(device_context)->stream();
    stream->ThenMemcpy(&gpu_dst_ptr, src_ptr, total_bytes);
    auto* dev_info = gpu_device->tensorflow_gpu_device_info();
    // Use of cpu_tensor may outlive stack scope, so keep a ref.
    Tensor* input_ref = new Tensor(*cpu_tensor);
    dev_info->event_mgr->ThenExecute(stream, [stream, done, input_ref]() {
      delete input_ref;
      if (!stream->ok()) {
        done(errors::Internal("CopyCPUTensorToGPU: GPU Memcpy failed"));
      } else {
        done(Status::OK());
      }
    });
  } else {
    // empty tensor case
    done(Status::OK());
  }
}

Status GPUUtil::Sync(Device* gpu_device) {
  VLOG(1) << "GPUUtil::Sync";
  auto* dev_info = gpu_device->tensorflow_gpu_device_info();
  if (!dev_info) {
    return errors::Internal("Failed to find dest device GPUDeviceInfo");
  }
  dev_info->stream->BlockHostUntilDone();
  if (!dev_info->stream->ok()) {
    LOG(FATAL) << "GPU sync failed";
  }
  return Status::OK();
}

Status GPUUtil::SyncAll(Device* gpu_device) {
  VLOG(1) << "GPUUtil::SyncAll";
  auto* dev_info = gpu_device->tensorflow_gpu_device_info();
  if (!dev_info) {
    return errors::Internal("Failed to find dest device GPUDeviceInfo");
  }
  if (!dev_info->stream->parent()->SynchronizeAllActivity() ||
      !dev_info->stream->ok()) {
    LOG(FATAL) << "GPU sync failed";
  }
  return Status::OK();
}

string GPUUtil::MemoryDebugString(const Device* device, Tensor* tensor) {
  string ret;
  CHECK(tensor);
  const int64 num_bytes = std::min<int64>(
      FLAGS_brain_gpu_util_debug_string_maxlen, tensor->TotalBytes());
  void* ptr = (num_bytes > 0) ? DMAHelper::base(tensor) : nullptr;
  strings::Appendf(&ret, "%p:", ptr);
  if (num_bytes > 0) {
    auto* dev_info = device->tensorflow_gpu_device_info();
    if (!dev_info) {
      strings::StrAppend(
          &ret, PrintMemory(reinterpret_cast<const char*>(ptr), num_bytes));
    } else {
      string buf;
      buf.resize(num_bytes);
      DeviceMemoryBase gpu_ptr(ptr, num_bytes);
      Status s = dev_info->stream->parent()->SynchronousMemcpyD2H(
          gpu_ptr, num_bytes, gtl::string_as_array(&buf));
      strings::StrAppend(&ret,
                         PrintMemory(gtl::string_as_array(&buf), num_bytes));
    }
  }
  return ret;
}

// TODO(pbar) Checksum is called from places without a valid device context.
uint64 GPUUtil::Checksum(Device* gpu_device,
                         const DeviceContext* device_context,
                         const Tensor& tensor) {
  Tensor copy(tensor.dtype(), tensor.shape());
  Status s;
  Notification n;
  CopyGPUTensorToCPU(gpu_device, device_context, &tensor, &copy,
                     [&s, &n](Status status) {
                       s.Update(status);
                       n.Notify();
                     });
  n.WaitForNotification();
  CHECK(s.ok()) << s;
  return Checksum(copy);
}

uint64 GPUUtil::Checksum(const Tensor& tensor) {
  const float* fptr = reinterpret_cast<const float*>(DMAHelper::base(&tensor));
  size_t num_bytes = tensor.TotalBytes();
  size_t num_floats = num_bytes / sizeof(float);
  for (size_t i = 0; i < num_floats; ++i) {
    CHECK(!std::isnan(fptr[i])) << " i " << i;
  }
  // TODO(tucker): consider using crc32c instead.
  return Hash64(reinterpret_cast<const char*>(DMAHelper::base(&tensor)),
                tensor.TotalBytes(), 0);
}

}  // namespace tensorflow
