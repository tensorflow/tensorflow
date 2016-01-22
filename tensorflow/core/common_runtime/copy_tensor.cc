#include "tensorflow/core/common_runtime/copy_tensor.h"

#include <vector>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tracing.h"

namespace tensorflow {
namespace {

static bool initialization_done = false;

struct RegistrationInfo {
  RegistrationInfo(DeviceType s, DeviceType r, CopyTensor::CopyFunction cf)
      : sender_device_type(s), receiver_device_type(r), copy_function(cf) {}
  DeviceType sender_device_type;
  DeviceType receiver_device_type;
  CopyTensor::CopyFunction copy_function;
};

// We use a vector instead of a map since we expect there to be very
// few registrations.
std::vector<RegistrationInfo>* MutableRegistry() {
  static std::vector<RegistrationInfo>* registry =
      new std::vector<RegistrationInfo>;
  return registry;
}

}  // namespace

// static
void CopyTensor::ViaDMA(const string& edge_name,
                        DeviceContext* send_dev_context,
                        DeviceContext* recv_dev_context, Device* src,
                        Device* dst, const AllocatorAttributes src_alloc_attr,
                        const AllocatorAttributes dst_alloc_attr,
                        const Tensor* input, Tensor* output,
                        StatusCallback done) {
  initialization_done = true;
  port::Tracing::ScopedAnnotation annotation(edge_name);
  VLOG(1) << "CopyViaDMA " << edge_name;
  const size_t total_bytes = input->TotalBytes();

  // Note that 0-size tensors have no backing buffer.
  if (total_bytes > 0) {
    const DeviceType src_device_type(src_alloc_attr.on_host()
                                         ? DEVICE_CPU
                                         : src->attributes().device_type());
    const DeviceType dst_device_type(dst_alloc_attr.on_host()
                                         ? DEVICE_CPU
                                         : dst->attributes().device_type());
    const bool non_cpu_src = src_device_type != DeviceType(DEVICE_CPU);
    const bool non_cpu_dst = dst_device_type != DeviceType(DEVICE_CPU);

    if (non_cpu_src) {
      if (non_cpu_dst) {
        // Device to device copy.  Look through registry for an appropriate
        // CopyFunction.
        std::vector<RegistrationInfo>* registry = MutableRegistry();
        for (const RegistrationInfo& ri : *registry) {
          if (ri.sender_device_type == src_device_type &&
              ri.receiver_device_type == dst_device_type) {
            ri.copy_function(send_dev_context, recv_dev_context, src, dst,
                             src_alloc_attr, dst_alloc_attr, input, output,
                             done);
            return;
          }
        }

        // TODO(josh11b): If no CopyFunction is found, we currently fail
        // but we could copy between devices via CPU.
        done(errors::Unimplemented(
            "No function registered to copy from devices of type ",
            src_device_type.type(), " to devices of type ",
            dst_device_type.type()));
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
      *output = *input;
      done(Status::OK());
    }
  } else {
    // buffer is empty
    done(Status::OK());
  }
}

// static
Status CopyTensor::Register(DeviceType sender_device_type,
                            DeviceType receiver_device_type,
                            CopyFunction copy_function) {
  if (initialization_done) {
    return errors::FailedPrecondition(
        "May only register CopyTensor functions during before the first tensor "
        "is copied.");
  }
  std::vector<RegistrationInfo>* registry = MutableRegistry();
  registry->emplace_back(sender_device_type, receiver_device_type,
                         copy_function);
  return Status::OK();
}

}  // namespace tensorflow
