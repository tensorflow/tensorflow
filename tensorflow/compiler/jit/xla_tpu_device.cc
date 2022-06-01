/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/xla_tpu_device.h"

#include "absl/types/optional.h"
#include "tensorflow/compiler/jit/kernels/xla_ops.h"
#include "tensorflow/compiler/jit/xla_device.h"
#include "tensorflow/compiler/jit/xla_device_ops.h"
#include "tensorflow/compiler/tf2xla/layout_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/kernel_def.pb.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/core/tpu/tpu_node_device_util.h"
#include "tensorflow/core/tpu/virtual_device.h"
#include "tensorflow/stream_executor/tpu/c_api_conversions.h"
#include "tensorflow/stream_executor/tpu/status_helper.h"
#include "tensorflow/stream_executor/tpu/tpu_node_context.h"
#include "tensorflow/stream_executor/tpu/tpu_platform.h"
#include "tensorflow/stream_executor/tpu/tpu_platform_interface.h"
#include "tensorflow/stream_executor/tpu/tpu_stream_interface.h"

namespace tensorflow {
namespace {

static bool tpu_autoclustering_flag = false;
static bool tpu_xla_device_failure_closes_chips_flag = true;
static bool tpu_use_substreams_for_cross_tpu_device_transfers_flag = true;

// Given a tensor of `shape` and `type`, as what shape should it be stored on
// the TPU device? This function tranposes or flattens the excessively-padded
// tensors to rank 1, but leaves other tensor shapes alone.
StatusOr<xla::Shape> TpuShapeRepresentation(
    const TensorShape& shape, DataType type, bool use_fast_memory,
    XlaLayoutPreference layout_preference) {
  xla::Shape xla_shape;
  TF_RETURN_IF_ERROR(
      tensorflow::TensorShapeToXLAShape(type, shape, &xla_shape));
  ApiConverter::StackHelper<XLA_Shape> se_shape(xla_shape);
  ApiConverter::StackHelper<XLA_Shape> tpu_shape;
  StatusHelper status;
  tpu::ExecutorApiFn()->XlaShapeToTpuShapeRepresentationFn(
      &se_shape.value, type, use_fast_memory, &tpu_shape.value,
      status.c_status);
  if (!status.status().ok()) {
    return status.status();
  }
  return tpu_shape.AsCpp<xla::Shape>();
}

// Given a tensor, returns the shape of its representation on device,
// fully padded. Contents of `shape` are undefined on error.
Status TpuPaddedShapeFn(const Tensor& tensor, xla::Shape* shape) {
  const tensorflow::XlaTensor* xla_tensor =
      tensorflow::XlaTensor::FromTensor(&tensor);
  if (xla_tensor == nullptr) {
    return errors::InvalidArgument(
        "Expected an XlaTensor when computing padded shape");
  }

  if (!xla_tensor->has_shaped_buffer()) {
    return errors::InvalidArgument(
        "XlaTensor is expected to have device memory allocated when "
        "computing padded shape");
  }

  const xla::Shape& on_device_shape =
      xla_tensor->shaped_buffer().on_device_shape();

  StatusHelper status;
  ApiConverter::StackHelper<XLA_Shape> se_shape(on_device_shape);
  ApiConverter::StackHelper<XLA_Shape> tpu_shape;
  tpu::ExecutorApiFn()->XlaShapeToTpuPaddedShapeFn(
      &se_shape.value, &tpu_shape.value, status.c_status);
  if (!status.ok()) {
    return status.status();
  }
  *shape = tpu_shape.AsCpp<xla::Shape>();
  return OkStatus();
}

// Check if TPU has been initialized. TPU initialization is not necessary
// for 1x1.
Status CheckIfTPUInitialized() {
  auto* tpu_platform = tpu::TpuPlatformInterface::GetRegisteredPlatform();
  if (!tpu_platform->Initialized()) {
    return errors::FailedPrecondition(
        "The TPU system has not been initialized.");
  }
  return OkStatus();
}

// Implementation of TPU->TPU device copies that copies over the dedicated TPU
// interconnects, which is much faster than PCIe or the host network.
// TODO(b/117426293): This implementation is only called for direct interconnect
// transfers between TPU devices attached to the same host. Ideally, we would
// generalize this support to direct interconnect transfers across hosts, but
// currently the CopyTensor infrastructure seems to the network topology is
// strictly hierarchical, that is, transfers between devices on different hosts
// can only take place using the host network.
void TpuDeviceToDeviceCopy(DeviceContext* src_dev_context,
                           DeviceContext* dst_dev_context, Device* src,
                           Device* dst, AllocatorAttributes src_allocator_attrs,
                           AllocatorAttributes dst_allocator_attrs,
                           const Tensor* input, Tensor* output,
                           int dev_to_dev_stream_index, StatusCallback done) {
  XlaDeviceContext* const src_xla_context =
      static_cast<XlaDeviceContext*>(src_dev_context);
  XlaDeviceContext* const dst_xla_context =
      static_cast<XlaDeviceContext*>(dst_dev_context);
  static const bool should_use_substream =
      tpu_use_substreams_for_cross_tpu_device_transfers_flag;

  auto impl = [&]() -> Status {
    if (src->name() != dst->name()) {
      Status s = CheckIfTPUInitialized();
      if (!s.ok()) {
        done(s);
        return OkStatus();
      }
    }
    if (input->shape().num_elements() == 0) {
      // Zero-element tensors have no backing buffers.
      done(OkStatus());
      return OkStatus();
    }

    se::Stream* const src_compute_stream = src_xla_context->stream();
    TF_RET_CHECK(src_compute_stream != nullptr);
    TF_RET_CHECK(input->dtype() == output->dtype())
        << "input type: " << DataTypeString(input->dtype()) << " output type "
        << DataTypeString(output->dtype());
    TF_RET_CHECK(input->shape() == output->shape());
    TF_RET_CHECK(DMAHelper::CanUseDMA(input));
    auto* const src_compute_stream_impl = static_cast<tpu::TpuStreamInterface*>(
        src_compute_stream->implementation());

    se::Stream* dst_compute_stream = dst_xla_context->stream();
    auto* const dst_compute_stream_impl = static_cast<tpu::TpuStreamInterface*>(
        dst_compute_stream->implementation());

    if (src_compute_stream_impl->IsSameSharedMemoryLocation(
            dst_compute_stream_impl)) {
      // Surprisingly, this path does get triggered in practice.
      *output = *input;
      done(OkStatus());
      return OkStatus();
    }

    // To avoid stream exhaustion, we pick a substream from a pool if enabled.
    se::Stream* const device_to_device_master_stream =
        should_use_substream ? dst_xla_context->device_to_device_stream(0)
                             : nullptr;
    se::Stream* const dst_device_to_device_stream =
        should_use_substream
            ? device_to_device_master_stream->GetOrCreateSubStream()
            : dst_xla_context->GetDeviceToDeviceStream();
    TF_RET_CHECK(dst_device_to_device_stream != nullptr);
    auto return_substream = gtl::MakeCleanup(
        [device_to_device_master_stream, dst_device_to_device_stream] {
          if (device_to_device_master_stream) {
            device_to_device_master_stream->ReturnSubStream(
                dst_device_to_device_stream);
          }
        });

    auto* const dst_device_to_device_stream_impl =
        static_cast<tpu::TpuStreamInterface*>(
            dst_device_to_device_stream->implementation());

    const int dst_device_ordinal =
        dst_xla_context->stream()->parent()->device_ordinal();

    XlaTensor* const xla_input = XlaTensor::FromTensor(input);
    TF_RET_CHECK(xla_input != nullptr && xla_input->has_shaped_buffer());
    XlaTensor* const xla_output = XlaTensor::FromTensor(output);
    TF_RET_CHECK(xla_output != nullptr && !xla_output->has_shaped_buffer());
    TF_RET_CHECK(input->shape() == output->shape());

    const auto& shape_determination_fns =
        dst_xla_context->shape_determination_fns();
    XlaLayoutPreference layout_preference =
        shape_determination_fns.layout_preference_fn(
            input->shape(), input->dtype(), absl::nullopt);
    TF_ASSIGN_OR_RETURN(xla::Shape shape,
                        shape_determination_fns.shape_representation_fn(
                            input->shape(), input->dtype(),
                            /*use_fast_memory=*/false, layout_preference));
    TF_RETURN_IF_ERROR(xla_output->AllocateShapedBuffer(
        input->dtype(), shape, dst_xla_context->client(), dst_device_ordinal));

    VLOG(2) << "TpuDeviceToDeviceCopy: src: "
            << src_compute_stream->parent()->device_ordinal() << ", "
            << " dst: " << dst_compute_stream->parent()->device_ordinal()
            << ", "
            << " input buffers: " << xla_input->shaped_buffer().ToString()
            << " output buffers: " << xla_output->shaped_buffer().ToString();

    // Wait for definition event of the source tensor so the input buffers are
    // available.
    xla_input->WaitForDefinitionEventOnStream(dst_device_to_device_stream);

    // Wait for the destination tensor buffers to be ready, if they are not
    // available for an immediate write.
    if (!dst_xla_context->transfer_manager()->CanShapedBufferBeAccessedNow(
            dst_compute_stream->parent(), xla_output->shaped_buffer())) {
      dst_device_to_device_stream->ThenWaitFor(dst_compute_stream);
      // If the representation is a tuple, we also must wait for the tuple index
      // buffers to be available on the destination host to device transfer
      // stream.
      if (xla_output->shaped_buffer().on_device_shape().IsTuple()) {
        dst_xla_context->host_to_device_stream()->ThenWaitFor(
            dst_compute_stream);
      }
    }

    for (const auto& leaf : xla_input->shaped_buffer().buffers().leaves()) {
      const xla::ShapeIndex& index = leaf.first;
      const se::DeviceMemoryBase& input_buffer = leaf.second;
      const se::DeviceMemoryBase& output_buffer =
          xla_output->shaped_buffer().buffer(index);
      TF_RET_CHECK(input_buffer.size() == output_buffer.size())
          << "input: " << input_buffer.size()
          << " output: " << output_buffer.size();
      TF_RETURN_IF_ERROR(
          dst_device_to_device_stream_impl->EnqueueOnTpuDeviceSendRecvLocal(
              input_buffer, output_buffer));
    }

    // If the on-device shape is a tuple, write new tuple index buffers.
    if (xla_output->shaped_buffer().on_device_shape().IsTuple()) {
      TF_RETURN_IF_ERROR(
          dst_xla_context->transfer_manager()->WriteTupleIndexTablesAsync(
              dst_xla_context->host_to_device_stream(),
              xla_output->shaped_buffer()));

      // We need a single definition event for an XlaTensor, so make the
      // device to device stream wait for the stream that wrote the tuple index
      // tables on the destination device. Should this prove to be a problem,
      // we can always extend XlaTensor to take a pair of definition events that
      // must all be satisfied, or add an Event::Merge() API that allows us to
      // build an event that is triggered when all of its dependencies are
      // triggered.
      dst_device_to_device_stream->ThenWaitFor(
          dst_xla_context->host_to_device_stream());
    }

    auto definition_event =
        std::make_shared<se::Event>(dst_xla_context->stream()->parent());
    TF_RET_CHECK(definition_event->Init()) << "Event failed to initialize!";
    dst_device_to_device_stream->ThenRecordEvent(definition_event.get());
    xla_output->ResetDefinitionEvent(std::move(definition_event),
                                     dst_device_to_device_stream);

    // The input must remain alive until the transfer completes, so we keep a
    // reference. We also wait until the transfer completes before calling
    // done().
    // The latter may be too conservative, but given the host is involved in
    // waiting for the transfer to complete anyway there is probably little
    // downside. If we were to add the ability for computations to wait directly
    // on transfers, then we might want to rethink this property.
    // Also ideally this host callback should be on source stream rather than
    // destination stream, but when this function returns, the send requests
    // might not be enqueued to the stream yet, we put it on destination stream.
    TensorReference input_reference(*input);
    std::move(return_substream).release();
    dst_device_to_device_stream->ThenDoHostCallback(
        [input_reference, done = std::move(done),
         device_to_device_master_stream, dst_device_to_device_stream] {
          if (device_to_device_master_stream) {
            device_to_device_master_stream->ReturnSubStream(
                dst_device_to_device_stream);
          }
          input_reference.Unref();
          done(OkStatus());
        });

    return OkStatus();
  };
  Status status = impl();
  if (!status.ok()) {
    done(status);
  }
}

class TpuNodeDeviceFactory : public DeviceFactory {
 public:
  Status ListPhysicalDevices(std::vector<string>* devices) override;
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override;
};

Status TpuNodeDeviceFactory::ListPhysicalDevices(std::vector<string>* devices) {
  tpu::TpuPlatformInterface* platform =
      tpu::TpuPlatformInterface::GetRegisteredPlatform();
  if (platform == nullptr) {
    // If we don't have a platform registered, then we have no devices.
    return OkStatus();
  }

  int device_count = platform->VisibleDeviceCount();

  for (int i = 0; i < device_count; ++i) {
    const string device_name = absl::StrCat("/physical_device:TPU:", i);
    devices->push_back(device_name);
  }

  return OkStatus();
}

Status TpuNodeDeviceFactory::CreateDevices(
    const SessionOptions& session_options, const string& name_prefix,
    std::vector<std::unique_ptr<Device>>* devices) {
  tpu::TpuPlatformInterface* platform =
      tpu::TpuPlatformInterface::GetRegisteredPlatform();
  if (platform == nullptr) {
    // If we don't have a platform registered, then we should not create any.
    return OkStatus();
  }

  if (platform != nullptr && platform->ShouldRegisterTpuDeviceToDeviceCopy()) {
    RegisterTpuDeviceToDeviceCopy();
  }

  XlaOpRegistry::DeviceRegistration registration;
  registration.compilation_device_name = DEVICE_TPU_XLA_JIT;
  registration.autoclustering_policy =
      tpu_autoclustering_flag
          ? XlaOpRegistry::AutoclusteringPolicy::kAlways
          : XlaOpRegistry::AutoclusteringPolicy::kIfExplicitlyRequested;

  registration.cluster_resource_variable_ops_unsafely = true;
  registration.cluster_stack_ops = false;
  registration.cluster_tensor_array_ops = true;
  registration.cluster_stateful_rng_ops = true;
  registration.cluster_control_trigger = true;
  registration.elide_assert_and_checknumerics = true;
  registration.cluster_variant_ops = true;
  registration.cluster_slow_ops = true;
  registration.cluster_inaccurate_ops = true;
  XlaOpRegistry::RegisterCompilationDevice(DEVICE_TPU_NODE, registration);

  static XlaDeviceOpRegistrations* registrations =
      RegisterXlaDeviceKernels(DEVICE_TPU_NODE, DEVICE_TPU_XLA_JIT);
  (void)registrations;

  int device_count = platform->VisibleDeviceCount();
  VLOG(1) << "Creating " << device_count << " TPU devices";
  for (int i = 0; i < device_count; ++i) {
    TF_RETURN_IF_ERROR(tpu::TpuNodeContext::Initialize(i));

    XlaDevice::Options options;
    options.platform = platform;
    options.device_name_prefix = name_prefix;
    options.device_name = DEVICE_TPU_NODE;
    options.device_ordinal = i;
    options.compilation_device_name = DEVICE_TPU_XLA_JIT;
    options.use_multiple_streams = true;
    // We set `use_global_compute_stream` to true for TPUs as TPUs can only
    // have one program running on each core at the same time.
    options.use_global_compute_stream = true;
    XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns{
        UseNoPreferenceLayoutFn(), &TpuShapeRepresentation};
    options.shape_determination_fns = {shape_determination_fns};
    options.padded_shape_fn = &TpuPaddedShapeFn;
    auto device = absl::make_unique<XlaDevice>(session_options, options);

    // The AcceleratorDeviceInfo actually provides information not only for GPU
    // devices but also for TPU. The name is a legacy from the pre-TPU
    // dark ages.
    Status status = device->UseAcceleratorDeviceInfo();
    if (!status.ok()) {
      errors::AppendToMessage(&status, "while setting up ", DEVICE_TPU_XLA_JIT,
                              " device number ", i);
      return status;
    }
    device->SetAllowsSyncOnCompletion(false);
    if (tpu_xla_device_failure_closes_chips_flag) {
      device->SetHandleDeviceErrorCallback(&tpu::TpuNodeContext::CloseTpuHost);
    }

    devices->push_back(std::move(device));
  }

  return OkStatus();
}

class TpuSystemDeviceFactory : public DeviceFactory {
 public:
  Status ListPhysicalDevices(std::vector<string>* devices) override;
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override;
};

Status TpuSystemDeviceFactory::ListPhysicalDevices(
    std::vector<string>* devices) {
  int device_count = 0;
  TF_RETURN_IF_ERROR(tpu::TpuPlatform::TpusPerHost(&device_count));
  if (device_count == 0) {
    VLOG(1) << "Host has no TPUs, not creating a TPU_SYSTEM device";
    return OkStatus();
  }

  devices->push_back("/physical_device:TPU_SYSTEM:0");

  return OkStatus();
}

Status TpuSystemDeviceFactory::CreateDevices(
    const SessionOptions& options, const string& name_prefix,
    std::vector<std::unique_ptr<Device>>* devices) {
  int device_count = 0;
  TF_RETURN_IF_ERROR(tpu::TpuPlatform::TpusPerHost(&device_count));
  if (device_count == 0) {
    VLOG(1) << "Host has no TPUs, not creating a TPU_SYSTEM device";
    return OkStatus();
  }

  int64_t memory_limit;
  TF_RETURN_IF_ERROR(tpu::TpuPlatform::TpuMemoryLimit(&memory_limit));

  // Creates a device that represents a TPU distributed system.
  const DeviceAttributes attrs = Device::BuildDeviceAttributes(
      absl::StrCat(name_prefix, "/device:", DEVICE_TPU_SYSTEM, ":", 0),
      DeviceType(DEVICE_TPU_SYSTEM), Bytes(memory_limit), DeviceLocality(),
      absl::StrCat("device: ", DEVICE_TPU_SYSTEM, " device"));
  devices->push_back(absl::make_unique<VirtualDevice>(options.env, attrs));
  VLOG(1) << "Created TPU_SYSTEM device. This host has " << device_count
          << " TPUs";

  return OkStatus();
}

}  // namespace

void RegisterTpuDeviceToDeviceCopy() {
  static auto* const register_tpu_tpu_copy = new CopyTensor::Registration(
      DEVICE_TPU_NODE, DEVICE_TPU_NODE, TpuDeviceToDeviceCopy);
  (void)register_tpu_tpu_copy;
}

void RegisterTpuNodeDevice(
    bool tpu_autoclustering, bool tpu_xla_device_failure_closes_chips,
    bool tpu_use_substreams_for_cross_tpu_device_transfers) {
  tpu_autoclustering_flag = tpu_autoclustering;
  tpu_xla_device_failure_closes_chips_flag =
      tpu_xla_device_failure_closes_chips;
  tpu_use_substreams_for_cross_tpu_device_transfers_flag =
      tpu_use_substreams_for_cross_tpu_device_transfers;

  REGISTER_XLA_LAUNCH_KERNEL(DEVICE_TPU_NODE, XlaLocalLaunchOp, kTpuAllTypes);
  REGISTER_XLA_COMPILE_KERNEL(DEVICE_TPU_NODE, XlaCompileOp, kTpuAllTypes);
  REGISTER_XLA_RUN_KERNEL(DEVICE_TPU_NODE, XlaRunOp, kTpuAllTypes);
  REGISTER_XLA_DEVICE_KERNELS(DEVICE_TPU_NODE, kTpuAllTypes);
  REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_TPU_NODE, TpuNodeDeviceFactory);
}

void RegisterTpuSystemDevice() {
  REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_TPU_SYSTEM, TpuSystemDeviceFactory);
}

#if !defined(PLATFORM_GOOGLE)

// We automatically register this if we are building for open source. For
// Google platforms, we initialize these devices in other places.

REGISTER_XLA_LAUNCH_KERNEL(DEVICE_TPU_NODE, XlaLocalLaunchOp, kTpuAllTypes);
REGISTER_XLA_COMPILE_KERNEL(DEVICE_TPU_NODE, XlaCompileOp, kTpuAllTypes);
REGISTER_XLA_RUN_KERNEL(DEVICE_TPU_NODE, XlaRunOp, kTpuAllTypes);
REGISTER_XLA_DEVICE_KERNELS(DEVICE_TPU_NODE, kTpuAllTypes);
REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_TPU_NODE, TpuNodeDeviceFactory);
REGISTER_LOCAL_DEVICE_FACTORY(DEVICE_TPU_SYSTEM, TpuSystemDeviceFactory);

#endif  // PLATFORM_GOOGLE

}  // namespace tensorflow
