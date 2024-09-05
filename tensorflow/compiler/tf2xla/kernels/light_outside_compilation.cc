/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/kernels/light_outside_compilation.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/const_init.h"  // NOLINT incorrectly flagged as unused
#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/kernels/callback.pb.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/xla_builder.h"
#include "xla/layout_util.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_finder.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "xla/util.h"
#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/common_runtime/process_state.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/statusor.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#endif

namespace tensorflow {

namespace {

const char* const kTfCallbackCustomCall = "GenericTfCallbackGPU";

// Each unique TfCallbackData in a call to Compile() generates a new
// KernelInstantiation.  We cache these in a global hashtable, because creating
// OpKernels is expensive.
//
// The KernelInstantiation contains a list of OpKernel+DeviceBase objects.
// These are all equivalent to each other.  When we want to run the kernel, we
// "check out" an element from the list, removing it.  When we're done, we add
// it back.  If there are no available elements in the list, we create one.
struct KernelInstantiation {
  static absl::StatusOr<std::unique_ptr<KernelInstantiation>> Create(
      TfCallbackData callback_data) {
    auto instantiation = std::make_unique<KernelInstantiation>();
    instantiation->input_shapes.reserve(callback_data.inputs_size());
    for (const auto& input : callback_data.inputs()) {
      TensorShape shape;
      TF_RETURN_IF_ERROR(TensorShape::BuildTensorShape(
          input.buffer_description().shape(), &shape));
      instantiation->input_shapes.push_back(std::move(shape));
    }
    instantiation->callback_data = std::move(callback_data);
    return instantiation;
  }

  // Cache of
  //   TensorShape::BuildTensorShape(
  //     callback_data.inputs(i).buffer_description().shape()),
  // because calling this for every invocation of the op is expensive.
  std::vector<TensorShape> input_shapes;

  TfCallbackData callback_data;

  // In order to run an Instantiation, we need a TF OpKernel object.  Creating
  // one is expensive, so we cache them.  When we run an op, we "check out" an
  // element from this vector, and then we return it when we're done.
  absl::Mutex mu;
  std::vector<std::pair<std::unique_ptr<DeviceBase>, std::unique_ptr<OpKernel>>>
      devices_and_kernels ABSL_GUARDED_BY(mu);
};

absl::StatusOr<std::string> MakeOpaque(TfCallbackData callback_data) {
  // Clear the `name` field in the callback_data, because this contains the full
  // TF op name scope.  We want ops with different names to map to the same
  // Instantiation (so that if the same op with the same shape appears 10 times
  // in a model, they all have the same logical TfCallbackData.)
  callback_data.mutable_op()->clear_name();

  std::string serialized_data;
  if (!tsl::SerializeToStringDeterministic(callback_data, &serialized_data)) {
    return absl::InternalError(
        "Failed in serializing TfCallbackData to string");
  }
  auto fingerprint = tsl::Fingerprint128(serialized_data);
  return absl::StrFormat(
      "fingerprint128=%s serialized=%s (%s)",
      absl::Base64Escape(absl::string_view(
          reinterpret_cast<char*>(&fingerprint), sizeof(fingerprint))),
      absl::Base64Escape(serialized_data), callback_data.op().op());
}

absl::StatusOr<KernelInstantiation*> GetInstantiation(
    absl::string_view opaque) {
  constexpr absl::string_view kFingerprintPrefix = "fingerprint128=";
  if (!absl::StartsWith(opaque, kFingerprintPrefix)) {
    return xla::Internal("Invalid opaque; must start with '%s', but was '%s'",
                         kFingerprintPrefix, opaque);
  }
  opaque.remove_prefix(kFingerprintPrefix.length());

  // We use a 128-bit fingerprint, encoded as base64.  It's 24 chars long
  // including padding.
  constexpr int kFingerprintLen = 24;
  absl::string_view fingerprint_str = opaque.substr(0, kFingerprintLen);
  opaque.remove_prefix(kFingerprintLen);
  if (fingerprint_str.length() != kFingerprintLen) {
    return xla::Internal("Invalid opaque; fingerprint is wrong length: '%s'",
                         fingerprint_str);
  }

  static absl::Mutex mu{absl::kConstInit};
  static auto& instantiations ABSL_GUARDED_BY(mu) =
      *new absl::flat_hash_map<std::string /*base64-encoded fingerprint*/,
                               std::unique_ptr<KernelInstantiation>>();

  absl::MutexLock lock(&mu);
  std::unique_ptr<KernelInstantiation>& instantiation =
      instantiations[fingerprint_str];
  if (instantiation == nullptr) {
    // Cache miss; create the instantiation.  To do this, we need to parse out
    // the serialized TfCallbackData from the opaque.
    constexpr absl::string_view kSerializedPrefix = " serialized=";
    if (!absl::StartsWith(opaque, kSerializedPrefix)) {
      return xla::Internal("Invalid opaque; must start with '%s', but was '%s'",
                           kSerializedPrefix, opaque);
    }
    opaque.remove_prefix(kSerializedPrefix.length());

    // Find the end of the base64-encoded serialized proto.
    opaque = opaque.substr(0, opaque.find_first_of(' '));

    // Unescape the base64 string, then parse the proto.
    std::string unescaped_opaque;
    if (!absl::Base64Unescape(opaque, &unescaped_opaque)) {
      return xla::Internal("Failed to base64 decode opaque %s", opaque);
    }
    TfCallbackData callback_data;
    if (!callback_data.ParseFromString(unescaped_opaque)) {
      return xla::Internal("Failed to parse TfCallbackData from opaque %s",
                           opaque);
    }
    TF_ASSIGN_OR_RETURN(instantiation,
                        KernelInstantiation::Create(std::move(callback_data)));

    // Log a warning every power of 2 if `instantiations` gets large.
    if (size_t n = instantiations.size();
        n >= (1 << 12) && (n & (n - 1)) == 0) {
      LOG(WARNING)  //
          << "Light outside compilation has compiled " << n
          << " unique op+shape combinations.  Each of these permanently "
             "leaks CPU memory.  Exactly how much depends on the op, but "
             "expect ~1kb per op.  This is probably happening because you're "
             "recompiling an XLA model many times with different shapes.  "
             "This message will be logged again at the next power of 2.";
    }
  }
  return instantiation.get();
}

}  // namespace

static absl::StatusOr<Tensor> TensorFromProto(const TensorProto& proto) {
  Tensor out;
  if (!out.FromProto(proto)) {
    return tsl::errors::Internal("Failed deserializing a TensorProto");
  }
  return out;
}

Status LightOutsideCompilationOp::CompileToCustomCallCallingTfKernel(
    int graph_def_version, const NodeDef& node_def, XlaOpKernelContext* ctx) {
  const OpRegistrationData* data = OpRegistry::Global()->LookUp(node_def.op());
  int num_inputs = ctx->num_inputs();
  int num_outputs = ctx->num_outputs();

  std::vector<Tensor> tensor_storage(num_inputs);
  std::vector<const Tensor*> input_tensors(num_inputs);
  std::vector<shape_inference::ShapeHandle> input_shapes;

  shape_inference::InferenceContext ic(
      graph_def_version, node_def, data->op_def,
      std::vector<shape_inference::ShapeHandle>(num_inputs), {}, {}, {});
  TF_RETURN_IF_ERROR(ic.construction_status());

  TfCallbackData callback_data;
  *callback_data.mutable_op() = node_def;

  TF_ASSIGN_OR_RETURN(
      std::vector<int> constant_inputs,
      XlaOpRegistry::CompileTimeConstantInputs(node_def, data->op_def));
  VLOG(1) << "Constant inputs we got: " << absl::StrJoin(constant_inputs, ", ");

  std::vector<xla::Shape> operand_shapes_with_layout;
  std::vector<xla::XlaOp> operands;
  for (int i = 0; i < num_inputs; ++i) {
    TF_ASSIGN_OR_RETURN(xla::Shape xla_shape, ctx->InputXlaShape(i));
    if (absl::c_any_of(xla_shape.dynamic_dimensions(),
                       [](const bool is_dynamic) { return is_dynamic; })) {
      // TODO(cheshire): Support input dynamic dimensions.
      return tsl::errors::Internal(
          "Input dynamic dimensions are not supported for light outside "
          "compilation");
    }
    // TODO(cheshire): Use InputXlaShape.
    TensorShape shape = ctx->InputShape(i);
    TfCallbackData::InputBufferDescription input_description;

    *input_description.mutable_buffer_description()->mutable_shape() =
        shape.AsProto();
    input_description.mutable_buffer_description()->set_type(
        ctx->input_type(i));

    if (absl::c_linear_search(constant_inputs, i)) {
      // Assuming kernels want to read INT32 datatypes.
      TF_ASSIGN_OR_RETURN(Tensor input_tensor, ctx->ConstantInputTensor(i));
      tensor_storage[i] = input_tensor;
      input_tensors[i] = &tensor_storage.at(i);
      input_tensor.AsProtoTensorContent(input_description.mutable_value());
    } else {
      input_tensors[i] = nullptr;
      operands.push_back(ctx->Input(i));
      operand_shapes_with_layout.push_back(xla_shape);
      xla::LayoutUtil::SetToDefaultLayout(&operand_shapes_with_layout.back());
    }

    *callback_data.add_inputs() = input_description;

    TF_ASSIGN_OR_RETURN(shape_inference::ShapeHandle handle,
                        ic.MakeShapeFromShapeTensor(shape));
    ic.SetInput(i, handle);
  }

  ic.set_input_tensors(input_tensors);

  TF_RETURN_IF_ERROR(data->shape_inference_fn(&ic));
  TF_ASSIGN_OR_RETURN(OutputDimensionBoundsMap output_dimension_bounds,
                      DynamicOutputDimensions(node_def, ctx));

  std::vector<xla::Shape> output_xla_shapes;
  for (int i = 0; i < num_outputs; ++i) {
    const DimensionBoundsMap& dimension_bounds = output_dimension_bounds[i];
    TfCallbackData::OutputBufferDescription output_description;
    output_description.mutable_buffer_description()->set_type(
        ctx->expected_output_dtype(i));

    TensorShapeProto output_tensor_shape_proto =
        ic.ShapeHandleToProto(ic.output(i));
    if (output_tensor_shape_proto.unknown_rank()) {
      return tsl::errors::Internal("Output ", i, " has unknown rank");
    }

    int rank = output_tensor_shape_proto.dim_size();
    std::vector<bool> dynamic_dimensions(rank, false);

    // Modify output tensor shape proto to replace dynamic dimensions with upper
    // bounds: that is the information we will be storing in the callback.
    for (int d = 0; d < output_tensor_shape_proto.dim_size(); ++d) {
      auto* dim = output_tensor_shape_proto.mutable_dim(d);
      auto it = dimension_bounds.find(d);

      if (dim->size() < 0) {
        if (it == dimension_bounds.end()) {
          return tsl::errors::Internal(
              "Bound for unknown dimension not found for dimension ", d);
        }
        dim->set_size(it->second);
        dynamic_dimensions[d] = true;
        output_description.set_is_dynamically_padded(true);
      } else {
        if (it == dimension_bounds.end()) {
          continue;
        }
        if (it->second < dim->size()) {
          dim->set_size(it->second);
        }
      }
    }

    *output_description.mutable_buffer_description()->mutable_shape() =
        output_tensor_shape_proto;
    *callback_data.add_outputs() = output_description;

    TF_ASSIGN_OR_RETURN(
        TensorShape output_tensor_shape,
        TensorShape::BuildTensorShape(output_tensor_shape_proto));

    TF_ASSIGN_OR_RETURN(xla::Shape output_shape,
                        TensorShapeToXLAShape(ctx->expected_output_dtype(i),
                                              output_tensor_shape));

    // Set corresponding dynamic bounds on the output xla::Shape.
    for (int64_t d = 0; d < dynamic_dimensions.size(); ++d) {
      output_shape.set_dynamic_dimension(d, dynamic_dimensions[d]);
    }
    output_xla_shapes.push_back(output_shape);
  }

  xla::Shape output_shape =
      xla::ShapeUtil::MakeMaybeTupleShape(output_xla_shapes);

  VLOG(1) << "Created output shape: " << output_shape.ToString();

  TF_ASSIGN_OR_RETURN(std::string opaque, MakeOpaque(std::move(callback_data)));
  xla::XlaOp out = xla::CustomCallWithLayout(
      ctx->builder(), kTfCallbackCustomCall, operands, output_shape,
      operand_shapes_with_layout, opaque,
      /*has_side_effect=*/false,
      /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
      xla::CustomCallSchedule::SCHEDULE_NONE,
      xla::CustomCallApiVersion::API_VERSION_STATUS_RETURNING);

  for (int i = 0; i < num_outputs; ++i) {
    ctx->SetOutput(i,
                   output_shape.IsTuple() ? xla::GetTupleElement(out, i) : out);
  }

  return absl::OkStatus();
}

namespace {

class WriteIntoXlaBufferAllocator : public Allocator {
 public:
  WriteIntoXlaBufferAllocator(void* xla_buffer, size_t buffer_size,
                              absl::string_view description)
      : xla_buffer_(xla_buffer),
        buffer_size_(buffer_size),
        description_(description) {}

  std::string Name() override {
    return absl::StrCat("allocator-xla-", description_);
  }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    VLOG(1) << "Faking allocation of " << num_bytes << " bytes into xla buffer "
            << description_;

    if (num_bytes > buffer_size_) {
      LOG(ERROR) << "Failed allocation: requested larger size than the "
                    "underlying buffer";
      return nullptr;
    }
    return xla_buffer_;
  }

  // Do not perform our own memory management.
  void DeallocateRaw(void* ptr) override {
    VLOG(1) << "Not deallocating pointer " << ptr << " for " << description_;
  }

 private:
  void* xla_buffer_;
  size_t buffer_size_;
  std::string description_;
};

int GetNumConstants(const TfCallbackData& callback_data) {
  return absl::c_count_if(callback_data.inputs(),
                          [&](const auto& input) { return input.has_value(); });
}

int GetOutputBufferId(int output_num, const TfCallbackData& callback_data) {
  return (callback_data.inputs_size() - GetNumConstants(callback_data)) +
         output_num;
}

int64_t BufferSize(const TfCallbackData::BufferDescription& descr) {
  // Do this without calling TensorShape::BuildTensorShape, because that's
  // nontrivially expensive.
  int64_t num_elems = 1;
  for (const auto& d : descr.shape().dim()) {
    num_elems *= d.size();
  }
  return num_elems * DataTypeSize(descr.type());
}

class TfCallbackDevice : public DeviceBase {
 public:
  TfCallbackDevice()
      : DeviceBase(Env::Default()),
        cpu_allocator_(
            ProcessState::singleton()->GetCPUAllocator(/*numa_node=*/0)) {}

  void SetUpForCall(se::Stream* stream, void** buffers,
                    const TfCallbackData& callback_data) {
    stream_ = stream;

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    gpu_allocator_ = GPUProcessState::singleton()->GetGPUAllocator(
        *BaseGPUDevice::FindTfDeviceId(stream));
#endif

    allocators_.clear();
    for (int i = 0; i < callback_data.outputs_size(); ++i) {
      int buffer_num = GetOutputBufferId(i, callback_data);
      VLOG(1) << "Binding output " << i << " to buffer " << buffers[buffer_num];
      int64_t buffer_size =
          BufferSize(callback_data.outputs(i).buffer_description());
      allocators_.emplace_back(buffers[buffer_num], buffer_size,
                               absl::StrCat("xla-output-", i));
    }

    accelerator_device_info_.stream = stream;
    set_tensorflow_accelerator_device_info(&accelerator_device_info_);
  }

  const string& name() const override { return name_; }

  PerOpGpuDevice* MakeGpuDevice() override {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    return new ConcretePerOpGpuDevice();
#else
    LOG(FATAL) << "CUDA-enabled build is required";  // Crash OK
#endif
  }

  Status ReinitializeGpuDevice(OpKernelContext* context, PerOpGpuDevice* device,
                               DeviceContext* dc,
                               Allocator* allocator) override {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    auto concrete_device = static_cast<ConcretePerOpGpuDevice*>(device);
    concrete_device->Reinitialize(
        context, stream_->platform_specific_handle().stream,
        /*platform_device_id=*/
        tsl::PlatformDeviceId(stream_->parent()->device_ordinal()), allocator,
        // TODO(cheshire): Pass meaningful scratch buffer.
        /*scratch=*/nullptr);
    return OkStatus();
#else
    LOG(FATAL) << "CUDA-enabled build is required";  // Crash OK
#endif
  }

  Allocator* GetScopedAllocator(AllocatorAttributes attrs,
                                int64_t step_id) override {
    return &allocators_[attrs.scope_id - 1];
  }

  Allocator* GetAllocator(AllocatorAttributes attr) override {
    if (attr.on_host()) {
      if (attr.gpu_compatible()) {
        GPUProcessState* ps = GPUProcessState::singleton();
        // TODO(jlebar): The very first call to GetGpuHostAllocator sets its
        // memory limits.  So passing {} for the options here means that if
        // nobody gets this allocator before us, we will not respect any limits
        // the user might have set on host memory allocation.  Our call to
        // GetGPUAllocator in the constructor has the same problem.
        return ps->GetGpuHostAllocator(/*options=*/{}, 0);
      } else {
        return cpu_allocator_;
      }
    } else {
      return gpu_allocator_;
    }
  }

 private:
  std::vector<WriteIntoXlaBufferAllocator> allocators_;
  se::Stream* stream_ = nullptr;  // NOLINT (used under GOOGLE_CUDA)
  Allocator* gpu_allocator_ = nullptr;
  Allocator* cpu_allocator_ = nullptr;
  AcceleratorDeviceInfo accelerator_device_info_;
  std::string name_ = "tf_callback_device";
};

// Populate the output with actual dimensions of the allocated shapes.
//
// Populates the vector on the host and then copies it over to the GPU.
Status PopulateMetadataBufferIfNeeded(OpKernelContext& ctx,
                                      const TfCallbackData& callback_data,
                                      void** buffers, se::Stream* stream) {
  for (int i = 0; i < ctx.num_outputs(); i++) {
    if (callback_data.outputs(i).is_dynamically_padded()) {
      Tensor* allocated = ctx.mutable_output(i);
      TensorShape allocated_shape = allocated->shape();
      int num_dimensions = allocated_shape.dims();
      std::vector<int32_t> shape_info(num_dimensions);
      for (int d = 0; d < allocated_shape.dims(); d++) {
        int dim_size = allocated_shape.dim_size(d);
        shape_info[d] = dim_size;
      }

      TF_ASSIGN_OR_RETURN(
          xla::Shape xla_shape,
          tensorflow::TensorShapeToXLAShape(
              callback_data.outputs(i).buffer_description().type(),
              callback_data.outputs(i).buffer_description().shape()));
      void* location = static_cast<char*>(allocated->data()) +
                       xla::ShapeUtil::ByteSizeOf(xla_shape);
      se::DeviceMemoryBase m{location, num_dimensions * sizeof(int32_t)};
      TF_RETURN_IF_ERROR(stream->Memcpy(&m, shape_info.data(),
                                        num_dimensions * sizeof(int32_t)));
    }
  }
  return absl::OkStatus();
}

class FakeDeviceContext : public DeviceContext {
 public:
  explicit FakeDeviceContext(se::Stream* stream) { stream_ = stream; }
  se::Stream* stream() const override { return stream_; }

 private:
  se::Stream* stream_;
};

Status CallTfKernel(void* stream_handle, void** buffers, const char* opaque,
                    int opaque_len) {
  // Look up the platform only once, for a small performance gain.
  static Status* platform_status = nullptr;
  static se::Platform* platform = [&]() -> se::Platform* {
    absl::StatusOr<se::Platform*> p =
        se::PlatformManager::PlatformWithName("CUDA");
    if (!p.ok()) {
      platform_status = new Status(p.status());
      return nullptr;
    }
    return *p;
  }();
  if (platform_status != nullptr) return *platform_status;

  TF_ASSIGN_OR_RETURN(se::Stream * stream,
                      stream_executor::FindStream(platform, stream_handle));
  if (!stream) {
    return xla::Internal("Stream not found for %p", stream_handle);
  }

  TF_ASSIGN_OR_RETURN(KernelInstantiation * instantiation,
                      GetInstantiation(absl::string_view(opaque, opaque_len)));
  const TfCallbackData& callback_data = instantiation->callback_data;

  // Get an existing TfCallbackDevice + OpKernel pair from the instantiation, or
  // create a new set.
  std::unique_ptr<TfCallbackDevice> device;
  std::unique_ptr<OpKernel> kernel;
  {
    absl::MutexLock lock(&instantiation->mu);

    if (instantiation->devices_and_kernels.empty()) {
      auto device = std::make_unique<TfCallbackDevice>();

      absl::Status nested_status;
      auto kernel =
          CreateOpKernel(DeviceType(DEVICE_GPU),
                         /*device=*/device.get(),
                         // NB: real allocator is passed with device, the one
                         // here is only called during the kernel construction.
                         // TODO(cheshire): Pass scratch allocator.
                         /*allocator=*/nullptr, callback_data.op(),
                         /*graph_def_version=*/1, &nested_status);
      TF_RETURN_IF_ERROR(nested_status);

      instantiation->devices_and_kernels.push_back(
          {std::move(device), std::move(kernel)});
    }

    // Grab the last element of devices_and_kernels.
    auto& dk = instantiation->devices_and_kernels.back();
    device =
        absl::WrapUnique(static_cast<TfCallbackDevice*>(dk.first.release()));
    kernel = std::move(dk.second);
    instantiation->devices_and_kernels.pop_back();
  }

  // Put callback_device and kernel back in `devices_and_kernels` when we're
  // done with them.
  auto cleanup = absl::MakeCleanup([&] {
    absl::MutexLock lock(&instantiation->mu);
    instantiation->devices_and_kernels.push_back(
        std::make_pair(std::move(device), std::move(kernel)));
  });

  // Point this fake device at our stream and buffers.
  device->SetUpForCall(stream, buffers, callback_data);

  std::vector<AllocatorAttributes> allocator_attributes;
  for (int output_idx = 0; output_idx < callback_data.outputs_size();
       ++output_idx) {
    AllocatorAttributes attr;
    // Repurpose `scope_id` to communicate which output is it.
    // Shift by one to make it greater than zero.
    attr.scope_id = output_idx + 1;
    allocator_attributes.push_back(attr);
  }

  auto device_context =
      core::RefCountPtr<FakeDeviceContext>(new FakeDeviceContext(stream));

  OpKernelContext::Params params;
  params.output_attr_array = allocator_attributes.data();
  params.op_kernel = kernel.get();
  params.device = device.get();
  params.ensure_eigen_gpu_device();
  params.op_device_context = device_context.get();

  absl::InlinedVector<TensorValue, 16> inputs;

  // Deque usage is important to avoid moving objects.
  std::deque<WriteIntoXlaBufferAllocator> input_allocators;
  std::deque<Tensor> input_tensors;

  int constant_offset = 0;

  TF_RET_CHECK(callback_data.inputs_size() ==
               instantiation->input_shapes.size());

  for (int i = 0; i < callback_data.inputs_size(); ++i) {
    DataType dt = callback_data.inputs(i).buffer_description().type();
    const TensorShape& shape = instantiation->input_shapes[i];

    VLOG(2) << "Input shape: " << shape.DebugString();
    int64_t input_size = shape.num_elements() * DataTypeSize(dt);

    if (callback_data.inputs(i).has_value()) {
      // Value provided at compile time: reconstruct the tensor.
      TF_ASSIGN_OR_RETURN(Tensor input,
                          TensorFromProto(callback_data.inputs(i).value()));

      VLOG(1) << "Input " << i << " is a tensor: " << input.DebugString();
      input_tensors.push_back(std::move(input));
      constant_offset++;
    } else {
      VLOG(1) << "Reading into the buffer for the input " << i;

      // We only get backing input buffer for those inputs which are *not*
      // forced to be constant at compile time.
      input_allocators.emplace_back(buffers[i - constant_offset], input_size,
                                    absl::StrCat("input-", i));
      input_tensors.emplace_back(&input_allocators[i], dt, shape);
    }
    inputs.emplace_back(&input_tensors.back());
  }

  params.inputs = absl::MakeSpan(inputs);
  OpKernelContext ctx(&params, callback_data.outputs_size());
  kernel->Compute(&ctx);

  bool has_dynamic_outputs = absl::c_any_of(
      callback_data.outputs(),
      [](const auto& out) { return out.is_dynamically_padded(); });

  if (has_dynamic_outputs) {
    TF_RETURN_IF_ERROR(
        PopulateMetadataBufferIfNeeded(ctx, callback_data, buffers, stream));
  }

  TF_RETURN_IF_ERROR(ctx.status());
  return absl::OkStatus();
}

void GenericTfCallback(void* stream_handle, void** buffers, const char* opaque,
                       int opaque_len, XlaCustomCallStatus* status) {
  Status s = CallTfKernel(stream_handle, buffers, opaque, opaque_len);
  if (!s.ok()) {
    auto msg = s.message();
    XlaCustomCallStatusSetFailure(status, msg.data(), msg.size());
  }
}

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(kTfCallbackCustomCall,
                                         GenericTfCallback, "CUDA");

}  // namespace

LightOutsideCompilationOp::LightOutsideCompilationOp(
    OpKernelConstruction* context)
    : XlaOpKernel(context),
      def_(context->def()),
      graph_def_version_(context->graph_def_version()) {}

void LightOutsideCompilationOp::Compile(XlaOpKernelContext* ctx) {
  OP_REQUIRES_OK(
      ctx, CompileToCustomCallCallingTfKernel(graph_def_version_, def_, ctx));
}

}  // namespace tensorflow
