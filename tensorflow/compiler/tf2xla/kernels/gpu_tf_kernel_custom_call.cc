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

#include "tensorflow/compiler/tf2xla/kernels/gpu_tf_kernel_custom_call.h"

#include <algorithm>
#include <deque>
#include <memory>
#include <numeric>
#include <string>
#include <utility>

#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/kernels/callback.pb.h"
#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/service/custom_call_status.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/common_runtime/gpu/gpu_cudamalloc_allocator.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/common_runtime/gpu/gpu_device.h"
#include "tensorflow/stream_executor/gpu/gpu_executor.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/stream_executor/gpu/gpu_types.h"
#endif

#include "tensorflow/core/common_runtime/gpu/gpu_process_state.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {

namespace {

const char* const kTfCallbackCustomCall = "GenericTfCallbackGPU";

}

static StatusOr<Tensor> TensorFromProto(const TensorProto& proto) {
  Tensor out;
  if (!out.FromProto(proto)) {
    return se::port::InternalError("Failed deserializing a TensorProto");
  }
  return out;
}

static StatusOr<TfCallbackData> CallbackDataFromProto(const char* opaque,
                                                      size_t opaque_len) {
  TfCallbackData callback_data;
  absl::string_view data{opaque, opaque_len};
  callback_data.ParseFromString(std::string{data});  // NOLINT: OSS req
  return callback_data;
}

static StatusOr<std::string> SerializeCallbackData(const TfCallbackData& data) {
  return data.SerializeAsString();
}

Status CallTfKernelOp::CompileToCustomCallCallingTfKernel(
    int graph_def_version, const NodeDef& node_def,
    XlaOpKernelContext* ctx) const {
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
      return se::port::InternalError(
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
      return se::port::InternalError(
          absl::StrCat("Output ", i, " has unknown rank"));
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
          return se::port::InternalError(absl::StrCat(
              "Bound for unknown dimension not found for dimension ", d));
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

  TF_ASSIGN_OR_RETURN(std::string callback_data_serialized,
                      SerializeCallbackData(callback_data));

  xla::XlaOp out = xla::CustomCallWithLayout(
      ctx->builder(), kTfCallbackCustomCall, operands, output_shape,
      /*operand_shapes_with_layout=*/operand_shapes_with_layout,
      /*opaque=*/callback_data_serialized,
      /*has_side_effect=*/false,
      /*output_operand_aliasing=*/{},
      /*literal=*/nullptr, xla::CustomCallSchedule::SCHEDULE_NONE,
      xla::CustomCallApiVersion::API_VERSION_STATUS_RETURNING);

  for (int i = 0; i < num_outputs; ++i) {
    ctx->SetOutput(i,
                   output_shape.IsTuple() ? xla::GetTupleElement(out, i) : out);
  }

  return OkStatus();
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
  TensorShape shape;
  CHECK(TensorShape::BuildTensorShape(descr.shape(), &shape).ok());  // Crash OK
  return shape.num_elements() * DataTypeSize(descr.type());
}

class TfCallbackDevice : public DeviceBase {
 public:
  explicit TfCallbackDevice(se::Stream* stream, void** buffers,
                            const TfCallbackData& callback_data)
      : DeviceBase(Env::Default()),
        stream_(stream),
        gpu_allocator_(GPUProcessState::singleton()->GetGPUAllocator(
            TfDeviceId{stream_->parent()->device_ordinal()})),
        cpu_allocator_(
            ProcessState::singleton()->GetCPUAllocator(/*numa_node=*/0)) {
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
    const void* gpu_stream = reinterpret_cast<const void*>(
        stream_->implementation()->GpuStreamMemberHack());
    concrete_device->Reinitialize(
        context, gpu_stream,
        /*platform_device_id=*/
        PlatformDeviceId(stream_->parent()->device_ordinal()), allocator,
        // TODO(cheshire): Pass meaningful scratch
        // buffer.
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
        return ps->GetGpuHostAllocator(0);
      } else {
        return cpu_allocator_;
      }
    } else {
      return gpu_allocator_;
    }
  }

 private:
  std::vector<WriteIntoXlaBufferAllocator> allocators_;
  se::Stream* stream_;  // NOLINT (used under GOOGLE_CUDA)
  Allocator* gpu_allocator_;
  Allocator* cpu_allocator_;
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
      stream->ThenMemcpy(&m, shape_info.data(),
                         num_dimensions * sizeof(int32_t));
    }
  }
  return OkStatus();
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
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      se::MultiPlatformManager::PlatformWithName("CUDA"));
  se::StreamExecutorConfig config;
  config.gpu_stream = stream_handle;
  TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                      platform->GetExecutor(config));
  se::Stream* stream = executor->FindAllocatedStream(stream_handle);
  if (!stream) {
    return xla::InternalError("Stream not found for %p", stream_handle);
  }
  TF_ASSIGN_OR_RETURN(TfCallbackData callback_data,
                      CallbackDataFromProto(opaque, opaque_len));
  TfCallbackDevice device(stream, buffers, callback_data);

  std::vector<AllocatorAttributes> allocator_attributes;
  for (int output_idx = 0; output_idx < callback_data.outputs_size();
       ++output_idx) {
    AllocatorAttributes attr;
    // Repurpose `scope_id` to communicate which output is it.
    // Shift by one to make it greater than zero.
    attr.scope_id = output_idx + 1;
    allocator_attributes.push_back(attr);
  }

  Status nested_status;
  std::unique_ptr<OpKernel> kernel =
      CreateOpKernel(DeviceType(DEVICE_GPU),
                     /*device=*/&device,

                     // NB: real allocator is passed with device, the one here
                     // is only called during the kernel construction.
                     // TODO(cheshire): Pass scratch allocator.
                     /*allocator=*/nullptr, callback_data.op(),
                     /*graph_def_version=*/1, &nested_status);
  TF_RETURN_IF_ERROR(nested_status);

  auto device_context =
      core::RefCountPtr<FakeDeviceContext>(new FakeDeviceContext(stream));

  OpKernelContext::Params params;
  params.output_attr_array = allocator_attributes.data();
  params.op_kernel = kernel.get();
  params.device = &device;
  params.ensure_eigen_gpu_device();
  params.op_device_context = device_context.get();

  absl::InlinedVector<TensorValue, 4> inputs;

  // Deque usage is important to avoid moving objects.
  std::deque<WriteIntoXlaBufferAllocator> input_allocators;
  std::deque<Tensor> input_tensors;

  int constant_offset = 0;

  for (int i = 0; i < callback_data.inputs_size(); ++i) {
    DataType dt = callback_data.inputs(i).buffer_description().type();

    TensorShape shape;
    TF_RETURN_IF_ERROR(TensorShape::BuildTensorShape(
        callback_data.inputs(i).buffer_description().shape(), &shape));

    VLOG(2) << "Input shape: " << shape.DebugString();
    int64_t input_size = shape.num_elements() * DataTypeSize(dt);

    if (callback_data.inputs(i).has_value()) {
      // Value provided at compile time: reconstruct the tensor.
      TF_ASSIGN_OR_RETURN(Tensor input,
                          TensorFromProto(callback_data.inputs(i).value()));
      input_tensors.push_back(input);

      constant_offset++;
      VLOG(1) << "Input " << i << " is a tensor: " << input.DebugString();
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

  params.inputs = inputs;
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
  return OkStatus();
}

void GenericTfCallback(void* stream_handle, void** buffers, const char* opaque,
                       int opaque_len, XlaCustomCallStatus* status) {
  Status s = CallTfKernel(stream_handle, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, s.error_message().c_str(),
                                  s.error_message().size());
  }
}

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM(kTfCallbackCustomCall,
                                         GenericTfCallback, "CUDA");

}  // namespace

CallTfKernelOp::CallTfKernelOp(OpKernelConstruction* context)
    : XlaOpKernel(context),
      def_(context->def()),
      graph_def_version_(context->graph_def_version()) {}

void CallTfKernelOp::Compile(XlaOpKernelContext* ctx) {
  OP_REQUIRES_OK(
      ctx, CompileToCustomCallCallingTfKernel(graph_def_version_, def_, ctx));
}

}  // namespace tensorflow
