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

#include "tensorflow/compiler/xla/service/gpu/runtime/conv.h"

#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "llvm/ADT/Sequence.h"
#include "tensorflow/compiler/xla/mlir/runtime/transforms/custom_call_encoding.h"
#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_asm_opts_util.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_algorithm_picker.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_runner.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_serializable_autotuner.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory_allocator.h"
#include "tensorflow/compiler/xla/translate/mhlo_to_hlo/attribute_exporter.h"
#include "tensorflow/compiler/xla/xla.pb.h"

#if GOOGLE_CUDA
#include "tensorflow/compiler/xla/stream_executor/gpu/redzone_allocator.h"
#endif

namespace xla {

using tensorflow::AutotuneResult;
using xla::runtime::AggregateAttrDef;
using xla::runtime::AggregateAttrEncoding;
using xla::runtime::CustomCall;
using xla::runtime::EnumAttrEncoding;
using xla::runtime::FlatMemrefView;
using xla::runtime::State;
using xla::runtime::StridedMemrefView;
using xla::runtime::Tagged;

namespace lmhlo_gpu = ::mlir::lmhlo_gpu;
namespace mhlo = ::mlir::mhlo;

//===----------------------------------------------------------------------===//
// Structs for encoding convolution attributes defined in MHLO dialect.
//===----------------------------------------------------------------------===//

namespace gpu {

struct ConvDimensionNumbers {
  int64_t input_batch_dim;
  int64_t input_feature_dim;
  absl::Span<const int64_t> input_spatial_dims;

  int64_t kernel_in_feature_dim;
  int64_t kernel_out_feature_dim;
  absl::Span<const int64_t> kernel_spatial_dims;

  int64_t output_batch_dim;
  int64_t output_feature_dim;
  absl::Span<const int64_t> output_spatial_dims;
};

struct ConvBackendConfig {
  int64_t algorithm;
  bool tensor_ops_enabled;
  bool is_cudnn_frontend;
  bool is_cudnn_reordered_int8;
  absl::Span<const int64_t> knob_ids;
  absl::Span<const int64_t> knob_values;
  absl::Span<const int64_t> operand_0_layout;
  absl::Span<const int64_t> operand_1_layout;
  absl::Span<const int64_t> result_layout;
  int64_t workspace_size;
};

}  // namespace gpu

//===----------------------------------------------------------------------===//
// Register convolution attributes decoding with the Xla runtime.
//===----------------------------------------------------------------------===//

namespace runtime {

XLA_RUNTIME_REGISTER_ENUM_ATTR_DECODING(se::dnn::ActivationMode);

XLA_RUNTIME_REGISTER_AGGREGATE_ATTR_DECODING(
    xla::gpu::ConvDimensionNumbers,
    // --- input dimensions
    AggregateMember<int64_t>("input_batch_dim"),
    AggregateMember<int64_t>("input_feature_dim"),
    AggregateMember<absl::Span<const int64_t>>("input_spatial_dims"),
    // --- kernel dimensions
    AggregateMember<int64_t>("kernel_in_feature_dim"),
    AggregateMember<int64_t>("kernel_out_feature_dim"),
    AggregateMember<absl::Span<const int64_t>>("kernel_spatial_dims"),
    // --- output dimensions
    AggregateMember<int64_t>("output_batch_dim"),
    AggregateMember<int64_t>("output_feature_dim"),
    AggregateMember<absl::Span<const int64_t>>("output_spatial_dims"));

XLA_RUNTIME_REGISTER_AGGREGATE_ATTR_DECODING(
    xla::gpu::ConvBackendConfig,  //
    AggregateMember<int64_t>("algorithm"),
    AggregateMember<bool>("tensor_ops_enabled"),
    AggregateMember<bool>("is_cudnn_frontend"),
    AggregateMember<bool>("is_cudnn_reordered_int8"),
    AggregateMember<absl::Span<const int64_t>>("knob_ids"),
    AggregateMember<absl::Span<const int64_t>>("knob_values"),
    AggregateMember<absl::Span<const int64_t>>("operand_0_layout"),
    AggregateMember<absl::Span<const int64_t>>("operand_1_layout"),
    AggregateMember<absl::Span<const int64_t>>("result_layout"),
    AggregateMember<int64_t>("workspace_size"));

}  // namespace runtime

//===----------------------------------------------------------------------===//
// Type names for encoded attributes.
//===----------------------------------------------------------------------===//

namespace gpu {

void RegisterConvTypeIdNames(runtime::TypeIDNameRegistry& registry) {
  registry.Register<Tagged<ConvDimensionNumbers>>("__type_id_conv_dim_numbers");
  registry.Register<Tagged<ConvBackendConfig>>("__type_id_conv_backend_config");
}

//===----------------------------------------------------------------------===//
// Encoding from MHLO attributes to Xla runtime aggregate attributes.
//===----------------------------------------------------------------------===//

// TODO(ezhulenev): We have to support enum encoding that can fail instead of
// always getting the value from returned StatusOr.
static auto EncodeConvActivation(lmhlo_gpu::Activation activation) {
  return ConvertConvActivationMode(activation).value();
}

void PopulateConvAttrEncoding(runtime::CustomCallAttrEncodingSet& encoding) {
  {  // --- Encode `lmhlo_gpu::ActivationAttr`.
    encoding
        .Add<EnumAttrEncoding<lmhlo_gpu::ActivationAttr, lmhlo_gpu::Activation,
                              se::dnn::ActivationMode>>(EncodeConvActivation);
  }

  {  // --- Encode `mhlo::ConvDimensionNumbersAttr`.
    using Attr = mhlo::ConvDimensionNumbersAttr;
    encoding.Add<AggregateAttrEncoding<Attr, ConvDimensionNumbers>>(
        encoding,
        AggregateAttrDef<Attr>()
            .Add("input_batch_dim", &Attr::getInputBatchDimension)
            .Add("input_feature_dim", &Attr::getInputFeatureDimension)
            .Add("input_spatial_dims", &Attr::getInputSpatialDimensions)
            .Add("kernel_in_feature_dim", &Attr::getKernelInputFeatureDimension)
            .Add("kernel_out_feature_dim",
                 &Attr::getKernelOutputFeatureDimension)
            .Add("kernel_spatial_dims", &Attr::getKernelSpatialDimensions)
            .Add("output_batch_dim", &Attr::getOutputBatchDimension)
            .Add("output_feature_dim", &Attr::getOutputFeatureDimension)
            .Add("output_spatial_dims", &Attr::getOutputSpatialDimensions));
  }

  {  // --- Encode `lmhlo_gpu::ConvolutionBackendConfigAttr`.
    using Attr = lmhlo_gpu::ConvolutionBackendConfigAttr;
    encoding.Add<AggregateAttrEncoding<Attr, ConvBackendConfig>>(
        encoding,
        AggregateAttrDef<Attr>()
            .Add("algorithm", &Attr::getAlgorithm)
            .Add("tensor_ops_enabled", &Attr::getTensorOpsEnabled)
            .Add("is_cudnn_frontend", &Attr::getIsCudnnFrontend)
            .Add("is_cudnn_reordered_int8", &Attr::getIsCudnnReorderedInt8)
            .Add("knob_ids", &Attr::getKnobIds)
            .Add("knob_values", &Attr::getKnobValues)
            .Add("operand_0_layout", &Attr::getOperand_0Layout)
            .Add("operand_1_layout", &Attr::getOperand_1Layout)
            .Add("result_layout", &Attr::getResultLayout)
            .Add("workspace_size", &Attr::getWorkspaceSize));
  }
}

//===----------------------------------------------------------------------===//
// Convolution runners caching.
//===----------------------------------------------------------------------===//

StreamExecutorConvRunners* ConvRunners::operator()(
    se::StreamExecutor* executor) {
  absl::MutexLock lock(&mutex_);
  return &runners_[executor];
}

//===----------------------------------------------------------------------===//
// Convolution custom call implementation.
//===----------------------------------------------------------------------===//

namespace {

struct Window {
  absl::Span<const int64_t> window_strides;
  absl::Span<const int64_t> padding;
  absl::Span<const int64_t> lhs_dilation;
  absl::Span<const int64_t> rhs_dilation;
  absl::Span<const int64_t> window_reversal;
};

struct ConvAttrs {
  int64_t feature_group_count;
  double result_scale;
};

struct FusedConvAttrs {
  se::dnn::ActivationMode activation_mode;
};

struct SideInputAttrs {
  double side_input_scale;
};

}  // namespace

static GpuConvDescriptor GetConvDescriptor(
    CudnnConvKind kind,
    // Arguments
    StridedMemrefView operand0, StridedMemrefView operand1,
    StridedMemrefView output, FlatMemrefView scratch,
    // Attributes
    ConvDimensionNumbers dims, Window w, ConvBackendConfig b, ConvAttrs attrs,
    // Conv-specific arguments and attributes
    std::optional<FusedConvAttrs> fused = std::nullopt,
    std::optional<SideInputAttrs> side_input = std::nullopt) {
  // Build a convolution descriptor from the attributes.
  GpuConvDescriptor descriptor;
  descriptor.kind = kind;

  // Apply backend config layout to the shape.
  auto apply_layout = [](StridedMemrefView& memref,
                         absl::Span<const int64_t> minor_to_major) {
    Shape shape = ToShape(memref);
    return ShapeUtil::MakeShapeWithDenseLayout(
        shape.element_type(), shape.dimensions(), minor_to_major);
  };

  descriptor.operand0_shape = apply_layout(operand0, b.operand_0_layout);
  descriptor.operand1_shape = apply_layout(operand1, b.operand_1_layout);
  descriptor.result_shape = apply_layout(output, b.result_layout);

  // Set up convolution dimensions numbers.
  ConvolutionDimensionNumbers dns;
  dns.set_input_batch_dimension(dims.input_batch_dim);
  dns.set_input_feature_dimension(dims.input_feature_dim);
  dns.set_kernel_input_feature_dimension(dims.kernel_in_feature_dim);
  dns.set_kernel_output_feature_dimension(dims.kernel_out_feature_dim);
  dns.set_output_batch_dimension(dims.output_batch_dim);
  dns.set_output_feature_dimension(dims.output_feature_dim);
  for (int64_t d : dims.input_spatial_dims) dns.add_input_spatial_dimensions(d);
  for (int64_t d : dims.kernel_spatial_dims)
    dns.add_kernel_spatial_dimensions(d);
  for (int64_t d : dims.output_spatial_dims)
    dns.add_output_spatial_dimensions(d);
  descriptor.dnums = std::move(dns);

  // Put together convolution window config.
  for (auto index : llvm::seq<int>(0, w.window_strides.size())) {
    WindowDimension* dim = descriptor.window.add_dimensions();
    // Window size for a convolution is the same as the kernel size.
    // Kernel size of the convolution is operand1_shape. We need to look at
    // the convolution dimension numbers kernel spatial dimensions to get
    // the window size.
    int kernel_dim = descriptor.dnums.kernel_spatial_dimensions(index);
    dim->set_size(descriptor.operand0_shape.dimensions(kernel_dim));
    dim->set_stride(w.window_strides[index]);
    dim->set_padding_low(w.padding[index]);
    dim->set_padding_high(w.padding[index]);
    dim->set_base_dilation(w.lhs_dilation[index]);
    dim->set_window_dilation(w.rhs_dilation[index]);
    dim->set_window_reversal(w.window_reversal[index]);
  }

  descriptor.scratch_size = scratch.size_in_bytes;
  descriptor.feature_group_count = attrs.feature_group_count;
  descriptor.backend_config.set_conv_result_scale(attrs.result_scale);
  descriptor.backend_config.set_reordered_int8_nchw_vect(
      b.is_cudnn_reordered_int8);

  // Set up convolution algorigthm.
  auto* algo = descriptor.backend_config.mutable_algorithm();
  algo->set_algo_id(b.algorithm);
  algo->set_math_type(b.tensor_ops_enabled
                          ? se::dnn::AlgorithmProto::TENSOR_OP_MATH
                          : se::dnn::AlgorithmProto::DEFAULT_MATH);
  algo->set_is_cudnn_frontend(b.is_cudnn_frontend);

  if (b.workspace_size >= 0)
    algo->mutable_workspace_size()->set_value(b.workspace_size);

  for (unsigned i = 0; i < b.knob_ids.size(); ++i) {
    algo->mutable_tuning_knobs()->insert({b.knob_ids[i], b.knob_values[i]});
  }

  // Set attributes specific for fused convolutions.
  if (fused.has_value())
    descriptor.backend_config.set_activation_mode(fused->activation_mode);

  // Set attributes specific for convolutions with side input.
  if (side_input.has_value())
    descriptor.backend_config.set_side_input_scale(
        side_input->side_input_scale);

  return descriptor;
}

#if GOOGLE_CUDA
// Do runtime autotuning and set the picked algorithm to ConvRunner.
StatusOr<AutotuneResult> DoRuntimeAutotuning(
    ConvRunner* conv, se::DeviceMemoryBase& scratch_buffer,
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options,
    const std::vector<se::DeviceMemoryBase> buffers,
    const se::DeviceMemoryBase result_buffer) {
  GpuConvConfig conv_config = conv->config;
  Shape output_shape = conv_config.output_shape;
  HloModuleConfig hlo_module_config;
  se::Stream* stream = run_options->stream();
  se::StreamExecutor* stream_exec = stream->parent();
  se::DeviceMemoryAllocator* allocator = stream->parent()->GetAllocator();
  se::RedzoneAllocator input_output_allocator(
      stream, allocator, PtxOptsFromDebugOptions(*debug_options),
      /*memory_limit=*/std::numeric_limits<int64_t>::max(),
      se::RedzoneAllocator::kDefaultRedzoneSize);

  DeviceConfig device_config = {stream_exec, allocator};
  GpuConvAlgorithmPicker conv_algorithm_picker(device_config);

  GpuConvAlgorithmPicker::AutotuneRuntimeArguments autotune_runtime_arguments =
      {output_shape,  hlo_module_config,       buffers,
       result_buffer, &input_output_allocator, conv_config,
       std::nullopt};

  return conv_algorithm_picker.PickBestAlgorithmNoCacheCuda(
      /* instr */ nullptr, allocator, stream,
      /* instruction_info */ std::nullopt, autotune_runtime_arguments);
}
#endif

template <CudnnConvKind kind>
static absl::Status ConvImpl(
    const ServiceExecutableRunOptions* run_options,
    const DebugOptions* debug_options, State<ConvRunner> runner,
    // Arguments
    StridedMemrefView operand0, StridedMemrefView operand1,
    std::optional<FlatMemrefView> bias,
    std::optional<StridedMemrefView> side_input, StridedMemrefView output,
    FlatMemrefView scratch, int64_t uid,
    // Convolution config
    ConvDimensionNumbers conv_dims,
    // Window config
    absl::Span<const int64_t> window_strides, absl::Span<const int64_t> padding,
    absl::Span<const int64_t> lhs_dilation,
    absl::Span<const int64_t> rhs_dilation,
    absl::Span<const int64_t> window_reversal,
    // Backend config attributes
    ConvBackendConfig backend_config,
    // Remaining attributes
    int64_t feature_group_count, double result_scale,
    // Optional attributes for fused convolutions.
    std::optional<se::dnn::ActivationMode> activation_mode = std::nullopt,
    std::optional<double> side_input_scale = std::nullopt) {
  // Build config for optional attributes.
  std::optional<FusedConvAttrs> fused_attrs = std::nullopt;
  if (activation_mode.has_value()) fused_attrs = {*activation_mode};

  std::optional<SideInputAttrs> side_input_attrs = std::nullopt;
  if (side_input_scale.has_value()) side_input_attrs = {*side_input_scale};

  bool runtime_autotuning = false;
  if (backend_config.algorithm == -1) {
    // Set the algorithm back to the default algorithm to avoid error from
    // cuDNN.
    backend_config.algorithm = 0;
    runtime_autotuning = true;
  }

  // Get or create the convolution runner state.
  absl::StatusOr<ConvRunner*> conv =
      runner.GetOrCreate([&]() -> absl::StatusOr<ConvRunner> {
        GpuConvDescriptor descriptor = GetConvDescriptor(
            kind, operand0, operand1, output, scratch, conv_dims,
            {window_strides, padding, lhs_dilation, rhs_dilation,
             window_reversal},
            backend_config, {feature_group_count, result_scale}, fused_attrs,
            side_input_attrs);

        StatusOr<GpuConvConfig> conv_config = GetGpuConvConfig(descriptor, "");
        if (!conv_config.ok()) return ToAbslStatus(conv_config.status());

        return ConvRunner(*std::move(conv_config));
      });
  if (!conv.ok()) return conv.status();

  // Prepare buffer arguments.
  std::vector<se::DeviceMemoryBase> buffers = {GetDeviceAddress(operand0),
                                               GetDeviceAddress(operand1)};
  if (bias.has_value()) buffers.push_back(GetDeviceAddress(*bias));
  if (side_input.has_value()) buffers.push_back(GetDeviceAddress(*side_input));

  se::DeviceMemoryBase result_buffer = GetDeviceAddress(output);
  se::DeviceMemoryBase scratch_buffer = GetDeviceAddress(scratch);

  int64_t scratch_buffer_size = scratch_buffer.size();

  // Do runtime conv autotuning.
#if GOOGLE_CUDA
  if (runtime_autotuning) {
    auto autotune_result =
        DoRuntimeAutotuning(conv.value(), scratch_buffer, run_options,
                            debug_options, buffers, result_buffer);
    if (!autotune_result.ok()) return ToAbslStatus(autotune_result.status());

    // Set algorithm in the convolution runner state.
    AutotuneResult best_algo = autotune_result.value();
    se::dnn::AlgorithmDesc algo_desc(best_algo.conv().algorithm(),
                                     best_algo.conv().tensor_ops_enabled());
    (*conv)->config.algorithm = algo_desc;

    // Set scratch buffer size according to the selected algorithm.
    scratch_buffer_size = best_algo.scratch_bytes();
  }
#endif

  RunConvOptions opts;
  opts.runner_cache = &(*conv)->runner;

  if (scratch_buffer_size > scratch_buffer.size()) {
    // Need to reallocate scratch buffer.
    auto stream_exec = run_options->stream()->parent();
    auto allocator = stream_exec->GetAllocator();
    StatusOr<se::OwningDeviceMemory> allocated_buffer =
        allocator->Allocate(stream_exec->device_ordinal(), scratch_buffer_size);
    if (!allocated_buffer.ok()) return ToAbslStatus(allocated_buffer.status());
    se::DeviceMemoryBase new_scratch_buffer(allocated_buffer->ptr(),
                                            scratch_buffer_size);

    // Run the convolution using the new scratch buffer.
    auto st = RunGpuConv((*conv)->config, buffers, result_buffer,
                         new_scratch_buffer, run_options->stream(), opts);
    if (!st.ok() || !run_options->stream()->ok()) {
      return ToAbslStatus(st);
    }
    return absl::OkStatus();
  }

  // Run the convolution.
  auto st = RunGpuConv((*conv)->config, buffers, result_buffer, scratch_buffer,
                       run_options->stream(), opts);
  if (!st.ok() || !run_options->stream()->ok()) {
    return ToAbslStatus(st);
  }

  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// Convolution custom calls bindings and registration.
//===----------------------------------------------------------------------===//

using Kind = CudnnConvKind;

template <typename... Ts>
static auto BindConvAttributes(runtime::CustomCallBinding<Ts...> binding) {
  return std::move(binding)
      // Unique convolution id for caching state.
      .template Attr<int64_t>("uid")
      // Convolution dimensions numbers
      .template Attr<ConvDimensionNumbers>("conv_dims")
      // Window config
      .template Attr<absl::Span<const int64_t>>("window_strides")
      .template Attr<absl::Span<const int64_t>>("padding")
      .template Attr<absl::Span<const int64_t>>("lhs_dilation")
      .template Attr<absl::Span<const int64_t>>("rhs_dilation")
      .template Attr<absl::Span<const int64_t>>("window_reversal")
      // Backend config attributes
      .template Attr<ConvBackendConfig>("backend_config")
      // Remaining attributes.
      .template Attr<int64_t>("feature_group_count")
      .template Attr<double>("result_scale");
}

XLA_RUNTIME_DEFINE_CUSTOM_CALL_TEMPLATE(
    Kind kind, Conv, FunctionWrapper<ConvImpl<kind>>(), checks,
    BindConvAttributes(
        CustomCall::Bind("xla.gpu.conv")
            .UserData<const ServiceExecutableRunOptions*>()
            .UserData<const DebugOptions*>()
            .State<ConvRunner>("uid")                   // runner
            .Arg<StridedMemrefView>()                   // operand0
            .Arg<StridedMemrefView>()                   // operand1
            .Value(std::optional<FlatMemrefView>())     // bias
            .Value(std::optional<StridedMemrefView>())  // side_input
            .Arg<StridedMemrefView>()                   // output
            .Arg<FlatMemrefView>()                      // scratch
        )
        .Value(std::optional<se::dnn::ActivationMode>())  // activation_mode
        .Value(std::optional<double>())                   // side_input_scale
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    ConvFused, FunctionWrapper<ConvImpl<Kind::kForwardActivation>>(), checks,
    BindConvAttributes(
        CustomCall::Bind("xla.gpu.conv.fused")
            .UserData<const ServiceExecutableRunOptions*>()
            .UserData<const DebugOptions*>()
            .State<ConvRunner>("uid")                   // runner
            .Arg<StridedMemrefView>()                   // operand0
            .Arg<StridedMemrefView>()                   // operand1
            .Arg<FlatMemrefView>()                      // bias
            .Value(std::optional<StridedMemrefView>())  // side_input
            .Arg<StridedMemrefView>()                   // output
            .Arg<FlatMemrefView>()                      // scratch
        )
        .Attr<se::dnn::ActivationMode>("activation_mode")
        .Value(std::optional<double>())  // side_input_scale
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    ConvFusedSideInput, FunctionWrapper<ConvImpl<Kind::kForwardActivation>>(),
    checks,
    BindConvAttributes(CustomCall::Bind("xla.gpu.conv.fused.side_input")
                           .UserData<const ServiceExecutableRunOptions*>()
                           .UserData<const DebugOptions*>()
                           .State<ConvRunner>("uid")  // runner
                           .Arg<StridedMemrefView>()  // operand0
                           .Arg<StridedMemrefView>()  // operand1
                           .Arg<FlatMemrefView>()     // bias
                           .Arg<StridedMemrefView>()  // side_input
                           .Arg<StridedMemrefView>()  // output
                           .Arg<FlatMemrefView>()     // scratch
                       )
        .Attr<se::dnn::ActivationMode>("activation_mode")
        .Attr<double>("side_input_scale"));

//===----------------------------------------------------------------------===//

void RegisterConvCustomCalls(runtime::DirectCustomCallRegistry& registry) {
  auto conv = [](std::string name) { return "xla.gpu.conv." + name; };
  registry.Register(conv("forward"), Conv<Kind::kForward>);
  registry.Register(conv("backward.input"), Conv<Kind::kBackwardInput>);
  registry.Register(conv("backward.filter"), Conv<Kind::kBackwardFilter>);
  registry.Register(conv("forward.fused"), ConvFused);
  registry.Register(conv("forward.fused.side_input"), ConvFusedSideInput);
}

}  // namespace gpu
}  // namespace xla
