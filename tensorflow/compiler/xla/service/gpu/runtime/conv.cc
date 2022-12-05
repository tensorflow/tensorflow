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

#include <optional>
#include <utility>
#include <vector>

#include "llvm/ADT/Sequence.h"
#include "tensorflow/compiler/xla/mlir/runtime/transforms/custom_call_encoding.h"
#include "tensorflow/compiler/xla/runtime/custom_call.h"
#include "tensorflow/compiler/xla/runtime/executable.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_runner.h"
#include "tensorflow/compiler/xla/service/gpu/runtime/support.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/translate/mhlo_to_hlo/attribute_exporter.h"

namespace xla {
namespace gpu {

using xla::runtime::CustomCall;
using xla::runtime::Executable;
using xla::runtime::FlatMemrefView;
using xla::runtime::StridedMemrefView;

using llvm::ArrayRef;
using mlir::StringRef;

namespace lmhlo_gpu = ::mlir::lmhlo_gpu;
namespace mhlo = ::mlir::mhlo;

// TODO(jacksonstokes): Add caching layer for convolution configs and runners.

// TODO(ezhulenev): We need to find a better way to pass structured attributes
// to JitRt custom calls.

namespace {

struct Window {
  ArrayRef<int64_t> window_strides;
  ArrayRef<int64_t> padding;
  ArrayRef<int64_t> lhs_dilation;
  ArrayRef<int64_t> rhs_dilation;
  ArrayRef<int64_t> window_reversal;
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

absl::StatusOr<ConvRunnerCache::Entry> ConvRunnerCache::GetOrCreate(
    Key key, absl::FunctionRef<absl::StatusOr<GpuConvConfig>()> config) {
  absl::MutexLock lock(&mutex_);
  auto it = runners_.find(key);
  if (it != runners_.end()) return Entry{&it->second.first, &it->second.second};

  absl::StatusOr<GpuConvConfig> cfg = config();
  if (!cfg.ok()) return cfg.status();

  auto emplaced = runners_.try_emplace(key, *cfg, *cfg);
  return Entry{&emplaced.first->second.first, &emplaced.first->second.second};
}

static GpuConvDescriptor GetConvDescriptor(
    CudnnConvKind kind,
    // Arguments
    runtime::StridedMemrefView operand0, runtime::StridedMemrefView operand1,
    runtime::StridedMemrefView output, runtime::FlatMemrefView scratch,
    // Attributes
    ConvDimensionNumbers dims, Window w, ConvBackendConfig b, ConvAttrs attrs,
    // Conv-specific arguments and attributes
    std::optional<FusedConvAttrs> fused = std::nullopt,
    std::optional<SideInputAttrs> side_input = std::nullopt) {
  // Build a convolution descriptor from the attributes.
  GpuConvDescriptor descriptor;
  descriptor.kind = kind;

  // Apply backend config layout to the shape.
  auto apply_layout = [](runtime::StridedMemrefView& memref,
                         ArrayRef<int64_t> minor_to_major) {
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

namespace {
struct Conv {
  LLVM_ATTRIBUTE_ALWAYS_INLINE
  absl::Status operator()(
      const ServiceExecutableRunOptions* run_options,
      const DebugOptions* debug_options, runtime::StridedMemrefView operand0,
      runtime::StridedMemrefView operand1,
      std::optional<runtime::FlatMemrefView> bias,
      std::optional<runtime::StridedMemrefView> side_input,
      runtime::StridedMemrefView output, runtime::FlatMemrefView scratch,
      int64_t uid, ConvRunnerCache* runners, ConvDimensionNumbers conv_dims,
      // Window config
      ArrayRef<int64_t> window_strides, ArrayRef<int64_t> padding,
      ArrayRef<int64_t> lhs_dilation, ArrayRef<int64_t> rhs_dilation,
      ArrayRef<int64_t> window_reversal,
      // Backend config attributes
      ConvBackendConfig backend_config,
      // Remaining attributes
      int64_t feature_group_count, double result_scale,
      // Optional attributes for fused convolutions.
      std::optional<se::dnn::ActivationMode> activation_mode = std::nullopt,
      std::optional<double> side_input_scale = std::nullopt) const {
    // Build config for optional attributes.
    std::optional<FusedConvAttrs> fused_attrs = std::nullopt;
    if (activation_mode.has_value()) fused_attrs = {*activation_mode};

    std::optional<SideInputAttrs> side_input_attrs = std::nullopt;
    if (side_input_scale.has_value()) side_input_attrs = {*side_input_scale};

    // Get the convolution runner from the cache.
    absl::StatusOr<ConvRunnerCache::Entry> runner = runners->GetOrCreate(
        {run_options->stream(), uid}, [&]() -> absl::StatusOr<GpuConvConfig> {
          GpuConvDescriptor descriptor = GetConvDescriptor(
              kind, operand0, operand1, output, scratch, conv_dims,
              {window_strides, padding, lhs_dilation, rhs_dilation,
               window_reversal},
              backend_config, {feature_group_count, result_scale}, fused_attrs,
              side_input_attrs);

          StatusOr<GpuConvConfig> conv_config =
              GetGpuConvConfig(descriptor, "");
          if (!conv_config.ok()) return ToAbslStatus(conv_config.status());

          return *conv_config;
        });
    if (!runner.ok()) return runner.status();

    // Prepare buffer arguments.
    std::vector<se::DeviceMemoryBase> buffers = {GetDeviceAddress(operand0),
                                                 GetDeviceAddress(operand1)};
    if (bias.has_value()) buffers.push_back(GetDeviceAddress(*bias));
    if (side_input.has_value())
      buffers.push_back(GetDeviceAddress(*side_input));

    se::DeviceMemoryBase result_buffer = GetDeviceAddress(output);
    se::DeviceMemoryBase scratch_buffer = GetDeviceAddress(scratch);

    RunConvOptions opts;
    opts.runner_cache = runner->runner;

    // Run the convolution.
    auto st = RunGpuConv(*runner->config, buffers, result_buffer,
                         scratch_buffer, run_options->stream(), opts);
    if (!st.ok() || !run_options->stream()->ok()) {
      return ToAbslStatus(st);
    }

    return absl::OkStatus();
  }

  static Conv Handler(CudnnConvKind kind) { return Conv{kind}; }

  CudnnConvKind kind;
};

}  // namespace

// Adds custom call bindings for convolution operations.
template <typename... Ts>
static auto BindConvAttributes(runtime::CustomCallBinding<Ts...> binding) {
  return std::move(binding)
      // Unique convolution id for caching state.
      .template Attr<int64_t>("uid")
      .template UserData<ConvRunnerCache*>()
      // Convolution dimensions numbers
      .template Attr<ConvDimensionNumbers>("conv_dims")
      // Window config
      .template Attr<ArrayRef<int64_t>>("window_strides")
      .template Attr<ArrayRef<int64_t>>("padding")
      .template Attr<ArrayRef<int64_t>>("lhs_dilation")
      .template Attr<ArrayRef<int64_t>>("rhs_dilation")
      .template Attr<ArrayRef<int64_t>>("window_reversal")
      // Backend config attributes
      .template Attr<ConvBackendConfig>("backend_config")
      // Remaining attributes.
      .template Attr<int64_t>("feature_group_count")
      .template Attr<double>("result_scale");
}

template <CudnnConvKind kind>
static bool ConvFn(runtime::ExecutionContext* ctx, void** args, void** attrs,
                   void** rets) {
  static auto* handler =
      BindConvAttributes(
          CustomCall::Bind("xla.gpu.conv")
              .UserData<const ServiceExecutableRunOptions*>()
              .UserData<const DebugOptions*>()
              .Arg<runtime::StridedMemrefView>()                   // operand0
              .Arg<runtime::StridedMemrefView>()                   // operand1
              .Value(std::optional<runtime::FlatMemrefView>())     // bias
              .Value(std::optional<runtime::StridedMemrefView>())  // side_input
              .Arg<runtime::StridedMemrefView>()                   // output
              .Arg<runtime::FlatMemrefView>()                      // scratch
          )
          .To<checks>(Conv::Handler(kind))
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

template <CudnnConvKind kind>
static bool ConvFusedFn(runtime::ExecutionContext* ctx, void** args,
                        void** attrs, void** rets) {
  static auto* handler =
      BindConvAttributes(
          CustomCall::Bind("xla.gpu.conv.fused")
              .UserData<const ServiceExecutableRunOptions*>()
              .UserData<const DebugOptions*>()
              .Arg<runtime::StridedMemrefView>()                   // operand0
              .Arg<runtime::StridedMemrefView>()                   // operand1
              .Arg<runtime::FlatMemrefView>()                      // bias
              .Value(std::optional<runtime::StridedMemrefView>())  // side_input
              .Arg<runtime::StridedMemrefView>()                   // output
              .Arg<runtime::FlatMemrefView>()                      // scratch
          )
          .Attr<se::dnn::ActivationMode>("activation_mode")
          .To<checks>(Conv::Handler(kind))
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

template <CudnnConvKind kind>
static bool ConvFuseSideInputdFn(runtime::ExecutionContext* ctx, void** args,
                                 void** attrs, void** rets) {
  static auto* handler =
      BindConvAttributes(CustomCall::Bind("xla.gpu.conv.fused.side_input")
                             .UserData<const ServiceExecutableRunOptions*>()
                             .UserData<const DebugOptions*>()
                             .Arg<runtime::StridedMemrefView>()  // operand0
                             .Arg<runtime::StridedMemrefView>()  // operand1
                             .Arg<runtime::FlatMemrefView>()     // bias
                             .Arg<runtime::StridedMemrefView>()  // side_input
                             .Arg<runtime::StridedMemrefView>()  // output
                             .Arg<runtime::FlatMemrefView>()     // scratch
                         )
          .Attr<se::dnn::ActivationMode>("activation_mode")
          .Attr<double>("side_input_scale")
          .To<checks>(Conv::Handler(kind))
          .release();

  return succeeded(Executable::Call(ctx, *handler, args, attrs, rets));
}

void PopulateConvAttrEncoding(runtime::CustomCallAttrEncodingSet& encoding) {
  encoding.Add<runtime::EnumAttrEncoding<lmhlo_gpu::ActivationAttr,
                                         lmhlo_gpu::Activation,
                                         se::dnn::ActivationMode>>(
      [](lmhlo_gpu::Activation value) -> se::dnn::ActivationMode {
        return ConvertConvActivationMode(value).value();
      });

  using ConvDimsAttr = mhlo::ConvDimensionNumbersAttr;
  encoding.Add<
      xla::runtime::AggregateAttrEncoding<ConvDimsAttr, ConvDimensionNumbers>>(
      encoding,
      xla::runtime::AggregateAttrDef<ConvDimsAttr>()
          .Add("input_batch_dim", &ConvDimsAttr::getInputBatchDimension)
          .Add("input_feature_dim", &ConvDimsAttr::getInputFeatureDimension)
          .Add("input_spatial_dims", &ConvDimsAttr::getInputSpatialDimensions)
          .Add("kernel_in_feature_dim",
               &ConvDimsAttr::getKernelInputFeatureDimension)
          .Add("kernel_out_feature_dim",
               &ConvDimsAttr::getKernelOutputFeatureDimension)
          .Add("kernel_spatial_dims", &ConvDimsAttr::getKernelSpatialDimensions)
          .Add("output_batch_dim", &ConvDimsAttr::getOutputBatchDimension)
          .Add("output_feature_dim", &ConvDimsAttr::getOutputFeatureDimension)
          .Add("output_spatial_dims",
               &ConvDimsAttr::getOutputSpatialDimensions));

  using ConvConfigAttr = lmhlo_gpu::ConvolutionBackendConfigAttr;
  encoding.Add<
      xla::runtime::AggregateAttrEncoding<ConvConfigAttr, ConvBackendConfig>>(
      encoding,
      xla::runtime::AggregateAttrDef<ConvConfigAttr>()
          .Add("algorithm", &ConvConfigAttr::getAlgorithm)
          .Add("tensor_ops_enabled", &ConvConfigAttr::getTensorOpsEnabled)
          .Add("is_cudnn_frontend", &ConvConfigAttr::getIsCudnnFrontend)
          .Add("knob_ids", &ConvConfigAttr::getKnobIds)
          .Add("knob_values", &ConvConfigAttr::getKnobValues)
          .Add("operand_0_layout", &ConvConfigAttr::getOperand_0Layout)
          .Add("operand_1_layout", &ConvConfigAttr::getOperand_1Layout)
          .Add("result_layout", &ConvConfigAttr::getResultLayout)
          .Add("workspace_size", &ConvConfigAttr::getWorkspaceSize));
}

void RegisterConvCustomCalls(runtime::DirectCustomCallRegistry& registry) {
  auto conv = [](StringRef name) { return ("xla.gpu.conv." + name).str(); };
  registry.Register(conv("forward"), &ConvFn<CudnnConvKind::kForward>);
  registry.Register(conv("backward.input"),
                    &ConvFn<CudnnConvKind::kBackwardInput>);
  registry.Register(conv("backward.filter"),
                    &ConvFn<CudnnConvKind::kBackwardFilter>);
  registry.Register(conv("forward.fused"),
                    &ConvFusedFn<CudnnConvKind::kForwardActivation>);
  registry.Register(conv("forward.fused.side_input"),
                    &ConvFuseSideInputdFn<CudnnConvKind::kForwardActivation>);
}

}  // namespace gpu
}  // namespace xla
