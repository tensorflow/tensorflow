// Copyright 2021 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Pattern to lower lmhlo convolution ops to tfrt_gpu dialect.
#include <sys/types.h>

#include <functional>

#include "mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/pattern_utils.h"
#include "tensorflow/compiler/mlir/xla/attribute_exporter.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_conv_runner.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_type_conversion_util.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime
#include "tfrt/gpu/wrapper/cudnn_wrapper.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime

namespace tensorflow {
namespace {

template <class ConvolutionOpType>
void FillConvDescriptor(ConvolutionOpType op, Value result,
                        xla::gpu::GpuConvDescriptor& descriptor) {
  auto apply_layout = [](const xla::Shape& shape,
                         mlir::ArrayAttr layout_attrib) {
    mlir::SmallVector<int64_t, 4> minor_to_major = llvm::to_vector<4>(
        llvm::map_range(layout_attrib, [](mlir::Attribute a) -> int64_t {
          return static_cast<int64_t>(a.cast<mlir::IntegerAttr>().getInt());
        }));
    return xla::ShapeUtil::MakeShapeWithLayout(
        shape.element_type(), shape.dimensions(), minor_to_major);
  };

  descriptor.operand0_shape =
      apply_layout(xla::gpu::GetShape(op->getOperand(0)),
                   op.backend_config().operand_0_layout());
  descriptor.operand1_shape =
      apply_layout(xla::gpu::GetShape(op->getOperand(1)),
                   op.backend_config().operand_1_layout());
  descriptor.result_shape = apply_layout(xla::gpu::GetShape(result),
                                         op.backend_config().result_layout());
  descriptor.dnums = xla::ConvertConvDimensionNumbers(op.dimension_numbers());
  descriptor.scratch_size = 0;  // Not used for op lowering.
  mlir::DenseIntElementsAttr window_strides = op.window_strides().getValue();
  mlir::DenseIntElementsAttr lhs_dilation = op.lhs_dilation().getValue();
  mlir::DenseIntElementsAttr rhs_dilation = op.rhs_dilation().getValue();
  mlir::DenseElementsAttr window_reversal = op.window_reversal().getValue();
  for (auto index : llvm::seq<int>(0, window_strides.getNumElements())) {
    xla::WindowDimension* dim = descriptor.window.add_dimensions();
    // Window size for a convolution is the same as the kernel size.
    // Kernel size of the convolution is operand1_shape. We need to look at
    // the convolution dimension numbers kernel spatial dimensions to get
    // the window size.
    int kernel_dim = descriptor.dnums.kernel_spatial_dimensions(index);
    dim->set_size(descriptor.operand0_shape.dimensions(kernel_dim));
    dim->set_stride(window_strides.getValues<int64_t>()[index]);
    dim->set_base_dilation(lhs_dilation.getValues<int64_t>()[index]);
    dim->set_window_dilation(rhs_dilation.getValues<int64_t>()[index]);
    dim->set_window_reversal(window_reversal.getValues<bool>()[index]);
    if (op.padding().hasValue()) {
      mlir::DenseIntElementsAttr padding = op.padding().getValue();
      dim->set_padding_low(padding.getValues<int64_t>()[index]);
      dim->set_padding_high(padding.getValues<int64_t>()[index]);
    }
  }
  descriptor.feature_group_count = op.feature_group_count();
  {
    auto* algorithm = descriptor.backend_config.mutable_algorithm();
    algorithm->set_algo_id(op.backend_config().algorithm().getInt());
    algorithm->set_math_type(op.backend_config().tensor_ops_enabled().getValue()
                                 ? se::dnn::AlgorithmProto::TENSOR_OP_MATH
                                 : se::dnn::AlgorithmProto::DEFAULT_MATH);
    for (int i = 0; i < op.backend_config().knob_ids().size(); ++i) {
      // N.B. tuning_knobs is a map rather than a repeated field, so this
      // doesn't require reserving space up front.
      auto knob_id = op.backend_config()
                         .knob_ids()[i]
                         .template cast<mlir::IntegerAttr>()
                         .getInt();
      auto knob_value = op.backend_config()
                            .knob_values()[i]
                            .template cast<mlir::IntegerAttr>()
                            .getInt();
      (*algorithm->mutable_tuning_knobs())[knob_id] = knob_value;
    }
    algorithm->set_is_cudnn_frontend(
        op.backend_config().is_cudnn_frontend().getValue());
    auto workspace_size = op.backend_config().workspace_size().getInt();
    if (workspace_size >= 0) {
      algorithm->mutable_workspace_size()->set_value(workspace_size);
    }
  }
  descriptor.backend_config.set_conv_result_scale(
      op.result_scale().convertToDouble());
}

Value CreateBuildUnfusedConvOp(Value input, Value output, Value handle,
                               mlir::Location loc,
                               const xla::gpu::GpuConvConfig& config,
                               cudnnBackendDescriptorType_t backend_type,
                               ConversionPatternRewriter& rewriter) {
  auto get_element_type = [](Value value) {
    return value.getType().cast<mlir::MemRefType>().getElementType();
  };
  cudnnDataType_t input_type = MlirTypeToCudnnDataType(
      get_element_type(input), config.input_descriptor.layout());
  cudnnDataType_t output_type = MlirTypeToCudnnDataType(
      get_element_type(output), config.output_descriptor.layout());

  int vector_size, vector_dim;
  std::tie(vector_size, vector_dim) =
      tfrt::gpu::wrapper::GetTensorVectorizedSizeAndDim(input_type);
  std::vector<int64_t> input_dims = config.input_descriptor.vectorized_dims(
      se::dnn::DataLayout::kBatchDepthYX, vector_size, vector_dim);
  std::vector<int64_t> input_strides =
      config.input_descriptor.vectorized_strides(
          se::dnn::DataLayout::kBatchDepthYX, vector_size, vector_dim);

  std::tie(vector_size, vector_dim) =
      tfrt::gpu::wrapper::GetTensorVectorizedSizeAndDim(output_type);
  std::vector<int64_t> output_dims = config.output_descriptor.vectorized_dims(
      se::dnn::DataLayout::kBatchDepthYX, vector_size, vector_dim);
  std::vector<int64_t> output_strides =
      config.output_descriptor.vectorized_strides(
          se::dnn::DataLayout::kBatchDepthYX, vector_size, vector_dim);

  std::tie(vector_size, vector_dim) =
      tfrt::gpu::wrapper::GetTensorVectorizedSizeAndDim(input_type);
  std::vector<int64_t> filter_dims = config.filter_descriptor.vectorized_dims(
      se::dnn::FilterLayout::kOutputInputYX, vector_size, vector_dim);
  std::vector<int64_t> filter_strides =
      config.filter_descriptor.vectorized_strides(
          se::dnn::FilterLayout::kOutputInputYX, vector_size, vector_dim);

  const auto* conv_desc = &config.conv_desc;
  cudnnConvolutionMode_t conv_mode =
      config.conv_desc.convolution_not_crosscorr() ? CUDNN_CONVOLUTION
                                                   : CUDNN_CROSS_CORRELATION;
  int conv_dim = config.conv_desc.ndims();
  auto conv_dialations = xla::llvm_ir::AsArrayRef(conv_desc->dilations());
  auto conv_padding = xla::llvm_ir::AsArrayRef(conv_desc->padding());
  auto conv_strides = xla::llvm_ir::AsArrayRef(conv_desc->strides());

  std::vector<int64_t> tuning_knob_ids, tuning_knob_values;
  tuning_knob_ids.reserve(config.algorithm.TuningKnobs().size());
  tuning_knob_values.reserve(config.algorithm.TuningKnobs().size());
  for (auto iter : config.algorithm.TuningKnobs()) {
    tuning_knob_ids.push_back(iter.first);
    tuning_knob_values.push_back(iter.second);
  }

  return rewriter.create<tfrt::gpu::DnnBuildConvolutionOp>(
      loc, handle, input_type, output_type,
      rewriter.getI64ArrayAttr(input_dims),
      rewriter.getI64ArrayAttr(input_strides),
      rewriter.getI64ArrayAttr(output_dims),
      rewriter.getI64ArrayAttr(output_strides),
      rewriter.getI64ArrayAttr(filter_dims),
      rewriter.getI64ArrayAttr(filter_strides), conv_mode, conv_dim,
      rewriter.getI64ArrayAttr(conv_dialations),
      rewriter.getI64ArrayAttr(conv_padding),
      rewriter.getI64ArrayAttr(conv_strides), backend_type,
      config.algorithm.algo_id(), rewriter.getI64ArrayAttr(tuning_knob_ids),
      rewriter.getI64ArrayAttr(tuning_knob_values));
}

Value CreateBuildFusedConvOp(Value input, Value output, Value bias,
                             Value handle, mlir::Location loc,
                             const xla::gpu::GpuConvConfig& config,
                             cudnnBackendDescriptorType_t backend_type,
                             ConversionPatternRewriter& rewriter) {
  se::dnn::BatchDescriptor bias_descriptor(config.output_descriptor.ndims());
  bias_descriptor.set_count(1)
      .set_height(1)
      .set_width(1)
      .set_feature_map_count(config.output_descriptor.feature_map_count())
      .set_layout([&] {
        if (config.algorithm.is_cudnn_frontend()) {
          // For the purposes of the cudnn graph, say that the bias tensor has
          // the same layout as the output tensor.  It doesn't actually matter,
          // because bias is a 1D array.  But we need to get the correct
          // vectorization, otherwise the cudnn graph API rejects this tensor,
          // even though vectorized float tensors aren't even a thing in cuDNN.
          return config.output_descriptor.layout();
        }
        // Normalize NCHW_VECT_C to NCHW for layout of `bias`, even though it's
        // actually the same (because `bias` only has one dimension):  cudnn
        // does not accept NCHW_VECT_C for `bias`.
        se::dnn::DataLayout layout = config.output_descriptor.layout();
        switch (layout) {
          case se::dnn::DataLayout::kBatchDepthYX4:
          case se::dnn::DataLayout::kBatchDepthYX32:
            return se::dnn::DataLayout::kBatchDepthYX;
          default:
            return layout;
        }
      }());
  if (bias_descriptor.ndims() == 3) {
    bias_descriptor.set_spatial_dim(se::dnn::DimIndex::Z, 1);
  }

  auto get_element_type = [](Value value) {
    return value.getType().cast<mlir::MemRefType>().getElementType();
  };
  cudnnDataType_t input_type = MlirTypeToCudnnDataType(
      get_element_type(input), config.input_descriptor.layout());
  cudnnDataType_t output_type = MlirTypeToCudnnDataType(
      get_element_type(output), config.output_descriptor.layout());
  cudnnDataType_t bias_type =
      MlirTypeToCudnnDataType(get_element_type(bias), bias_descriptor.layout());

  int vector_size, vector_dim;
  std::tie(vector_size, vector_dim) =
      tfrt::gpu::wrapper::GetTensorVectorizedSizeAndDim(input_type);
  std::vector<int64_t> input_dims = config.input_descriptor.vectorized_dims(
      se::dnn::DataLayout::kBatchDepthYX, vector_size, vector_dim);
  std::vector<int64_t> input_strides =
      config.input_descriptor.vectorized_strides(
          se::dnn::DataLayout::kBatchDepthYX, vector_size, vector_dim);

  std::tie(vector_size, vector_dim) =
      tfrt::gpu::wrapper::GetTensorVectorizedSizeAndDim(output_type);
  std::vector<int64_t> output_dims = config.output_descriptor.vectorized_dims(
      se::dnn::DataLayout::kBatchDepthYX, vector_size, vector_dim);
  std::vector<int64_t> output_strides =
      config.output_descriptor.vectorized_strides(
          se::dnn::DataLayout::kBatchDepthYX, vector_size, vector_dim);

  std::tie(vector_size, vector_dim) =
      tfrt::gpu::wrapper::GetTensorVectorizedSizeAndDim(input_type);
  std::vector<int64_t> filter_dims = config.filter_descriptor.vectorized_dims(
      se::dnn::FilterLayout::kOutputInputYX, vector_size, vector_dim);
  std::vector<int64_t> filter_strides =
      config.filter_descriptor.vectorized_strides(
          se::dnn::FilterLayout::kOutputInputYX, vector_size, vector_dim);

  std::tie(vector_size, vector_dim) =
      tfrt::gpu::wrapper::GetTensorVectorizedSizeAndDim(input_type);
  std::vector<int64_t> bias_dims = bias_descriptor.vectorized_dims(
      se::dnn::DataLayout::kBatchDepthYX, vector_size, vector_dim);
  std::vector<int64_t> bias_strides = bias_descriptor.vectorized_strides(
      se::dnn::DataLayout::kBatchDepthYX, vector_size, vector_dim);

  const auto* conv_desc = &config.conv_desc;
  cudnnConvolutionMode_t conv_mode =
      config.conv_desc.convolution_not_crosscorr() ? CUDNN_CONVOLUTION
                                                   : CUDNN_CROSS_CORRELATION;
  int conv_dim = config.conv_desc.ndims();
  auto conv_dialations = xla::llvm_ir::AsArrayRef(conv_desc->dilations());
  auto conv_padding = xla::llvm_ir::AsArrayRef(conv_desc->padding());
  auto conv_strides = xla::llvm_ir::AsArrayRef(conv_desc->strides());

  std::vector<int64_t> tuning_knob_ids, tuning_knob_values;
  tuning_knob_ids.reserve(config.algorithm.TuningKnobs().size());
  tuning_knob_values.reserve(config.algorithm.TuningKnobs().size());
  for (auto iter : config.algorithm.TuningKnobs()) {
    tuning_knob_ids.push_back(iter.first);
    tuning_knob_values.push_back(iter.second);
  }

  auto alpha = rewriter.create<tfrt::compiler::ConstantF64Op>(
      loc, llvm::APFloat(config.conv_result_scale));
  auto alpha2 = rewriter.create<tfrt::compiler::ConstantF64Op>(
      loc, llvm::APFloat(config.fusion->side_input_scale));
  cudnnActivationMode_t activaton_mode = config.fusion->mode == se::dnn::kRelu
                                             ? CUDNN_ACTIVATION_RELU
                                             : CUDNN_ACTIVATION_IDENTITY;

  return rewriter.create<tfrt::gpu::DnnBuildFusedConvolutionOp>(
      loc, handle, input_type, output_type, bias_type,
      rewriter.getI64ArrayAttr(input_dims),
      rewriter.getI64ArrayAttr(input_strides),
      rewriter.getI64ArrayAttr(output_dims),
      rewriter.getI64ArrayAttr(output_strides),
      rewriter.getI64ArrayAttr(filter_dims),
      rewriter.getI64ArrayAttr(filter_strides),
      rewriter.getI64ArrayAttr(bias_dims),
      rewriter.getI64ArrayAttr(bias_strides), conv_mode, conv_dim,
      rewriter.getI64ArrayAttr(conv_dialations),
      rewriter.getI64ArrayAttr(conv_padding),
      rewriter.getI64ArrayAttr(conv_strides), backend_type, alpha, alpha2,
      activaton_mode, config.algorithm.algo_id(),
      rewriter.getI64ArrayAttr(tuning_knob_ids),
      rewriter.getI64ArrayAttr(tuning_knob_values));
}

// Specialization for convolution forward
Status SetConvKind(lmhlo_gpu::ConvForwardOp op,
                   xla::gpu::GpuConvDescriptor& descriptor) {
  descriptor.kind = xla::gpu::CudnnConvKind::kForward;
  return Status::OK();
}
Value GetResult(lmhlo_gpu::ConvForwardOp op) { return op.output(); }
Value CreateBuildConvOp(lmhlo_gpu::ConvForwardOp op, Value handle,
                        const xla::gpu::GpuConvConfig& config,
                        cudnnBackendDescriptorType_t backend_type,
                        ConversionPatternRewriter& rewriter) {
  return CreateBuildUnfusedConvOp(op.input(), op.output(), handle, op.getLoc(),
                                  config, backend_type, rewriter);
}
Value CreateRunConvolutionOp(lmhlo_gpu::ConvForwardOpAdaptor adaptor,
                             mlir::Location loc, Value handle, Value conv_plan,
                             Value chain, ConversionPatternRewriter& rewriter) {
  return rewriter.create<tfrt::gpu::DnnRunConvolutionOp>(
      loc, handle, conv_plan, adaptor.input(), adaptor.output(),
      adaptor.filter(), adaptor.scratch(), chain);
}

// Specialization for convolution backward input
Status SetConvKind(lmhlo_gpu::ConvBackwardInputOp op,
                   xla::gpu::GpuConvDescriptor& descriptor) {
  descriptor.kind = xla::gpu::CudnnConvKind::kBackwardInput;
  return Status::OK();
}
Value GetResult(lmhlo_gpu::ConvBackwardInputOp op) { return op.d_input(); }
Value CreateBuildConvOp(lmhlo_gpu::ConvBackwardInputOp op, Value handle,
                        const xla::gpu::GpuConvConfig& config,
                        cudnnBackendDescriptorType_t backend_type,
                        ConversionPatternRewriter& rewriter) {
  return CreateBuildUnfusedConvOp(op.d_input(), op.d_output(), handle,
                                  op.getLoc(), config, backend_type, rewriter);
}
Value CreateRunConvolutionOp(lmhlo_gpu::ConvBackwardInputOpAdaptor adaptor,
                             mlir::Location loc, Value handle, Value conv_plan,
                             Value chain, ConversionPatternRewriter& rewriter) {
  return rewriter.create<tfrt::gpu::DnnRunConvolutionOp>(
      loc, handle, conv_plan, adaptor.d_input(), adaptor.d_output(),
      adaptor.filter(), adaptor.scratch(), chain);
}

// Specialization for convolution backward filter
Status SetConvKind(lmhlo_gpu::ConvBackwardFilterOp op,
                   xla::gpu::GpuConvDescriptor& descriptor) {
  descriptor.kind = xla::gpu::CudnnConvKind::kBackwardFilter;
  return Status::OK();
}
Value GetResult(lmhlo_gpu::ConvBackwardFilterOp op) { return op.d_filter(); }
Value CreateBuildConvOp(lmhlo_gpu::ConvBackwardFilterOp op, Value handle,
                        const xla::gpu::GpuConvConfig& config,
                        cudnnBackendDescriptorType_t backend_type,
                        ConversionPatternRewriter& rewriter) {
  return CreateBuildUnfusedConvOp(op.input(), op.d_output(), handle,
                                  op.getLoc(), config, backend_type, rewriter);
}
Value CreateRunConvolutionOp(lmhlo_gpu::ConvBackwardFilterOpAdaptor adaptor,
                             mlir::Location loc, Value handle, Value conv_plan,
                             Value chain, ConversionPatternRewriter& rewriter) {
  return rewriter.create<tfrt::gpu::DnnRunConvolutionOp>(
      loc, handle, conv_plan, adaptor.input(), adaptor.d_output(),
      adaptor.d_filter(), adaptor.scratch(), chain);
}

// Specialization for convolution forward fused
Status SetConvKind(lmhlo_gpu::ConvForwardFusedOp op,
                   xla::gpu::GpuConvDescriptor& descriptor) {
  descriptor.kind = xla::gpu::CudnnConvKind::kForwardActivation;
  auto activation_mode_or =
      xla::ConvertConvActivationMode(op.activation_mode());
  if (!activation_mode_or.ok()) {
    return activation_mode_or.status();
  }
  auto activation_mode = activation_mode_or.ValueOrDie();
  descriptor.backend_config.set_activation_mode(
      static_cast<int64_t>(activation_mode));
  return Status::OK();
}
Value GetResult(lmhlo_gpu::ConvForwardFusedOp op) { return op.output(); }
Value CreateBuildConvOp(lmhlo_gpu::ConvForwardFusedOp op, Value handle,
                        const xla::gpu::GpuConvConfig& config,
                        cudnnBackendDescriptorType_t backend_type,
                        ConversionPatternRewriter& rewriter) {
  return CreateBuildFusedConvOp(op.input(), op.output(), op.bias(), handle,
                                op.getLoc(), config, backend_type, rewriter);
}
Value CreateRunConvolutionOp(lmhlo_gpu::ConvForwardFusedOpAdaptor adaptor,
                             mlir::Location loc, Value handle, Value conv_plan,
                             Value chain, ConversionPatternRewriter& rewriter) {
  return rewriter.create<tfrt::gpu::DnnRunFusedConvolutionOp>(
      loc, handle, conv_plan, adaptor.input(), adaptor.output(),
      adaptor.filter(), adaptor.output(), adaptor.bias(), adaptor.scratch(),
      chain);
}

// Specialization for convolution forward fused side input
Status SetConvKind(lmhlo_gpu::ConvForwardFusedSideInputOp op,
                   xla::gpu::GpuConvDescriptor& descriptor) {
  descriptor.kind = xla::gpu::CudnnConvKind::kForwardActivation;
  auto activation_mode_or =
      xla::ConvertConvActivationMode(op.activation_mode());
  if (!activation_mode_or.ok()) {
    return activation_mode_or.status();
  }
  auto activation_mode = activation_mode_or.ValueOrDie();
  descriptor.backend_config.set_activation_mode(
      static_cast<int64_t>(activation_mode));
  descriptor.backend_config.set_side_input_scale(
      op.side_input_scale().convertToDouble());
  return Status::OK();
}
Value GetResult(lmhlo_gpu::ConvForwardFusedSideInputOp op) {
  return op.output();
}
Value CreateBuildConvOp(lmhlo_gpu::ConvForwardFusedSideInputOp op, Value handle,
                        const xla::gpu::GpuConvConfig& config,
                        cudnnBackendDescriptorType_t backend_type,
                        ConversionPatternRewriter& rewriter) {
  return CreateBuildFusedConvOp(op.input(), op.output(), op.bias(), handle,
                                op.getLoc(), config, backend_type, rewriter);
}
Value CreateRunConvolutionOp(
    lmhlo_gpu::ConvForwardFusedSideInputOpAdaptor adaptor, mlir::Location loc,
    Value handle, Value conv_plan, Value chain,
    ConversionPatternRewriter& rewriter) {
  return rewriter.create<tfrt::gpu::DnnRunFusedConvolutionOp>(
      loc, handle, conv_plan, adaptor.input(), adaptor.output(),
      adaptor.filter(), adaptor.side_input(), adaptor.bias(), adaptor.scratch(),
      chain);
}

template <class ConvolutionOpType>
struct ConvolutionRewritePattern
    : tfrt::gpu::GpuAsyncOpConversionPattern<ConvolutionOpType> {
  using typename tfrt::gpu::GpuAsyncOpConversionPattern<
      ConvolutionOpType>::OpAdaptor;
  using tfrt::gpu::GpuAsyncOpConversionPattern<
      ConvolutionOpType>::GpuAsyncOpConversionPattern;
  FailureOr<Value> matchAndRewriteOp(
      ConvolutionOpType op, OpAdaptor adaptor, Value chain, Value stream,
      ConversionPatternRewriter& rewriter) const override {
    xla::gpu::GpuConvDescriptor descriptor;
    auto status = SetConvKind(op, descriptor);
    if (!status.ok()) {
      return rewriter.notifyMatchFailure(op, status.error_message());
    }
    FillConvDescriptor(op, GetResult(op), descriptor);
    auto config_or = xla::gpu::GetGpuConvConfig(descriptor, "");
    if (!config_or.ok()) {
      return rewriter.notifyMatchFailure(
          op, "Failed to get GPU convolution config.");
    }
    xla::gpu::GpuConvConfig config = config_or.ValueOrDie();

    if (config.fusion.has_value()) {
      if (config.fusion->mode != se::dnn::kNone &&
          config.fusion->mode != se::dnn::kRelu) {
        return rewriter.notifyMatchFailure(op,
                                           "Unimplemented activation mode.");
      }
    }
    if (config.conv_desc.pad_alignment() ==
        se::dnn::PadAlignment::kTensorFlowPadding) {
      return rewriter.notifyMatchFailure(
          op, "TensorFlow padding alignment is not supported.");
    }

    cudnnBackendDescriptorType_t backend_type;
    switch (descriptor.kind) {
      case xla::gpu::CudnnConvKind::kForward:
      case xla::gpu::CudnnConvKind::kForwardActivation:
        backend_type = CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
        break;
      case xla::gpu::CudnnConvKind::kBackwardInput:
        backend_type =
            CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR;
        break;
      case xla::gpu::CudnnConvKind::kBackwardFilter:
        backend_type =
            CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR;
        break;
      default:
        return rewriter.notifyMatchFailure(op, "Unexpected convolution kind.");
    }

    auto saved_point = rewriter.saveInsertionPoint();

    // Create a function that returns the convolution plan.
    rewriter.setInsertionPoint(op->template getParentOfType<mlir::FuncOp>());
    mlir::Type handle_type = rewriter.getType<tfrt::gpu::DnnHandleType>();
    mlir::Type conv_plan_type =
        rewriter.getType<tfrt::gpu::DnnConvolutionPlanType>();
    std::string function_name =
        absl::StrCat("get_", op->getName().stripDialect().str(), "_plan");
    mlir::FuncOp conv_plan_func = rewriter.create<mlir::FuncOp>(
        op.getLoc(), function_name,
        rewriter.getFunctionType(handle_type, conv_plan_type));
    rewriter.setInsertionPointToEnd(conv_plan_func.addEntryBlock());
    Value conv_plan = CreateBuildConvOp(op, conv_plan_func.getArgument(0),
                                        config, backend_type, rewriter);
    rewriter.create<tfrt::compiler::ReturnOp>(op.getLoc(), conv_plan);

    // Once-initialize the convolution plan.
    rewriter.restoreInsertionPoint(saved_point);
    Value handle = rewriter.create<tfrt::gpu::DnnCreateOp>(op.getLoc(), stream);
    auto once_op = rewriter.create<tfrt::compiler::OnceOp>(
        op.getLoc(), conv_plan_func.getType().getResults(), handle,
        conv_plan_func.getName());

    Value out_chain = CreateRunConvolutionOp(
        adaptor, op.getLoc(), handle, once_op.getResult(0), chain, rewriter);
    rewriter.eraseOp(op);
    return out_chain;
  }
};

}  // namespace

void populateConvolutionConversionPattern(RewritePatternSet& patterns,
                                          TypeConverter& converter) {
  patterns
      .add<ConvolutionRewritePattern<lmhlo_gpu::ConvForwardOp>,
           ConvolutionRewritePattern<lmhlo_gpu::ConvBackwardInputOp>,
           ConvolutionRewritePattern<lmhlo_gpu::ConvBackwardFilterOp>,
           ConvolutionRewritePattern<lmhlo_gpu::ConvForwardFusedOp>,
           ConvolutionRewritePattern<lmhlo_gpu::ConvForwardFusedSideInputOp>>(
          converter, patterns.getContext());
}

}  // namespace tensorflow
