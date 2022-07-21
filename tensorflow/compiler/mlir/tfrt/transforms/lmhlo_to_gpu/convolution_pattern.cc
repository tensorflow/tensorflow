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

#include <algorithm>
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
#include "tfrt/gpu/wrapper/dnn_wrapper.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime

namespace tensorflow {
namespace {

template <class ConvolutionOpType>
void FillConvDescriptor(ConvolutionOpType op, Value result,
                        xla::gpu::GpuConvDescriptor& descriptor) {
  auto apply_layout = [](const xla::Shape& shape,
                         mlir::ArrayRef<int64_t> minor_to_major) {
    return xla::ShapeUtil::MakeShapeWithLayout(
        shape.element_type(), shape.dimensions(), minor_to_major);
  };

  descriptor.operand0_shape =
      apply_layout(xla::gpu::GetShape(op->getOperand(0)),
                   op.getBackendConfig().getOperand_0Layout());
  descriptor.operand1_shape =
      apply_layout(xla::gpu::GetShape(op->getOperand(1)),
                   op.getBackendConfig().getOperand_1Layout());
  descriptor.result_shape = apply_layout(
      xla::gpu::GetShape(result), op.getBackendConfig().getResultLayout());
  descriptor.dnums = xla::ConvertConvDimensionNumbers(op.getDimensionNumbers());
  descriptor.scratch_size = 0;  // Not used for op lowering.
  mlir::DenseIntElementsAttr window_strides = op.getWindowStrides().getValue();
  mlir::DenseIntElementsAttr lhs_dilation = op.getLhsDilation().getValue();
  mlir::DenseIntElementsAttr rhs_dilation = op.getRhsDilation().getValue();
  mlir::DenseElementsAttr window_reversal = op.getWindowReversal().getValue();
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
    if (op.getPadding().has_value()) {
      mlir::DenseIntElementsAttr padding = op.getPadding().getValue();
      dim->set_padding_low(padding.getValues<int64_t>()[index]);
      dim->set_padding_high(padding.getValues<int64_t>()[index]);
    }
  }
  descriptor.feature_group_count = op.getFeatureGroupCount();
  {
    auto* algorithm = descriptor.backend_config.mutable_algorithm();
    algorithm->set_algo_id(op.getBackendConfig().getAlgorithm());
    algorithm->set_math_type(op.getBackendConfig().getTensorOpsEnabled()
                                 ? se::dnn::AlgorithmProto::TENSOR_OP_MATH
                                 : se::dnn::AlgorithmProto::DEFAULT_MATH);
    for (int i = 0; i < op.getBackendConfig().getKnobIds().size(); ++i) {
      // N.B. tuning_knobs is a map rather than a repeated field, so this
      // doesn't require reserving space up front.
      auto knob_id = op.getBackendConfig().getKnobIds()[i];
      auto knob_value = op.getBackendConfig().getKnobValues()[i];
      (*algorithm->mutable_tuning_knobs())[knob_id] = knob_value;
    }
    algorithm->set_is_cudnn_frontend(
        op.getBackendConfig().getIsCudnnFrontend());
    auto workspace_size = op.getBackendConfig().getWorkspaceSize();
    if (workspace_size >= 0) {
      algorithm->mutable_workspace_size()->set_value(workspace_size);
    }
  }
  descriptor.backend_config.set_conv_result_scale(
      op.getResultScale().convertToDouble());
}

mlir::Type GetMemRefElementType(Value value) {
  return value.getType().cast<mlir::MemRefType>().getElementType();
}

// Converts (via narrowing) a Span<const int64_t> to a vector<int>, and checks
// that the elements have not changed due to the conversion.
std::vector<int> CheckedNarrowing(absl::Span<const int64_t> wide_span) {
  std::vector<int> narrow_vector(wide_span.size());
  std::transform(
      wide_span.cbegin(), wide_span.cend(), narrow_vector.begin(),
      [](int64_t wide) {
        int narrow = wide;
        assert(narrow == wide &&
               "checked narrowing failed; values not equal post-conversion");
        return narrow;
      });
  return narrow_vector;
}

// Create ops to describe tensors (e.g., input, output, or bias) when using
// legacy cudnn.
template <class ConvolutionOpType>
FailureOr<Value> CreateLegacyTensorDescriptor(
    ConvolutionOpType op, const se::dnn::BatchDescriptor& batch_descriptor,
    tfrt::gpu::wrapper::DnnDataType elem_type, Value chain,
    ConversionPatternRewriter& rewriter) {
  std::vector<int64_t> dims64, strides64;
  switch (batch_descriptor.layout()) {
    case se::dnn::DataLayout::kBatchYXDepth:
    case se::dnn::DataLayout::kBatchDepthYX: {
      // cuDNN requires the strides and dims to be ordered as BDYX.
      dims64 = batch_descriptor.full_dims(se::dnn::DataLayout::kBatchDepthYX);
      strides64 =
          batch_descriptor.full_strides(se::dnn::DataLayout::kBatchDepthYX);
      break;
    }
    case se::dnn::DataLayout::kBatchDepthYX4:
    case se::dnn::DataLayout::kBatchDepthYX32: {
      const int64_t n = batch_descriptor.count();
      const int64_t c = batch_descriptor.feature_map_count();
      const int64_t h = batch_descriptor.height();
      const int64_t w = batch_descriptor.width();
      const int64_t v =
          batch_descriptor.layout() == se::dnn::DataLayout::kBatchDepthYX4 ? 4
                                                                           : 32;
      assert(c / v > 0 && "Vectorized feature map count is non-positive.");
      dims64 = {n, c / v, h, w};
      strides64 = {c / v * h * w, h * w, w, 1};
      break;
    }
    default:
      return rewriter.notifyMatchFailure(op, "Unsupported tensor format.");
  }

  // cuDNN requires arrays of ints.
  std::vector<int> dims = CheckedNarrowing(dims64);
  std::vector<int> strides = CheckedNarrowing(strides64);
  return rewriter
      .create<tfrt::gpu::DnnCreateTensorDescriptorOp>(
          op.getLoc(), elem_type, rewriter.getI32ArrayAttr(dims),
          rewriter.getI32ArrayAttr(strides))
      .getResult();
}

template <class FusedConvOpType, class FusedConvOpAdaptorType>
FailureOr<Value> CreateLegacyFusedConvOp(
    FusedConvOpType op, FusedConvOpAdaptorType adaptor, Type mlir_scale_type,
    Value handle, Value stream, Value input_desc, Value output_desc,
    Value filter_desc, Value conv_desc, Value algorithm, Value side_input,
    Value chain, const xla::gpu::GpuConvConfig& config,
    ConversionPatternRewriter& rewriter) {
  // Create bias descriptor.
  se::dnn::BatchDescriptor bias_descriptor = GetBiasDescriptor(config);
  cudnnDataType_t bias_type = MlirTypeToDnnDataType(
      GetMemRefElementType(op.getBias()), bias_descriptor.layout());
  FailureOr<Value> bias_desc_or = CreateLegacyTensorDescriptor(
      op, bias_descriptor, bias_type, chain, rewriter);
  if (failed(bias_desc_or)) {
    return bias_desc_or;
  }

  // Create activation descriptor.
  auto loc = op.getLoc();
  auto coefficient =
      rewriter.create<tfrt::compiler::ConstantF64Op>(loc, llvm::APFloat(0.0));
  cudnnActivationMode_t activaton_mode = config.fusion->mode == se::dnn::kRelu
                                             ? CUDNN_ACTIVATION_RELU
                                             : CUDNN_ACTIVATION_IDENTITY;
  auto activation_desc =
      rewriter.create<tfrt::gpu::DnnCreateActivationDescriptorOp>(
          loc, coefficient, activaton_mode, CUDNN_NOT_PROPAGATE_NAN);

  tfrt::gpu::wrapper::DnnDataType scale_type =
      MlirTypeToDnnDataType(mlir_scale_type);
  auto alpha1 = MakeScalingFactorConstant(
      rewriter, loc, mlir_scale_type, llvm::APFloat(config.conv_result_scale),
      llvm::APFloat(0.0));
  auto alpha2 = MakeScalingFactorConstant(
      rewriter, loc, mlir_scale_type,
      llvm::APFloat(config.fusion->side_input_scale), llvm::APFloat(0.0));
  return rewriter
      .create<tfrt::gpu::DnnConvolutionBiasActivationForwardOp>(
          loc, handle, stream, scale_type, alpha1, input_desc,
          adaptor.getInput(), filter_desc, adaptor.getFilter(), conv_desc,
          algorithm, adaptor.getScratch(), alpha2, output_desc, side_input,
          *bias_desc_or, adaptor.getBias(), activation_desc, output_desc,
          adaptor.getOutput(), chain)
      .getResult();
}

// Create op to build a convolution plan, which can be used to run the
// convolution. This is the unfused variant (not fused with activation).
Value CreateBuildUnfusedConvPlanOp(Value input, Value output, Value handle,
                                   mlir::Location loc,
                                   const xla::gpu::GpuConvConfig& config,
                                   cudnnBackendDescriptorType_t backend_type,
                                   ConversionPatternRewriter& rewriter) {
  tfrt::gpu::wrapper::DnnDataType input_type = MlirTypeToDnnDataType(
      GetMemRefElementType(input), config.input_descriptor.layout());
  tfrt::gpu::wrapper::DnnDataType output_type = MlirTypeToDnnDataType(
      GetMemRefElementType(output), config.output_descriptor.layout());

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

// Create op to build a convolution plan, which can be used to run the
// convolution. This is the variant with fused activation.
Value CreateBuildFusedConvPlanOp(Value input, Value output, Value bias,
                                 Value handle, mlir::Location loc,
                                 const xla::gpu::GpuConvConfig& config,
                                 cudnnBackendDescriptorType_t backend_type,
                                 ConversionPatternRewriter& rewriter) {
  se::dnn::BatchDescriptor bias_descriptor = GetBiasDescriptor(config);
  // For the purposes of the cudnn graph, say that the bias tensor has the same
  // layout as the output tensor.  It doesn't actually matter, because bias is a
  // 1D array.  But we need to get the correct vectorization, otherwise the
  // cudnn graph API rejects this tensor, even though vectorized float tensors
  // aren't even a thing in cuDNN.
  bias_descriptor.set_layout(config.output_descriptor.layout());

  tfrt::gpu::wrapper::DnnDataType input_type = MlirTypeToDnnDataType(
      GetMemRefElementType(input), config.input_descriptor.layout());
  tfrt::gpu::wrapper::DnnDataType output_type = MlirTypeToDnnDataType(
      GetMemRefElementType(output), config.output_descriptor.layout());
  tfrt::gpu::wrapper::DnnDataType bias_type = MlirTypeToDnnDataType(
      GetMemRefElementType(bias), bias_descriptor.layout());

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
Value GetInput(lmhlo_gpu::ConvForwardOp op) { return op.getInput(); }
Value GetOutput(lmhlo_gpu::ConvForwardOp op) { return op.getOutput(); }
Value GetResult(lmhlo_gpu::ConvForwardOp op) { return op.getOutput(); }
Value CreateBuildConvPlanOp(lmhlo_gpu::ConvForwardOp op, Value handle,
                            const xla::gpu::GpuConvConfig& config,
                            cudnnBackendDescriptorType_t backend_type,
                            ConversionPatternRewriter& rewriter) {
  return CreateBuildUnfusedConvPlanOp(op.getInput(), op.getOutput(), handle,
                                      op.getLoc(), config, backend_type,
                                      rewriter);
}
Value CreateRunConvolutionOp(lmhlo_gpu::ConvForwardOpAdaptor adaptor,
                             mlir::Location loc, Value handle, Value conv_plan,
                             Value chain, Value stream,
                             ConversionPatternRewriter& rewriter) {
  return rewriter.create<tfrt::gpu::DnnRunConvolutionOp>(
      loc, handle, stream, conv_plan, adaptor.getInput(), adaptor.getOutput(),
      adaptor.getFilter(), adaptor.getScratch(), chain);
}
FailureOr<Value> CreateLegacyConvOp(
    lmhlo_gpu::ConvForwardOp op, lmhlo_gpu::ConvForwardOpAdaptor adaptor,
    Type mlir_scale_type, Value handle, Value stream, Value input_desc,
    Value output_desc, Value filter_desc, Value conv_desc, int64_t algorithm,
    Value chain, const xla::gpu::GpuConvConfig& config,
    ConversionPatternRewriter& rewriter) {
  tfrt::gpu::wrapper::DnnDataType scale_type =
      MlirTypeToDnnDataType(mlir_scale_type);
  Value algo = rewriter.create<tfrt::gpu::DnnConvolutionForwardAlgorithmOp>(
      op.getLoc(), tfrt::gpu::wrapper::DnnConvFwdAlgo(
                       algorithm, tfrt::gpu::wrapper::Platform::CUDA));
  return rewriter
      .create<tfrt::gpu::DnnConvolutionForwardOp>(
          op.getLoc(), handle, stream, scale_type, input_desc,
          adaptor.getInput(), filter_desc, adaptor.getFilter(), conv_desc, algo,
          adaptor.getScratch(), output_desc, adaptor.getOutput(), chain)
      .getResult();
}

// Specialization for convolution backward input
Status SetConvKind(lmhlo_gpu::ConvBackwardInputOp op,
                   xla::gpu::GpuConvDescriptor& descriptor) {
  descriptor.kind = xla::gpu::CudnnConvKind::kBackwardInput;
  return Status::OK();
}
Value GetInput(lmhlo_gpu::ConvBackwardInputOp op) { return op.getDInput(); }
Value GetOutput(lmhlo_gpu::ConvBackwardInputOp op) { return op.getDOutput(); }
Value GetResult(lmhlo_gpu::ConvBackwardInputOp op) { return op.getDInput(); }
Value CreateBuildConvPlanOp(lmhlo_gpu::ConvBackwardInputOp op, Value handle,
                            const xla::gpu::GpuConvConfig& config,
                            cudnnBackendDescriptorType_t backend_type,
                            ConversionPatternRewriter& rewriter) {
  return CreateBuildUnfusedConvPlanOp(op.getDInput(), op.getDOutput(), handle,
                                      op.getLoc(), config, backend_type,
                                      rewriter);
}
Value CreateRunConvolutionOp(lmhlo_gpu::ConvBackwardInputOpAdaptor adaptor,
                             mlir::Location loc, Value handle, Value conv_plan,
                             Value chain, Value stream,
                             ConversionPatternRewriter& rewriter) {
  return rewriter.create<tfrt::gpu::DnnRunConvolutionOp>(
      loc, handle, stream, conv_plan, adaptor.getDInput(), adaptor.getDOutput(),
      adaptor.getFilter(), adaptor.getScratch(), chain);
}
FailureOr<Value> CreateLegacyConvOp(
    lmhlo_gpu::ConvBackwardInputOp op,
    lmhlo_gpu::ConvBackwardInputOpAdaptor adaptor, Type mlir_scale_type,
    Value handle, Value stream, Value input_desc, Value output_desc,
    Value filter_desc, Value conv_desc, int64_t algorithm, Value chain,
    const xla::gpu::GpuConvConfig& config,
    ConversionPatternRewriter& rewriter) {
  tfrt::gpu::wrapper::DnnDataType scale_type =
      MlirTypeToDnnDataType(mlir_scale_type);
  Value algo =
      rewriter.create<tfrt::gpu::DnnConvolutionBackwardDataAlgorithmOp>(
          op.getLoc(), tfrt::gpu::wrapper::DnnConvBwdDataAlgo(
                           algorithm, tfrt::gpu::wrapper::Platform::CUDA));
  return rewriter
      .create<tfrt::gpu::DnnConvolutionBackwardDataOp>(
          op.getLoc(), handle, stream, scale_type, filter_desc,
          adaptor.getFilter(), output_desc, adaptor.getDOutput(), conv_desc,
          algo, adaptor.getScratch(), input_desc, adaptor.getDInput(), chain)
      .getResult();
}

// Specialization for convolution backward filter
Status SetConvKind(lmhlo_gpu::ConvBackwardFilterOp op,
                   xla::gpu::GpuConvDescriptor& descriptor) {
  descriptor.kind = xla::gpu::CudnnConvKind::kBackwardFilter;
  return Status::OK();
}
Value GetInput(lmhlo_gpu::ConvBackwardFilterOp op) { return op.getInput(); }
Value GetOutput(lmhlo_gpu::ConvBackwardFilterOp op) { return op.getDOutput(); }
Value GetResult(lmhlo_gpu::ConvBackwardFilterOp op) { return op.getDFilter(); }
Value CreateBuildConvPlanOp(lmhlo_gpu::ConvBackwardFilterOp op, Value handle,
                            const xla::gpu::GpuConvConfig& config,
                            cudnnBackendDescriptorType_t backend_type,
                            ConversionPatternRewriter& rewriter) {
  return CreateBuildUnfusedConvPlanOp(op.getInput(), op.getDOutput(), handle,
                                      op.getLoc(), config, backend_type,
                                      rewriter);
}
Value CreateRunConvolutionOp(lmhlo_gpu::ConvBackwardFilterOpAdaptor adaptor,
                             mlir::Location loc, Value handle, Value conv_plan,
                             Value chain, Value stream,
                             ConversionPatternRewriter& rewriter) {
  return rewriter.create<tfrt::gpu::DnnRunConvolutionOp>(
      loc, handle, stream, conv_plan, adaptor.getInput(), adaptor.getDOutput(),
      adaptor.getDFilter(), adaptor.getScratch(), chain);
}
FailureOr<Value> CreateLegacyConvOp(
    lmhlo_gpu::ConvBackwardFilterOp op,
    lmhlo_gpu::ConvBackwardFilterOpAdaptor adaptor, Type mlir_scale_type,
    Value handle, Value stream, Value input_desc, Value output_desc,
    Value filter_desc, Value conv_desc, int64_t algorithm, Value chain,
    const xla::gpu::GpuConvConfig& config,
    ConversionPatternRewriter& rewriter) {
  tfrt::gpu::wrapper::DnnDataType scale_type =
      MlirTypeToDnnDataType(mlir_scale_type);
  Value algo =
      rewriter.create<tfrt::gpu::DnnConvolutionBackwardFilterAlgorithmOp>(
          op.getLoc(), tfrt::gpu::wrapper::DnnConvBwdFilterAlgo(
                           algorithm, tfrt::gpu::wrapper::Platform::CUDA));
  return rewriter
      .create<tfrt::gpu::DnnConvolutionBackwardFilterOp>(
          op.getLoc(), handle, stream, scale_type, input_desc,
          adaptor.getInput(), output_desc, adaptor.getDOutput(), conv_desc,
          algo, adaptor.getScratch(), filter_desc, adaptor.getDFilter(), chain)
      .getResult();
}

// Specialization for convolution forward fused
Status SetConvKind(lmhlo_gpu::ConvForwardFusedOp op,
                   xla::gpu::GpuConvDescriptor& descriptor) {
  descriptor.kind = xla::gpu::CudnnConvKind::kForwardActivation;
  auto activation_mode_or =
      xla::ConvertConvActivationMode(op.getActivationMode());
  if (!activation_mode_or.ok()) {
    return activation_mode_or.status();
  }
  auto activation_mode = activation_mode_or.ValueOrDie();
  descriptor.backend_config.set_activation_mode(
      static_cast<int64_t>(activation_mode));
  return Status::OK();
}
Value GetInput(lmhlo_gpu::ConvForwardFusedOp op) { return op.getInput(); }
Value GetOutput(lmhlo_gpu::ConvForwardFusedOp op) { return op.getOutput(); }
Value GetResult(lmhlo_gpu::ConvForwardFusedOp op) { return op.getOutput(); }
Value CreateBuildConvPlanOp(lmhlo_gpu::ConvForwardFusedOp op, Value handle,
                            const xla::gpu::GpuConvConfig& config,
                            cudnnBackendDescriptorType_t backend_type,
                            ConversionPatternRewriter& rewriter) {
  return CreateBuildFusedConvPlanOp(op.getInput(), op.getOutput(), op.getBias(),
                                    handle, op.getLoc(), config, backend_type,
                                    rewriter);
}
Value CreateRunConvolutionOp(lmhlo_gpu::ConvForwardFusedOpAdaptor adaptor,
                             mlir::Location loc, Value handle, Value conv_plan,
                             Value chain, Value stream,
                             ConversionPatternRewriter& rewriter) {
  return rewriter.create<tfrt::gpu::DnnRunFusedConvolutionOp>(
      loc, handle, stream, conv_plan, adaptor.getInput(), adaptor.getOutput(),
      adaptor.getFilter(), adaptor.getOutput(), adaptor.getBias(),
      adaptor.getScratch(), chain);
}
FailureOr<Value> CreateLegacyConvOp(
    lmhlo_gpu::ConvForwardFusedOp op,
    lmhlo_gpu::ConvForwardFusedOpAdaptor adaptor, Type mlir_scale_type,
    Value handle, Value stream, Value input_desc, Value output_desc,
    Value filter_desc, Value conv_desc, int64_t algorithm, Value chain,
    const xla::gpu::GpuConvConfig& config,
    ConversionPatternRewriter& rewriter) {
  Value algo = rewriter.create<tfrt::gpu::DnnConvolutionForwardAlgorithmOp>(
      op.getLoc(), tfrt::gpu::wrapper::DnnConvFwdAlgo(
                       algorithm, tfrt::gpu::wrapper::Platform::CUDA));
  return CreateLegacyFusedConvOp(op, adaptor, mlir_scale_type, handle, stream,
                                 input_desc, output_desc, filter_desc,
                                 conv_desc, algo, adaptor.getOutput(), chain,
                                 config, rewriter);
}

// Specialization for convolution forward fused side input
Status SetConvKind(lmhlo_gpu::ConvForwardFusedSideInputOp op,
                   xla::gpu::GpuConvDescriptor& descriptor) {
  descriptor.kind = xla::gpu::CudnnConvKind::kForwardActivation;
  auto activation_mode_or =
      xla::ConvertConvActivationMode(op.getActivationMode());
  if (!activation_mode_or.ok()) {
    return activation_mode_or.status();
  }
  auto activation_mode = activation_mode_or.ValueOrDie();
  descriptor.backend_config.set_activation_mode(
      static_cast<int64_t>(activation_mode));
  descriptor.backend_config.set_side_input_scale(
      op.getSideInputScale().convertToDouble());
  return Status::OK();
}
Value GetInput(lmhlo_gpu::ConvForwardFusedSideInputOp op) {
  return op.getInput();
}
Value GetOutput(lmhlo_gpu::ConvForwardFusedSideInputOp op) {
  return op.getOutput();
}
Value GetResult(lmhlo_gpu::ConvForwardFusedSideInputOp op) {
  return op.getOutput();
}
Value CreateBuildConvPlanOp(lmhlo_gpu::ConvForwardFusedSideInputOp op,
                            Value handle, const xla::gpu::GpuConvConfig& config,
                            cudnnBackendDescriptorType_t backend_type,
                            ConversionPatternRewriter& rewriter) {
  return CreateBuildFusedConvPlanOp(op.getInput(), op.getOutput(), op.getBias(),
                                    handle, op.getLoc(), config, backend_type,
                                    rewriter);
}
Value CreateRunConvolutionOp(
    lmhlo_gpu::ConvForwardFusedSideInputOpAdaptor adaptor, mlir::Location loc,
    Value handle, Value conv_plan, Value chain, Value stream,
    ConversionPatternRewriter& rewriter) {
  return rewriter.create<tfrt::gpu::DnnRunFusedConvolutionOp>(
      loc, handle, stream, conv_plan, adaptor.getInput(), adaptor.getOutput(),
      adaptor.getFilter(), adaptor.getSideInput(), adaptor.getBias(),
      adaptor.getScratch(), chain);
}
FailureOr<Value> CreateLegacyConvOp(
    lmhlo_gpu::ConvForwardFusedSideInputOp op,
    lmhlo_gpu::ConvForwardFusedSideInputOpAdaptor adaptor, Type mlir_scale_type,
    Value handle, Value stream, Value input_desc, Value output_desc,
    Value filter_desc, Value conv_desc, int64_t algorithm, Value chain,
    const xla::gpu::GpuConvConfig& config,
    ConversionPatternRewriter& rewriter) {
  Value side_input = config.fusion->side_input_scale == 0
                         ? adaptor.getOutput()
                         : adaptor.getSideInput();
  Value algo = rewriter.create<tfrt::gpu::DnnConvolutionForwardAlgorithmOp>(
      op.getLoc(), tfrt::gpu::wrapper::DnnConvFwdAlgo(
                       algorithm, tfrt::gpu::wrapper::Platform::CUDA));
  return CreateLegacyFusedConvOp(
      op, adaptor, mlir_scale_type, handle, stream, input_desc, output_desc,
      filter_desc, conv_desc, algo, side_input, chain, config, rewriter);
}

template <class ConvolutionOpType, class OpAdaptor>
FailureOr<Value> LegacyConvolutionRewritePattern(
    ConvolutionOpType op, OpAdaptor adaptor, Value chain, Value stream,
    const xla::gpu::GpuConvConfig& config,
    ConversionPatternRewriter& rewriter) {
  tfrt::gpu::wrapper::DnnDataType input_type = MlirTypeToDnnDataType(
      GetMemRefElementType(GetInput(op)), config.input_descriptor.layout());
  tfrt::gpu::wrapper::DnnDataType output_type = MlirTypeToDnnDataType(
      GetMemRefElementType(GetOutput(op)), config.output_descriptor.layout());
  tfrt::gpu::wrapper::DnnDataType filter_type = MlirTypeToDnnDataType(
      GetMemRefElementType(GetInput(op)), config.filter_descriptor.layout());

  // Create input descriptor.
  FailureOr<Value> input_desc_or = CreateLegacyTensorDescriptor(
      op, config.input_descriptor, input_type, chain, rewriter);
  if (failed(input_desc_or)) {
    return input_desc_or;
  }

  // Create output descriptor.
  FailureOr<Value> output_desc_or = CreateLegacyTensorDescriptor(
      op, config.output_descriptor, output_type, chain, rewriter);
  if (failed(output_desc_or)) {
    return output_desc_or;
  }

  // Create filter descriptor.
  cudnnTensorFormat_t tensor_format;
  switch (config.filter_descriptor.layout()) {
    case se::dnn::FilterLayout::kOutputInputYX:
      tensor_format = CUDNN_TENSOR_NCHW;
      break;
    case se::dnn::FilterLayout::kOutputYXInput:
      tensor_format = CUDNN_TENSOR_NHWC;
      break;
    case se::dnn::FilterLayout::kOutputInputYX4:
    case se::dnn::FilterLayout::kOutputInputYX32: {
      tensor_format = CUDNN_TENSOR_NCHW_VECT_C;
      break;
    }
    default:
      return rewriter.notifyMatchFailure(op, "Unexpected filter layout.");
  }
  std::vector<int> dims(2 + config.filter_descriptor.ndims());
  dims[0] = config.filter_descriptor.output_feature_map_count();
  dims[1] = config.filter_descriptor.input_feature_map_count();
  absl::Span<const int64_t> spatial_dims =
      config.filter_descriptor.input_filter_dims();
  std::copy(spatial_dims.begin(), spatial_dims.end(), dims.begin() + 2);
  auto loc = op.getLoc();
  Value filter_desc = rewriter.create<tfrt::gpu::DnnCreateFilterDescriptorOp>(
      loc, filter_type, tensor_format, rewriter.getI32ArrayAttr(dims));

  // Create convolution descriptor.
  mlir::Type mlir_compute_type = GetMemRefElementType(GetInput(op));
  tfrt::gpu::wrapper::DnnDataType compute_type =
      MlirTypeToDnnDataType(mlir_compute_type);
  cudnnConvolutionMode_t conv_mode =
      config.conv_desc.convolution_not_crosscorr() ? CUDNN_CONVOLUTION
                                                   : CUDNN_CROSS_CORRELATION;
  const auto& convolution_descriptor = config.conv_desc;
  // cuDNN requires arrays of ints.
  std::vector<int> strides = CheckedNarrowing(convolution_descriptor.strides());
  std::vector<int> padding = CheckedNarrowing(convolution_descriptor.padding());
  std::vector<int> dilations =
      CheckedNarrowing(convolution_descriptor.dilations());
  cudnnMathType_t math_type = config.algorithm.tensor_ops_enabled()
                                  ? CUDNN_TENSOR_OP_MATH
                                  : CUDNN_FMA_MATH;
  Value conv_desc =
      rewriter.create<tfrt::gpu::DnnCreateConvolutionDescriptorOp>(
          loc, compute_type, conv_mode, math_type,
          rewriter.getI32ArrayAttr(padding), rewriter.getI32ArrayAttr(strides),
          rewriter.getI32ArrayAttr(dilations));

  // Create convolution op.
  mlir::Type mlir_scale_type =
      mlir_compute_type.isF64() ? rewriter.getF64Type() : rewriter.getF32Type();
  Value context = rewriter.create<tfrt::gpu::StreamGetContextOp>(loc, stream);
  Value handle = rewriter.create<tfrt::gpu::DnnCreateOp>(loc, context);
  auto out_chain_or = CreateLegacyConvOp(
      op, adaptor, mlir_scale_type, handle, stream, *input_desc_or,
      *output_desc_or, filter_desc, conv_desc, config.algorithm.algo_id(),
      chain, config, rewriter);
  if (succeeded(out_chain_or)) {
    rewriter.eraseOp(op);
  }
  return out_chain_or;
}

template <class ConvolutionOpType>
struct ConvolutionRewritePattern
    : tfrt::gpu::StreamifyOpConversionPattern<ConvolutionOpType> {
  using typename tfrt::gpu::StreamifyOpConversionPattern<
      ConvolutionOpType>::OpAdaptor;
  using tfrt::gpu::StreamifyOpConversionPattern<
      ConvolutionOpType>::StreamifyOpConversionPattern;
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

    bool use_legacy_conv = [&] {
      if (!config.algorithm.is_cudnn_frontend()) return true;

      auto print_reason = [&](const char* reason) {
        LOG(ERROR)
            << "Disabling cuDNN frontend for the following convolution:\n"
            << "  input: " << config.input_descriptor.ToString() << "\n"
            << "  filter: " << config.filter_descriptor.ToString() << "\n"
            << "  conv: " << config.conv_desc.ToString() << "\n... because "
            << reason;
      };

      if (config.input_descriptor.layout() ==
          se::dnn::DataLayout::kBatchDepthYX32)
        // Current versions of the frontend API lack support for Tx32.
        return print_reason("Tx32 convolutions are unsupported."), true;

      if (CUDNN_VERSION < 8100)
        // cuDNN frontend support became sufficiently stable to use in 8.1.
        return print_reason("the cuDNN version does not support it."), true;

      return false;
    }();

    if (use_legacy_conv) {
      return LegacyConvolutionRewritePattern(op, adaptor, chain, stream, config,
                                             rewriter);
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
    Location loc = op->getLoc();

    // Create a function that returns the convolution plan.
    mlir::SymbolTable symbol_table(
        op->template getParentOfType<mlir::ModuleOp>());
    rewriter.setInsertionPoint(
        op->template getParentOfType<mlir::func::FuncOp>());
    mlir::Type handle_type = rewriter.getType<tfrt::gpu::DnnHandleType>();
    mlir::Type conv_plan_type =
        rewriter.getType<tfrt::gpu::DnnConvolutionPlanType>();
    std::string function_name =
        absl::StrCat("get_", op->getName().stripDialect().str(), "_plan");
    mlir::func::FuncOp conv_plan_func = rewriter.create<mlir::func::FuncOp>(
        loc, function_name,
        rewriter.getFunctionType(handle_type, conv_plan_type));
    symbol_table.insert(conv_plan_func);
    rewriter.setInsertionPointToEnd(conv_plan_func.addEntryBlock());
    Value conv_plan = CreateBuildConvPlanOp(op, conv_plan_func.getArgument(0),
                                            config, backend_type, rewriter);
    rewriter.create<tfrt::compiler::ReturnOp>(loc, conv_plan);

    // Once-initialize the convolution plan.
    rewriter.restoreInsertionPoint(saved_point);
    Value context = rewriter.create<tfrt::gpu::StreamGetContextOp>(loc, stream);
    Value handle = rewriter.create<tfrt::gpu::DnnCreateOp>(loc, context);
    auto once_op = rewriter.create<tfrt::compiler::OnceOp>(
        loc, conv_plan_func.getFunctionType().getResults(), handle,
        conv_plan_func.getName());

    chain = CreateRunConvolutionOp(adaptor, loc, handle, once_op.getResult(0),
                                   chain, stream, rewriter);
    rewriter.eraseOp(op);
    return chain;
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
