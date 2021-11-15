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
  }
  descriptor.backend_config.set_conv_result_scale(
      op.result_scale().convertToDouble());
}

// Specialization for convolution forward
xla::gpu::CudnnConvKind GetConvKind(lmhlo_gpu::ConvForwardOp op) {
  return xla::gpu::CudnnConvKind::kForward;
}
Value GetResult(lmhlo_gpu::ConvForwardOp op) { return op.output(); }
Value GetConvOp(lmhlo_gpu::ConvForwardOp op,
                lmhlo_gpu::ConvForwardOpAdaptor adaptor,
                cudnnDataType_t compute_type, Value handle,
                Value input_tensor_desc, Value output_tensor_desc,
                Value filter_desc, Value conv_desc, Value algo_const,
                ConversionPatternRewriter& rewriter) {
  return rewriter.create<tfrt::gpu::DnnConvolutionForwardOp>(
      op.getLoc(), handle, compute_type, input_tensor_desc, adaptor.input(),
      filter_desc, adaptor.filter(), conv_desc, algo_const, adaptor.scratch(),
      output_tensor_desc, adaptor.output());
}

// Specialization for convolution backward input
xla::gpu::CudnnConvKind GetConvKind(lmhlo_gpu::ConvBackwardInputOp op) {
  return xla::gpu::CudnnConvKind::kBackwardInput;
}
Value GetResult(lmhlo_gpu::ConvBackwardInputOp op) { return op.d_input(); }
Value GetConvOp(lmhlo_gpu::ConvBackwardInputOp op,
                lmhlo_gpu::ConvBackwardInputOpAdaptor adaptor,
                cudnnDataType_t compute_type, Value handle,
                Value input_tensor_desc, Value output_tensor_desc,
                Value filter_desc, Value conv_desc, Value algo_const,
                ConversionPatternRewriter& rewriter) {
  return rewriter.create<tfrt::gpu::DnnConvolutionBackwardDataOp>(
      op.getLoc(), handle, compute_type, filter_desc, adaptor.filter(),
      output_tensor_desc, adaptor.d_output(), conv_desc, algo_const,
      adaptor.scratch(), input_tensor_desc, adaptor.d_input());
}

// Specialization for convolution backward filter
xla::gpu::CudnnConvKind GetConvKind(lmhlo_gpu::ConvBackwardFilterOp op) {
  return xla::gpu::CudnnConvKind::kBackwardFilter;
}
Value GetResult(lmhlo_gpu::ConvBackwardFilterOp op) { return op.d_filter(); }
Value GetConvOp(lmhlo_gpu::ConvBackwardFilterOp op,
                lmhlo_gpu::ConvBackwardFilterOpAdaptor adaptor,
                cudnnDataType_t compute_type, Value handle,
                Value input_tensor_desc, Value output_tensor_desc,
                Value filter_desc, Value conv_desc, Value algo_const,
                ConversionPatternRewriter& rewriter) {
  return rewriter.create<tfrt::gpu::DnnConvolutionBackwardFilterOp>(
      op.getLoc(), handle, compute_type, input_tensor_desc, adaptor.input(),
      output_tensor_desc, adaptor.d_output(), conv_desc, algo_const,
      adaptor.scratch(), filter_desc, adaptor.d_filter());
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
    descriptor.kind = GetConvKind(op);
    FillConvDescriptor(op, GetResult(op), descriptor);
    auto config_or = xla::gpu::GetGpuConvConfig(descriptor, "");
    if (!config_or.ok()) {
      return rewriter.notifyMatchFailure(
          op, "Failed to get GPU convolution config.");
    }
    xla::gpu::GpuConvConfig config = config_or.ValueOrDie();

    mlir::Type element_type = [](Value value) {
      return value.getType().cast<mlir::MemRefType>().getElementType();
    }(GetResult(op));
    cudnnDataType_t data_type = MlirTypeToCudnnDataType(element_type);

    auto i32_array_attr = [&](ArrayRef<int64_t> v) {
      std::vector<int32_t> w(v.begin(), v.end());
      return rewriter.getI32ArrayAttr(w);
    };

    auto filter_layout = config.filter_descriptor.layout();
    auto filter_dims = config.filter_descriptor.full_dims(filter_layout);
    cudnnTensorFormat_t tensor_format;
    switch (filter_layout) {
      case stream_executor::dnn::kOutputInputYX:
        tensor_format = CUDNN_TENSOR_NCHW;
        break;
      case stream_executor::dnn::kOutputYXInput:
        tensor_format = CUDNN_TENSOR_NHWC;
        break;
      case stream_executor::dnn::kOutputInputYX4:
      case stream_executor::dnn::kOutputInputYX32:
        tensor_format = CUDNN_TENSOR_NCHW_VECT_C;
        break;
      default:
        return rewriter.notifyMatchFailure(op, "Unexpected filter layout.");
    }
    auto filter_desc = rewriter.create<tfrt::gpu::DnnCreateFilterDescriptorOp>(
        op.getLoc(), data_type, tensor_format, i32_array_attr(filter_dims),
        chain);

    auto input_layout = config.input_descriptor.layout();
    auto input_dims = config.input_descriptor.full_dims(input_layout);
    auto input_strides = config.input_descriptor.full_strides(input_layout);
    auto input_tensor_desc =
        rewriter.create<tfrt::gpu::DnnCreateTensorDescriptorOp>(
            op.getLoc(), data_type, i32_array_attr(input_dims),
            i32_array_attr(input_strides), chain);

    auto output_layout = config.output_descriptor.layout();
    auto output_dims = config.output_descriptor.full_dims(output_layout);
    auto output_strides = config.output_descriptor.full_strides(output_layout);
    auto output_tensor_desc =
        rewriter.create<tfrt::gpu::DnnCreateTensorDescriptorOp>(
            op.getLoc(), data_type, i32_array_attr(output_dims),
            i32_array_attr(output_strides), chain);

    // Use FP32 compute type for lower FP16 precision data types by default.
    Type mlir_compute_type =
        element_type.isF16() ? rewriter.getF32Type() : element_type;
    cudnnDataType_t compute_type = MlirTypeToCudnnDataType(mlir_compute_type);
    cudnnConvolutionMode_t conv_mode =
        config.conv_desc.convolution_not_crosscorr() ? CUDNN_CONVOLUTION
                                                     : CUDNN_CROSS_CORRELATION;
    const auto* conv_descriptor = &config.conv_desc;
    auto pad = xla::llvm_ir::AsArrayRef(conv_descriptor->padding());
    auto filter_stride = xla::llvm_ir::AsArrayRef(conv_descriptor->strides());
    auto dilation = xla::llvm_ir::AsArrayRef(conv_descriptor->dilations());
    auto conv_desc =
        rewriter.create<tfrt::gpu::DnnCreateConvolutionDescriptorOp>(
            op.getLoc(), compute_type, conv_mode, i32_array_attr(pad),
            i32_array_attr(filter_stride), i32_array_attr(dilation), chain);

    auto handle = rewriter.create<tfrt::gpu::DnnCreateOp>(op.getLoc(), stream);
    auto algo_const = rewriter.create<tfrt::compiler::ConstantUI64Op>(
        op.getLoc(), config.algorithm.algorithm().value().algo_id());
    auto out_chain = GetConvOp(op, adaptor, compute_type, handle,
                               input_tensor_desc, output_tensor_desc,
                               filter_desc, conv_desc, algo_const, rewriter);
    rewriter.eraseOp(op);
    return out_chain;
  }
};

}  // namespace

void populateConvolutionConversionPattern(RewritePatternSet& patterns,
                                          TypeConverter& converter) {
  patterns.add<ConvolutionRewritePattern<lmhlo_gpu::ConvForwardOp>,
               ConvolutionRewritePattern<lmhlo_gpu::ConvBackwardInputOp>,
               ConvolutionRewritePattern<lmhlo_gpu::ConvBackwardFilterOp>>(
      converter, patterns.getContext());
}

}  // namespace tensorflow
