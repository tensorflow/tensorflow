// Copyright 2022 The TensorFlow Runtime Authors
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

// Pattern to lower lmhlo.fft op to tfrt dialect.
#include <cstdint>
#include <functional>
#include <string>

#include "mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "llvm/ADT/StringRef.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/pattern_utils.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime
#include "tfrt/gpu/wrapper/cublas_wrapper.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime

namespace tensorflow {
namespace {

struct FftRewritePattern
    : tfrt::gpu::GpuAsyncOpConversionPattern<lmhlo::FftOp> {
  using tfrt::gpu::GpuAsyncOpConversionPattern<lmhlo::FftOp>::OpAdaptor;
  using tfrt::gpu::GpuAsyncOpConversionPattern<
      lmhlo::FftOp>::GpuAsyncOpConversionPattern;
  FailureOr<Value> matchAndRewriteOp(
      lmhlo::FftOp op, OpAdaptor adaptor, Value chain, Value stream,
      ConversionPatternRewriter& rewriter) const override {
    xla::Shape input_shape = xla::gpu::GetShape(op.operand());
    int64_t input_rank = input_shape.rank();
    int64_t input_dims[3];
    for (int i = 0; i < input_rank; i++)
      input_dims[i] = input_shape.dimensions(i);
    auto input_shape_attr =
        rewriter.getI64ArrayAttr(ArrayRef<int64_t>(input_dims, input_rank));

    xla::Shape output_shape = xla::gpu::GetShape(op.output());
    int64_t output_rank = output_shape.rank();
    int64_t output_dims[3];
    for (int i = 0; i < output_rank; i++)
      output_dims[i] = output_shape.dimensions(i);
    auto output_shape_attr =
        rewriter.getI64ArrayAttr(ArrayRef<int64_t>(output_dims, output_rank));

    bool double_precision = input_shape.element_type() == xla::F64 ||
                            input_shape.element_type() == xla::C128;

    llvm::StringRef old_fft_type = adaptor.fft_typeAttr().getValue();
    tfrt::gpu::wrapper::FftType fft_type;
    if (old_fft_type == "FFT") {
      fft_type = double_precision ? tfrt::gpu::wrapper::FftType::kZ2ZForward
                                  : tfrt::gpu::wrapper::FftType::kC2CForward;
    } else if (old_fft_type == "RFFT") {
      fft_type = double_precision ? tfrt::gpu::wrapper::FftType::kD2Z
                                  : tfrt::gpu::wrapper::FftType::kR2C;
    } else if (old_fft_type == "IFFT") {
      fft_type = double_precision ? tfrt::gpu::wrapper::FftType::kZ2ZInverse
                                  : tfrt::gpu::wrapper::FftType::kC2CInverse;
    } else if (old_fft_type == "IRFFT") {
      fft_type = double_precision ? tfrt::gpu::wrapper::FftType::kZ2D
                                  : tfrt::gpu::wrapper::FftType::kC2R;
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported fft type");
    }

    auto handle =
        rewriter.create<tfrt::gpu::FftCreateHandleOp>(op->getLoc(), stream);
    // TODO : remove casts
    chain = rewriter.create<tfrt::gpu::FftCreatePlanOp>(
        op->getLoc(), stream, handle, (uint64_t)fft_type,
        adaptor.fft_lengthAttr(), input_shape_attr, output_shape_attr, chain);
    chain = rewriter.create<tfrt::gpu::FftExecOp>(
        op->getLoc(), stream, handle, adaptor.operand(), adaptor.output(),
        adaptor.fft_lengthAttr(), (uint64_t)fft_type, input_shape_attr,
        output_shape_attr, chain);
    rewriter.eraseOp(op);
    return chain;
  }
};

}  // namespace

void populateFftConversionPattern(RewritePatternSet& patterns,
                                  TypeConverter& converter) {
  patterns.add<FftRewritePattern>(converter, patterns.getContext());
}

}  // namespace tensorflow
