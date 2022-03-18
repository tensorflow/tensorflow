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
#include <numeric>
#include <string>
#include <utility>

#include "mlir-hlo/Dialect/lhlo_gpu/IR/lhlo_gpu_ops.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ScopedPrinter.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/lmhlo_to_gpu/pattern_utils.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime
#include "tfrt/gpu/wrapper/cufft_wrapper.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime

namespace tensorflow {

static llvm::Expected<tfrt::gpu::wrapper::FftType> GetFftType(
    llvm::StringRef type, bool double_precision) {
  llvm::Expected<int> value =
      llvm::StringSwitch<llvm::Expected<int>>(type)
          .Case("FFT", double_precision ? CUFFT_Z2Z : CUFFT_C2C)
          .Case("IFFT", double_precision ? CUFFT_Z2Z : CUFFT_C2C)
          .Case("RFFT", double_precision ? CUFFT_D2Z : CUFFT_R2C)
          .Case("IRFFT", double_precision ? CUFFT_Z2D : CUFFT_C2R)
          .Default(tfrt::MakeStringError("Unsupported FFT type: ", type));
  if (!value) return value.takeError();
  return tfrt::gpu::wrapper::FftType(*value, kGpuTargetPlatform);
}

static llvm::Expected<tfrt::gpu::wrapper::FftDirection> GetFftDirection(
    llvm::StringRef type) {
  llvm::Expected<int> value =
      llvm::StringSwitch<llvm::Expected<int>>(type)
          .Case("FFT", CUFFT_FORWARD)
          .Case("IFFT", CUFFT_INVERSE)
          .Case("RFFT", CUFFT_FORWARD)
          .Case("IRFFT", CUFFT_INVERSE)
          .Default(tfrt::MakeStringError("Unsupported FFT type: ", type));
  if (!value) return value.takeError();
  return tfrt::gpu::wrapper::FftDirection(*value, kGpuTargetPlatform);
}

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
    xla::Shape output_shape = xla::gpu::GetShape(op.output());
    if (input_shape.is_dynamic() || output_shape.is_dynamic())
      return rewriter.notifyMatchFailure(op, "expected static shapes");
    if (!xla::LayoutUtil::IsMonotonicWithDim0Major(input_shape.layout()) ||
        !xla::LayoutUtil::IsMonotonicWithDim0Major(output_shape.layout())) {
      return rewriter.notifyMatchFailure(op, "expected dense row-major");
    }

    bool double_precision = input_shape.element_type() == xla::F64 ||
                            input_shape.element_type() == xla::C128;
    auto type = GetFftType(mlir::mhlo::stringifyFftType(adaptor.fft_type()),
                           double_precision);
    auto direction =
        GetFftDirection(mlir::mhlo::stringifyFftType(adaptor.fft_type()));
    if (!type || !direction) {
      auto error = joinErrors(type.takeError(), direction.takeError());
      return rewriter.notifyMatchFailure(op, llvm::toString(std::move(error)));
    }

    llvm::SmallVector<int64_t, 3> dimensions;
    llvm::copy(op.fft_length().getValues<int64_t>(),
               std::back_inserter(dimensions));
    int rank = dimensions.size();

    auto batch_dims = input_shape.dimensions();
    uint64_t batch =
        std::accumulate(batch_dims.begin(), batch_dims.end() - rank, 1,
                        std::multiplies<int64_t>());

    auto get_strides = [](absl::Span<const int64_t> dims) {
      llvm::SmallVector<int64_t, 4> strides(dims.size() + 1, 1);
      std::partial_sum(dims.rbegin(), dims.rend(), strides.rbegin() + 1,
                       std::multiplies<int64_t>());
      return strides;
    };
    llvm::SmallVector<int64_t, 4> input_strides =
        get_strides(input_shape.dimensions().last(rank));
    llvm::SmallVector<int64_t, 4> output_strides =
        get_strides(output_shape.dimensions().last(rank));

    mlir::Location loc = op->getLoc();
    Value context = rewriter.create<tfrt::gpu::StreamGetContextOp>(loc, stream);

    auto handle = rewriter.create<tfrt::gpu::FftCreateOp>(
        loc, context, *type, batch, rewriter.getI64ArrayAttr(dimensions),
        rewriter.getI64ArrayAttr(input_strides),
        rewriter.getI64ArrayAttr(output_strides));

    // Note: we could determine the workspace size during lowering similar to
    // convolutions because the dimensions are static. But it's unclear if we
    // really want the compiler to depend on cuFFT/hipFFT, and the expensive
    // part is the allocation, which is currently not hoisted.
    mlir::Value workspace_size =
        rewriter.create<tfrt::gpu::FftGetWorkspaceSizeOp>(loc, handle);
    mlir::Value allocator =
        rewriter.create<tfrt::gpu::AllocatorCreateOp>(loc, context);
    mlir::Value workspace = rewriter.create<tfrt::gpu::MemAllocateOp>(
        loc, allocator, stream, workspace_size, chain);

    chain = rewriter.create<tfrt::gpu::FftExecuteOp>(
        loc, stream, handle, adaptor.operand(), adaptor.output(), workspace,
        *direction, chain);

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
