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

// Pattern to lower lmhlo_gpu.cholesky op to tfrt_gpu dialect.
#include <functional>
#include <string>

#include "mlir-hlo/Dialect/mhlo/IR/lhlo_gpu_ops.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime
#include "tfrt/gpu/wrapper/cublas_wrapper.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime

namespace tensorflow {
namespace {

static cudaDataType_t MlirTypeToCudaDataType(mlir::Type type) {
  if (type.isF16()) return CUDA_R_16F;
  if (type.isF32()) return CUDA_R_32F;
  if (type.isF64()) return CUDA_R_64F;
  if (auto complex_type = type.dyn_cast<mlir::ComplexType>()) {
    auto element_type = complex_type.getElementType();
    if (element_type.isF32()) return CUDA_C_32F;
    if (element_type.isF64()) return CUDA_C_64F;
  }
  llvm_unreachable("unsupported type");
}

struct CholeskyRewritePattern
    : tfrt::gpu::GpuAsyncOpConversionPattern<lmhlo_gpu::CholeskyOp> {
  using tfrt::gpu::GpuAsyncOpConversionPattern<
      lmhlo_gpu::CholeskyOp>::GpuAsyncOpConversionPattern;
  FailureOr<Value> matchAndRewriteOp(
      lmhlo_gpu::CholeskyOp op, OpAdaptor adaptor, Value chain, Value stream,
      ConversionPatternRewriter& rewriter) const override {
    if (!llvm::all_of(adaptor.getOperands(), [](Value operand) {
          return operand.getType().isa<tfrt::gpu::BufferType>();
        }))
      return rewriter.notifyMatchFailure(op, "expected buffer operands");

    chain = rewriter
                .create<tfrt::gpu::MemCopyOp>(op.getLoc(), adaptor.output(),
                                              adaptor.input(), stream, chain)
                .getResult();

    auto handle =
        rewriter.create<tfrt::gpu::SolverCreateOp>(op.getLoc(), stream)
            .getResult();

    cublasFillMode_t fill_mode =
        op.is_lower() ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

    mlir::Type element_type =
        op.input().getType().cast<mlir::MemRefType>().getElementType();
    auto data_type = MlirTypeToCudaDataType(element_type);

    const xla::Shape shape = xla::gpu::GetShape(op.input());
    int rank = shape.dimensions_size();
    assert(rank >= 2);
    auto n = rewriter.create<tfrt::compiler::ConstantI32Op>(
        op.getLoc(), shape.dimensions(rank - 1));

    const auto& dims = shape.dimensions();
    int64_t batch_size = std::accumulate(
        dims.begin(), dims.end() - 2, int64_t{1}, std::multiplies<int64_t>());
    auto batch =
        rewriter.create<tfrt::compiler::ConstantI32Op>(op.getLoc(), batch_size);

    chain = rewriter
                .create<tfrt::gpu::SolverPotrfBatchOp>(
                    op.getLoc(), handle, fill_mode, n, data_type,
                    adaptor.output(), n, adaptor.info(), batch, chain)
                .getResult();
    rewriter.eraseOp(op);
    return chain;
  }
};

}  // namespace

void populateCholeskyConversionPattern(RewritePatternSet& patterns,
                                       TypeConverter& converter) {
  patterns.add<CholeskyRewritePattern>(converter, patterns.getContext());
}

}  // namespace tensorflow
