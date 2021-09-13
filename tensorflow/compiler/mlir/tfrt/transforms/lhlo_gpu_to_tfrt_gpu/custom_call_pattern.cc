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

//===- custom_call_pattern.cc
//-------------------------------------------------------------------------===//
//
// Pattern to lower lmhlo.custom_call op to tfrt_gpu/xlir dialect.
//
//===----------------------------------------------------------------------===//
#include "tensorflow/compiler/mlir/tfrt/transforms/lhlo_gpu_to_tfrt_gpu/custom_call_pattern.h"

#include <functional>
#include <string>

#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/service/gpu/xlir_ops.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/pass/pass.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime

namespace tensorflow {
namespace {

struct CustomCallRewritePattern
    : tfrt::gpu::GpuAsyncOpConversionPattern<lmhlo::CustomCallOp> {
  using tfrt::gpu::GpuAsyncOpConversionPattern<
      lmhlo::CustomCallOp>::GpuAsyncOpConversionPattern;
  FailureOr<Value> matchAndRewriteOp(
      lmhlo::CustomCallOp op, Value chain, Value stream,
      ArrayRef<Value> operands,
      ConversionPatternRewriter& rewriter) const override {
    if (!all_of(operands, [](Value operand) {
          return operand.getType().isa<tfrt::gpu::BufferType>();
        }))
      return rewriter.notifyMatchFailure(op, "expected buffer operands");

    if (operands.size() != op.args().size() + op.output().size()) {
      return rewriter.notifyMatchFailure(
          op,
          "Number of buffer operands does not match the number of op inputs "
          "and outputs.");
    }

    int64_t target_args_count, target_results_count;
    llvm::SmallVector<int64_t, 4> args_to_target_args,
        results_to_target_results;

    if (op.target_arg_mapping()) {
      lmhlo::CustomCallTargetArgMapping target_mapping =
          *op.target_arg_mapping();

      target_args_count = target_mapping.num_args().getInt();
      target_results_count = target_mapping.num_results().getInt();

      for (const auto& attr : target_mapping.args_to_target_args().getValue()) {
        args_to_target_args.push_back(attr.dyn_cast<IntegerAttr>().getInt());
      }
      for (const auto& attr :
           target_mapping.results_to_target_results().getValue()) {
        results_to_target_results.push_back(
            attr.dyn_cast<IntegerAttr>().getInt());
      }
    } else {
      target_args_count = op.args().size();
      target_results_count = op.output().size();
    }

    mlir::Type chain_type = rewriter.getType<tfrt::compiler::ChainType>();
    auto out_chain =
        rewriter
            .create<xla::gpu::CustomCallOp>(
                op.getLoc(), chain_type, stream, chain, operands,
                rewriter.getI64ArrayAttr(args_to_target_args),
                op.backend_config().str(),
                rewriter.getI64ArrayAttr(results_to_target_results),
                target_args_count, target_results_count)
            .getResult();
    rewriter.eraseOp(op);
    return out_chain;
  }
};

}  // namespace

void populateCustomCallConversionPattern(RewritePatternSet& patterns) {
  patterns.add<CustomCallRewritePattern>(patterns.getContext());
}

}  // namespace tensorflow
