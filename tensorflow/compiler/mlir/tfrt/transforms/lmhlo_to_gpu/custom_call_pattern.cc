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
#include <functional>
#include <iterator>
#include <string>

#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "tensorflow/compiler/xla/service/gpu/xlir_ops.h"
#include "tfrt/gpu/kernels/gpu_ops.h"  // from @tf_runtime
#include "tfrt/gpu/passes/passes.h"  // from @tf_runtime
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime

namespace tensorflow {
namespace {

struct CustomCallRewritePattern
    : tfrt::gpu::StreamifyOpConversionPattern<lmhlo::CustomCallOp> {
  using tfrt::gpu::StreamifyOpConversionPattern<
      lmhlo::CustomCallOp>::StreamifyOpConversionPattern;
  FailureOr<Value> matchAndRewriteOp(
      lmhlo::CustomCallOp op, OpAdaptor adaptor, Value chain, Value stream,
      ConversionPatternRewriter& rewriter) const override {
    llvm::SmallVector<int32_t, 4> indices;
    if (auto mapping = op.getTargetArgMapping()) {
      auto num_args = mapping->getNumArgs();
      auto num_results = mapping->getNumResults();
      indices.resize(num_args + num_results, -1);
      for (const auto& pair : llvm::enumerate(mapping->getArgsToTargetArgs())) {
        indices[pair.value()] = pair.index();
      }
      for (const auto& pair :
           llvm::enumerate(mapping->getResultsToTargetResults())) {
        indices[pair.value() + num_args] = pair.index() + op.getArgs().size();
      }
    } else {
      int32_t num_indices = op.getArgs().size() + op.getOutput().size();
      indices.reserve(num_indices);
      llvm::copy(llvm::seq(0, num_indices), std::back_inserter(indices));
    }

    Value result = rewriter.create<xla::gpu::CustomCallOp>(
        op.getLoc(), chain.getType(), stream, adaptor.getOperands(), chain,
        op.getCallTargetNameAttr(), rewriter.getI32ArrayAttr(indices),
        op.getBackendConfigAttr());
    rewriter.eraseOp(op);
    return result;
  }
};

}  // namespace

void populateCustomCallConversionPattern(RewritePatternSet& patterns,
                                         TypeConverter& converter) {
  patterns.add<CustomCallRewritePattern>(converter, patterns.getContext());
}

}  // namespace tensorflow
