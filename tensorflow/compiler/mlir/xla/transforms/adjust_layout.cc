/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <vector>

#include "absl/types/span.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/type_to_shape.h"
#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/stream_executor/tpu/c_api_conversions.h"

namespace mlir {
namespace mhlo {
namespace {
class AdjustLayout : public PassWrapper<AdjustLayout, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mhlo::MhloDialect>();
  }

 public:
  StringRef getArgument() const final { return "xla-adjust-layout"; }
  StringRef getDescription() const final {
    return "Adjust layouts so infeed send & receive use the same format.";
  }

  static FailureOr<std::vector<int64_t>> GetTPUInfeedLayoutFromAPI(
      RankedTensorType t) {
    // Call the TPU API to determine the right infeed layout. Note that
    // this can fail if we're not running on a TPU-enabled node.
    // TODO(kramm): Move this into a separate pass. See b/184944903
    xla::Shape old_shape = xla::TypeToShape(t);
    XLA_Shape old_shape_c = {};
    XLA_Shape new_shape_c = {};
    TfTpu_ExecutorApiFn *executor = tensorflow::tpu::ExecutorApiFn();
    if (!tensorflow::tpu::IsInitialized(executor)) {
      return failure();
    }
    ApiConverter::ToC(old_shape, &old_shape_c);
    executor->TpuTransferManager_GetInfeedLayoutFn(&old_shape_c, &new_shape_c);
    xla::Shape new_shape = ApiConverter::FromC(&new_shape_c);
    ApiConverter::Free(&old_shape_c);
    ApiConverter::Free(&new_shape_c);

    auto minor_to_major = new_shape.layout().minor_to_major();
    return std::vector<int64_t>(minor_to_major.begin(), minor_to_major.end());
  }

  static FailureOr<Attribute> GetLayout(const Type &type, OpBuilder &rewriter) {
    auto i64_type = rewriter.getIntegerType(64);
    if (type.isa<TupleType>()) {
      auto tuple_type = type.dyn_cast<TupleType>();
      const auto &types = tuple_type.getTypes();
      llvm::SmallVector<mlir::Attribute> v;
      v.reserve(types.size());
      for (const mlir::Type &t : types) {
        auto layout = GetLayout(t, rewriter);
        if (failed(layout)) return failure();
        v.push_back(layout.getValue());
      }
      ArrayRef<Attribute> shape(v);
      return rewriter.getArrayAttr(shape);
    } else if (auto t = type.dyn_cast<RankedTensorType>()) {
      if (!t.hasStaticShape()) return failure();
      auto layout = GetTPUInfeedLayoutFromAPI(t);
      std::vector<int64_t> minor_to_major;
      if (succeeded(layout)) {
        minor_to_major = layout.getValue();
      } else {
        /* If we're not running on a TPU node, we might not be able to
         * actually call the part of the TPU API that gives us layout.
         * This happens e.g. for unit tests. Below we just create a reasonable
         * layout.  We sort by dimension size, which makes the layout agree with
         * the "correct" TPU layout in surprisingly many cases.
         * Note that the corresponding InfeedEnqueue op will be generated
         * through another path, and might still generate an (incompatible)
         * layout using the TPU API. Running legalize_tf.cc on non-TPU nodes
         * thus is a potential source of bugs.
         */
        minor_to_major.resize(t.getRank());
        std::iota(minor_to_major.begin(), minor_to_major.end(), 0);
        std::sort(minor_to_major.begin(), minor_to_major.end(),
                  [=](int64_t a, int64_t b) {
                    int64_t da = t.getDimSize(a);
                    int64_t db = t.getDimSize(b);
                    return da > db || (da == db && a > b);
                  });
      }
      std::vector<Attribute> elements;
      elements.reserve(minor_to_major.size());
      for (auto e : minor_to_major) {
        elements.push_back(rewriter.getIntegerAttr(i64_type, e));
      }
      return rewriter.getArrayAttr(elements);
    } else {
      return rewriter.getUnitAttr();  // e.g. tokens
    }
  }

  static void runOnInfeedOp(::mlir::mhlo::InfeedOp op) {
    OpBuilder builder(op.getContext());
    auto layout = GetLayout(op.getType(), builder);
    if (failed(layout)) return;
    op->setAttr("layout", layout.getValue());
  }

  void runOnFunction() override { getFunction().walk(runOnInfeedOp); }
};
}  // anonymous namespace

// Header for this is in passes.h, which pulls into many deps. NOLINTNEXTLINE
std::unique_ptr<Pass> CreateAdjustLayoutPass() {
  return std::make_unique<AdjustLayout>();
}

void RegisterAdjustLayoutPass() { static PassRegistration<AdjustLayout> pass; }

}  // namespace mhlo
}  // namespace mlir
