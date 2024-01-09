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

#include "tensorflow/compiler/mlir/tensorflow/transforms/set_tpu_infeed_layout.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "xla/layout.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/shape.h"
#include "xla/stream_executor/tpu/c_api_conversions.h"
#include "xla/stream_executor/tpu/tpu_api.h"
#include "xla/translate/mhlo_to_hlo/type_to_shape.h"

namespace mlir {

static FailureOr<std::vector<int64_t>> GetTPUInfeedLayoutFromAPI(
    RankedTensorType t) {
  // Call the TPU API to determine the right infeed layout. Note that
  // this can fail if we're not running on a TPU-enabled node.
  // TODO(kramm): Move this into a separate pass. See b/184944903
  xla::Shape old_shape = xla::TypeToShape(t);
  XLA_Shape old_shape_c = {};
  XLA_Shape new_shape_c = {};
  TfTpu_ExecutorApiFn *executor = stream_executor::tpu::ExecutorApiFn();
  if (!stream_executor::tpu::IsInitialized(executor)) {
    return failure();
  }
  ApiConverter::ToC(old_shape, &old_shape_c);
  executor->TpuTransferManager_GetInfeedLayoutFn(&old_shape_c, &new_shape_c);
  xla::Shape new_shape = ApiConverter::FromC(&new_shape_c);
  ApiConverter::Destroy(&old_shape_c);
  ApiConverter::Destroy(&new_shape_c);

  auto minor_to_major = new_shape.layout().minor_to_major();
  return std::vector<int64_t>(minor_to_major.begin(), minor_to_major.end());
}

FailureOr<Attribute> GetTPUInfeedLayout(const ArrayRef<Type> types,
                                        OpBuilder &rewriter) {
  auto i64_type = rewriter.getIntegerType(64);
  if (types.size() > 1) {
    llvm::SmallVector<mlir::Attribute> v;
    v.reserve(types.size());
    for (const mlir::Type &t : types) {
      if (t.isa<mhlo::TokenType>()) continue;
      auto layout = GetTPUInfeedLayout({t}, rewriter);
      if (failed(layout)) return failure();
      v.push_back(layout.value());
    }
    ArrayRef<Attribute> shape(v);
    return rewriter.getArrayAttr(shape);
  } else if (types[0].isa<TupleType>()) {
    auto tuple_type = types[0].dyn_cast<TupleType>();
    const auto &types = tuple_type.getTypes();
    llvm::SmallVector<mlir::Attribute> v;
    v.reserve(types.size());
    for (const mlir::Type &t : types) {
      if (t.isa<mhlo::TokenType>()) continue;
      auto layout = GetTPUInfeedLayout({t}, rewriter);
      if (failed(layout)) return failure();
      v.push_back(layout.value());
    }
    ArrayRef<Attribute> shape(v);
    return rewriter.getArrayAttr(shape);
  } else if (auto t = types[0].dyn_cast<RankedTensorType>()) {
    if (!t.hasStaticShape()) return failure();
    auto layout = GetTPUInfeedLayoutFromAPI(t);
    std::vector<int64_t> minor_to_major;
    if (succeeded(layout)) {
      minor_to_major = layout.value();
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
    // types.size() == 1 and types[0] == TokenType
    // For this case, we return an empty array attribute.
    return rewriter.getArrayAttr({});
  }
}

bool SetTPUInfeedLayout(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module) {
  auto res = mlir_module->walk([&](mlir::TF::InfeedDequeueTupleOp op) {
    mlir::OpBuilder builder(op.getContext());
    std::vector<mlir::Type> result_types;

    for (mlir::Type t : op.getResultTypes()) {
      auto ty = t.cast<mlir::TensorType>();
      if (!ty.hasStaticShape()) return mlir::WalkResult::interrupt();
      result_types.push_back(t);
    }

    auto layout = mlir::GetTPUInfeedLayout(
        mlir::TupleType::get(builder.getContext(), result_types), builder);
    if (failed(layout)) return mlir::WalkResult::interrupt();
    // Do not append a UnitAttr for the "token" operand here to avoid
    // compilation failure when exporting the "layouts" attribute to a graph
    // node. Instead, add the UnitAttr during LegalizeTF pass.
    op->setAttr("layouts", layout.value());

    return mlir::WalkResult::advance();
  });
  return !res.wasInterrupted();
}

}  // namespace mlir
