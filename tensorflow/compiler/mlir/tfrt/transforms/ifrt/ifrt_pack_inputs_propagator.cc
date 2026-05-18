/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <memory>
#include <optional>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/host_runtime/tfrt_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/ifrt_constants.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

#define GEN_PASS_DEF_IFRTPACKINPUTSPROPAGATORPASS
#define GEN_PASS_DECL_IFRTPACKINPUTSPROPAGATORPASS
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/passes.h.inc"  // IWYU pragma: keep

inline int64_t RoundUp(int64_t value, int64_t align) {
  return (value + align - 1) & ~(align - 1);
}

// Mirror of planner helpers — kept in-pass for self-containment.
int64_t ElementByteSize(mlir::Type elt) {
  if (elt.isIntOrFloat()) {
    int64_t bw = elt.getIntOrFloatBitWidth();
    if (bw < 8 || bw % 8 != 0) return -1;
    return bw / 8;
  }
  if (llvm::isa<mlir::TF::Quint8Type, mlir::TF::Qint8Type>(elt)) return 1;
  if (llvm::isa<mlir::TF::Quint16Type, mlir::TF::Qint16Type>(elt)) return 2;
  if (llvm::isa<mlir::TF::Qint32Type>(elt)) return 4;
  return -1;
}

template <typename OpT>
std::optional<llvm::SmallVector<int64_t>> ResolveBoundsFromStaticShapes(
    OpT call, mlir::func::FuncOp callee, unsigned arg_idx) {
  auto idx_attr = callee.getArgAttrOfType<mlir::IntegerAttr>(
      arg_idx, "tf._static_shape_arg_idx");
  if (!idx_attr) return std::nullopt;
  const int orig_num_args = static_cast<int>(call.getArgs().size());
  int64_t ss_idx = idx_attr.getInt() - orig_num_args;
  mlir::ValueRange static_shapes = call.getStaticShapes();
  if (ss_idx < 0 || ss_idx >= static_cast<int64_t>(static_shapes.size())) {
    return std::nullopt;
  }
  mlir::Value shape_val = static_shapes[ss_idx];
  while (auto id = shape_val.getDefiningOp<mlir::TF::IdentityOp>()) {
    shape_val = id.getInput();
  }
  auto const_op = shape_val.getDefiningOp<mlir::TF::ConstOp>();
  if (!const_op) return std::nullopt;
  auto dense = llvm::dyn_cast<mlir::DenseIntElementsAttr>(const_op.getValue());
  if (!dense) return std::nullopt;
  llvm::SmallVector<int64_t> bounds;
  bounds.reserve(dense.getNumElements());
  for (const llvm::APInt& v : dense.getValues<llvm::APInt>()) {
    bounds.push_back(v.getSExtValue());
  }
  return bounds;
}

template <typename OpT>
int64_t BoundedOperandByteSize(OpT call, mlir::func::FuncOp callee,
                               unsigned i) {
  mlir::Value value = call.getArgs()[i];
  auto ranked = llvm::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!ranked) return -1;
  int64_t bytes_per_elt = ElementByteSize(ranked.getElementType());
  if (bytes_per_elt < 0) return -1;
  if (ranked.hasStaticShape()) {
    return ranked.getNumElements() * bytes_per_elt;
  }
  if (!callee) return -1;
  auto bounds = ResolveBoundsFromStaticShapes(call, callee, i);
  if (!bounds || static_cast<int>(bounds->size()) != ranked.getRank()) {
    return -1;
  }
  int64_t num_elements = 1;
  for (int d = 0; d < ranked.getRank(); ++d) {
    int64_t dim = ranked.getDimSize(d);
    if (mlir::ShapedType::isDynamic(dim)) {
      dim = (*bounds)[d];
    } else if (dim != (*bounds)[d]) {
      return -1;
    }
    num_elements *= dim;
  }
  return num_elements * bytes_per_elt;
}

class IfrtPackInputsPropagatorPass
    : public impl::IfrtPackInputsPropagatorPassBase<
          IfrtPackInputsPropagatorPass> {
 public:
  template <typename OpT>
  mlir::WalkResult ProcessCall(
      OpT call,
      const llvm::DenseMap<int64_t, mlir::func::FuncOp>& program_to_func,
      mlir::OpBuilder& builder) {
    auto group_ids_attr =
        call->template getAttrOfType<mlir::ArrayAttr>(kIfrtPackGroupIdsAttr);
    if (!group_ids_attr) return mlir::WalkResult::advance();

    llvm::SmallVector<int64_t> group_ids;
    group_ids.reserve(group_ids_attr.size());
    bool has_packed = false;
    for (mlir::Attribute a : group_ids_attr) {
      int64_t gid = llvm::cast<mlir::IntegerAttr>(a).getInt();
      group_ids.push_back(gid);
      if (gid >= 0) has_packed = true;
    }
    if (!has_packed) return mlir::WalkResult::advance();

    auto args = call.getArgs();
    if (static_cast<int>(args.size()) != static_cast<int>(group_ids.size())) {
      call.emitError() << "ifrt_pack_group_ids length (" << group_ids.size()
                       << ") does not match operand count (" << args.size()
                       << ")";
      return mlir::WalkResult::interrupt();
    }

    int64_t program_id = call.getProgramId();
    auto func_it = program_to_func.find(program_id);
    if (func_it == program_to_func.end()) {
      call.emitError() << "No callee func with program_id " << program_id;
      return mlir::WalkResult::interrupt();
    }
    mlir::func::FuncOp callee = func_it->second;

    constexpr int64_t kAlignBytes = 16;
    llvm::DenseMap<int64_t, int64_t> running_offset_per_group;
    llvm::SmallVector<int64_t> pack_offsets(args.size(), 0);

    for (size_t i = 0; i < args.size(); ++i) {
      if (group_ids[i] < 0) continue;
      int64_t bytes =
          BoundedOperandByteSize(call, callee, static_cast<unsigned>(i));
      if (bytes < 0) {
        call.emitError() << "Operand " << i
                         << " is group-packed but its byte size couldn't be "
                            "resolved";
        return mlir::WalkResult::interrupt();
      }
      int64_t& offset = running_offset_per_group[group_ids[i]];
      offset = RoundUp(offset, kAlignBytes);
      pack_offsets[i] = offset;
      offset += bytes;
    }

    call->setAttr(kIfrtPackOffsetsAttr, builder.getI64ArrayAttr(pack_offsets));
    callee->setAttr(kIfrtPackGroupIdsAttr, group_ids_attr);
    callee->setAttr(kIfrtPackOffsetsAttr,
                    builder.getI64ArrayAttr(pack_offsets));
    return mlir::WalkResult::advance();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(&getContext());

    llvm::DenseMap<int64_t, mlir::func::FuncOp> program_to_func;
    for (auto func : module.getOps<mlir::func::FuncOp>()) {
      if (auto pid = func->getAttrOfType<mlir::IntegerAttr>(
              "tfrt_ifrt_serving.program_id")) {
        program_to_func[pid.getInt()] = func;
      }
    }

    bool interrupted = false;
    module.walk([&](mlir::TF::IfrtCallOp call) {
      if (ProcessCall(call, program_to_func, builder).wasInterrupted()) {
        interrupted = true;
      }
    });
    if (!interrupted) {
      module.walk([&](mlir::TF::AsyncIfrtCallOp call) {
        if (ProcessCall(call, program_to_func, builder).wasInterrupted()) {
          interrupted = true;
        }
      });
    }
    if (interrupted) return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtPackInputsPropagatorPass() {
  return std::make_unique<IfrtPackInputsPropagatorPass>();
}

}  // namespace ifrt_serving
}  // namespace tensorflow
