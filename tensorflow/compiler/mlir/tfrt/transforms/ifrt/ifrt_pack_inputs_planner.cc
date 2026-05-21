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
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
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

#define GEN_PASS_DEF_IFRTPACKINPUTSPLANNERPASS
#define GEN_PASS_DECL_IFRTPACKINPUTSPLANNERPASS
#include "tensorflow/compiler/mlir/tfrt/transforms/ifrt/passes.h.inc"  // IWYU pragma: keep

inline constexpr int64_t kPackGroupIndividual = -1;
inline constexpr int64_t kPackGroupDefault = 0;

// Layout-fussy TF ops (Filter 4). Packing an operand that feeds one of these
// triggers an on-device relayout copy after the bitcast_convert unpack. On
// TPU especially, that copy can dominate the H2D launch savings.
const llvm::StringSet<>& LayoutFussyTfOps() {
  static const auto* kSet = new llvm::StringSet<>{
      "tf.MatMul",
      "tf.BatchMatMul",
      "tf.BatchMatMulV2",
      "tf.BatchMatMulV3",
      "tf.Conv2D",
      "tf.Conv2DBackpropFilter",
      "tf.Conv2DBackpropInput",
      "tf.Conv3D",
      "tf.DepthwiseConv2dNative",
      "tf.MaxPool",
      "tf.MaxPool3D",
      "tf.AvgPool",
      "tf.AvgPool3D",
      "tf.Einsum",
  };
  return *kSet;
}

bool FeedsLayoutFussyOp(mlir::func::FuncOp atom_func, unsigned arg_index) {
  if (arg_index >= atom_func.getNumArguments()) return false;
  for (mlir::OpOperand& use : atom_func.getArgument(arg_index).getUses()) {
    llvm::StringRef name = use.getOwner()->getName().getStringRef();
    if (LayoutFussyTfOps().contains(name)) return true;
  }
  return false;
}

template <typename OpT>
bool IsVariableArg(OpT call, int32_t operand_index) {
  for (mlir::Attribute attr : call.getVariableArgIndices()) {
    if (auto int_attr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
      if (int_attr.getInt() == operand_index) return true;
    }
  }
  return false;
}

// Per-element byte size of a pack-eligible tensor element type. Returns -1
// for unsupported types (sub-byte, complex, string, resource).
int64_t ElementByteSize(mlir::Type elt) {
  if (elt.isIntOrFloat()) {
    int64_t bw = elt.getIntOrFloatBitWidth();
    if (bw < 8 || bw % 8 != 0) return -1;
    return bw / 8;
  }
  // TF quantized types: storage is plain (u)int{8,16,32}.
  if (llvm::isa<mlir::TF::Quint8Type, mlir::TF::Qint8Type>(elt)) return 1;
  if (llvm::isa<mlir::TF::Quint16Type, mlir::TF::Qint16Type>(elt)) return 2;
  if (llvm::isa<mlir::TF::Qint32Type>(elt)) return 4;
  return -1;
}

// If `callee.getArgument(arg_idx)` carries a `tf._static_shape_arg_idx`
// attribute (set by PropagateStaticShapesPass), returns the upper-bound dim
// values resolved from the corresponding `static_shapes` operand of the
// caller. Returns nullopt if no bound is attached or the static-shape
// operand isn't a tf.Const.
template <typename OpT>
std::optional<llvm::SmallVector<int64_t>> ResolveBoundsFromStaticShapes(
    OpT call, mlir::func::FuncOp callee, unsigned arg_idx) {
  auto idx_attr = callee.getArgAttrOfType<mlir::IntegerAttr>(
      arg_idx, "tf._static_shape_arg_idx");
  if (!idx_attr) {
    LOG(INFO) << "IFRT Pack-Inputs: _static_shape_arg_idx is null";
    return std::nullopt;
  }
  LOG(INFO) << "IFRT Pack-Inputs: idx_attr: ";
  ;

  // tf._static_shape_arg_idx is an index into the callee's NEW signature
  // (originals + appended static_shape args). The corresponding caller-side
  // operand lives in `call.getStaticShapes()` at the matching offset.
  const int orig_num_args = static_cast<int>(call.getArgs().size());
  int64_t ss_idx = idx_attr.getInt() - orig_num_args;
  mlir::ValueRange static_shapes = call.getStaticShapes();
  if (ss_idx < 0 || ss_idx >= static_cast<int64_t>(static_shapes.size())) {
    LOG(INFO) << "IFRT Pack-Inputs: ss_idx out of bounds";
    return std::nullopt;
  }

  // Walk past identities to find the underlying tf.Const.
  mlir::Value shape_val = static_shapes[ss_idx];
  while (auto id = shape_val.getDefiningOp<mlir::TF::IdentityOp>()) {
    shape_val = id.getInput();
  }
  auto const_op = shape_val.getDefiningOp<mlir::TF::ConstOp>();
  if (!const_op) return std::nullopt;
  LOG(INFO) << "IFRT Pack-Inputs: const_op: ";
  auto dense = llvm::dyn_cast<mlir::DenseIntElementsAttr>(const_op.getValue());
  if (!dense) return std::nullopt;
  LOG(INFO) << "IFRT Pack-Inputs: dense: ";

  llvm::SmallVector<int64_t> bounds;
  bounds.reserve(dense.getNumElements());
  for (const llvm::APInt& v : dense.getValues<llvm::APInt>()) {
    bounds.push_back(v.getSExtValue());
  }
  return bounds;
}

// Returns byte size of operand at index `i`, treating bounded-dynamic dims
// via the upper bound from the IfrtCall's `static_shapes` operand. Returns
// -1 if not pack-eligible (unranked, truly unbounded, sub-byte, etc.).
template <typename OpT>
int64_t BoundedOperandByteSize(OpT call, mlir::func::FuncOp callee,
                               unsigned i) {
  mlir::Value value = call.getArgs()[i];
  auto ranked = llvm::dyn_cast<mlir::RankedTensorType>(value.getType());
  if (!ranked) return -1;
  int64_t bytes_per_elt = ElementByteSize(ranked.getElementType());
  LOG(INFO) << "IFRT Pack-Inputs: ElementByteSize: " << bytes_per_elt;
  if (bytes_per_elt < 0) {
    LOG(INFO) << "IFRT Pack-Inputs: bytes_per_elt < 0";
    return -1;
  }

  if (ranked.hasStaticShape()) {
    LOG(INFO) << "IFRT Pack-Inputs: ranked has static shape: "
              << ranked.getNumElements() * bytes_per_elt;
    return ranked.getNumElements() * bytes_per_elt;
  }

  // Bounded-dynamic: lift bounds out of the caller's static_shapes operand
  // via the callee's tf._static_shape_arg_idx attribute on this arg.
  // Requires that PropagateStaticShapesPass has already run, which the
  // production pipeline guarantees (it runs before SinkVariableAsNamedArray,
  // which is itself before this planner).
  if (!callee) {
    LOG(INFO) << "IFRT Pack-Inputs: callee is null";
    return -1;
  }
  auto bounds = ResolveBoundsFromStaticShapes(call, callee, i);
  if (!bounds || static_cast<int>(bounds->size()) != ranked.getRank()) {
    LOG(INFO) << "IFRT Pack-Inputs: bounds not resolved" << bounds.has_value();
    return -1;
  }
  int64_t num_elements = 1;
  for (int d = 0; d < ranked.getRank(); ++d) {
    int64_t dim = ranked.getDimSize(d);
    if (mlir::ShapedType::isDynamic(dim)) {
      dim = (*bounds)[d];
    } else if (dim != (*bounds)[d]) {
      LOG(INFO) << "IFRT Pack-Inputs: declared static dim " << dim
                << " disagrees with claimed upper bound " << (*bounds)[d];
      return -1;  // declared static dim disagrees with claimed upper bound
    }
    num_elements *= dim;
  }
  LOG(INFO) << "IFRT Pack-Inputs: num_elements: " << num_elements;
  return num_elements * bytes_per_elt;
}

class IfrtPackInputsPlannerPass
    : public impl::IfrtPackInputsPlannerPassBase<IfrtPackInputsPlannerPass> {
 public:
  using impl::IfrtPackInputsPlannerPassBase<
      IfrtPackInputsPlannerPass>::IfrtPackInputsPlannerPassBase;

  template <typename OpT>
  void ProcessCall(
      OpT call,
      const llvm::DenseMap<int64_t, mlir::func::FuncOp>& program_to_func,
      mlir::OpBuilder& builder) {
    const int num_args = static_cast<int>(call.getArgs().size());
    llvm::SmallVector<int64_t> group_ids(num_args, kPackGroupIndividual);

    auto callee_it = program_to_func.find(call.getProgramId());
    mlir::func::FuncOp callee = (callee_it == program_to_func.end())
                                    ? mlir::func::FuncOp{}
                                    : callee_it->second;

    for (int i = 0; i < num_args; ++i) {
      // Filter 3: skip on-device variable handles.
      if (IsVariableArg(call, i)) {
        LOG(INFO) << "IFRT Pack-Inputs: skipping operand " << i
                  << " (is variable)";
        continue;
      }
      // Filter 1 (mechanical) + 2 (economic): byte size resolvable (static
      // or bounded via static_shapes), under threshold.
      int64_t bytes =
          BoundedOperandByteSize(call, callee, static_cast<unsigned>(i));
      if (bytes < 0) {
        LOG(INFO) << "IFRT Pack-Inputs: skipping operand " << i
                  << " (unresolvable or ineligible shape/type)";
        continue;
      }
      if (bytes > size_threshold_bytes_) {
        LOG(INFO) << "IFRT Pack-Inputs: skipping operand " << i << " (" << bytes
                  << " bytes) (exceeds threshold " << size_threshold_bytes_
                  << ")";
        continue;
      }
      // Filter 4: skip args feeding layout-fussy consumers.
      if (callee && FeedsLayoutFussyOp(callee, static_cast<unsigned>(i))) {
        LOG(INFO) << "IFRT Pack-Inputs: skipping operand " << i
                  << " (feeds layout-fussy consumer)";
        continue;
      }
      VLOG(1) << "IFRT Pack-Inputs: packing operand " << i << " (" << bytes
              << " bytes) into group " << kPackGroupDefault;
      group_ids[i] = kPackGroupDefault;
    }

    call->setAttr(kIfrtPackGroupIdsAttr, builder.getI64ArrayAttr(group_ids));
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    VLOG(1) << "IFRT Pack-Inputs: Running Planner pass on module "
            << module.getName().value_or("").str();
    mlir::OpBuilder builder(&getContext());

    llvm::DenseMap<int64_t, mlir::func::FuncOp> program_to_func;
    for (auto func : module.getOps<mlir::func::FuncOp>()) {
      if (auto pid = func->getAttrOfType<mlir::IntegerAttr>(
              "tfrt_ifrt_serving.program_id")) {
        program_to_func[pid.getInt()] = func;
      }
    }

    module.walk([&](mlir::TF::IfrtCallOp call) {
      ProcessCall(call, program_to_func, builder);
    });
    module.walk([&](mlir::TF::AsyncIfrtCallOp call) {
      ProcessCall(call, program_to_func, builder);
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtPackInputsPlannerPass() {
  return std::make_unique<IfrtPackInputsPlannerPass>();
}

}  // namespace ifrt_serving
}  // namespace tensorflow
