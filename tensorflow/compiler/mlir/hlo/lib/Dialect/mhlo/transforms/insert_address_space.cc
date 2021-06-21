
#include <string>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/MLIRContext.h"              // TF:llvm-project
#include "mlir/Pass/Pass.h"                   // TF:local_config_mlir
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/utils/hlo_utils.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/utils/lhlo_utils.h"

using llvm::StringRef;
using std::string;

namespace mlir {

namespace lmhlo {
namespace {

Value MaybeGetCastedMemref(OpBuilder& builder, Value& memref) {
  auto type = memref.getType().dyn_cast<ShapedType>();
  if (type.getNumDynamicDims() < type.getRank()) {
    for (auto user : memref.getUsers()) {
      if (isa<memref::CastOp>(user)) {
        builder.setInsertionPointAfter(user);
        auto casted_memref = user->getResult(0);
        return MaybeGetCastedMemref(builder, casted_memref);
      }
    }
    // If no MemrefCastOp found, probably the alloc itself is partial dynamic
    // (e.g. output of fully connnected layer is partial dyanmic since the
    // shape of weight is static). Returns directly in this situation.
    return memref;
  }
  return memref;
}

struct InsertAddressSpace
    : public PassWrapper<InsertAddressSpace, OperationPass<ModuleOp>> {
  InsertAddressSpace() = default;
  InsertAddressSpace(const InsertAddressSpace& o) {}
  StringRef getArgument() const final { return "insert-address-space"; }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::vector<FuncOp> func_ops;
    module.walk([&](FuncOp func_op) { func_ops.emplace_back(func_op); });

    for (auto func : func_ops) {
      func.walk([&](memref::AllocOp alloc) {
        if (!isDeviceAlloc(alloc)) {
          return;
        }
        // Assume there is only one func.
        // TODO: To handle control flow.
        OpBuilder builder(alloc);
        builder.setInsertionPointAfter(alloc);
        SmallVector<Value, 4> operands;
        auto memref = alloc.getResult();
        // If the memref is static shaped, looking for the memref::CastOp
        // The marker can only be attached to a dynamic shaped memref.
        // The Cananonlizer pass is responsible to add the memref::CastOp
        memref = MaybeGetCastedMemref(builder, memref);
        // assert(!memref.getType().dyn_cast<ShapedType>().hasStaticShape());
        operands.push_back(memref);

        auto memref_type = memref.getType().cast<MemRefType>();
        string addr_space_str = "device_type_";
        absl::StrAppendFormat(&addr_space_str, "%dd", memref_type.getRank());
        if (memref_type.getElementType() == builder.getF32Type()) {
          absl::StrAppend(&addr_space_str, "f32");
        } else if (memref_type.getElementType() == builder.getF16Type()) {
          absl::StrAppend(&addr_space_str, "f16");
        } else if (auto int_type =
                       memref_type.getElementType().dyn_cast<IntegerType>()) {
          absl::StrAppend(&addr_space_str,
                          Twine("i").concat(Twine(int_type.getWidth())).str());
        } else {
          alloc.dump();
          emitError(module.getLoc(),
                    "Element type of device_type not supported.");
          signalPassFailure();
        }
        assert(memref_type.hasRank());

        // create new allocOp with address space
        Attribute memSpace = StringAttr::get(&getContext(), addr_space_str);
        auto new_memref_ty = MemRefType::get(
            memref_type.getShape(), memref_type.getElementType(),
            memref_type.getAffineMaps(), memSpace);
        auto new_alloc_op = builder.create<memref::AllocOp>(
            alloc.getLoc(), new_memref_ty, alloc.getOperands());

        // Replace all uses with new AllocOp.
        // Process Func
        for (Operation* user : memref.getUsers()) {
          // process function
          if (isa<CallOp>(user)) {
            auto call_op = cast<CallOp>(user);
            // locate target operand
            Operation::operand_range operands = call_op.operands();
            int operand_idx = -1;
            for (auto indexed_operand : llvm::enumerate(operands)) {
              if (indexed_operand.value() == memref) {
                operand_idx = indexed_operand.index();
                break;
              }
            }
            assert(operand_idx != -1);
            // Get FuncOp
            auto module = call_op->getParentOfType<ModuleOp>();
            auto callee = module.lookupSymbol<FuncOp>(call_op.getCallee());
            FunctionType funcType = callee.getType();
            Block* entry_block = &callee.getBody().front();
            //  Replace old argument by new argument in block,
            Type argType = funcType.getInput(operand_idx);
            BlockArgument oldArgument = entry_block->getArgument(operand_idx);
            BlockArgument newArgument =
                entry_block->insertArgument(operand_idx, new_memref_ty);
            //  Replace all uses with new argument and
            oldArgument.replaceAllUsesWith(newArgument);
            entry_block->eraseArgument(oldArgument.getArgNumber());
            SmallVector<Type, 4> input_types;
            // Replace func type of operand_idx's argument
            for (int arg_idx = 0; arg_idx < funcType.getInputs().size();
                 ++arg_idx) {
              if (arg_idx == operand_idx) {
                input_types.push_back(new_memref_ty);
              } else {
                input_types.push_back(funcType.getInput(arg_idx));
              }
            }
            callee.setType(mlir::FunctionType::get(
                &getContext(), input_types, callee.getType().getResults()));
          }  // if isa<CallOp>(user)
        }    // for user
        // Process other Value
        memref.replaceAllUsesWith(new_alloc_op);
        alloc->erase();
      });
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createInsertAddressSpace() {
  return absl::make_unique<InsertAddressSpace>();
}

}  // namespace lmhlo
}  // namespace mlir
