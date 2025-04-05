#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"

using namespace mlir;

namespace {

unsigned getScratchSize128(Operation *) { return 128; }

enum class GetScratchSizeFunction {
  None,
  ValidConstant,
};

struct TestAllocationPass
    : public PassWrapper<TestAllocationPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAllocationPass);

  TestAllocationPass() = default;
  TestAllocationPass(const TestAllocationPass &other)
      : PassWrapper<TestAllocationPass, OperationPass<ModuleOp>>(other) {}

  StringRef getArgument() const final { return "test-print-allocation"; }
  StringRef getDescription() const final {
    return "print the result of the allocation pass";
  }

  ModuleAllocation getModuleAllocation() {
    switch (getScratchSizeFunction) {
    case GetScratchSizeFunction::None:
      return {getOperation()};
    case GetScratchSizeFunction::ValidConstant:
      return {getOperation(), getScratchSize128};
    }
    llvm_unreachable("Unhandled case");
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    // Convert to std::string can remove quotes from opName
    ModuleAllocation moduleAllocation = getModuleAllocation();
    moduleOp.walk([&](triton::FuncOp funcOp) {
      auto opName = SymbolTable::getSymbolName(funcOp).getValue().str();
      mlir::emitRemark(funcOp.getLoc(), opName);
      auto *allocation = moduleAllocation.getFuncData(funcOp);
      funcOp.walk([&](Operation *op) {
        auto scratchBufferId = allocation->getBufferId(op);
        if (scratchBufferId != Allocation::InvalidBufferId) {
          size_t offset = allocation->getOffset(scratchBufferId);
          size_t size = allocation->getAllocatedSize(scratchBufferId);
          mlir::emitRemark(op->getLoc())
              << (allocation->isVirtualBuffer(scratchBufferId) ? "virtual"
                                                               : "scratch")
              << " offset = " << offset << ", size = " << size;
        }
        if (op->getNumResults() < 1)
          return;
        for (Value result : op->getResults()) {
          auto bufferId = allocation->getBufferId(result);
          if (bufferId != Allocation::InvalidBufferId) {
            size_t offset = allocation->getOffset(bufferId);
            size_t size = allocation->getAllocatedSize(bufferId);
            mlir::emitRemark(op->getLoc())
                << "offset = " << offset << ", size = " << size;
          }
        }
      });
      mlir::emitRemark(funcOp.getLoc())
          << "size = " << allocation->getSharedMemorySize();
    });
  }

  Option<GetScratchSizeFunction> getScratchSizeFunction{
      *this, "get-scratch-size-function",
      llvm::cl::desc("Custom scratch size function to use"),
      llvm::cl::init(GetScratchSizeFunction::None),
      llvm::cl::values(
          clEnumValN(GetScratchSizeFunction::None, "None", "None (default)"),
          clEnumValN(GetScratchSizeFunction::ValidConstant, "ValidConstant",
                     "ValidConstant"))};
};

} // namespace

namespace mlir {
namespace test {
void registerTestAllocationPass() { PassRegistration<TestAllocationPass>(); }
} // namespace test
} // namespace mlir
