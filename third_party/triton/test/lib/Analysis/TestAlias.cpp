#include "mlir/IR/AsmState.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Alias.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;

namespace {

struct TestAliasPass
    : public PassWrapper<TestAliasPass, OperationPass<triton::FuncOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAliasPass);

  static std::string getValueOperandName(Value value, AsmState &state) {
    std::string opName;
    llvm::raw_string_ostream ss(opName);
    value.printAsOperand(ss, state);
    return opName;
  }

  static void emit(Location loc, StringRef name,
                   SmallVector<std::string> &vals) {
    if (vals.empty())
      return;
    InFlightDiagnostic diag = mlir::emitRemark(loc);
    diag << name << " -> ";
    size_t i = 0;
    for (auto val : vals) {
      if (i != 0)
        diag << ",";
      diag << val;
      ++i;
    }
  }

  StringRef getArgument() const final { return "test-print-alias"; }
  StringRef getDescription() const final {
    return "print the result of the alias analysis pass";
  }

  void runOnOperation() override {
    Operation *operation = getOperation();

    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    SharedMemoryAliasAnalysis *analysis =
        solver->load<SharedMemoryAliasAnalysis>();
    if (failed(solver->initializeAndRun(operation)))
      return signalPassFailure();

    AsmState state(operation->getParentOfType<ModuleOp>());
    // Get operation ids of value's aliases
    auto getLocalAllocOpNames = [&](Value value) {
      dataflow::Lattice<AliasInfo> *latticeElement =
          analysis->getLatticeElement(value);
      SmallVector<std::string> opNames;
      if (latticeElement) {
        auto &info = latticeElement->getValue();
        for (auto &alias : info.getAllocs()) {
          auto opName =
              getValueOperandName(alias.getDefiningOp()->getResult(0), state);
          opNames.push_back(std::move(opName));
        }
      }
      // Ensure deterministic output
      std::sort(opNames.begin(), opNames.end());
      return opNames;
    };

    operation->walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (op->getNumResults() < 1) {
        // cond br, br
        if (auto branch = dyn_cast<BranchOpInterface>(op)) {
          auto *block = branch->getBlock();
          for (auto arg : llvm::enumerate(block->getArguments())) {
            auto operand = block->getArgument(arg.index());
            auto opNames = getLocalAllocOpNames(operand);
            auto argName = getValueOperandName(arg.value(), state);
            emit(op->getLoc(), argName, opNames);
          }
        }
        return;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        for (auto arg : llvm::enumerate(forOp.getRegionIterArgs())) {
          auto operand = forOp.getTiedLoopInit(arg.value())->get();
          auto opNames = getLocalAllocOpNames(operand);
          auto argName = getValueOperandName(arg.value(), state);
          emit(op->getLoc(), argName, opNames);
        }
      }
      for (auto result : llvm::enumerate(op->getResults())) {
        auto opNames = getLocalAllocOpNames(result.value());
        auto resultName = getValueOperandName(result.value(), state);
        emit(op->getLoc(), resultName, opNames);
      }
    });
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestAliasPass() { PassRegistration<TestAliasPass>(); }
} // namespace test
} // namespace mlir
