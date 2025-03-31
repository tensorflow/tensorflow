#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct TestAxisInfoPass
    : public PassWrapper<TestAxisInfoPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAxisInfoPass);

  StringRef getArgument() const final { return "test-print-alignment"; }
  StringRef getDescription() const final {
    return "print the result of the alignment analysis pass";
  }

  void runOnOperation() override {
    Operation *operation = getOperation();
    ModuleOp moduleOp = cast<ModuleOp>(operation);
    ModuleAxisInfoAnalysis moduleAxisInfoAnalysis(moduleOp);
    moduleOp.walk([&](FuncOp funcOp) {
      funcOp.walk([&](Operation *op) {
        if (op->getNumResults() < 1)
          return;
        for (Value result : op->getResults()) {
          InFlightDiagnostic diag = mlir::emitRemark(op->getLoc());
          diag << result;
          diag << " => ";
          auto *axisInfo = moduleAxisInfoAnalysis.getAxisInfo(result);
          if (axisInfo) {
            std::string str;
            llvm::raw_string_ostream os(str);
            axisInfo->print(os);
            diag << str;
          }
        }
      });
    });
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestAlignmentPass() { PassRegistration<TestAxisInfoPass>(); }
} // namespace test
} // namespace mlir
