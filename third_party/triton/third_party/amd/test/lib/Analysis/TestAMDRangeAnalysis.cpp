#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Analysis/RangeAnalysis.h"
#include "triton/Analysis/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct TestAMDRangeAnalysisPass
    : PassWrapper<TestAMDRangeAnalysisPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAMDRangeAnalysisPass)

  StringRef getArgument() const final {
    return "test-tritonamdgpu-range-analysis";
  }
  StringRef getDescription() const final {
    return "print the result of the tritonamdgpu-range-analysis pass";
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp mod = getOperation();

    // Collect assumptions in the function
    DenseMap<Value, SetVector<Operation *>> assumptions =
        AMD::TritonIntegerRangeAnalysis::collectAssumptions(getOperation());
    std::shared_ptr<DataFlowSolver> solver = createDataFlowSolver();
    solver->load<AMD::TritonIntegerRangeAnalysis>(assumptions);
    if (failed(solver->initializeAndRun(getOperation())))
      return signalPassFailure();

    auto nonNegativePred = [&solver](Value v) -> bool {
      if (const auto *r =
              solver->lookupState<dataflow::IntegerValueRangeLattice>(v)) {
        if (r->getValue().isUninitialized())
          return false;
        if (AMD::isEmptyInitializedRange(r->getValue().getValue()))
          return false;
      }
      return succeeded(dataflow::staticallyNonNegative(*solver, v));
    };

    mod->walk<WalkOrder::PreOrder>([&solver, nonNegativePred](Operation *op) {
      auto results = op->getResults();
      if (auto outputRanges = AMD::collectRanges(*solver, results)) {
        int i = 0;
        for (const auto &[res, outR] : llvm::zip(results, *outputRanges)) {
          std::string rangeS;
          llvm::raw_string_ostream rangeSt(rangeS);
          if (results.size() > 1)
            rangeSt << " result " << i << ": " << outR;
          else
            rangeSt << outR;
          i++;
          emitRemark(res.getLoc(), rangeS);
        }

        if (auto cmpOp = llvm::dyn_cast<arith::CmpIOp>(op)) {
          if (AMD::cmpIIsStaticallyTrue(*solver, cmpOp))
            emitRemark(op->getLoc(), "result is true");
        }
      }

      int i = 0;
      for (auto result : results) {
        if (nonNegativePred(result)) {
          std::string nonNegs;
          llvm::raw_string_ostream nonNegSt(nonNegs);
          if (results.size() > 1)
            nonNegSt << " result " << i << ": non-neg";
          else
            nonNegSt << "non-neg";
          emitRemark(result.getLoc(), nonNegs);
        }
        i++;
      }
    });
  }
};

} // namespace

namespace mlir::test {
void registerTestTritonAMDGPURangeAnalysis() {
  PassRegistration<TestAMDRangeAnalysisPass>();
}
} // namespace mlir::test
