#include "mlir/IR/MLIRContext.h"
#include "mlir/TensorFlow/ControlFlowOps.h"

using namespace mlir;

// Register the TFControlFlow ops with the MLIRContext.
void initializeMLIRContext(MLIRContext *ctx) {
  TFControlFlow::registerOperations(*ctx);
}
