#include "./RegisterTritonDialects.h"

#include "mlir/Tools/mlir-reduce/MlirReduceMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registerTritonDialects(registry);

  mlir::MLIRContext context(registry);
  return mlir::failed(mlir::mlirReduceMain(argc, argv, context));
}
