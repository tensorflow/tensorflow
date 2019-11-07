// RUN: mlir-translate -verify-diagnostics -mlir-to-llvmir %s

// expected-error @+1 {{unsupported module-level operation}}
func @foo() {
  llvm.return
}
