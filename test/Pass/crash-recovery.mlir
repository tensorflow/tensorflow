// RUN: mlir-opt %s -pass-pipeline='func(test-function-pass, test-pass-crash)' -pass-pipeline-crash-reproducer=%t -verify-diagnostics
// RUN: cat %t | FileCheck -check-prefix=REPRO %s

// expected-error@+1 {{A crash has been detected while processing the MLIR module}}
module {
  func @foo() {
    return
  }
}

// REPRO: configuration: -pass-pipeline='func(test-function-pass, test-pass-crash)'

// REPRO: module
// REPRO: func @foo() {
// REPRO-NEXT: return
