// RUN: not tf-mlir-translate -mlir-tf-str-attr-to-mlir %s 2>&1 | FileCheck %s

"totally @invalid MLIR module {here} <-"

// CHECK: Invalid argument: could not parse MLIR module-:1:1: error: custom op 'totally' is unknown
