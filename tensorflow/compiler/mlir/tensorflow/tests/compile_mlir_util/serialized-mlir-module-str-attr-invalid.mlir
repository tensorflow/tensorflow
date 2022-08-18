// RUN: not tf-mlir-translate -mlir-tf-str-attr-to-mlir %s 2>&1 | FileCheck %s

"builtin.totally @invalid MLIR module {here} <-"

// CHECK: could not parse MLIR module-:1:1: error: custom op 'builtin.totally' is unknown
