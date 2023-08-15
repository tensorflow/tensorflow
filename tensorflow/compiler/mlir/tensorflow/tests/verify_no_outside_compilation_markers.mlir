// RUN: tf-opt -split-input-file -verify-diagnostics -verify-no-outside-compilation-markers %s | FileCheck %s

// CHECK-LABEL: func @has_compilation_markers
func.func @has_compilation_markers(%arg0: tensor<i32>) -> () {
  return
}
