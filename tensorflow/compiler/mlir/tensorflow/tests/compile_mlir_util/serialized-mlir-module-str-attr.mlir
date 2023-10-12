// RUN: tf-mlir-translate -mlir-tf-str-attr-to-mlir %s -mlir-print-debuginfo -mlir-print-local-scope | FileCheck %s

"\0A\0Amodule attributes {tf.versions = {producer = 888 : i32}} {\0A func.func @main(%arg0: tensor<?xi32>) -> tensor<?xi32> {\0A %0 = \22tf.Identity\22(%arg0) : (tensor<?xi32>) -> tensor<?xi32> loc(unknown)\0A return %0 : tensor<?xi32> loc(unknown)\0A } loc(unknown)\0A} loc(unknown)"

// Test simple serialized computation consisting of a function named `main`
// with a tf.Identity op forwarding the function single argument to the function
// single result.

// CHECK-LABEL: module
// CHECK-SAME:  attributes {tf.versions = {producer = 888 : i32}} {
// CHECK-NEXT:   func @main([[ARG0:%.+]]: tensor<?xi32> loc({{.*}})) -> tensor<?xi32> {
// CHECK-NEXT:     [[IDENTITY:%.+]] = "tf.Identity"([[ARG0]]) : (tensor<?xi32>) -> tensor<?xi32> loc(unknown)
// CHECK-NEXT:     return [[IDENTITY]] : tensor<?xi32> loc(unknown)
// CHECK-NEXT:   } loc(unknown)
// CHECK-NEXT: } loc(unknown)
