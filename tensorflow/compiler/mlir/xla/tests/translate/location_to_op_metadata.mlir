// RUN: tf-mlir-translate -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s --dump-input=always

// CHECK-LABEL: %main
func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc(unknown)
  return %0 : !mhlo.token
}

// CHECK: after-all
// CHECK-NOT: metadata

// -----

// CHECK-LABEL: %main
func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc("AfterAll")
  return %0 : !mhlo.token
}

// CHECK: after-all
// CHECK-SAME: metadata={op_name="AfterAll"}

// -----

// CHECK-LABEL: %main
func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc("name@function")
  return %0 : !mhlo.token
}

// CHECK: after-all
// CHECK-SAME: metadata={op_name="name"}

// -----

// CHECK-LABEL: %main
func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc("file_name":2:8)
  return %0 : !mhlo.token
}

// CHECK: after-all
// CHECK-SAME: metadata={source_file="file_name" source_line=2}
