// RUN: tf-mlir-translate -split-input-file -mlir-hlo-to-hlo-text --legalize-node-names=false %s | FileCheck %s --dump-input=always --check-prefixes=CHECK,NOLNN
// RUN: tf-mlir-translate -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s --dump-input=always --check-prefixes=CHECK,LNN

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

// -----

// CHECK-LABEL: %main
func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc("name(with)[]")
  return %0 : !mhlo.token
}

// CHECK: after-all
// NOLNN-SAME: metadata={op_name="name(with)[]"}
// LNN-SAME: metadata={op_name="name.with..."}

// -----

// CHECK-LABEL: %main
func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc("name(anothername)"("file_name":2:8))
  return %0 : !mhlo.token
}

// CHECK: after-all
// NOLNN-SAME: metadata={op_name="name(anothername)" source_file="file_name" source_line=2}
// LNN-SAME: metadata={op_name="name.anothername." source_file="file_name" source_line=2}
