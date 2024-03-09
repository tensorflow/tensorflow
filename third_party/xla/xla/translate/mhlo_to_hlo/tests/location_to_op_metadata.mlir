// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s --dump-input=always --check-prefixes=CHECK

// CHECK-LABEL: %main
func.func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc(unknown)
  func.return %0 : !mhlo.token
}

// CHECK: after-all
// CHECK-NOT: metadata

// -----

// CHECK-LABEL: %main
func.func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc("AfterAll")
  func.return %0 : !mhlo.token
}

// CHECK: after-all
// CHECK-SAME: metadata={op_name="AfterAll"}

// -----

// CHECK-LABEL: %main
func.func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc("name@function")
  func.return %0 : !mhlo.token
}

// CHECK: after-all
// CHECK-SAME: metadata={op_name="name"}

// -----

// CHECK-LABEL: %main
func.func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc("file_name":2:8)
  func.return %0 : !mhlo.token
}

// CHECK: after-all
// CHECK-SAME: metadata={source_file="file_name" source_line=2}

// -----

// CHECK-LABEL: %main
func.func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc("name(with)[]")
  func.return %0 : !mhlo.token
}

// CHECK: after-all
// CHECK-SAME: metadata={op_name="name(with)[]"}

// -----

// CHECK-LABEL: %main
func.func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc("name(anothername)"("file_name":2:8))
  func.return %0 : !mhlo.token
}

// CHECK: after-all
// CHECK-SAME: metadata={op_name="name(anothername)" source_file="file_name" source_line=2}

// -----

// CHECK-LABEL: %main
func.func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc(fused["fused/location/file", "source.txt":42:5])
  func.return %0 : !mhlo.token
}

// CHECK: after-all
// CHECK-SAME: metadata={op_name="fused/location/file" source_file="source.txt" source_line=42}

// -----

// CHECK-LABEL: %main
func.func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc(fused["name1", fused["nested_fusion":5:42, "name2"]])
  func.return %0 : !mhlo.token
}

// CHECK: after-all
// CHECK-SAME: metadata={op_name="name1;name2" source_file="nested_fusion" source_line=5}

// -----

// CHECK-LABEL: %main
func.func @main(%arg0: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token loc(fused["multiple_sources", "source1":1:2, "source2":3:4])
  func.return %0 : !mhlo.token
}

// CHECK: after-all
// CHECK-SAME: metadata={op_name="multiple_sources" source_file="source2" source_line=3}
