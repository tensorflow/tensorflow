// RUN: mlir-opt -test-legalize-patterns -test-legalize-mode=full %s | FileCheck %s

// CHECK-LABEL: func @multi_level_mapping
func @multi_level_mapping() {
  // CHECK: "test.type_producer"() : () -> f64
  // CHECK: "test.type_consumer"(%{{.*}}) : (f64) -> ()
  %result = "test.type_producer"() : () -> i32
  "test.type_consumer"(%result) : (i32) -> ()
  "test.return"() : () -> ()
}
