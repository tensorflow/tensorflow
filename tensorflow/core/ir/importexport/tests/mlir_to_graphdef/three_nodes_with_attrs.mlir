// RUN: tfg-translate -mlir-to-graphdef %s | FileCheck %s

tfg.graph #tf_type.version<producer = 34, min_consumer = 5> {
  %ctl = tfg.Add(%Placeholder, %Placeholder_1) name("SomeAdd") {T = i32} : (!tf_type.tensor, !tf_type.tensor) -> ()
  %Placeholder, %ctl_0 = tfg.Placeholder name("Placeholder1") {dtype = i32} : () -> (!tf_type.tensor)
  %Placeholder_1, %ctl_2 = tfg.Placeholder name("Placeholder2") {dtype = i32} : () -> (!tf_type.tensor)
}

// CHECK: node {
// CHECK-NEXT:   name: "SomeAdd"
// CHECK-NEXT:   op: "Add"
// CHECK-NEXT:   input: "Placeholder1"
// CHECK-NEXT:   input: "Placeholder2"
// CHECK-NEXT:   attr {
// CHECK-NEXT:     key: "T"
// CHECK-NEXT:     value {
// CHECK-NEXT:       type: DT_INT32
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK:      node {
// CHECK-NEXT:   name: "Placeholder1"
// CHECK-NEXT:   op: "Placeholder"
// CHECK-NEXT:  attr {
// CHECK-NEXT:    key: "dtype"
// CHECK-NEXT:    value {
// CHECK-NEXT:      type: DT_INT32
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK: node {
// CHECK-NEXT:  name: "Placeholder2"
// CHECK-NEXT:  op: "Placeholder"
// CHECK-NEXT:  attr {
// CHECK-NEXT:    key: "dtype"
// CHECK-NEXT:    value {
// CHECK-NEXT:      type: DT_INT32
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK: versions {
// CHECK-NEXT:  producer: 34
// CHECK-NEXT:  min_consumer: 5
// CHECK-NEXT: }
