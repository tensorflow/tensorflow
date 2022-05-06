// RUN: tfg-transforms-opt %s --pass-pipeline="tfg.graph(tfg-toposort), tfg.func(tfg-toposort)" | FileCheck %s

// Test with region ops
// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  // CHECK-NEXT: %[[INDEX:.*]], %{{.*}} = Index
  // CHECK-NEXT: %[[FOO:.*]], %{{.*}} = Foo(%[[INDEX]]) :
  // CHECK-NEXT: %{{.*}}, %[[CTLCASE:.*]] = CaseRegion %[[INDEX]] {
  // CHECK-NEXT:   %[[B:.*]], %{{.*}} = B(%[[A:.*]]) :
  // CHECK-NEXT:   %[[A]], %{{.*}} = A(%[[FOO]]) :
  // CHECK-NEXT:   yield(%[[B]])
  // CHECK-NEXT: }
  // CHECK-NEXT: NoOp [%[[CTLCASE]]]
  %ctlNoOp = NoOp [%ctlCase]
  %Case, %ctlCase = CaseRegion %Index {
    %B, %ctlB = B(%A) : (tensor<*xi32>) -> (tensor<*xi32>)
    %A, %ctlA = A(%Foo) : (tensor<*xi32>) -> (tensor<*xi32>)
    yield(%B) : tensor<*xi32>
  } : (tensor<*xi32>) -> (tensor<*xi32>)
  %Foo, %ctlFoo = Foo(%Index) : (tensor<*xi32>) -> (tensor<*xi32>)
  %Index, %ctlIndex = Index : () -> (tensor<*xi32>)
}
