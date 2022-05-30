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

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 1, min_consumer = 1> {
  // CHECK-NEXT: %[[A:.*]], %{{.*}} = Placeholder name("a")
  // CHECK-NEXT: %[[COND:.*]], %{{.*}} = Placeholder name("cond")
  // CHECK-NEXT: IfRegion %[[COND]] then {
  // CHECK-NEXT:   %[[C:.*]], %{{.*}} = Placeholder name("c")
  // CHECK-NEXT:   %[[D:.*]], %{{.*}} = Foo(%[[C]]) name("d")
  // CHECK-NEXT:   yield(%[[D]])
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   yield(%[[A]])
  %a, %ctlA = Placeholder name("a") : () -> (tensor<*xi32>)
  %cond, %ctlCond = Placeholder name("cond") : () -> (tensor<*xi1>)
  %b, %ctlB = IfRegion %cond then {
    %c, %ctlC = Placeholder name("c") : () -> (tensor<*xi32>)
    %d, %ctlD = Foo(%c) name("d") : (tensor<*xi32>) -> (tensor<*xi32>)
    yield(%d) : tensor<*xi32>
  } else {
    yield(%a) : tensor<*xi32>
  } : (tensor<*xi1>) -> (tensor<*xi32>)
}
