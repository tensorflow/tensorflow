// RUN: tfg-opt-no-passes %s --canonicalize | FileCheck %s

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  // CHECK: %[[A:.*]], %{{.*}} = A
  %A, %ctl = A : () -> (tensor<1xi32>)
  // CHECK-NOT: cast %[[A]]
  %cast, %ctl_0 = Cast(%A) {SrcT = i32, DstT = i32} : (tensor<1xi32>) -> (tensor<1xi32>)
  // CHECK: B(%[[A]])
  %B, %ctl_1 = B(%cast) : (tensor<1xi32>) -> (tensor<1xf32>)
}
