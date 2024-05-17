// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s

// We need a "main" function, or the flatbuffer exporter won't export anything.
// CHECK-LABEL: @main
func.func @main(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  func.return %arg0: tensor<1xf32>
}
// CHECK-NEXT: return

// CHECK-LABEL: @unneeded_control_node_gets_deleted
func.func @unneeded_control_node_gets_deleted(%arg0: tensor<1xf32>)->tensor<1xf32> {
  %tmp0, %ctl0  = tfl.control_node controls "tfl.neg"(%arg0): (tensor<1xf32>) -> tensor<1xf32>
  func.return %tmp0: tensor<1xf32>
}
// CHECK-NEXT: %[[tmp:.*]] = "tfl.neg"(%arg0)
// CHECK-NEXT: return %[[tmp]]

// CHECK-LABEL: @long_chain
func.func @long_chain(%arg0: tensor<1xf32>)->tensor<1xf32> {
  %tmp0, %ctl0 = tfl.control_node controls "tfl.neg"(%arg0): (tensor<1xf32>) -> tensor<1xf32>
  %tmp1, %ctl1 = tfl.control_node(%ctl0) controls "tfl.neg"(%tmp0): (tensor<1xf32>) -> tensor<1xf32>
  %tmp2, %ctl2 = tfl.control_node(%ctl1) controls "tfl.neg"(%tmp1): (tensor<1xf32>) -> tensor<1xf32>
  %tmp3, %ctl3 = tfl.control_node(%ctl2) controls "tfl.neg"(%tmp2): (tensor<1xf32>) -> tensor<1xf32>
  func.return %tmp3: tensor<1xf32>
}
// CHECK-NEXT: %[[t0:.*]], %[[c0:.*]] = tfl.control_node controls "tfl.neg"(%arg0)
// CHECK-NEXT: %[[t1:.*]], %[[c1:.*]] = tfl.control_node(%[[c0]]) controls "tfl.neg"(%[[t0]])
// CHECK-NEXT: %[[t2:.*]], %[[c2:.*]] = tfl.control_node(%[[c1]]) controls "tfl.neg"(%[[t1]])
// CHECK-NEXT: %[[t3:.*]], %[[c3:.*]] = tfl.control_node(%[[c2]]) controls "tfl.neg"(%[[t2]])
// CHECK-NEXT: return %[[t3]]


// CHECK-LABEL: @overlapping_chains
func.func @overlapping_chains(%arg0: tensor<1xf32>)->tensor<1xf32> {
  %tmp0, %ctl0 = tfl.control_node controls "tfl.neg"(%arg0): (tensor<1xf32>) -> tensor<1xf32>
  %tmp1, %ctl1 = tfl.control_node controls "tfl.neg"(%tmp0): (tensor<1xf32>) -> tensor<1xf32>
  %tmp2, %ctl2 = tfl.control_node(%ctl0) controls "tfl.neg"(%tmp1): (tensor<1xf32>) -> tensor<1xf32>
  %tmp3, %ctl3 = tfl.control_node(%ctl1) controls "tfl.neg"(%tmp2): (tensor<1xf32>) -> tensor<1xf32>
  func.return %tmp3: tensor<1xf32>
}
// CHECK-NEXT: %[[t0:.*]], %[[c0:.*]] = tfl.control_node controls "tfl.neg"(%arg0)
// CHECK-NEXT: %[[t1:.*]], %[[c1:.*]] = tfl.control_node controls "tfl.neg"(%[[t0]])
// CHECK-NEXT: %[[t2:.*]], %[[c2:.*]] = tfl.control_node(%[[c0]]) controls "tfl.neg"(%[[t1]])
// CHECK-NEXT: %[[t3:.*]], %[[c3:.*]] = tfl.control_node(%[[c1]]) controls "tfl.neg"(%[[t2]])
// CHECK-NEXT: return %[[t3]]

// CHECK-LABEL: @multiple_node_args
func.func @multiple_node_args(%arg0: tensor<1xf32>)->tensor<1xf32> {
  %tmp0, %ctl0 = tfl.control_node controls "tfl.neg"(%arg0): (tensor<1xf32>) -> tensor<1xf32>
  %tmp1, %ctl1 = tfl.control_node controls "tfl.neg"(%tmp0): (tensor<1xf32>) -> tensor<1xf32>
  %tmp2, %ctl2 = tfl.control_node controls "tfl.neg"(%tmp1): (tensor<1xf32>) -> tensor<1xf32>
  %tmp3, %ctl3 = tfl.control_node controls "tfl.neg"(%tmp2): (tensor<1xf32>) -> tensor<1xf32>
  %tmp4, %ctl4 = tfl.control_node(%ctl2, %ctl0, %ctl3, %ctl2, %ctl1) controls "tfl.neg"(%tmp3): (tensor<1xf32>) -> tensor<1xf32>
  func.return %tmp4: tensor<1xf32>
}
// CHECK-NEXT: %[[t0:.*]], %[[c0:.*]] = tfl.control_node controls "tfl.neg"(%arg0)
// CHECK-NEXT: %[[t1:.*]], %[[c1:.*]] = tfl.control_node controls "tfl.neg"(%[[t0]])
// CHECK-NEXT: %[[t2:.*]], %[[c2:.*]] = tfl.control_node controls "tfl.neg"(%[[t1]])
// CHECK-NEXT: %[[t3:.*]], %[[c3:.*]] = tfl.control_node controls "tfl.neg"(%[[t2]])
// CHECK-NEXT: %[[t4:.*]], %[[c4:.*]] = tfl.control_node(%[[c0]], %[[c1]], %[[c2]], %[[c3]]) controls "tfl.neg"(%[[t3]])
// CHECK-NEXT: return %[[t4]]
