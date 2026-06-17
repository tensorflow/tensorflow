// RUN: tfg-transforms-opt --tfg-dedupe-hoist-constant %s | FileCheck %s

// CHECK-LABEL:   tfg.graph
tfg.graph #tf_type.version<producer = 1015, min_consumer = 0> {
  %Const, %ctl = Const device("/job:host/task:0/device:CPU:0") name("apple") {dtype = i32, value = dense<[218, 128]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %Const_1, %ctl_1 = Const [%ctl] device("/job:host/task:0/device:CPU:0") name("banana") {dtype = i32, value = dense<[218, 128]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %Const_2, %ctl_2 = Const [%ctl_1] device("/job:host/task:0/device:CPU:1") name("pear") {dtype = i32, value = dense<[218, 128]> : tensor<2xi32>} : () -> (tensor<2xi32>)
}

// CHECK:   %[[VAL_0:.*]], %[[VAL_1:.*]] = Const device("/job:host/task:0/device:CPU:0") name("apple")
// CHECK:   %[[VAL_2:.*]], %[[VAL_3:.*]] = Const {{\[}}%[[VAL_1]]] device("/job:host/task:0/device:CPU:1") name("pear")
