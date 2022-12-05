// RUN: tfg-transforms-opt --tfg-dedupe-hoist-constant=assume-strict-calls %s | FileCheck %s --check-prefix=STRICT
// RUN: tfg-transforms-opt --tfg-dedupe-hoist-constant %s | FileCheck %s

// CHECK-LABEL:   tfg.graph
tfg.graph #tf_type.version<producer = 1015, min_consumer = 0> {
  %Const, %ctl = Const device("/job:host/task:0/device:CPU:0") name("apple") {_here = 1, dtype = i32, value = dense<[218, 128]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %Const_1, %ctl_1 = Const [%ctl] device("/job:host/task:0/device:CPU:0") name("banana") {_here = 1, dtype = i32, value = dense<[218, 128]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %Const_2, %ctl_2 = Const [%ctl_1] device("/job:host/task:0/device:CPU:0") name("pear") {_here = 1, dtype = i32, value = dense<[218, 128]> : tensor<2xi32>} : () -> (tensor<2xi32>)
  %res_2, %ctl_3 = foo(%Const_2) device("/job:host/task:0/device:CPU:0") name("call") : (tensor<2xi32>) -> (tensor<2xi32>)
// CHECK:   %[[VAL_0:.*]], %[[VAL_1:.*]] = Const device("/job:host/task:0/device:CPU:0") name("apple")
// CHECK-SAME: _here = 1
// CHECK:   %[[ID:.*]], %[[CTL:.*]] = Identity(%[[VAL_0]]) [%[[VAL_1]]] device("/job:host/task:0/device:CPU:0") name("const_dedupe_hoist/pear")
// Verify that the dtype and value attributes were not propagated but T is set.
// CHECK-SAME: {T = i32, _here = 1 : i64}
// CHECK:   foo(%[[ID]]) device
// STRICT:   %[[VAL_0:.*]], %[[VAL_1:.*]] = Const device("/job:host/task:0/device:CPU:0") name("apple")
// STRICT-SAME: _here = 1
// STRICT:   foo(%[[VAL_0]]) device
}

tfg.func @foo(%arg0 : tensor<2xi32> {tfg.name = "input1"},
              %arg1 : tensor<2xi32> {tfg.name = "input2"}) -> (tensor<2xi32> {tfg.name = "result1"})
    attributes {description = "function foo"} {
  return(%arg0) : tensor<2xi32>
}
