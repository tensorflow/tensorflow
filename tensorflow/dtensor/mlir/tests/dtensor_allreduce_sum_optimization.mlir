// RUN: dtensor-opt %s -split-input-file -dtensor-allreduce-sum-optimization -verify-diagnostics | FileCheck %s

// Check that DTensorAllReduce op with Add/AddN/AddV2 operations are optimized.
// CHECK-LABEL: func @main
func.func @main() -> (tensor<916x8192xbf16>)  {
  // CHECK:       %[[GROUP_ASSIGNMENT:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[VAL1:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[VAL2:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[ADD_OUT:.*]] = "tf.AddN"(%[[VAL1]], %[[VAL2]])
  // CHECK-NEXT:  %[[REDUCTION_OUT:.*]] = "tf.DTensorAllReduce"(%[[ADD_OUT]], %[[GROUP_ASSIGNMENT]])
  // CHECK-NEXT:  %[[B_OUT:.*]] = "tf.B"(%[[REDUCTION_OUT]])
  %0 = "tf.Const"() {value = dense<0> : tensor<2x64xi32>} : () -> tensor<2x64xi32>
  %1 = "tf.Const"() {value = dense<0.0> : tensor<916x8192xbf16>} : () -> tensor<916x8192xbf16>
  %2 = "tf.Const"() {value = dense<0.0> : tensor<916x8192xbf16>} : () ->tensor<916x8192xbf16>
  %3= "tf.DTensorAllReduce"(%1, %0) {_layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<916x8192xbf16>, tensor<2x64xi32>) -> tensor<916x8192xbf16>
  %4= "tf.DTensorAllReduce"(%2, %0) {_layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<916x8192xbf16>, tensor<2x64xi32>) -> tensor<916x8192xbf16>
  %5= "tf.AddN"(%3, %4) { device = ""} : (tensor<916x8192xbf16>, tensor<916x8192xbf16>) -> tensor<916x8192xbf16>
  %6 = "tf.B"(%5) : (tensor<916x8192xbf16>) -> tensor<916x8192xbf16>

  // CHECK:       %[[GROUP_ASSIGNMENT_2:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[VAL3:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[VAL4:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[ADD_OUT_2:.*]] = "tf.Add"(%[[VAL3]], %[[VAL4]])
  // CHECK-NEXT:  %[[REDUCTION_OUT_2:.*]] = "tf.DTensorAllReduce"(%[[ADD_OUT_2]], %[[GROUP_ASSIGNMENT_2]])
  // CHECK-NEXT:  %[[C_OUT:.*]] = "tf.C"(%[[REDUCTION_OUT_2]])
  %7 = "tf.Const"() {value = dense<0> : tensor<2x64xi32>} : () -> tensor<2x64xi32>
  %8 = "tf.Const"() {value = dense<0.0> : tensor<916x8192xbf16>} : () -> tensor<916x8192xbf16>
  %9 = "tf.Const"() {value = dense<0.0> : tensor<916x8192xbf16>} : () ->tensor<916x8192xbf16>
  %10= "tf.DTensorAllReduce"(%8, %7) {_layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<916x8192xbf16>, tensor<2x64xi32>) -> tensor<916x8192xbf16>
  %11= "tf.DTensorAllReduce"(%9, %7) {_layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<916x8192xbf16>, tensor<2x64xi32>) -> tensor<916x8192xbf16>
  %12= "tf.Add"(%10, %11) { device = ""} : (tensor<916x8192xbf16>, tensor<916x8192xbf16>) -> tensor<916x8192xbf16>
  %13 = "tf.C"(%12) : (tensor<916x8192xbf16>) -> tensor<916x8192xbf16>

  // CHECK:       %[[GROUP_ASSIGNMENT_3:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[VAL5:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[VAL6:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[ADD_OUT_3:.*]] = "tf.AddV2"(%[[VAL5]], %[[VAL6]])
  // CHECK-NEXT:  %[[REDUCTION_OUT_3:.*]] = "tf.DTensorAllReduce"(%[[ADD_OUT_3]], %[[GROUP_ASSIGNMENT_3]])
  // CHECK-NEXT:  %[[D_OUT:.*]] = "tf.D"(%[[REDUCTION_OUT_3]])
  %14 = "tf.Const"() {value = dense<0> : tensor<2x64xi32>} : () -> tensor<2x64xi32>
  %15 = "tf.Const"() {value = dense<0.0> : tensor<916x8192xbf16>} : () -> tensor<916x8192xbf16>
  %16 = "tf.Const"() {value = dense<0.0> : tensor<916x8192xbf16>} : () ->tensor<916x8192xbf16>
  %17= "tf.DTensorAllReduce"(%15, %14) {_layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<916x8192xbf16>, tensor<2x64xi32>) -> tensor<916x8192xbf16>
  %18= "tf.DTensorAllReduce"(%16, %14) {_layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<916x8192xbf16>, tensor<2x64xi32>) -> tensor<916x8192xbf16>
  %19= "tf.AddV2"(%17, %18) { device = ""} : (tensor<916x8192xbf16>, tensor<916x8192xbf16>) -> tensor<916x8192xbf16>
  %20 = "tf.D"(%19) : (tensor<916x8192xbf16>) -> tensor<916x8192xbf16>
  func.return %6 : tensor<916x8192xbf16>
}

// -----

// Check that DTensorAllReduce op with operation with group assignment from different constant with same values are optimized correctly.
// CHECK-LABEL: func @main
func.func @main() -> (tensor<916x8192xbf16>) {
  // CHECK:       %[[VAL_1:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[GROUP_ASSIGNMENT_1:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[VAL_2:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[GROUP_ASSIGNMENT_2:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[ADD_OUT_1:.*]] = "tf.AddN"(%[[VAL_1]], %[[VAL_2]])
  // CHECK-NEXT:  %[[REDUCTION_OUT:.*]] = "tf.DTensorAllReduce"(%[[ADD_OUT_1]], %[[GROUP_ASSIGNMENT_1]])
  // CHECK-NEXT:  return %[[REDUCTION_OUT]]
  %0 = "tf.Const"() {value = dense<0.0> : tensor<916x8192xbf16>} : () -> tensor<916x8192xbf16>
  %1 = "tf.Const"() {value = dense<0> : tensor<2x64xi32>} : () -> tensor<2x64xi32>
  %4= "tf.DTensorAllReduce"(%0, %1) {_layout = ["sharding_specs:x,unsharded, mesh:tpu_mesh|x=2,y=64|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<916x8192xbf16>, tensor<2x64xi32>) -> tensor<916x8192xbf16>

  %2 = "tf.Const"() {value = dense<0.0> : tensor<916x8192xbf16>} : () -> tensor<916x8192xbf16>
  %3= "tf.Const"() {value = dense<0> : tensor<2x64xi32>} : () -> tensor<2x64xi32>
  %5= "tf.DTensorAllReduce"(%2, %3) {_layout = ["sharding_specs:x,unsharded, mesh:tpu_mesh|x=2,y=64|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<916x8192xbf16>, tensor<2x64xi32>) -> tensor<916x8192xbf16>

  %6= "tf.AddN"(%4, %5): (tensor<916x8192xbf16>, tensor<916x8192xbf16>) -> tensor<916x8192xbf16>
  func.return %6: tensor<916x8192xbf16>
}

// -----

// Check that DTensorAllReduce op with operation type that is not `sum` are not optimized.
// CHECK-LABEL: func @main
func.func @main() -> (tensor<916x8192xbf16>) {
  // CHECK:  %[[REDUCE_OUT_1:.*]] = "tf.DTensorAllReduce"
  // CHECK:  %[[REDUCE_OUT_2:.*]] = "tf.DTensorAllReduce"
  // CHECK:  %[[ADD_OUT:.*]] = "tf.AddN"(%[[REDUCE_OUT_1]], %[[REDUCE_OUT_2]])
  %0 = "tf.Const"() {value = dense<0.0> : tensor<916x8192xbf16>} : () -> tensor<916x8192xbf16>
  %1 = "tf.Const"() {value = dense<0> : tensor<2x64xi32>} : () -> tensor<2x64xi32>
  %4= "tf.DTensorAllReduce"(%0, %1) {_layout = ["sharding_specs:x,unsharded, mesh:tpu_mesh|x=2,y=64|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Mean"} : (tensor<916x8192xbf16>, tensor<2x64xi32>) -> tensor<916x8192xbf16>
  %2 = "tf.Const"() {value = dense<0.0> : tensor<916x8192xbf16>} : () -> tensor<916x8192xbf16>
  %3= "tf.Const"() {value = dense<0> : tensor<2x64xi32>} : () -> tensor<2x64xi32>
  %5= "tf.DTensorAllReduce"(%2, %3) {_layout = ["sharding_specs:x,unsharded, mesh:tpu_mesh|x=2,y=64|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Max"} : (tensor<916x8192xbf16>, tensor<2x64xi32>) -> tensor<916x8192xbf16>
  %6= "tf.AddN"(%4, %5): (tensor<916x8192xbf16>, tensor<916x8192xbf16>) -> tensor<916x8192xbf16>
  func.return %6: tensor<916x8192xbf16>
}

// -----

// Check that DTensorAllReduce op with different group assignment are not optimized away.
// CHECK-LABEL: func @main
func.func @main() -> (tensor<916x8192xbf16>) {
  // CHECK:  %[[REDUCE_OUT_1:.*]] = "tf.DTensorAllReduce"
  // CHECK:  %[[REDUCE_OUT_2:.*]] = "tf.DTensorAllReduce"
  // CHECK:  %[[ADD_OUT:.*]] = "tf.AddN"(%[[REDUCE_OUT_1]], %[[REDUCE_OUT_2]])
  %0 = "tf.Const"() {value = dense<0.0> : tensor<916x8192xbf16>} : () -> tensor<916x8192xbf16>
  %1 = "tf.Const"() {value = dense<0> : tensor<2x64xi32>} : () -> tensor<2x64xi32>
  %4= "tf.DTensorAllReduce"(%0, %1) {_layout = ["sharding_specs:x,unsharded, mesh:tpu_mesh|x=2,y=64|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<916x8192xbf16>, tensor<2x64xi32>) -> tensor<916x8192xbf16>
  %2 = "tf.Const"() {value = dense<0.0> : tensor<916x8192xbf16>} : () -> tensor<916x8192xbf16>
  %3= "tf.Const"() {value = dense<1> : tensor<2x64xi32>} : () -> tensor<2x64xi32>
  %5= "tf.DTensorAllReduce"(%2, %3) {_layout = ["sharding_specs:x,unsharded, mesh:tpu_mesh|x=2,y=64|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<916x8192xbf16>, tensor<2x64xi32>) -> tensor<916x8192xbf16>
  %6= "tf.AddN"(%4, %5): (tensor<916x8192xbf16>, tensor<916x8192xbf16>) -> tensor<916x8192xbf16>
  func.return %6: tensor<916x8192xbf16>
}

// -----

// Check that DTensorAllReduce op with malformed layout specification is disallowed.
func.func @main() -> (tensor<916x8192xbf16>) {
  %0 = "tf.Const"() {value = dense<0.0> : tensor<916x8192xbf16>} : () -> tensor<916x8192xbf16>
  %1 = "tf.Const"() {value = dense<0> : tensor<2x64xi32>} : () -> tensor<2x64xi32>
  // expected-error @+1 {{Malformed layout specification for DTensorAllReduce op found}}
  %4= "tf.DTensorAllReduce"(%0, %1) {_layout = ["1234"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<916x8192xbf16>, tensor<2x64xi32>) -> tensor<916x8192xbf16>
  %2 = "tf.Const"() {value = dense<0.0> : tensor<916x8192xbf16>} : () -> tensor<916x8192xbf16>
  %3= "tf.Const"() {value = dense<0> : tensor<2x64xi32>} : () -> tensor<2x64xi32>
  %5= "tf.DTensorAllReduce"(%2, %3) {_layout = ["sharding_specs:x,unsharded, mesh:tpu_mesh|x=2,y=64|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<916x8192xbf16>, tensor<2x64xi32>) -> tensor<916x8192xbf16>
  %6= "tf.AddN"(%4, %5): (tensor<916x8192xbf16>, tensor<916x8192xbf16>) -> tensor<916x8192xbf16>
  func.return %6: tensor<916x8192xbf16>
}

// -----

// Check that DTensorAllReduce op with missing layout specification is disallowed.
func.func @main() -> (tensor<916x8192xbf16>) {
  %0 = "tf.Const"() {value = dense<0.0> : tensor<916x8192xbf16>} : () -> tensor<916x8192xbf16>
  %1 = "tf.Const"() {value = dense<0> : tensor<2x64xi32>} : () -> tensor<2x64xi32>
  // expected-error @+1 {{DTensorAllReduce op must have layout specification}}
  %4= "tf.DTensorAllReduce"(%0, %1) {device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<916x8192xbf16>, tensor<2x64xi32>) -> tensor<916x8192xbf16>
  %2 = "tf.Const"() {value = dense<0.0> : tensor<916x8192xbf16>} : () -> tensor<916x8192xbf16>
  %3= "tf.Const"() {value = dense<0> : tensor<2x64xi32>} : () -> tensor<2x64xi32>
  %5= "tf.DTensorAllReduce"(%2, %3) {_layout = ["sharding_specs:x,unsharded, mesh:tpu_mesh|x=2,y=64|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<916x8192xbf16>, tensor<2x64xi32>) -> tensor<916x8192xbf16>
  %6= "tf.AddN"(%4, %5): (tensor<916x8192xbf16>, tensor<916x8192xbf16>) -> tensor<916x8192xbf16>
  func.return %6: tensor<916x8192xbf16>
}

// -----

// Check that DTensorAllReduce op inside while loop is optimized away correctly.
// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: tensor<i32>
func.func @main(%arg0: tensor<i32>) {
  // CHECK:       %[[VAL_1:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[VAL_1_ID:.*]] = "tf.Identity"(%[[VAL_1]])
  // CHECK-NEXT:  %[[WHILE_OUT:.*]]:2 = "tf.WhileRegion"(%[[VAL_1_ID]], %[[ARG0]])
  // CHECK:         "tf.A"
  // CHECK-NEXT:    "tf.Yield"
  // CHECK:         ^bb0(%[[BARG0:.*]]: tensor<4xf32>, %[[BARG1:.*]]: tensor<i32>)
  // CHECK:          %[[INPUT0:.*]] = "tf.Const"() {value = dense<0> : tensor<4xi32>} : () -> tensor<4xi32>
  // CHECK-NEXT:     %[[GROUP:.*]] = "tf.Const"() {value = dense<0> : tensor<2x64xi32>} : () -> tensor<2x64xi32>
  // CHECK-NEXT:     %[[CAST_OUT:.*]] = "tf.Cast"(%[[INPUT0]])
  // CHECK-NEXT:     %[[ADD_OUT:.*]] = "tf.AddV2"(%[[CAST_OUT]], %[[BARG0]])
  // CHECK-NEXT:     %[[OUT:.*]] = "tf.Identity"(%[[ADD_OUT]])
  // CHECK-NEXT:     "tf.Yield"
  // CHECK:      %[[NEW_GROUP_ASSIGN:.*]] = "tf.Const"() {value = dense<0> : tensor<2x64xi32>} : () -> tensor<2x64xi32>
  // CHECK:      %[[ALL_REDUCE_OUT:.*]] = "tf.DTensorAllReduce"(%[[WHILE_OUT]]#0, %[[NEW_GROUP_ASSIGN]])
  %0 = "tf.Const"() {value = dense<0.0> : tensor<4xf32>} : () -> tensor<4xf32>
  %2 = "tf.Identity"(%0) : (tensor<4xf32>) -> tensor<4xf32>

  %9:2 = "tf.WhileRegion"(%2, %arg0) ({
    ^bb0(%carg0: tensor<4xf32>, %carg1: tensor<i32>):
       %1 = "tf.A"(%carg1) : (tensor<i32>) -> (tensor<i1>)
       "tf.Yield"(%1) : (tensor<i1>) -> ()
  },  {
    ^bb0(%barg0: tensor<4xf32>, %barg1: tensor<i32>):
      %3 = "tf.Const"() {value = dense<0> : tensor<4xi32>} : () -> tensor<4xi32>
      %4= "tf.Const"() {value = dense<0> : tensor<2x64xi32>} : () -> tensor<2x64xi32>
      %5= "tf.DTensorAllReduce"(%3, %4) {_layout = ["sharding_specs:x,unsharded, mesh:tpu_mesh|x=2,y=64|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<4xi32>, tensor<2x64xi32>) -> tensor<4xi32>
      %6 = "tf.Cast"(%5) : (tensor<4xi32>) ->  tensor<4xf32>
      %7 = "tf.AddV2"(%6, %barg0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %8 = "tf.Identity"(%7) : (tensor<4xf32>) -> tensor<4xf32>
      "tf.Yield"(%8, %barg1) : (tensor<4xf32>, tensor<i32>) -> ()
  }) {is_stateless = true} : (tensor<4xf32>, tensor<i32>) -> (tensor<4xf32>, tensor<i32>)

  "tf.C"(%9#0) : (tensor<4xf32>) -> ()
  func.return
}

// -----

// Check that while op optimization is ignored if loop variant input is used for
// while loop condition logic.
// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: tensor<i32>
func.func @main(%arg0: tensor<i32>) {
  // CHECK:       %[[WHILE_OUT:.*]]:2 = "tf.WhileRegion"
  // CHECK:         "tf.A"
  // CHECK-NEXT:    "tf.Yield"
  // CHECK:         ^bb0(%[[BARG0:.*]]: tensor<4xf32>, %[[BARG1:.*]]: tensor<i32>)
  // CHECK:          "tf.Const"()
  // CHECK-NEXT:     "tf.Const"()
  // CHECK-NEXT:     "tf.DTensorAllReduce"
  // CHECK-NEXT:     "tf.AddV2"
  // CHECK-NEXT:     "tf.Yield"
  %0 = "tf.Const"() {value = dense<0.0> : tensor<4xf32>} : () -> tensor<4xf32>

  %9:2 = "tf.WhileRegion"(%0, %arg0) ({
    ^bb0(%carg0: tensor<4xf32>, %carg1: tensor<i32>):
       %1 = "tf.A"(%carg0) : (tensor<4xf32>) -> (tensor<i1>)
       "tf.Yield"(%1) : (tensor<i1>) -> ()
  },  {
    ^bb0(%barg0: tensor<4xf32>, %barg1: tensor<i32>):
      %3 = "tf.Const"() {value = dense<0.0> : tensor<4xf32>} : () -> tensor<4xf32>
      %4= "tf.Const"() {value = dense<0> : tensor<2x64xi32>} : () -> tensor<2x64xi32>
      %5= "tf.DTensorAllReduce"(%3, %4) {_layout = ["sharding_specs:x,unsharded, mesh:tpu_mesh|x=2,y=64|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<4xf32>, tensor<2x64xi32>) -> tensor<4xf32>
      %6 = "tf.AddV2"(%5, %barg0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      "tf.Yield"(%6, %barg1) : (tensor<4xf32>, tensor<i32>) -> ()
  }) {is_stateless = true} : (tensor<4xf32>, tensor<i32>) -> (tensor<4xf32>, tensor<i32>)
  func.return
}

// -----

// Check that while op with input that is not constant zero will not trigger
// lazy all reduce optimization.
// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: tensor<i32>
func.func @main(%arg0: tensor<i32>) {
  // CHECK:       %[[WHILE_OUT:.*]]:2 = "tf.WhileRegion"
  // CHECK:         "tf.A"
  // CHECK-NEXT:    "tf.Yield"
  // CHECK:         ^bb0(%[[BARG0:.*]]: tensor<4xf32>, %[[BARG1:.*]]: tensor<i32>)
  // CHECK:          "tf.Const"()
  // CHECK-NEXT:     "tf.Const"()
  // CHECK-NEXT:     "tf.DTensorAllReduce"
  // CHECK-NEXT:     "tf.AddV2"
  // CHECK-NEXT:     "tf.Yield"
  %0 = "tf.Const"() {value = dense<[0.0, 1.0, 0.0, 0.0]> : tensor<4xf32>} : () -> tensor<4xf32>

  %9:2 = "tf.WhileRegion"(%0, %arg0) ({
    ^bb0(%carg0: tensor<4xf32>, %carg1: tensor<i32>):
       %1 = "tf.A"(%carg0) : (tensor<4xf32>) -> (tensor<i1>)
       "tf.Yield"(%1) : (tensor<i1>) -> ()
  },  {
    ^bb0(%barg0: tensor<4xf32>, %barg1: tensor<i32>):
      %3 = "tf.Const"() {value = dense<0.0> : tensor<4xf32>} : () -> tensor<4xf32>
      %4= "tf.Const"() {value = dense<0> : tensor<2x64xi32>} : () -> tensor<2x64xi32>
      %5= "tf.DTensorAllReduce"(%3, %4) {_layout = ["sharding_specs:x,unsharded, mesh:tpu_mesh|x=2,y=64|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<4xf32>, tensor<2x64xi32>) -> tensor<4xf32>
      %6 = "tf.AddV2"(%5, %barg0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      "tf.Yield"(%6, %barg1) : (tensor<4xf32>, tensor<i32>) -> ()
  }) {is_stateless = true} : (tensor<4xf32>, tensor<i32>) -> (tensor<4xf32>, tensor<i32>)
  func.return
}

// -----

// Check that optimization that does not reduce number of DTensorAllReduce is
// not applied.
// CHECK-LABEL: func @main
func.func @main() -> (tensor<4096x8192xf32>) {
  // CHECK:       %[[CONST_OUT_1:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[GROUP_ASSIGNMENT:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[REDUCE_OUT:.*]] = "tf.DTensorAllReduce"(%[[CONST_OUT_1]], %[[GROUP_ASSIGNMENT]])
  // CHECK-NEXT:  "tf.Cast"(%[[REDUCE_OUT]])
  %0 = "tf.Const"() {value = dense<0.0> : tensor<4096x8192xbf16>} : () -> tensor<4096x8192xbf16>
  %1 = "tf.Const"() {value = dense<0> : tensor<2x64xi32>} : () -> tensor<2x64xi32>
  %2= "tf.DTensorAllReduce"(%0, %1) {_layout = ["sharding_specs:x,unsharded, mesh:tpu_mesh|x=2,y=64|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<4096x8192xbf16>, tensor<2x64xi32>) -> tensor<4096x8192xbf16>
  %3= "tf.Cast"(%2) {Truncate = false, device = ""} : (tensor<4096x8192xbf16>) -> tensor<4096x8192xf32>
  func.return %3: tensor<4096x8192xf32>
}

// -----

// Check that  DTensorAllReduce op moved after Identity-like operations.
// CHECK-LABEL: func @main
func.func @main() -> (tensor<916x8192xf32>)  {
  // CHECK:       %[[GROUP_ASSIGNMENT:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[CONST_OUT_1:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[CONST_OUT_2:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[CAST_OUT:.*]] = "tf.Cast"(%[[CONST_OUT_2]])
  // CHECK-NEXT:  %[[IDENTITY_OUT:.*]] = "tf.Identity"(%[[CAST_OUT]])
  // CHECK-NEXT:  %[[RESHAPE_CONST:.*]] = "tf.Const"
  // CHECK-NEXT:  %[[RESHAPE_OUT:.*]] = "tf.Reshape"(%[[IDENTITY_OUT]], %[[RESHAPE_CONST]])
  // CHECK-NEXT:  %[[ADD_OUT:.*]] = "tf.AddN"(%[[RESHAPE_OUT]], %[[CONST_OUT_1]])
  // CHECK-NEXT:  "tf.DTensorAllReduce"(%[[ADD_OUT]], %[[GROUP_ASSIGNMENT]])
  %0 = "tf.Const"() {value = dense<0> : tensor<2x64xi32>} : () -> tensor<2x64xi32>
  %1 = "tf.Const"() {value = dense<0.0> : tensor<916x8192xf32>} : () -> tensor<916x8192xf32>
  %2 = "tf.Const"() {value = dense<0.0> : tensor<8192x916xbf16>} : () ->tensor<8192x916xbf16>

  %3= "tf.DTensorAllReduce"(%1, %0) {_layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<916x8192xf32>, tensor<2x64xi32>) -> tensor<916x8192xf32>
  %4= "tf.DTensorAllReduce"(%2, %0) {_layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*TPU"], device_type = "/job:localhost/replica:0/task:0/device:TPU", reduce_op = "Add"} : (tensor<8192x916xbf16>, tensor<2x64xi32>) -> tensor<8192x916xbf16>
  %5 = "tf.Cast"(%4){Truncate = false, device = ""} : ( tensor<8192x916xbf16>) -> tensor<8192x916xf32>
  %6 = "tf.Identity"(%5){Truncate = false, device = ""} : (tensor<8192x916xf32>) -> tensor<8192x916xf32>
  %7 = "tf.Const"() {value = dense<[916,8192]> : tensor<2xi32>} : () -> tensor<2xi32>
  %8 = "tf.Reshape"(%6, %7) : (tensor<8192x916xf32>, tensor<2xi32>) -> tensor<916x8192xf32>

  %9= "tf.AddN"(%8, %3) { device = ""} : (tensor<916x8192xf32>, tensor<916x8192xf32>) -> tensor<916x8192xf32>
  %10 = "tf.B"(%9) : (tensor<916x8192xf32>) ->  tensor<916x8192xf32>
  func.return %10 :  tensor<916x8192xf32>
}
