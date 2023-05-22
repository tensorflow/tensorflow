// RUN: dtensor-opt %s -dtensor-annotate-global-shape -dtensor-layout-propagation -split-input-file -verify-diagnostics | FileCheck %s

// Check Unary op layout propagation.
// CHECK-LABEL: func @main
func.func @main() {
    // CHECK:      "tf_device.cluster"
    // CHECK: %1 =   "tf.Const"()
    // CHECK-SAME:   _layout = ["sharding_specs:unsharded, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
    // CHECK-SAME:   () -> tensor<i32>
    // CHECK:        %2 = "tf.Neg"(%1)
    // CHECK-SAME:   _layout = ["sharding_specs:unsharded, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
    // CHECK-SAME:   (tensor<i32>) -> tensor<i32>
    // CHECK:       tf_device.return
    %0 = "tf_device.cluster"() ({
      %1 = "tf.Const"() {_layout = ["sharding_specs:unsharded, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"], value = dense<10> : tensor<i32>} : () -> tensor<i32>
      %2 = "tf.Neg"(%1) : (tensor<i32>) -> tensor<i32>
      tf_device.return %2 : tensor<i32>
    }) {_mesh="CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU3"} : () -> (tensor<i32>)
    func.return
}

// -----

// Check elementwise op with operands having incompatible layouts is not
// allowed.
func.func @main() {
    %0 = "tf_device.cluster"() ({
      %1 = "tf.Const"() {_layout = ["sharding_specs:unsharded,unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"], value = dense<10> : tensor<i32>} : () -> tensor<i32>
      %2 = "tf.Const"() {_layout = ["sharding_specs:x,z, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"], value = dense<10> : tensor<i32>} : () -> tensor<i32>
      // expected-error @+1 {{mesh dimension not contained in mesh}}
      %3 = "tf.Add"(%1, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_device.return %3 : tensor<i32>
    }) : () -> (tensor<i32>)
    func.return
}

// -----

// Check elementwise op layout propagation with first operand missing layout.
// CHECK-LABEL: func @main
func.func @main() {
    // CHECK:      "tf_device.cluster"
    // CHECK:        %1 = "tf.Const"()
    // CHECK-SAME:   () -> tensor<2x2xi32>
    // CHECK:        %2 = "tf.Const"()
    // CHECK-SAME:   _layout = ["sharding_specs:unsharded,unsharded, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
    // CHECK-SAME:   () -> tensor<2x2xi32>
    // CHECK:        %3 = "tf.Add"(%1, %2)
    // CHECK-SAME:   _layout = ["sharding_specs:unsharded,unsharded, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
    // CHECK-SAME:   (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    // CHECK:       tf_device.return
    %0 = "tf_device.cluster"() ({
      %1 = "tf.Const"() {value = dense<10> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
      %2 = "tf.Const"() {_layout = ["sharding_specs:unsharded,unsharded, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"], value = dense<10> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
      %3 = "tf.Add"(%1, %2) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
      tf_device.return %3 : tensor<2x2xi32>
    }) {_mesh = "CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"}  : () -> (tensor<2x2xi32>)
    func.return
}

// -----

// Check elementwise op layout propagation
// CHECK-LABEL: func @main
func.func @main() {
    // CHECK:      "tf_device.cluster"
    // CHECK:        %1 = "tf.Const"()
    // CHECK-SAME:   _layout = ["sharding_specs:unsharded, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
    // CHECK-SAME:   () -> tensor<i32>
    // CHECK:        %2 = "tf.Const"()
    // CHECK-SAME:   _layout = ["sharding_specs:unsharded, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
    // CHECK-SAME:   () -> tensor<i32>
    // CHECK:        %3 = "tf.Add"(%1, %2)
    // CHECK-SAME:   _layout = ["sharding_specs:unsharded, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
    // CHECK-SAME:   (tensor<i32>, tensor<i32>) -> tensor<i32>
    // CHECK:       tf_device.return
    %0 = "tf_device.cluster"() ({
      %1 = "tf.Const"() {_layout = ["sharding_specs:unsharded, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"], value = dense<10> : tensor<i32>} : () -> tensor<i32>
      %2 = "tf.Const"() {_layout = ["sharding_specs:unsharded, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"], value = dense<10> : tensor<i32>} : () -> tensor<i32>
      %3 = "tf.Add"(%1, %2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_device.return %3 : tensor<i32>
    }) {_mesh = "mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"} : () -> (tensor<i32>)
    func.return
}

// -----

// Check layout propagation of elementwise op with broadcast propagation.
// CHECK-LABEL: func @main
func.func @main() {
    // CHECK:      "tf_device.cluster"
    // CHECK:        %1 = "tf.Const"()
    // CHECK-SAME:   _layout = ["sharding_specs:unsharded,unsharded, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
    // CHECK-SAME:   () -> tensor<10x10xi32>
    // CHECK:        %2 = "tf.Const"()
    // CHECK-SAME:   _layout = ["sharding_specs: mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
    // CHECK-SAME:   () -> tensor<i32>
    // CHECK:        %3 = "tf.Add"(%1, %2)
    // CHECK-SAME:   _layout = ["sharding_specs:unsharded,unsharded, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
    // CHECK-SAME:   (tensor<10x10xi32>, tensor<i32>) -> tensor<10x10xi32>
    // CHECK:      tf_device.return
    %0 = "tf_device.cluster"() ({
      %1 = "tf.Const"() {_layout = ["sharding_specs:unsharded,unsharded, mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"], value = dense<10> : tensor<10x10xi32>} : () -> tensor<10x10xi32>
      %2 = "tf.Const"() {_layout = ["sharding_specs: mesh:CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"], value = dense<10> : tensor<i32>} : () -> tensor<i32>
      %3 = "tf.Add"(%1, %2) : (tensor<10x10xi32>, tensor<i32>) -> tensor<10x10xi32>
      tf_device.return %3 : tensor<10x10xi32>
    }) {_mesh = "CPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"} : () -> (tensor<10x10xi32>)
    func.return
}


// -----

// Check layout propagation of elementwise op with multiple device cluster.
// CHECK-LABEL: func @main
func.func @main() {
    // CHECK:      "tf_device.cluster"
    // CHECK:        %2 = "tf.Const"()
    // CHECK-SAME:   _layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
    // CHECK-SAME:   () -> tensor<i32>
    // CHECK:        %3 = "tf.Const"()
    // CHECK-SAME:   _layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
    // CHECK-SAME:   () -> tensor<i32>
    // CHECK:        %4 = "tf.Add"(%2, %3)
    // CHECK-SAME:   _layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
    // CHECK-SAME:   (tensor<i32>, tensor<i32>) -> tensor<i32>
    // CHECK:        tf_device.return
    %0 = "tf_device.cluster"() ({
      %2 = "tf.Const"() {_layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"], value = dense<10> : tensor<i32>} : () -> tensor<i32>
      %3 = "tf.Const"() {_layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"], value = dense<10> : tensor<i32>} : () -> tensor<i32>
      %4 = "tf.Add"(%2, %3) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_device.return %4 : tensor<i32>
    }) {_layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]} : () -> (tensor<i32>)

    // CHECK:      "tf_device.cluster"
    //
    // CHECK: %2 = "tf.Const"()
    // CHECK-SAME: _layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
    // CHECK-SAME: () -> tensor<i32>
    //
    // CHECK: %3 = "tf.Const"()
    // CHECK-SAME: _layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
    // CHECK-SAME: () -> tensor<i32>
    //
    // CHECK: %4 = "tf.Add"(%2, %3)
    // CHECK-SAME: _layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
    // CHECK-SAME: (tensor<i32>, tensor<i32>) -> tensor<i32>
    //
    // CHECK:      tf_device.return
    %1 = "tf_device.cluster"() ({
      %2 = "tf.Const"() {_layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"], value = dense<10> : tensor<i32>} : () -> tensor<i32>
      %3 = "tf.Const"() {_layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"], value = dense<10> : tensor<i32>} : () -> tensor<i32>
      %4 = "tf.Add"(%2, %3) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      tf_device.return %4 : tensor<i32>
    }) {_mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"} : () -> (tensor<i32>)
    func.return
}

// -----

// Check layout propagation of elementwise op with multiple inputs.
// CHECK-LABEL: func @main
// CHECK:      "tf_device.cluster"() ({
// CHECK-NEXT:   %1 = "tf.Add"(%arg1, %arg2)
// CHECK-SAME:   _layout = ["sharding_specs:unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"]
// CHECK-SAME:   (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK:        tf_device.return
func.func @main(%arg0: tensor<i64>,
  %arg1: tensor<1xf32> { tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"},
  %arg2: tensor<1xf32> { tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=1|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"}) -> tensor<1xf32> {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Add"(%arg1, %arg2) {} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    tf_device.return %1 : tensor<1xf32>
  }) {} : () -> tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// -----

// Check layout propagation of pack op.
// CHECK-LABEL: func @main
func.func @main(%arg1: tensor<1xf32> { tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3" }) -> tensor<1xf32> {
  // CHECK:       "tf.Pack"
  // CHECK-SAME:  _layout = ["sharding_specs:unsharded,x,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
  // CHECK-SAME:  (tensor<10x10xf32>, tensor<10x10xf32>, tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<4x10x10xf32>
  // CHECK-NEXT:  tf_device.return
  // CHECK-SAME:  tensor<4x10x10xf32>
  %0 = "tf_device.cluster"() ({
    %1 = "tf.A"() {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]} : () -> tensor<10x10xf32>
    %2 = "tf.A"() {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]} : () -> tensor<10x10xf32>
    %3 = "tf.A"() {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]} : () -> tensor<10x10xf32>
    %4 = "tf.A"() {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]} : () -> tensor<10x10xf32>
    %5 = "tf.Pack"(%1, %2, %3, %4) {} : (tensor<10x10xf32>, tensor<10x10xf32>, tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<4x10x10xf32>
    tf_device.return %5 : tensor<4x10x10xf32>
  }) {} : () -> tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// -----

// Check layout propagation logic of tf.Pack op with a single operand.
// CHECK-LABEL: func @main
func.func @main(%arg1: tensor<1xf32> { tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3" }) -> tensor<1xf32> {
  // CHECK:      "tf.Pack"(%1)
  // CHECK-SAME: _layout = ["sharding_specs:x,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
  // CHECK-SAME: axis = 0
  // CHECK-SAME: (tensor<10x10xf32>) -> tensor<10x10xf32>
  // CHECK-NEXT: tf_device.return
  // CHECK-SAME: tensor<10x10xf32>
  %0 = "tf_device.cluster"() ({
    %1 = "tf.A"() {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]} : () -> tensor<10x10xf32>
    %2 = "tf.Pack"(%1) {axis = 0 : i64} : (tensor<10x10xf32>) -> tensor<10x10xf32>
    tf_device.return %2 : tensor<10x10xf32>
  }) {} : () -> tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// -----

// Check layout propagation of pack op with non-matching layouts.
func.func @main(%arg1: tensor<1xf32> { tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=2|*TPU" }) -> tensor<1xf32> {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.A"() {_layout = ["sharding_specs:x,y, mesh:|x=2,y=2|*TPU"]} : () -> tensor<10x10xf32>
    %2 = "tf.A"() {_layout = ["sharding_specs:x,unsharded, mesh:|x=2,y=2|*TPU"]} : () -> tensor<10x10xf32>
    %3 = "tf.A"() {_layout = ["sharding_specs:x,unsharded, mesh:|x=2,y=2|*TPU"]} : () -> tensor<10x10xf32>
    %4 = "tf.A"() {_layout = ["sharding_specs:y,x, mesh:|x=2,y=2|*TPU"]} : () -> tensor<10x10xf32>
    // expected-error @+1 {{'tf.Pack' op All arguments to pack must have the same layout.}}
    %5 = "tf.Pack"(%1, %2, %3, %4) {} : (tensor<10x10xf32>, tensor<10x10xf32>, tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<4x10x10xf32>
    tf_device.return %5 : tensor<4x10x10xf32>
  }) {} : () -> tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// -----

// Check layout propagation of reshape op with replicated inputs.
// CHECK-LABEL: func @main
func.func @main(%arg1: tensor<1xf32> { tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=2|*TPU" }) -> tensor<1xf32> {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Const" () {_layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|*TPU"], value = dense<10> : tensor<1xi32>}: () -> tensor<1xi32>
    %2 = "tf.Const" () {_layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|*TPU"], value = dense<-1> : tensor<1xi32>}: () -> tensor<1xi32>
    %3 = "tf.Pack" (%1, %2) : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %4 = "tf.A"() {_layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*TPU"]} : () -> tensor<10x10xf32>
    // CHECK:      "tf.Reshape"(%4, %3)
    // CHECK-SAME: (tensor<10x10xf32>, tensor<2xi32>) -> tensor<10x10xf32>
    %5 = "tf.Reshape"(%4, %3) : (tensor<10x10xf32>, tensor<2xi32>) -> tensor<10x10xf32>
    tf_device.return %5 : tensor<10x10xf32>
  }) {} : () -> tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// -----

// Check layout propagation of reshape op with replicated inputs with different rank.
// CHECK-LABEL: func @main
func.func @main(%arg1: tensor<1xf32> { tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=2|*TPU" }) -> tensor<1xf32> {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Const" () {_layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|*TPU"], value = dense<-1> : tensor<1xi32>}: () -> tensor<1xi32>
    %2 = "tf.A"() {_layout = ["sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*TPU"]} : () -> tensor<10x10xf32>
    // CHECK:      "tf.Reshape"(%2, %1)
    // CHECK-SAME: _layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"]
    // CHECK-SAME: (tensor<10x10xf32>, tensor<1xi32>) -> tensor<100xf32>
    %3 = "tf.Reshape"(%2, %1) : (tensor<10x10xf32>, tensor<1xi32>) -> tensor<100xf32>
    tf_device.return %3 : tensor<100xf32>
  }) {} : () -> tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// -----

// Check reshape op with batch sharded inputs.
// CHECK-LABEL: func @main
func.func @main(%arg1: tensor<1xf32> { tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=2|*TPU" }) -> tensor<1xf32> {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Const" () {_layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|*TPU"], value = dense<-1> : tensor<1xi32>}: () -> tensor<1xi32>
    %2 = "tf.Const" () {_layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|*TPU"], value = dense<100> : tensor<1xi32>}: () -> tensor<1xi32>
    %3 = "tf.Pack" (%1, %2) : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %4 = "tf.A"() {_layout = ["sharding_specs:x,unsharded,unsharded, mesh:|x=2,y=2|*TPU"]} : () -> tensor<10x10x10xf32>
    // CHECK:      "tf.Reshape"
    // CHECK-SAME: (tensor<10x10x10xf32>, tensor<2xi32>) -> tensor<10x100xf32>
    %5 = "tf.Reshape"(%4, %3) : (tensor<10x10x10xf32>, tensor<2xi32>) -> tensor<10x100xf32>
    tf_device.return %5 : tensor<10x100xf32>
  }) {} : () -> tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// -----

// Check that layout propagation of inputs that are sharded in non-batch dimension is disallowed.
func.func @main(%arg0: tensor<32x32xf32> { tf._layout = "sharding_specs:x,y, mesh:|x=2,y=2|*TPU"}, %arg1: tensor<32x32xf32> { tf._layout = "sharding_specs:x,y, mesh:|x=2,y=2|*TPU"}) {
  "tf_device.cluster"() ({
    // expected-error @+1 {{Features input to Softmax loss ops must be sharded only across batch dimension}}
    "tf.SoftmaxCrossEntropyWithLogits"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> (tensor<32x1xf32>, tensor<32x32xf32>)
    tf_device.return
 }) {layout = "sharding_specs:x,y, mesh:|x=2,y=2|*TPU"} : () -> ()
 func.return
}

// -----

// Check layout propagation of read variable op.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<!tf_type.resource<tensor<1xf32>>> { tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=2|*CPU" }) -> tensor<1xf32> {
  // CHECK:      "tf_device.cluster"
  //
  // CHECK: %1 = "tf.ReadVariableOp"(%arg0)
  // CHECK-SAME: _layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
  // CHECK-SAME: (tensor<!tf_type.resource<tensor<1xf32>>>) -> tensor<1xf32>
  //
  // CHECK:      tf_device.return
  %0 = "tf_device.cluster"() ({
    %1 = "tf.ReadVariableOp"(%arg0) : (tensor<!tf_type.resource<tensor<1xf32>>>) -> tensor<1xf32>
    tf_device.return %1 : tensor<1xf32>
  }) {} : () -> tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// -----

// Check layout propagation of const ops from it's consumers.
// CHECK-LABEL: func @main
func.func @main() {
    // CHECK:        "tf_device.cluster"
    //
    // CHECK: %1 = "tf.Const"()
    // CHECK-SAME: _layout = ["sharding_specs:x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
    // CHECK-SAME: () -> tensor<22xi32>
    //
    // CHECK: %2 = "tf.Const"()
    // CHECK-SAME: _layout = ["sharding_specs:x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
    // CHECK-SAME: () -> tensor<22xi32>
    //
    // CHECK: %3 = "tf.Const"()
    // CHECK-SAME: _layout = ["sharding_specs:x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
    // CHECK-SAME: () -> tensor<22xi32>
    //
    // CHECK: %4 = "tf.Add"(%1, %2)
    // CHECK-SAME: _layout = ["sharding_specs:x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
    // CHECK-SAME: (tensor<22xi32>, tensor<22xi32>) -> tensor<22xi32>
    // CHECK:      tf_device.return
    %0 = "tf_device.cluster"() ({
      %1 = "tf.Const"() {_layout = ["sharding_specs:x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"], value = dense<[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]> : tensor<22xi32>} : () -> tensor<22xi32>
      %2 = "tf.Const"() {value = dense<[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]> : tensor<22xi32>} : () -> tensor<22xi32>
      %3 = "tf.Const"() {_layout = ["sharding_specs:x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"], value = dense<[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]> : tensor<22xi32>} : () -> tensor<22xi32>
      %4 = "tf.Add"(%1,%2) {_layout = ["sharding_specs:x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]} : (tensor<22xi32>,tensor<22xi32>) -> tensor<22xi32>
      %5 = "tf.Add"(%2,%3) {_layout = ["sharding_specs:x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]} : (tensor<22xi32>,tensor<22xi32>) -> tensor<22xi32>
      tf_device.return %3 : tensor<22xi32>
    }) {_mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"} : () -> (tensor<22xi32>)

    func.return
}

// -----

// CHECK-LABEL: func @main
func.func @main() {
  // CHECK:      "tf_device.cluster"
  // CHECK:        %1 = "tf.Const"()
  // CHECK-SAME:   _layout = ["sharding_specs:x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
  // CHECK-SAME:   () -> tensor<2xi32>
  // CHECK:        %2 = "tf.Const"()
  // CHECK-SAME:   _layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
  // CHECK-SAME:   () -> tensor<2xi32>
  // CHECK:        %3 = "tf.Reshape"(%1, %2)
  // CHECK-SAME:   _layout = ["sharding_specs:x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
  // CHECK-SAME:   (tensor<2xi32>, tensor<2xi32>) -> tensor<1x2xi32>
  // CHECK:        tf_device.return
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Const"() {_layout = ["sharding_specs:x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"], value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
    %2 = "tf.Const"() {_layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"], value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
    %3 = "tf.Reshape"(%1, %2) {_layout = ["sharding_specs:x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]} : (tensor<2xi32>, tensor<2xi32>) -> tensor<1x2xi32>
    tf_device.return %3 : tensor<1x2xi32>
  }) {_layout = ["sharding_specs:x, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]} : () -> tensor<1x2xi32>
  func.return
}

// -----

// Check layout propagation of fill op.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>, %arg1: tensor<2xi32>, %arg2: tensor<f32>) -> (tensor<?x?xf32>{
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT:   "tf.Fill"
  // CHECK-SAME:   _layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"], device = ""}
  // CHECK-SAME:   tensor<2xi32>, tensor<f32>) -> tensor<?x?xf32>
  tf._default_layout = "sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"}) {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.Fill"(%arg1, %arg2) {device = ""} : (tensor<2xi32>, tensor<f32>) -> tensor<?x?xf32>
    tf_device.return %1 : tensor<?x?xf32>
  }) {} : () -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

// Check that layouts of ops in function definitions are propagated by inferring
// layouts from function default layout values.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> (tensor<?x?xi32>{
  // CHECK:      "tf_device.cluster"
  tf._default_layout = "sharding_specs:unsharded, mesh:|x=2,y=2|*CPU"}) {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.PartitionedCall"(%arg1, %arg2) {f = @callee1, config = "", config_proto = "", executor_type = ""} : (tensor<i32>, tensor<i32>) -> tensor<?x?xi32>
    tf_device.return %1 : tensor<?x?xi32>
  }) { _mesh = "|x=2,y=2|*CPU" } : () -> tensor<?x?xi32>
  func.return %0 : tensor<?x?xi32>
}

// CHECK-LABEL: func private @callee1
// CHECK-SAME:  %arg0: tensor<i32>
// CHECK-SAME:  %arg1: tensor<i32>
// CHECK:       tf._default_layout = "sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"
func.func private @callee1(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<?x?xi32> attributes {tf.signature.is_stateful} {
  %1 = "tf_device.cluster"() ({
    %0 = "tf.PartitionedCall"(%arg0, %arg1) {f = @callee2, config = "", config_proto = "", executor_type = ""} : (tensor<i32>, tensor<i32>) -> tensor<?x?xi32>
    tf_device.return %0 : tensor<?x?xi32>
  }) { _mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|*CPU" } : () -> tensor<?x?xi32>
  func.return %1 : tensor<?x?xi32>
}

// CHECK-LABEL: func private @callee2
// CHECK-SAME:  %arg0: tensor<i32>
// CHECK-SAME:  %arg1: tensor<i32>
// CHECK:       tf._default_layout = "sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"
func.func private @callee2(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<?x?xi32> attributes {tf.signature.is_stateful} {
  %1 = "tf_device.cluster"() ({
    %0 = "tf.Const"() { value = dense<[1, 2]> : tensor<2xi32>} : () -> tensor<2xi32>
    %1 = "tf.Fill"(%0, %arg1) {device = ""} : (tensor<2xi32>, tensor<i32>) -> tensor<?x?xi32>
    tf_device.return %1 : tensor<?x?xi32>
  }) { _mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3" } : () -> tensor<?x?xi32>
  func.return %1 : tensor<?x?xi32>
}

// -----

// Check that layouts of ops in function definitions are propagated by inferring
// layouts from function argument layouts.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>,
  %arg1: tensor<?x?xi32>{ tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=2|*CPU"},
  %arg2: tensor<?x?xi32>{ tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=2|*CPU"}) -> (tensor<?x?xi32>) {
  %0 = "tf_device.cluster"() ({
    %1 = "tf.PartitionedCall"(%arg1, %arg2) {f = @callee1, config = "", config_proto = "", executor_type = ""} : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
    tf_device.return %1 : tensor<?x?xi32>
  }) { _mesh = "|x=2,y=2|*CPU" } : () -> tensor<?x?xi32>
  func.return %0 : tensor<?x?xi32>
}

// CHECK-LABEL: func private @callee1
// CHECK-SAME:  %arg0: tensor<?x?xi32>
// CHECK-SAME:  tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"
// CHECK-SAME:  %arg1: tensor<?x?xi32>
// CHECK-SAME:  tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"
func.func private @callee1(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> attributes {tf.signature.is_stateful} {
  %1 = "tf_device.cluster"() ({
    %0 = "tf.PartitionedCall"(%arg0, %arg1) {f = @callee2, config = "", config_proto = "", executor_type = ""} : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
    tf_device.return %0 : tensor<?x?xi32>
  }) { _mesh = "mesh:CPU,x=2,y=2" } : () -> tensor<?x?xi32>
  func.return %1 : tensor<?x?xi32>
}

// CHECK-LABEL: func private @callee2
// CHECK-SAME:  %arg0: tensor<?x?xi32>
// CHECK-SAME:  tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"
// CHECK-SAME:  %arg1: tensor<?x?xi32>
// CHECK-SAME:  tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"
func.func private @callee2(%arg0: tensor<?x?xi32>, %arg1: tensor<?x?xi32>) -> tensor<?x?xi32> attributes {tf.signature.is_stateful} {
  // CHECK:      "tf_device.cluster"
  // CHECK:        "tf.Add"
  // CHECK-SAME:   _layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
  %1 = "tf_device.cluster"() ({
    %1 = "tf.Add"(%arg0, %arg1) {device = ""} : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
    tf_device.return %1 : tensor<?x?xi32>
  }) { _mesh = "|x=2,y=2|*CPU" } : () -> tensor<?x?xi32>
  func.return %1 : tensor<?x?xi32>
}

// -----

// Check that layouts of ops in functions with multiple outputs with different
// layouts are supported.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>,
  %arg1: tensor<2x2xi32>{
    tf._layout = "sharding_specs:unsharded,unsharded, mesh:|batch=2,x=2|*CPU"},
  %arg2: tensor<2x2xi32>{
    tf._layout = "sharding_specs:batch,unsharded, mesh:|batch=2,x=2|*CPU"})
-> (tensor<2x2xi32>) {
  %0 = "tf_device.cluster"() ({
    // CHECK:      "tf.PartitionedCall"
    // CHECK-SAME: _layout
    // CHECK-SAME: "sharding_specs:unsharded,unsharded, mesh:|batch=2,x=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"
    %1, %2 = "tf.PartitionedCall"(%arg1, %arg2) {f = @callee1, config = "", config_proto = "", executor_type = ""} : (tensor<2x2xi32>, tensor<2x2xi32>) -> (tensor<2x2xi32>, tensor<2x2xi32>)
    tf_device.return %1 : tensor<2x2xi32>
  }) {_mesh = "|batch=2,x=2|*CPU"} : () -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// CHECK-LABEL: func private @callee1
// CHECK-SAME:  %arg0: tensor<2x2xi32>
// CHECK-SAME:  tf._layout = "sharding_specs:unsharded,unsharded, mesh:|batch=2,x=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"
// CHECK-SAME:  %arg1: tensor<2x2xi32>
// CHECK-SAME:  tf._layout = "sharding_specs:batch,unsharded, mesh:|batch=2,x=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"
// CHECK:       tf._default_layout = "sharding_specs:unsharded,unsharded, mesh:|batch=2,x=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"
// CHECK:       tf._default_layout = "sharding_specs:batch,unsharded, mesh:|batch=2,x=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"
func.func private @callee1(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> (tensor<2x2xi32>, tensor<2x2xi32>)  attributes {tf.signature.is_stateful} {
  %5, %6 = "tf_device.cluster"() ({
    // CHECK:       "tf.Const"()
    // CHECK-SAME:   _layout = ["sharding_specs:unsharded,unsharded, mesh:|batch=2,x=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
    %1 = "tf.Const"() {value = dense<10> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    // CHECK:       "tf.Add"
    // CHECK-SAME:   _layout = ["sharding_specs:unsharded,unsharded, mesh:|batch=2,x=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
    %2 = "tf.Add"(%1, %arg0) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    // CHECK:       "tf.Const"()
    // CHECK-SAME:   _layout = ["sharding_specs:unsharded,unsharded, mesh:|batch=2,x=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
    %3 = "tf.Const"() {value = dense<10> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
    // CHECK:       "tf.Add"
    // CHECK-SAME:   _layout = ["sharding_specs:batch,unsharded, mesh:|batch=2,x=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"]
    %4 = "tf.Add"(%3, %arg1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
    tf_device.return %2, %4 : tensor<2x2xi32>, tensor<2x2xi32>
  }) {_mesh = "|batch=2,x=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1,/job:localhost/task:0/device:CPU:2,/job:localhost/task:0/device:CPU:3"} : () -> (tensor<2x2xi32>, tensor<2x2xi32>)
  func.return %5, %6 : tensor<2x2xi32>, tensor<2x2xi32>
}

// -----
// Unimplemented op throws an error.
func.func @main() {
  %0 = "tf_device.cluster"() ({
    // expected-error @+1 {{does not implement layout propagation}}
    %0 = "tf.A"() : () -> tensor<2xi32>
    tf_device.return %0 : tensor<2xi32>
  }) {_mesh = "|batch=2,x=2|*CPU"} : () -> tensor<2xi32>
  func.return
}

