// RUN: dtensor-opt %s -dtensor-device-mesh-cluster-coarsening -split-input-file -verify-diagnostics | FileCheck %s

// The layout string is the text format escaped string from a simple 2 device, 1d mesh.
//
// The corresponding proto to CPU used in below tests:
// mesh_config {
//   mesh_dimensions {
//     dimension {
//     name: "batch"
//   }
//   size: 2
// }
//   devices: "/job:localhost/task:0/device:CPU:0"
//   devices: "/job:localhost/task:0/device:CPU:1"
// }
//
// For TPU, just replace CPU string in devices with TPU.

// CHECK-LABEL: func @coarsen_cluster_with_same_device_config
func.func @coarsen_cluster_with_same_device_config() {
    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:      %[[A_OUT:.*]] = "tf.A"
    // CHECK-NEXT:      %[[B_OUT:.*]] = "tf.B"
    // CHECK-NEXT:      tf_device.return
    // CHECK-NEXT:    _mesh = "CPU|batch=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"
    %0 = "tf_device.cluster"() ({
      %1 = "tf.A"() : () -> tensor<i32>
      tf_device.return %1 : tensor<i32>
    }) {_mesh = "CPU|batch=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"} : () -> (tensor<i32>)

    %2 = "tf_device.cluster"() ({
      %3 = "tf.B"() : () -> tensor<f32>
      tf_device.return %3 : tensor<f32>
    }) {_mesh = "CPU|batch=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"} : () -> (tensor<f32>)
    func.return
}

// -----

// CHECK-LABEL: func @coarsening_clusters_with_different_configs
func.func @coarsening_clusters_with_different_configs() {
    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:      %[[A_OUT:.*]] = "tf.A"
    // CHECK-NEXT:      %[[B_OUT:.*]] = "tf.B"
    // CHECK-NEXT:      tf_device.return %[[A_OUT]], %[[B_OUT]]
    // CHECK-NEXT:    _mesh = "CPU|batch=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"
    //
    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:      %[[C_OUT:.*]] = "tf.C"
    // CHECK-NEXT:      %[[D_OUT:.*]] = "tf.D"
    // CHECK-NEXT:      tf_device.return %[[D_OUT]]
    // CHECK:        _mesh = "TPU|batch=2|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"
    //
    // CHECK:        "tf_device.cluster"
    // CHECK-NEXT:      %[[E_OUT:.*]] = "tf.E"
    // CHECK-NEXT:      tf_device.return %[[E_OUT]]
    // CHECK:        _mesh = "CPU|batch=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"
    %0 = "tf_device.cluster"() ({
      %1 = "tf.A"() : () -> tensor<i32>
      tf_device.return %1 : tensor<i32>
    }) {_mesh= "CPU|batch=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"} : () -> (tensor<i32>)

    %2 = "tf_device.cluster"() ({
      %3 = "tf.B"() : () -> tensor<f32>
      tf_device.return %3 : tensor<f32>
    }) {_mesh= "CPU|batch=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"} : () -> (tensor<f32>)

    %5 = "tf_device.cluster"() ({
      %4 = "tf.C"(%2, %0) : (tensor<f32>, tensor<i32>) -> tensor<f32>
      tf_device.return %4 : tensor<f32>
    }) {_mesh= "TPU|batch=2|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"} : () -> (tensor<f32>)

    %7 = "tf_device.cluster"() ({
      %6 = "tf.D"(%0, %5) : (tensor<i32>, tensor<f32>) -> tensor<f32>
      tf_device.return %6 : tensor<f32>
    }) {_mesh= "TPU|batch=2|0,1|0,1|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1"} : () -> (tensor<f32>)

    %9 = "tf_device.cluster"() ({
      %8 = "tf.E"(%7) : (tensor<f32>) -> tensor<f32>
      tf_device.return %8 : tensor<f32>
    }) {_mesh= "CPU|batch=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"} : () -> (tensor<i32>)

    func.return
}

// -----

func.func @check_cluster_without_mesh_attribute_disallowed() {
    // expected-error @+1 {{failed to merge mesh cluster as cluster does not have mesh attribute. This is likely due to problem in mesh propagation}}
    %0 = "tf_device.cluster"() ({
      %1 = "tf.A"() : () -> tensor<i32>
      tf_device.return %1 : tensor<i32>
    }) : () -> (tensor<i32>)

    %2 = "tf_device.cluster"() ({
      %3 = "tf.B"() : () -> tensor<f32>
      tf_device.return %3 : tensor<f32>
    }) {_mesh = "CPU|batch=2|0,1|0,1|/job:localhost/task:0/device:CPU:0,/job:localhost/task:0/device:CPU:1"} : () -> (tensor<f32>)
    func.return
}

// -----

// Check ops in tf.WhileRegions are grouped into cluster correctly.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>,
  %arg1: tensor<4xf32> {tf._layout = "sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3", tf._mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"})
-> (tensor<4xf32> {tf._default_layout = "sharding_specs:unsharded, mesh:|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"}) attributes {tf.entry_function = {control_outputs = "eager_operation", inputs = "device_id,op_input_0", outputs = "op_output_0"}} {
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT:   "tf.WhileRegion"
  // CHECK:          "tf_device.cluster"
  // CHECK-NEXT:       constant
  // CHECK-NEXT:       "tf.NotEqual"
  // CHECK-NEXT:       tf_device.return
  // CHECK-NEXT:     _mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"
  // CHECK-NEXT:     "tf.Yield"
  // CHECK:          "tf_device.cluster"
  // CHECK-NEXT:       constant
  // CHECK-NEXT:       "tf.Sub"
  // CHECK-NEXT:       tf_device.return
  // CHECK-NEXT:     _mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"
  // CHECK-NEXT:     "tf.Yield"
  // CHECK-NEXT:   (tensor<4xf32>, tensor<i32>) -> (tensor<4xf32>, tensor<i32>)
  // CHECK-NEXT:   "tf.Identity"
  // CHECK-NEXT:   tf_device.return
  // CHECK:      _mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"
  %15:2 = "tf_device.cluster"() ({
    %2:2 = "tf.WhileRegion"(%arg1, %arg0) ({
      ^bb0(%carg0: tensor<4xf32>, %carg1: tensor<i32>):
         %11 = "tf_device.cluster"() ({
           %limit = arith.constant dense<5> : tensor<i32>
           tf_device.return %limit : tensor<i32>
         }){_mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"} : () -> tensor<i32>


         %12 = "tf_device.cluster"() ({
           %cond = "tf.NotEqual"(%carg1, %11) : (tensor<i32>, tensor<i32>) -> tensor<i1>
           tf_device.return %cond : tensor<i1>
         }) {_mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"}: () -> tensor<i1>

         "tf.Yield"(%12) : (tensor<i1>) -> ()
    },  {
      ^bb0(%barg0: tensor<4xf32>, %barg1: tensor<i32>):
        %13 = "tf_device.cluster"() ({
          %one = arith.constant dense<1.0> : tensor<4xf32>
          tf_device.return %one: tensor<4xf32>
         }) {_mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"}: () -> tensor<4xf32>

        %14 = "tf_device.cluster"() ({
          %sub = "tf.Sub"(%barg0, %13) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
          tf_device.return %sub: tensor<4xf32>
         }) {_mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"}: () -> tensor<4xf32>

        "tf.Yield"(%14, %barg1) : (tensor<4xf32>, tensor<i32>) -> ()
    }) {is_stateless = true} : (tensor<4xf32>, tensor<i32>) -> (tensor<4xf32>, tensor<i32>)

    tf_device.return %2#0, %2#1 : tensor<4xf32>, tensor<i32>
  }) {_mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"} : () -> (tensor<4xf32>, tensor<i32>)

  %16 = "tf_device.cluster"() ({
    %5 = "tf.Identity"(%15#0) : (tensor<4xf32>) -> (tensor<4xf32>)
    tf_device.return %5 : tensor<4xf32>
  }){_mesh = "|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/replica:0/task:0/device:CPU:0,/job:localhost/replica:0/task:0/device:CPU:1,/job:localhost/replica:0/task:0/device:CPU:2,/job:localhost/replica:0/task:0/device:CPU:3"} : () -> tensor<4xf32>

  func.return %16 : tensor<4xf32>
}

