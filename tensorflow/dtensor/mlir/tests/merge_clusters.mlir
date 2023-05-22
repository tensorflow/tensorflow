// RUN: dtensor-opt %s -split-input-file -dtensor-merge-clusters -verify-diagnostics | FileCheck %s

// Check that multiple tf_device.Cluster ops with same mesh specification are
// merged correctly to a single global cluster.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>) -> (tensor<1xi32>,  tensor<i64>, tensor<1xi32>, tensor<i64>) {
  // CHECK:      "tf_device.cluster"
  // CHECK:        "tf.Cast"
  // CHECK-NEXT:   "tf.Const"
  // CHECK-NEXT:   "tf.FloorMod"
  // CHECK-NEXT:   "tf.XlaRecvFromHost"
  // CHECK-NEXT:   "tf.Const"
  // CHECK-NEXT:   "tf.Equal"
  // CHECK-NEXT:   "tf.IfRegion"
  // CHECK:        tf_device.return
  // CHECK-NEXT: _mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"
  // CHECK:      %[[CLUSTER_OUT:.*]]:4 = "tf_device.cluster"
  // CHECK:        "tf._TPUCompileMlirPlaceholderProgramKey"
  // CHECK-NEXT:   %[[CONST_OUT1:.*]] = "tf.Const"
  // CHECK-NEXT:   "tf.Const"
  // CHECK-NEXT:   "tf._XlaSendFromHostV2"
  // CHECK-NEXT:   "tf.Const"
  // CHECK-NEXT:   "tf._XlaSendFromHostV2"
  // CHECK-NEXT:   "tf.Const"
  // CHECK-NEXT:   "tf._XlaSendFromHostV2"
  // CHECK-NEXT:   %[[CONST_OUT2:.*]] = "tf.Const"
  // CHECK-NEXT:   "tf._XlaSendFromHostV2"
  // CHECK-NEXT:   "tf._TPUCompileMlirPlaceholderProgramKey"
  // CHECK-NEXT:   %[[CAST_OUT:.*]] = "tf.Cast"
  // CHECK-NEXT:   "tf.Const"
  // CHECK-NEXT:   "tf.FloorMod"
  // CHECK-NEXT:   %[[RECV_OUT:.*]] = "tf._XlaRecvAtHostV2"
  // CHECK-NEXT:   tf_device.return %[[CONST_OUT1]], %[[CONST_OUT2]], %[[RECV_OUT]], %[[CAST_OUT]]
  // CHECK-NEXT: _mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0
  // CHECK-NEXT: return %[[CLUSTER_OUT]]#0, %[[CLUSTER_OUT]]#1, %[[CLUSTER_OUT]]#2, %[[CLUSTER_OUT]]#3
  %7, %8 = "tf_device.cluster"() ({
    %0 = "tf._TPUCompileMlirPlaceholderProgramKey"() : () -> tensor<2x!tf_type.string>
    %1 = "tf.Const"() {_layout = ["sharding_specs:unsharded, mesh:CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"], value = dense<10> : tensor<1xi32>} : () -> tensor<1xi32>
    %2 = "tf.Const"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
    "tf._XlaSendFromHostV2"(%1, %0, %2) {key = "communication_key_sharding_specs:, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3_0"} : (tensor<1xi32>, tensor<2x!tf_type.string>, tensor<i64>) -> ()
    %3 = "tf.Const"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    "tf._XlaSendFromHostV2"(%1, %0, %3) {key = "communication_key_sharding_specs:, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3_0"} : (tensor<1xi32>, tensor<2x!tf_type.string>, tensor<i64>) -> ()
    %4 = "tf.Const"() {value = dense<2> : tensor<i64>} : () -> tensor<i64>
    "tf._XlaSendFromHostV2"(%1, %0, %4) {key = "communication_key_sharding_specs:, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3_0"} : (tensor<1xi32>, tensor<2x!tf_type.string>, tensor<i64>) -> ()
    %5 = "tf.Const"() {value = dense<3> : tensor<i64>} : () -> tensor<i64>
    "tf._XlaSendFromHostV2"(%1, %0, %5) {_layout = [], key = "communication_key_sharding_specs:, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3_0"} : (tensor<1xi32>, tensor<2x!tf_type.string>, tensor<i64>) -> ()
    tf_device.return %1, %5 : tensor<1xi32>,  tensor<i64>
  }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"} : () -> (tensor<1xi32>,  tensor<i64>)
  "tf_device.cluster"() ({
    %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<i32>) -> tensor<i64>
    %1 = "tf.Const"() {value = dense<4> : tensor<i64>} : () -> tensor<i64>
    %2 = "tf.FloorMod"(%0, %1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %3 = "tf.XlaRecvFromHost"() {_layout = ["sharding_specs:unsharded, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3_0"], key = "communication_key_sharding_specs:, mesh:TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3_0", shape = #tf_type.shape<1>} : () -> tensor<1xi32>
    %4 = "tf.Const"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %5 = "tf.Equal"(%2, %4) {incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
    "tf.IfRegion"(%5) ({
      "tf.XlaSendToHost"(%3) {key = "communication_key_sharding_specs:, CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0_2"} : (tensor<1xi32>) -> ()
      "tf.Yield"() : () -> ()
    },  {
      "tf.Yield"() : () -> ()
    }) {_layout = [], is_stateless = false} : (tensor<i1>) -> ()
    tf_device.return {_layout = []}
  }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> ()
  %9, %10 = "tf_device.cluster"() ({
    %0 = "tf._TPUCompileMlirPlaceholderProgramKey"() : () -> tensor<2x!tf_type.string>
    %1 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<i32>) -> tensor<i64>
    %2 = "tf.Const"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %3 = "tf.FloorMod"(%1, %2) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %4 = "tf._XlaRecvAtHostV2"(%0, %3) {_layout = ["sharding_specs:unsharded, mesh:CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"], key = "communication_key_sharding_specs:, CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0_2"} : (tensor<2x!tf_type.string>, tensor<i64>) -> tensor<1xi32>
    tf_device.return %4, %1 : tensor<1xi32>, tensor<i64>
  }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"} : () -> (tensor<1xi32>, tensor<i64>)
  func.return %7, %8, %9, %10 : tensor<1xi32>,  tensor<i64>, tensor<1xi32>, tensor<i64>
}

// -----

// Check that duplicate/nested tf_device.cluster ops are removed.

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  // CHECK:      "tf_device.cluster"
  // CHECK:        "tf.A"
  // CHECK-NEXT:   "tf.B"
  // CHECK-NEXT:   "tf.G"
  // CHECK-NEXT:   "tf.F"
  // CHECK-NEXT:   "tf.IfRegion"
  // CHECK-NEXT:     %[[D_OUT:.*]] = "tf.D"
  // CHECK-NEXT:     %[[I_OUT:.*]] = "tf.I"(%[[D_OUT]])
  // CHECK-NEXT:     "tf.J"(%[[I_OUT]])
  // CHECK-NEXT:     "tf.Yield"
  // CHECK:        %[[E_OUT:.*]] = "tf.E"
  // CHECK-NEXT:   tf_device.return %[[E_OUT]]
  // CHECK-NEXT: _mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"
  %2 = "tf_device.cluster"() ({
    %3 = "tf.A"() : () -> (tensor<?xi32>)
    %4 = "tf.B"() : () -> (tensor<?xi32>)
    %6 = "tf.G"() : () -> (tensor<i1>)
    %7 = "tf.F"() : () -> tensor<?xi32>
    "tf.IfRegion"(%6) ({
      %10 = "tf_device.cluster"() ({
        %8 = "tf.D"(%4, %3, %7) {} : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) -> (tensor<?xi32>)
        %9 = "tf.I"(%8) : (tensor<?xi32>) -> (tensor<?xi32>)
        tf_device.return %9 : tensor<?xi32>
      }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"} : () -> (tensor<?xi32>)

      "tf.J"(%10) : (tensor<?xi32>) -> ()

      "tf.Yield"() : () -> ()
    }, {
      "tf.Yield"() : () -> ()
    }) {is_stateless = false} : (tensor<i1>) -> ()

    %5 = "tf.E"() : () -> tensor<?xi32>
    tf_device.return %5 : tensor<?xi32>
  }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"} : () -> tensor<?xi32>
  func.return %2 : tensor<?xi32>
}

// -----

// Check clusters with no mesh specification are disallowed.

func.func @main(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  %2 = "tf_device.cluster"() ({
    %3 = "tf.A"() : () -> (tensor<?xi32>)
    %4 = "tf.B"() : () -> (tensor<?xi32>)
    %6 = "tf.G"() : () -> (tensor<i1>)
    %7 = "tf.F"() : () -> tensor<?xi32>
    "tf.IfRegion"(%6) ({

      // expected-error @+1 {{All clusters must have specified mesh}}
      "tf_device.cluster"() ({
        "tf.D"() : () -> ()
        tf_device.return
      }) : () -> ()

      "tf.Yield"() : () -> ()
    }, {
      "tf.Yield"() : () -> ()
    }) {is_stateless = false} : (tensor<i1>) -> ()

    %5 = "tf.E"() : () -> tensor<?xi32>
    tf_device.return %5 : tensor<?xi32>
  }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"} : () -> tensor<?xi32>

  func.return %2 : tensor<?xi32>
}

// -----

// Check nested clusters with input edges are disallowed.

func.func @main(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  %2 = "tf_device.cluster"() ({
    %3 = "tf.A"() : () -> (tensor<?xi32>)
    %4 = "tf.B"() : () -> (tensor<?xi32>)
    %6 = "tf.G"() : () -> (tensor<i1>)
    %7 = "tf.F"() : () -> tensor<?xi32>
    "tf.IfRegion"(%6) ({

      // expected-error @+1 {{found nested tf_device.Cluster op with inputs}}
      "tf_device.cluster"() ({
        "tf.D"(%4, %3, %7) {} : (tensor<?xi32>, tensor<?xi32>, tensor<?xi32>) -> ()
        tf_device.return
      }) {_mesh = "TPU|x=1|0|0|/job:localhost/task:0/device:TPU:0"} : () -> ()

      "tf.Yield"() : () -> ()
    }, {
      "tf.Yield"() : () -> ()
    }) {is_stateless = false} : (tensor<i1>) -> ()

    %5 = "tf.E"() : () -> tensor<?xi32>
    tf_device.return %5 : tensor<?xi32>
  }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"} : () -> tensor<?xi32>

  func.return %2 : tensor<?xi32>
}

// -----

// Check nested clusters with outputs edges are disallowed.

func.func @main(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  %2 = "tf_device.cluster"() ({
    %3 = "tf.A"() : () -> (tensor<?xi32>)
    %4 = "tf.B"() : () -> (tensor<?xi32>)
    %6 = "tf.G"() : () -> (tensor<i1>)
    %7 = "tf.F"() : () -> tensor<?xi32>
    "tf.IfRegion"(%6) ({

      // expected-error @+1 {{found nested tf_device.Cluster op with outputs}}
      %9 = "tf_device.cluster"() ({
        %8 = "tf.D"() : () -> tensor<?xi32>
        tf_device.return %8 : tensor<?xi32>
      }) {_mesh = "TPU|x=1|0|0|/job:localhost/task:0/device:TPU:0"} : () -> (tensor<?xi32>)

      "tf.Yield"() : () -> ()
    }, {
      "tf.Yield"() : () -> ()
    }) {is_stateless = false} : (tensor<i1>) -> ()

    %5 = "tf.E"() : () -> tensor<?xi32>
    tf_device.return %5 : tensor<?xi32>
  }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"} : () -> tensor<?xi32>

  func.return %2 : tensor<?xi32>
}


// -----

// Check tf.If control flow ops are decomposed correctly.

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  // CHECK:      "tf_device.cluster"
  // CHECK:        %[[PREDICATE_RECV_OUT:.*]] = "tf.DTensorRecv"
  // CHECK-SAME:   key = "SendRecvKeyForControlflow_0"
  // CHECK-NEXT:   "tf.IfRegion"(%[[PREDICATE_RECV_OUT]])
  // CHECK-NEXT:     "tf.D"
  // CHECK-NEXT:     "tf.Yield"
  // CHECK:        tf_device.return
  // CHECK-NEXT: _mesh = "TPU|x=1|0|0|/job:localhost/task:0/device:TPU:0"
  // CHECK-SAME: () -> ()

  // CHECK-NEXT: %[[CLUSTER_OUT:.*]] = "tf_device.cluster"
  // CHECK-NEXT:   "tf.A"
  // CHECK-NEXT:   "tf.B"
  // CHECK-NEXT:   %[[PREDICATE_OUT:.*]] = "tf.G"
  // CHECK-NEXT:   "tf.F"
  // CHECK-NEXT:   "tf.DTensorSend"(%[[PREDICATE_OUT]])
  // CHECK-NEXT:   "tf.IfRegion"(%[[PREDICATE_OUT]])
  // CHECK-NEXT:     "tf.Yield"
  // CHECK:          "tf.Yield"
  // CHECK:        %[[E_OUT:.*]] = "tf.E"
  // CHECK-NEXT:   tf_device.return %[[E_OUT]]
  // CHECK-NEXT: _mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"
  // CHECK-NEXT: return %[[CLUSTER_OUT]]
  %2 = "tf_device.cluster"() ({
    %3 = "tf.A"() : () -> (tensor<?xi32>)
    %4 = "tf.B"() : () -> (tensor<?xi32>)
    %6 = "tf.G"() : () -> (tensor<i1>)
    %7 = "tf.F"() : () -> tensor<?xi32>
    "tf.IfRegion"(%6) ({

      "tf_device.cluster"() ({
        "tf.D"() {} : () -> ()
        tf_device.return
      }) {_mesh = "TPU|x=1|0|0|/job:localhost/task:0/device:TPU:0"} : () -> ()

      "tf.Yield"() : () -> ()
    }, {
      "tf.Yield"() : () -> ()
    }) {is_stateless = false} : (tensor<i1>) -> ()

    %5 = "tf.E"() : () -> tensor<?xi32>
    tf_device.return %5 : tensor<?xi32>
  }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"} : () -> tensor<?xi32>

  func.return %2 : tensor<?xi32>
}

// -----

// Check decomposing 2 tf_device.cluster ops inside then/else branch of tf.If.

// CHECK-LABEL: func @main
// CHECK-SAME:  %[[ARG0:.*]]: tensor<i32>
func.func @main(%arg0: tensor<i32>) -> tensor<?xi32> {
  // CHECK:      "tf_device.cluster"
  // CHECK:        %[[PREDICATE_RECV_OUT:.*]] = "tf.DTensorRecv"
  // CHECK-SAME:   key = "SendRecvKeyForControlflow_0"
  // CHECK-NEXT:   "tf.IfRegion"(%[[PREDICATE_RECV_OUT]])
  // CHECK-NEXT:     "tf.D"
  // CHECK-NEXT:     "tf.Yield"
  // CHECK:          "tf.Yield"
  // CHECK:        tf_device.return
  // CHECK-NEXT: _mesh = "TPU|x=1|0|0|/job:localhost/task:0/device:TPU:0"
  // CHECK-SAME: () -> ()

  // CHECK-NEXT: "tf_device.cluster"
  // CHECK:        %[[PREDICATE_RECV_OUT_2:.*]] = "tf.DTensorRecv"
  // CHECK-SAME:   key = "SendRecvKeyForControlflow_1"
  // CHECK-NEXT:   "tf.IfRegion"(%[[PREDICATE_RECV_OUT_2]])
  // CHECK-NEXT:     "tf.Yield"
  // CHECK:          "tf.I"
  // CHECK-NEXT:     "tf.Yield"
  // CHECK:        tf_device.return
  // CHECK-NEXT: _mesh = "TPU|a=4|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"
  // CHECK-SAME: () -> ()

  // CHECK-NEXT: %[[CPU_CLUSTER_OUT:.*]] = "tf_device.cluster"
  // CHECK-NEXT:   "tf.A"()
  // CHECK-NEXT:   "tf.B"()
  // CHECK-NEXT:   %[[PREDICATE_OUT:.*]] = "tf.G"()
  // CHECK-NEXT:   "tf.F"()
  // CHECK-NEXT:   "tf.DTensorSend"(%[[PREDICATE_OUT]])
  // CHECK-SAME:   key = "SendRecvKeyForControlflow_0"
  // CHECK-NEXT:   "tf.DTensorSend"(%[[PREDICATE_OUT]])
  // CHECK-SAME:   key = "SendRecvKeyForControlflow_1"
  // CHECK-NEXT:   "tf.IfRegion"(%[[PREDICATE_OUT]])
  // CHECK-NEXT:     "tf.Yield"
  // CHECK:          "tf.Yield"
  // CHECK:        %[[E_OUT:.*]] = "tf.E"()
  // CHECK-NEXT:   tf_device.return %[[E_OUT]]
  // CHECK-NEXT: _mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"
  // CHECK-NEXT: return %[[CPU_CLUSTER_OUT]]
  %2 = "tf_device.cluster"() ({
    %3 = "tf.A"() : () -> (tensor<?xi32>)
    %4 = "tf.B"() : () -> (tensor<?xi32>)
    %6 = "tf.G"() : () -> (tensor<i1>)
    %7 = "tf.F"() : () -> tensor<?xi32>
    "tf.IfRegion"(%6) ({

      "tf_device.cluster"() ({
        "tf.D"() {} : () -> ()
        tf_device.return
      }) {_mesh = "TPU|x=1|0|0|/job:localhost/task:0/device:TPU:0"} : () -> ()

      "tf.Yield"() : () -> ()
    }, {
      "tf_device.cluster"() ({
        "tf.I"() {} : () -> ()
        tf_device.return
      }) {_mesh = "TPU|a=4|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> ()
      "tf.Yield"() : () -> ()
    }) {is_stateless = false} : (tensor<i1>) -> ()

    %5 = "tf.E"() : () -> tensor<?xi32>
    tf_device.return %5 : tensor<?xi32>
  }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"} : () -> tensor<?xi32>

  func.return %2 : tensor<?xi32>
}

// -----

// Check decomposing tf_device cluster inside tested tf.If op.

// CHECK-LABEL: func @main
// CHECK-SAME: %[[ARG0:.*]]: tensor<i32>
func.func @main(%arg0: tensor<i32>) -> tensor<?xi32> {
  // CHECK:       "tf_device.cluster"()
  // CHECK:         %[[OUTER_PREDICATE_RECV:.*]] = "tf.DTensorRecv"()
  // CHECK-SAME:    key = "SendRecvKeyForControlflow_1"
  // CHECK-NEXT:    "tf.IfRegion"(%[[OUTER_PREDICATE_RECV]])
  // CHECK-NEXT:      %[[INNER_PREDICATE_RECV:.*]] = "tf.DTensorRecv"()
  // CHECK-SAME:      key = "SendRecvKeyForControlflow_0"
  // CHECK-NEXT:      "tf.IfRegion"(%[[INNER_PREDICATE_RECV]])
  // CHECK-NEXT:        "tf.Yield"
  // CHECK:             "tf.I"
  // CHECK-NEXT:        "tf.D"
  // CHECK:             "tf.Yield"
  // CHECK:           "tf.Yield"
  // CHECK:           "tf.Yield"
  // CHECK:         tf_device.return
  // CHECK-NEXT:    _mesh = "TPU|x=1|0|0|/job:localhost/task:0/device:TPU:0"
  // CHECK-SAME:    () -> ()

  // CHECK:       "tf_device.cluster"
  // CHECK-NEXT:    "tf.A"
  // CHECK-NEXT:    "tf.B"
  // CHECK-NEXT:    %[[OUTER_PREDICATE:.*]] = "tf.G"
  // CHECK-NEXT:    "tf.DTensorSend"(%[[OUTER_PREDICATE]])
  // CHECK-NEXT:    "tf.IfRegion"(%[[OUTER_PREDICATE]])
  // CHECK-NEXT:      %[[INNER_PREDICATE:.*]] = "tf.H"
  // CHECK-NEXT:      "tf.DTensorSend"(%[[INNER_PREDICATE]])
  // CHECK-NEXT:      "tf.IfRegion"(%[[INNER_PREDICATE]])
  // CHECK-NEXT:        "tf.Yield"
  // CHECK:             "tf.Yield"
  // CHECK:           "tf.Yield"
  // CHECK:           "tf.Yield"
  // CHECK:         %[[E_OUT:.*]] = "tf.E"
  // CHECK-NEXT:    tf_device.return %[[E_OUT]]
  // CHECK-NEXT:  _mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"
  %2 = "tf_device.cluster"() ({
    %3 = "tf.A"() : () -> (tensor<?xi32>)
    %4 = "tf.B"() : () -> (tensor<?xi32>)
    %6 = "tf.G"() : () -> (tensor<i1>)

    "tf.IfRegion"(%6) ({
       %7 = "tf.H"(%4) : (tensor<?xi32>) -> (tensor<i1>)

      "tf.IfRegion"(%7)({
          "tf.Yield"() : () -> ()
        },
        {
          "tf_device.cluster"() ({
            %8 = "tf.I"() : () -> (tensor<?xi32>)
            "tf.D"(%8) : (tensor<?xi32>) -> ()
            tf_device.return
          }) {_mesh = "TPU|x=1|0|0|/job:localhost/task:0/device:TPU:0"} : () -> ()

          "tf.Yield"() : () -> ()
        }) {is_stateless = false} : (tensor<i1>) -> ()

      "tf.Yield"() : () -> ()
    }, {
      "tf.Yield"() : () -> ()
    }) { is_stateless = false} : (tensor<i1>) -> ()

    %5 = "tf.E"() : () -> tensor<?xi32>
    tf_device.return %5 : tensor<?xi32>
  }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"} : () -> tensor<?xi32>

  func.return %2 : tensor<?xi32>
}

// -----

// Check whether metadata attributes are cloned correctly during cluster
// merging.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>, %arg1: tensor<!tf_type.resource<tensor<2x4xf32>>>) -> () {
  // CHECK:      "tf_device.cluster"
  // CHECK:        "tf.B"
  // CHECK-NEXT:   tf_device.return
  // CHECK-NEXT: _mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"
  // CHECK-NEXT: "tf_device.cluster"
  // CHECK:         "tf.A"
  // CHECK-NEXT:   "tf.C"
  // CHECK-NEXT:   "tf.AssignVariableOp"
  // CHECK-NEXT:   tf_device.return
  // CHECK-NEXT: _inferred_resource_indices = dense<1>
  // CHECK-SAME: _inferred_resource_layouts = ["sharding_specs:unsharded,unsharded, mesh:CPU|x=1|0|0|CPU:0"]
  // CHECK-SAME: _mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"
  "tf_device.cluster"() ({
    "tf.A"() : () -> ()
    tf_device.return
  }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"} : () -> ()

  "tf_device.cluster"() ({
    "tf.B"() : () -> ()
    tf_device.return
  }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> ()

  "tf_device.cluster"() ({
    %0 = "tf.C"() : () -> (tensor<2x4xf32>)
    "tf.AssignVariableOp"(%arg1, %0) : (tensor<!tf_type.resource<tensor<2x4xf32>>>, tensor<2x4xf32>) -> ()
    tf_device.return
  }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0", _inferred_resource_indices = dense<1> : vector<1xi32>, _inferred_resource_layouts = ["sharding_specs:unsharded,unsharded, mesh:CPU|x=1|0|0|CPU:0"]} : () -> ()
  func.return
}

// -----

// Check whether metadata attributes are merged correctly.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>, %arg1: tensor<!tf_type.resource<tensor<2x4xf32>>>,  %arg2: tensor<!tf_type.resource<tensor<2x4xf32>>>) -> () {
  // CHECK:      "tf_device.cluster"
  // CHECK:        "tf.B"
  // CHECK-NEXT:   tf_device.return
  // CHECK-NEXT: _mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"
  // CHECK-NEXT: "tf_device.cluster"
  // CHECK:         "tf.A"
  // CHECK-NEXT:   "tf.AssignVariableOp"
  // CHECK-NEXT:   "tf.C"
  // CHECK-NEXT:   "tf.AssignVariableOp"
  // CHECK-NEXT:   tf_device.return
  // CHECK-NEXT: _inferred_resource_indices = dense<[2, 1]>
  // CHECK-SAME: _inferred_resource_layouts = ["sharding_specs:unsharded,unsharded, mesh:CPU|x=1|0|0|CPU:0", "sharding_specs:unsharded,unsharded, mesh:CPU|x=1|0|0|CPU:0"]
  // CHECK-SAME: _mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"
  "tf_device.cluster"() ({
    %0 = "tf.A"() : () -> (tensor<2x4xf32>)
    "tf.AssignVariableOp"(%arg2, %0) : (tensor<!tf_type.resource<tensor<2x4xf32>>>, tensor<2x4xf32>) -> ()
    tf_device.return
  }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0",  _inferred_resource_indices = dense<2> : vector<1xi32>, _inferred_resource_layouts = ["sharding_specs:unsharded,unsharded, mesh:CPU|x=1|0|0|CPU:0"]} : () -> ()

  "tf_device.cluster"() ({
    "tf.B"() : () -> ()
    tf_device.return
  }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> ()

  "tf_device.cluster"() ({
    %0 = "tf.C"() : () -> (tensor<2x4xf32>)
    "tf.AssignVariableOp"(%arg1, %0) : (tensor<!tf_type.resource<tensor<2x4xf32>>>, tensor<2x4xf32>) -> ()
    tf_device.return
  }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0", _inferred_resource_indices = dense<1> : vector<1xi32>, _inferred_resource_layouts = ["sharding_specs:unsharded,unsharded, mesh:CPU|x=1|0|0|CPU:0"]} : () -> ()
  func.return
}

// -----

// Check whether shape op metadata attributes are merged correctly.
// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>, %arg1: tensor<2x4xf32>, %arg2: tensor<2x4xf32>) -> () {
  // CHECK:      "tf_device.cluster"
  // CHECK:        "tf.B"
  // CHECK-NEXT:   tf_device.return
  // CHECK-NEXT: _mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"
  // CHECK-NEXT: "tf_device.cluster"
  // CHECK:        "tf.ShapeOp"
  // CHECK-NEXT:   "tf.ShapeOp"
  // CHECK-NEXT:   tf_device.return
  // CHECK-NEXT: _mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"
  // CHECK-SAME: _shape_input_indices = dense<[2, 1]
  // CHECK-SAME: _shape_input_layout = ["sharding_specs:unsharded,unsharded, mesh:CPU|x=1|0|0|CPU:0", "sharding_specs:unsharded,unsharded, mesh:CPU|x=1|0|0|CPU:0"]
  "tf_device.cluster"() ({
    %0 = "tf.ShapeOp"(%arg1) : (tensor<2x4xf32>) -> (tensor<1xf32>)
    tf_device.return
  }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0", _shape_input_indices = dense<[1, 2]> : vector<2xi32>, _shape_input_layout = ["sharding_specs:unsharded,unsharded, mesh:CPU|x=1|0|0|CPU:0", "sharding_specs:unsharded,unsharded, mesh:CPU|x=1|0|0|CPU:0"]} : () -> ()

  "tf_device.cluster"() ({
    "tf.B"() : () -> ()
    tf_device.return
  }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> ()

  "tf_device.cluster"() ({
    %0 = "tf.ShapeOp"(%arg2) : (tensor<2x4xf32>) -> (tensor<1xf32>)
    tf_device.return
  }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0", _shape_input_indices = dense<[1]> : vector<1xi32>, _shape_input_layout = ["sharding_specs:unsharded,unsharded, mesh:CPU|x=1|0|0|CPU:0"]} : () -> ()
  func.return
}

// -----

// Check whether conflicting metadata attributes disallowed.
func.func @main(%arg0: tensor<i32>, %arg1: tensor<!tf_type.resource<tensor<2x4xf32>>>,  %arg2: tensor<!tf_type.resource<tensor<2x4xf32>>>) -> () {
  "tf_device.cluster"() ({
    %0 = "tf.A"() : () -> (tensor<2x4xf32>)
    "tf.AssignVariableOp"(%arg2, %0) : (tensor<!tf_type.resource<tensor<2x4xf32>>>, tensor<2x4xf32>) -> ()
    tf_device.return
  }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0",  _inferred_resource_indices = dense<1> : vector<1xi32>, _inferred_resource_layouts = ["sharding_specs:x,unsharded, mesh:CPU|x=1|0|0|CPU:0"]} : () -> ()

  "tf_device.cluster"() ({
    "tf.B"() : () -> ()
    tf_device.return
  }) {_mesh = "TPU|x=2,y=2|0,1,2,3|0,1,2,3|/job:localhost/task:0/device:TPU:0,/job:localhost/task:0/device:TPU:1,/job:localhost/task:0/device:TPU:2,/job:localhost/task:0/device:TPU:3"} : () -> ()

  // expected-error @+1 {{Found conflicting metadata attributes while merging clusters}}
  "tf_device.cluster"() ({
    %0 = "tf.C"() : () -> (tensor<2x4xf32>)
    "tf.AssignVariableOp"(%arg1, %0) : (tensor<!tf_type.resource<tensor<2x4xf32>>>, tensor<2x4xf32>) -> ()
    tf_device.return
  }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0", _inferred_resource_indices = dense<1> : vector<1xi32>, _inferred_resource_layouts = ["sharding_specs:unsharded,unsharded, mesh:CPU|x=1|0|0|CPU:0"]} : () -> ()
  func.return
}

// -----

// Check that unused tf_device.cluster results are pruned away.

// CHECK-LABEL: func @main
// CHECK-SAME: %[[DEVICE_ID:.*]]: tensor<i32>
func.func @main(%arg0: tensor<i32>) {
  // CHECK:       "tf_device.cluster"()
  // CHECK-NEXT:    "tf.Const"
  // CHECK-NEXT:    tf_device.return
  // CHECK-NEXT:  _mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"
  // CHECK-SAME:  () -> ()
  %2 = "tf_device.cluster"() ({
    %0 = "tf.Const"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    tf_device.return %0 : tensor<i64>
  }) {_mesh = "CPU|x=1|0|0|/job:localhost/task:0/device:CPU:0"} : () -> tensor<i64>
 func.return
}
