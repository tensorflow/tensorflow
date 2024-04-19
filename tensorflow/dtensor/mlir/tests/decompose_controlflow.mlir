// RUN: dtensor-opt %s -split-input-file -dtensor-decompose-controlflow -dtensor-merge-clusters -verify-diagnostics | FileCheck %s

// -----

// Check tf.If control flow ops are decomposed correctly.

// CHECK-LABEL: module @test_if_decomposed
module @test_if_decomposed {
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
}

// -----

// Check decomposing 2 tf_device.cluster ops inside then/else branch of tf.If.

// CHECK-LABEL: module @test_if_then_else_branches
module @test_if_then_else_branches {
// CHECK: func @main
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
}

// -----


// Check decomposing tf_device cluster inside tested tf.If op.

// CHECK-LABEL: module @test_nested_cluster_inside_if
// CHECK: func @main
// CHECK-SAME: %[[ARG0:.*]]: tensor<i32>
module @test_nested_cluster_inside_if {
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

