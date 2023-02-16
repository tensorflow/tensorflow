// RUN: dtensor-opt %s -split-input-file -dtensor-annotate-global-shape -dtensor-layout-propagation-v2 -dtensor-spmd-expansion -verify-diagnostics | FileCheck %s

// Check SPMD of Softmax with no sharding.

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>,
           %arg1: tensor<6x4xf32> {tf._layout = "sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*CPU"},
           %arg2: tensor<6x4xf32> {tf._layout = "sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*CPU"}) -> tensor<6xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT: "tf.SoftmaxCrossEntropyWithLogits"
  // CHECK-NEXT: "tf.IdentityN"
  // CHECK-NEXT:        tf_device.return
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<6x4>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*CPU>} : (tensor<6x4xf32>) -> tensor<6x4xf32>
    %2 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<6x4>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*CPU>} : (tensor<6x4xf32>) -> tensor<6x4xf32>
    %loss, %backprop = "tf.SoftmaxCrossEntropyWithLogits"(%1, %2) : (tensor<6x4xf32>, tensor<6x4xf32>) -> (tensor<6xf32>, tensor<6x4xf32>)
    %3 = "tf.DTensorLayout"(%loss) {global_shape = #tf_type.shape<6>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|x=2,y=2|*CPU>} : (tensor<6xf32>) -> tensor<6xf32>
    %4 = "tf.DTensorLayout"(%backprop) {global_shape = #tf_type.shape<6x4>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*CPU>} : (tensor<6x4xf32>) -> tensor<6x4xf32>
    tf_device.return %3 : tensor<6xf32>
  }) {_mesh = "|x=2,y=2|*CPU", _layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|*CPU"]} : () -> (tensor<6xf32>)
  func.return %0 : tensor<6xf32>
}

// -----

// Check SPMD of Softmax with batch sharding but no class sharding.

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>,
           %arg1: tensor<6x4xf32> {tf._layout = "sharding_specs:x,unsharded, mesh:|x=2,y=2|*CPU"},
           %arg2: tensor<6x4xf32> {tf._layout = "sharding_specs:x,unsharded, mesh:|x=2,y=2|*CPU"}) -> tensor<6xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT: "tf.SoftmaxCrossEntropyWithLogits"
  // CHECK-NEXT: "tf.IdentityN"
  // CHECK-NEXT:        tf_device.return
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<6x4>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=2|*CPU>} : (tensor<6x4xf32>) -> tensor<6x4xf32>
    %2 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<6x4>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=2|*CPU>} : (tensor<6x4xf32>) -> tensor<6x4xf32>
    %loss, %backprop = "tf.SoftmaxCrossEntropyWithLogits"(%1, %2) : (tensor<6x4xf32>, tensor<6x4xf32>) -> (tensor<6xf32>, tensor<6x4xf32>)
    %3 = "tf.DTensorLayout"(%loss) {global_shape = #tf_type.shape<6>, layout = #dtensor.layout<sharding_specs:x, mesh:|x=2,y=2|*CPU>} : (tensor<6xf32>) -> tensor<6xf32>
    %4 = "tf.DTensorLayout"(%backprop) {global_shape = #tf_type.shape<6x4>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=2|*CPU>} : (tensor<6x4xf32>) -> tensor<6x4xf32>
    tf_device.return %3 : tensor<6xf32>
  }) {_mesh = "|x=2,y=2|*CPU", _layout = ["sharding_specs:x, mesh:|x=2,y=2|*CPU"]} : () -> (tensor<6xf32>)
  func.return %0 : tensor<6xf32>
}

// -----

// Check SPMD of Softmax with batch sharding and class sharding,

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>,
           %arg1: tensor<6x4xf32> {tf._layout = "sharding_specs:x,y, mesh:|x=2,y=2|*CPU"},
           %arg2: tensor<6x4xf32> {tf._layout = "sharding_specs:x,y, mesh:|x=2,y=2|*CPU"}) -> tensor<6xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK:      %[[LOCAL_MAX:.*]] = "tf.Max"(%arg1,
  // CHECK:      %[[MAX:.*]] = "tf.DTensorAllReduce"(%[[LOCAL_MAX]],
  // CHECK-SAME: "Max"
  // CHECK:      %[[SHIFTED_LOGITS:.*]] = "tf.Sub"(%arg1, %[[MAX]])
  // CHECK:      %[[EXP_LOGITS:.*]] = "tf.Exp"(%[[SHIFTED_LOGITS]])
  // CHECK:      %[[LOCAL_SUM:.*]] = "tf.Sum"(%[[EXP_LOGITS]],
  // CHECK:      %[[SUM:.*]] = "tf.DTensorAllReduce"(%[[LOCAL_SUM]],
  // CHECK-SAME: "Add"
  // CHECK:      %[[LOG_SUM:.*]] = "tf.Log"(%[[SUM]])
  // CHECK:      %[[LOG_SOFTMAX:.*]] = "tf.Sub"(%[[SHIFTED_LOGITS]], %[[LOG_SUM]])
  // CHECK:      %[[SOFTMAX:.*]] = "tf.Div"(%[[EXP_LOGITS]], %[[SUM]])
  // CHECK:      %[[IS_ZERO:.*]] = "tf.Equal"(%arg2,
  // CHECK:      %[[SAFE_LOG_SOFTMAX:.*]] = "tf.SelectV2"(%[[IS_ZERO]], %[[ZERO:.*]], %[[LOG_SOFTMAX]])
  // CHECK:      %[[PROD:.*]] = "tf.Mul"(%arg2, %[[SAFE_LOG_SOFTMAX]])
  // CHECK:      %[[LOCAL_NEG_LOSS:.*]] = "tf.Sum"(%[[PROD]],
  // CHECK:      %[[NEG_LOSS:.*]] = "tf.DTensorAllReduce"(%[[LOCAL_NEG_LOSS]],
  // CHECK-SAME: "Add"
  // CHECK:      %[[SQUEEZED_NEG_LOSS:.*]] = "tf.Squeeze"(%[[NEG_LOSS]])
  // CHECK:      %[[LOSS:.*]] = "tf.Neg"(%[[SQUEEZED_NEG_LOSS]])
  // CHECK:      %[[BACKPROP:.*]] = "tf.Sub"(%[[SOFTMAX]], %arg2)
  // CHECK-NEXT: "tf.IdentityN"(%[[LOSS]], %[[BACKPROP]])
  // CHECK-NEXT: tf_device.return
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<6x4>, layout = #dtensor.layout<sharding_specs:x,y, mesh:|x=2,y=2|*CPU>} : (tensor<6x4xf32>) -> tensor<6x4xf32>
    %2 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<6x4>, layout = #dtensor.layout<sharding_specs:x,y, mesh:|x=2,y=2|*CPU>} : (tensor<6x4xf32>) -> tensor<6x4xf32>
    %loss, %backprop = "tf.SoftmaxCrossEntropyWithLogits"(%1, %2) : (tensor<6x4xf32>, tensor<6x4xf32>) -> (tensor<6xf32>, tensor<6x4xf32>)
    %3 = "tf.DTensorLayout"(%loss) {global_shape = #tf_type.shape<6>, layout = #dtensor.layout<sharding_specs:x, mesh:|x=2,y=2|*CPU>} : (tensor<6xf32>) -> tensor<6xf32>
    %4 = "tf.DTensorLayout"(%backprop) {global_shape = #tf_type.shape<6x4>, layout = #dtensor.layout<sharding_specs:x,y, mesh:|x=2,y=2|*CPU>} : (tensor<6x4xf32>) -> tensor<6x4xf32>
    tf_device.return %3 : tensor<6xf32>
  }) {_mesh = "|x=2,y=2|*CPU", _layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|*CPU"]} : () -> (tensor<6xf32>)
  func.return %0 : tensor<6xf32>
}

// -----

// Check SPMD of SparseSoftmax with no sharding.

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>,
           %arg1: tensor<6x4xf32> {tf._layout = "sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*CPU"},
           %arg2: tensor<6xi32> {tf._layout = "sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*CPU"}) -> tensor<6xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT: "tf.SparseSoftmaxCrossEntropyWithLogits"
  // CHECK-NEXT: "tf.IdentityN"
  // CHECK-NEXT: tf_device.return
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<6x4>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*CPU>} : (tensor<6x4xf32>) -> tensor<6x4xf32>
    %2 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<6>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|x=2,y=2|*CPU>} : (tensor<6xi32>) -> tensor<6xi32>
    %loss, %backprop = "tf.SparseSoftmaxCrossEntropyWithLogits"(%1, %2) : (tensor<6x4xf32>, tensor<6xi32>) -> (tensor<6xf32>, tensor<6x4xf32>)
    %3 = "tf.DTensorLayout"(%loss) {global_shape = #tf_type.shape<6>, layout = #dtensor.layout<sharding_specs:unsharded, mesh:|x=2,y=2|*CPU>} : (tensor<6xf32>) -> tensor<6xf32>
    %4 = "tf.DTensorLayout"(%backprop) {global_shape = #tf_type.shape<6x4>, layout = #dtensor.layout<sharding_specs:unsharded,unsharded, mesh:|x=2,y=2|*CPU>} : (tensor<6x4xf32>) -> tensor<6x4xf32>
    tf_device.return %3 : tensor<6xf32>
  }) {_mesh = "|x=2,y=2|*CPU", _layout = ["sharding_specs:unsharded, mesh:|x=2,y=2|*CPU"]} : () -> (tensor<6xf32>)
  func.return %0 : tensor<6xf32>
}

// -----

// Check SPMD of Softmax with batch sharding but no class sharding.

// CHECK-LABEL: func @main
func.func @main(%arg0: tensor<i32>,
           %arg1: tensor<6x4xf32> {tf._layout = "sharding_specs:x,unsharded, mesh:|x=2,y=2|*CPU"},
           %arg2: tensor<6xi32> {tf._layout = "sharding_specs:x, mesh:|x=2,y=2|*CPU"}) -> tensor<6xf32> {
  // CHECK:      "tf_device.cluster"
  // CHECK-NEXT: "tf.SparseSoftmaxCrossEntropyWithLogits"
  // CHECK-NEXT: "tf.IdentityN"
  // CHECK-NEXT:        tf_device.return
  %0 = "tf_device.cluster"() ({
    %1 = "tf.DTensorLayout"(%arg1) {global_shape = #tf_type.shape<6x4>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=2|*CPU>} : (tensor<6x4xf32>) -> tensor<6x4xf32>
    %2 = "tf.DTensorLayout"(%arg2) {global_shape = #tf_type.shape<6>, layout = #dtensor.layout<sharding_specs:x, mesh:|x=2,y=2|*CPU>} : (tensor<6xi32>) -> tensor<6xi32>
    %loss, %backprop = "tf.SparseSoftmaxCrossEntropyWithLogits"(%1, %2) : (tensor<6x4xf32>, tensor<6xi32>) -> (tensor<6xf32>, tensor<6x4xf32>)
    %3 = "tf.DTensorLayout"(%loss) {global_shape = #tf_type.shape<6>, layout = #dtensor.layout<sharding_specs:x, mesh:|x=2,y=2|*CPU>} : (tensor<6xf32>) -> tensor<6xf32>
    %4 = "tf.DTensorLayout"(%backprop) {global_shape = #tf_type.shape<6x4>, layout = #dtensor.layout<sharding_specs:x,unsharded, mesh:|x=2,y=2|*CPU>} : (tensor<6x4xf32>) -> tensor<6x4xf32>
    tf_device.return %3 : tensor<6xf32>
  }) {_mesh = "|x=2,y=2|*CPU", _layout = ["sharding_specs:x, mesh:|x=2,y=2|*CPU"]} : () -> (tensor<6xf32>)
  func.return %0 : tensor<6xf32>
}

