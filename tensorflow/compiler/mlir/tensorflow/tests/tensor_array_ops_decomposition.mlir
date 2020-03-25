// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tensor-array-ops-decomposition | FileCheck %s -dump-input-on-failure

// Test read and write on a tensor list.

// CHECK-LABEL: func @main
func @main() -> tensor<3xf32> {
  %size = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[BUFFER:.*]] = "tf.BroadcastTo"(%2, %3)
  // CHECK-SAME: -> tensor<5x3xf32>
  // CHECK: %[[VAR:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<5x3xf32>>>
  // CHECK: "tf.AssignVariableOp"(%[[VAR]], %[[BUFFER]])
  %ta:2 = "tf.TensorArrayV3"(%size) {dtype = f32, element_shape = "tfshape$dim { size: 3 }", dynamic_size = false, clear_after_read = true, identical_element_shapes = true, tensor_array_name = "ta"} : (tensor<i32>) -> (tensor<!tf.resource>, tensor<f32>)
  // CHECK: %[[IND:.*]] = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %index = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[VAL:.*]] = "tf.Const"() {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>} : () -> tensor<3xf32>
  %value = "tf.Const"() {value = dense<[1.0, 2.0, 3.0]> : tensor<3xf32>} : () -> tensor<3xf32>
  // CHECK: %[[READ_VAR:.*]] = "tf.ReadVariableOp"(%[[VAR]])
  // CHECK: %[[UPDATE_SLICE:.*]] = "tf.Reshape"(%[[VAL]], %12)
  // CHECK-SAME: -> tensor<1x3xf32>
  // CHECK: %[[NEW_BUFFER:.*]] = "tf.XlaDynamicUpdateSlice"(%[[READ_VAR]], %[[UPDATE_SLICE]],
  // CHECK: "tf.AssignVariableOp"(%[[VAR]], %[[NEW_BUFFER]])
  %write = "tf.TensorArrayWriteV3"(%ta#0, %index, %value, %ta#1) : (tensor<!tf.resource>, tensor<i32>, tensor<3xf32>, tensor<f32>) -> tensor<f32>
  // CHECK: %[[READ_VAR2:.*]] = "tf.ReadVariableOp"(%[[VAR]])
  // CHECK: %[[READ_SLICE:.*]] = "tf.Slice"(%[[READ_VAR2]],
  // CHECK: %[[READ:.*]] = "tf.Reshape"(%[[READ_SLICE]],
  // CHECK-SAME: -> tensor<3xf32>
  %read = "tf.TensorArrayReadV3"(%ta#0, %index, %write) : (tensor<!tf.resource>, tensor<i32>, tensor<f32>) -> tensor<3xf32>
  // CHECK-NOT: TensorArrayCloseV3
  "tf.TensorArrayCloseV3"(%ta#0) : (tensor<!tf.resource>) -> ()
  // CHECK: return %[[READ]] : tensor<3xf32>
  return %read: tensor<3xf32>
}

// -----

// Test inferring shape from the first write.

// CHECK-LABEL: func @main
func @main() -> tensor<i32> {
  %size = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[BUFFER:.*]] = "tf.BroadcastTo"(%2, %3)
  // CHECK-SAME: -> tensor<5x3xf32>
  // CHECK: %[[VAR:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<5x3xf32>>>
  // CHECK: "tf.AssignVariableOp"(%[[VAR]], %[[BUFFER]])
  %ta:2 = "tf.TensorArrayV3"(%size) {dtype = f32, element_shape = "tfshape$unknown_rank: true", dynamic_size = false, clear_after_read = true, identical_element_shapes = true, tensor_array_name = "ta"} : (tensor<i32>) -> (tensor<!tf.resource>, tensor<f32>)
  %index = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %value = "tf.Const"() {value = dense<[1.0, 2.0, 3.0]> : tensor<3xf32>} : () -> tensor<3xf32>
  %write = "tf.TensorArrayWriteV3"(%ta#0, %index, %value, %ta#1) : (tensor<!tf.resource>, tensor<i32>, tensor<3xf32>, tensor<f32>) -> tensor<f32>
  // CHECK: %[[SIZE:.*]] = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  %size_out = "tf.TensorArraySizeV3"(%ta#0, %write) : (tensor<!tf.resource>, tensor<f32>) -> tensor<i32>
  // CHECK: return %[[SIZE]] : tensor<i32>
  return %size_out : tensor<i32>
}

// -----

// Test tensor array concat and split.

// CHECK-LABEL: func @main
func @main() -> () {
  %size = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[VAR:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<5x3xf32>>>
  // CHECK: "tf.AssignVariableOp"
  %ta:2 = "tf.TensorArrayV3"(%size) {dtype = f32, element_shape = "tfshape$dim { size: 3 }", dynamic_size = false, clear_after_read = true, identical_element_shapes = true, tensor_array_name = "ta"} : (tensor<i32>) -> (tensor<!tf.resource>, tensor<f32>)
  // CHECK: %[[READ:.*]] = "tf.ReadVariableOp"(%[[VAR]]) : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
  // CHECK: %[[CONCAT_RESHAPE:.*]] = "tf.Reshape"(%[[READ]],
  // CHECK-SAME: -> tensor<15xf32>
  // CHECK: %[[LENS:.*]] = "tf.Const"() {value = dense<3> : tensor<5xi64>} : () -> tensor<5xi64>
  %concat:2 = "tf.TensorArrayConcatV3"(%ta#0, %ta#1) {element_shape_except0 = "tfshape$unknown_rank: true"} : (tensor<!tf.resource>, tensor<f32>) -> (tensor<*xf32>, tensor<*xi64>)
  // CHECK: %[[SPLIT_RESHAPE:.*]] = "tf.Reshape"(%[[CONCAT_RESHAPE]],
  // CHECK-SAME: -> tensor<5x3xf32>
  // CHECK: %[[READ2:.*]] = "tf.ReadVariableOp"(%[[VAR]]) : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
  // CHECK: %[[ADD:.*]] = "tf.AddV2"(%[[READ2]], %[[SPLIT_RESHAPE]])
  // CHECK: "tf.AssignVariableOp"(%[[VAR]], %[[ADD]])
  %split = "tf.TensorArraySplitV3"(%ta#0, %concat#0, %concat#1, %ta#1) {element_shape_except0 = "tfshape$unknown_rank: true"} : (tensor<!tf.resource>, tensor<*xf32>, tensor<*xi64>, tensor<f32>) -> tensor<f32>
  return
}

// -----

// Test tensor array gather and scatter on contiguous indices.

// CHECK-LABEL: func @main
func @main() -> () {
  %size = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[VAR:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<5x3xf32>>>
  // CHECK: "tf.AssignVariableOp"
  %ta:2 = "tf.TensorArrayV3"(%size) {dtype = f32, element_shape = "tfshape$dim { size: 3 }", dynamic_size = false, clear_after_read = true, identical_element_shapes = true, tensor_array_name = "ta"} : (tensor<i32>) -> (tensor<!tf.resource>, tensor<f32>)
  %indices = "tf.Const"() {value = dense<[0, 1, 2, 3, 4]> : tensor<5xi32>} : () -> tensor<5xi32>
  // CHECK: %[[READ:.*]] = "tf.ReadVariableOp"(%[[VAR]]) : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
  // CHECK: %[[GATHER_SLICE:.*]] = "tf.Slice"(%[[READ]]
  // CHECK-SAME: (tensor<5x3xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<5x3xf32>
  %gather = "tf.TensorArrayGatherV3"(%ta#0, %indices, %ta#1) {element_shape = "tfshape$unknown_rank: true"} : (tensor<!tf.resource>, tensor<5xi32>, tensor<f32>) -> tensor<*xf32>
  // CHECK: %[[READ2:.*]] = "tf.ReadVariableOp"(%[[VAR]]) : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
  // CHECK: %[[ADD:.*]] = "tf.AddV2"(%[[READ2]], %[[GATHER_SLICE]])
  // CHECK: "tf.AssignVariableOp"(%[[VAR]], %[[ADD]])
  %scatter = "tf.TensorArrayScatterV3"(%ta#0, %indices, %gather, %ta#1) {element_shape_except0 = "tfshape$unknown_rank: true"} : (tensor<!tf.resource>, tensor<5xi32>, tensor<*xf32>, tensor<f32>) -> tensor<f32>
  return
}

// -----

// Test tensor array gather and scatter on non-contiguous indices.

// CHECK-LABEL: func @main
func @main() -> () {
  %size = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[VAR:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<5x3xf32>>>
  // CHECK: "tf.AssignVariableOp"
  %ta:2 = "tf.TensorArrayV3"(%size) {dtype = f32, element_shape = "tfshape$dim { size: 3 }", dynamic_size = false, clear_after_read = true, identical_element_shapes = true, tensor_array_name = "ta"} : (tensor<i32>) -> (tensor<!tf.resource>, tensor<f32>)
  // CHECK: %[[INDS:.*]] = "tf.Const"() {value = dense<[2, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  %indices = "tf.Const"() {value = dense<[2, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK: %[[READ:.*]] = "tf.ReadVariableOp"(%[[VAR]]) : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
  // CHECK: %[[AXIS:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[GATHER:.*]] = "tf.GatherV2"(%[[READ]], %[[INDS]], %[[AXIS]]) : (tensor<5x3xf32>, tensor<2xi32>, tensor<i32>) -> tensor<2x3xf32>
  %gather = "tf.TensorArrayGatherV3"(%ta#0, %indices, %ta#1) {element_shape = "tfshape$unknown_rank: true"} : (tensor<!tf.resource>, tensor<2xi32>, tensor<f32>) -> tensor<*xf32>
  // CHECK: %[[READ2:.*]] = "tf.ReadVariableOp"(%[[VAR]]) : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
  // CHECK: %[[SLICE_SIZE:.*]] = "tf.Const"() {value = dense<[1, 3]> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK: %[[IND_SLICE0_START:.*]] = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: %[[IND_SLICE0_SIZE:.*]] = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: %[[IND_SLICE0:.*]] = "tf.Slice"(%[[INDS]], %[[IND_SLICE0_START]], %[[IND_SLICE0_SIZE]]) : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  // CHECK: %[[SLICE0_START:.*]] = "tf.ConcatV2"(%[[IND_SLICE0]],
  // CHECK: %[[OLD_SLICE0:.*]] = "tf.Slice"(%[[READ2]], %[[SLICE0_START]],
  // CHECK-SAME: (tensor<5x3xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x3xf32>
  // CHECK: %[[UPDATE_SLICE0_START:.*]] = "tf.Const"() {value = dense<0> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK: %[[UPDATE_SLICE0:.*]] = "tf.Slice"(%[[GATHER]], %[[UPDATE_SLICE0_START]], %[[SLICE_SIZE]]) : (tensor<2x3xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x3xf32>
  // CHECK: %[[ADD0:.*]] = "tf.AddV2"(%[[OLD_SLICE0]], %[[UPDATE_SLICE0]])
  // CHECK: %[[UPDATE0:.*]] = "tf.XlaDynamicUpdateSlice"(%[[READ2]], %[[ADD0]]
  // CHECK-SAME: (tensor<5x3xf32>, tensor<1x3xf32>, tensor<2xi32>) -> tensor<5x3xf32>

  // CHECK: %[[IND_SLICE1_START:.*]] = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: %[[IND_SLICE1_SIZE:.*]] = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK: %[[IND_SLICE1:.*]] = "tf.Slice"(%[[INDS]], %[[IND_SLICE1_START]], %[[IND_SLICE1_SIZE]]) : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  // CHECK: %[[SLICE1_START:.*]] = "tf.ConcatV2"(%[[IND_SLICE1]],
  // CHECK: %[[OLD_SLICE1:.*]] = "tf.Slice"(%[[UPDATE0]], %[[SLICE1_START]],
  // CHECK-SAME: (tensor<5x3xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x3xf32>
  // CHECK: %[[UPDATE_SLICE1_START:.*]] = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK: %[[UPDATE_SLICE1:.*]] = "tf.Slice"(%[[GATHER]], %[[UPDATE_SLICE1_START]], %[[SLICE_SIZE]]) : (tensor<2x3xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x3xf32>
  // CHECK: %[[ADD1:.*]] = "tf.AddV2"(%[[OLD_SLICE1]], %[[UPDATE_SLICE1]])
  // CHECK: %[[UPDATE1:.*]] = "tf.XlaDynamicUpdateSlice"(%[[UPDATE0]], %[[ADD1]]
  // CHECK-SAME: (tensor<5x3xf32>, tensor<1x3xf32>, tensor<2xi32>) -> tensor<5x3xf32>
  // CHECK: "tf.AssignVariableOp"(%[[VAR]], %[[UPDATE1]])
  %scatter = "tf.TensorArrayScatterV3"(%ta#0, %indices, %gather, %ta#1) {element_shape_except0 = "tfshape$unknown_rank: true"} : (tensor<!tf.resource>, tensor<2xi32>, tensor<*xf32>, tensor<f32>) -> tensor<f32>
  return
}

// -----

// Test tensor list grads.

// CHECK-LABEL: func @main
func @main() {
  %size = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[VAR:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<5x3xf32>>>
  // CHECK: "tf.AssignVariableOp"(%[[VAR]],
  %ta:2 = "tf.TensorArrayV3"(%size) {dtype = f32, element_shape = "tfshape$dim { size: 3 }", dynamic_size = false, clear_after_read = true, identical_element_shapes = true, tensor_array_name = "ta"} : (tensor<i32>) -> (tensor<!tf.resource>, tensor<f32>)
  %index = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[VALUE:.*]] = "tf.Const"() {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>} : () -> tensor<3xf32>
  %value = "tf.Const"() {value = dense<[1.0, 2.0, 3.0]> : tensor<3xf32>} : () -> tensor<3xf32>
  // CHECK: %[[GVAR1:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<5x3xf32>>>
  // CHECK: "tf.AssignVariableOp"(%[[GVAR1]],
  %grad1:2 = "tf.TensorArrayGradV3"(%ta#0, %ta#1) {source = "a"} : (tensor<!tf.resource>, tensor<f32>) -> (tensor<!tf.resource>, tensor<f32>)
  %grad2:2 = "tf.TensorArrayGradV3"(%ta#0, %ta#1) {source = "a"} : (tensor<!tf.resource>, tensor<f32>) -> (tensor<!tf.resource>, tensor<f32>)
  // CHECK: %[[GVAR3:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<5x3xf32>>>
  // CHECK: "tf.AssignVariableOp"(%[[GVAR3]],
  %grad3:2 = "tf.TensorArrayGradV3"(%ta#0, %ta#1) {source = "b"} : (tensor<!tf.resource>, tensor<f32>) -> (tensor<!tf.resource>, tensor<f32>)
  // CHECK: %[[READ1:.*]] = "tf.ReadVariableOp"(%[[GVAR1]]) : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
  // CHECK: %[[OLD_SLICE1:.*]] = "tf.Slice"(%[[READ1]],
  // CHECK: %[[RESHAPE1:.*]] = "tf.Reshape"(%[[VALUE]],
  // CHECK: %[[ADD1:.*]] = "tf.AddV2"(%[[RESHAPE1]], %[[OLD_SLICE1]]) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  // CHECK: %[[UPDATE1:.*]] = "tf.XlaDynamicUpdateSlice"(%[[READ1]], %[[ADD1]],
  // CHECK: "tf.AssignVariableOp"(%[[GVAR1]], %[[UPDATE1]])
  %write1 = "tf.TensorArrayWriteV3"(%grad1#0, %index, %value, %grad1#1) : (tensor<!tf.resource>, tensor<i32>, tensor<3xf32>, tensor<f32>) -> tensor<f32>
  // CHECK: %[[READ2:.*]] = "tf.ReadVariableOp"(%[[GVAR1]]) : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
  // CHECK: %[[OLD_SLICE2:.*]] = "tf.Slice"(%[[READ2]],
  // CHECK: %[[RESHAPE2:.*]] = "tf.Reshape"(%[[VALUE]],
  // CHECK: %[[ADD2:.*]] = "tf.AddV2"(%[[RESHAPE2]], %[[OLD_SLICE2]]) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  // CHECK: %[[UPDATE2:.*]] = "tf.XlaDynamicUpdateSlice"(%[[READ2]], %[[ADD2]],
  // CHECK: "tf.AssignVariableOp"(%[[GVAR1]], %[[UPDATE2]])
  %write2 = "tf.TensorArrayWriteV3"(%grad2#0, %index, %value, %grad2#1) : (tensor<!tf.resource>, tensor<i32>, tensor<3xf32>, tensor<f32>) -> tensor<f32>
  // CHECK: %[[READ3:.*]] = "tf.ReadVariableOp"(%[[GVAR3]]) : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
  // CHECK: %[[OLD_SLICE3:.*]] = "tf.Slice"(%[[READ3]],
  // CHECK: %[[RESHAPE3:.*]] = "tf.Reshape"(%[[VALUE]],
  // CHECK: %[[ADD3:.*]] = "tf.AddV2"(%[[RESHAPE3]], %[[OLD_SLICE3]]) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  // CHECK: %[[UPDATE3:.*]] = "tf.XlaDynamicUpdateSlice"(%[[READ3]], %[[ADD3]],
  // CHECK: "tf.AssignVariableOp"(%[[GVAR3]], %[[UPDATE3]])
  %write3 = "tf.TensorArrayWriteV3"(%grad3#0, %index, %value, %grad3#1) : (tensor<!tf.resource>, tensor<i32>, tensor<3xf32>, tensor<f32>) -> tensor<f32>
  return
}
