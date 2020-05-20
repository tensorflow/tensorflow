// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-tensor-array-ops-decomposition | FileCheck %s -dump-input-on-failure

// Test read and write on a tensor list.

// CHECK-LABEL: func @main
func @main() -> tensor<3xf32> {
  %size = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[BUFFER:.*]] = "tf.BroadcastTo"(%2, %3)
  // CHECK-SAME: -> tensor<5x3xf32>
  // CHECK: %[[VAR:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<5x3xf32>>>
  // CHECK: "tf.AssignVariableOp"(%[[VAR]], %[[BUFFER]])
  %ta:2 = "tf.TensorArrayV3"(%size) {dtype = f32, element_shape = #tf.shape<3>, dynamic_size = false, clear_after_read = true, identical_element_shapes = true, tensor_array_name = "ta"} : (tensor<i32>) -> (tensor<!tf.resource>, tensor<f32>)
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
  %ta:2 = "tf.TensorArrayV3"(%size) {dtype = f32, element_shape = #tf.shape<*>, dynamic_size = false, clear_after_read = true, identical_element_shapes = true, tensor_array_name = "ta"} : (tensor<i32>) -> (tensor<!tf.resource>, tensor<f32>)
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
  %ta:2 = "tf.TensorArrayV3"(%size) {dtype = f32, element_shape = #tf.shape<3>, dynamic_size = false, clear_after_read = true, identical_element_shapes = true, tensor_array_name = "ta"} : (tensor<i32>) -> (tensor<!tf.resource>, tensor<f32>)
  // CHECK: %[[READ:.*]] = "tf.ReadVariableOp"(%[[VAR]]) : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
  // CHECK: %[[CONCAT_RESHAPE:.*]] = "tf.Reshape"(%[[READ]],
  // CHECK-SAME: -> tensor<15xf32>
  // CHECK: %[[LENS:.*]] = "tf.Const"() {value = dense<3> : tensor<5xi64>} : () -> tensor<5xi64>
  %concat:2 = "tf.TensorArrayConcatV3"(%ta#0, %ta#1) {element_shape_except0 = #tf.shape<*>} : (tensor<!tf.resource>, tensor<f32>) -> (tensor<*xf32>, tensor<*xi64>)
  // CHECK: %[[SPLIT_RESHAPE:.*]] = "tf.Reshape"(%[[CONCAT_RESHAPE]],
  // CHECK-SAME: -> tensor<5x3xf32>
  // CHECK: %[[READ2:.*]] = "tf.ReadVariableOp"(%[[VAR]]) : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
  // CHECK: %[[ADD:.*]] = "tf.AddV2"(%[[READ2]], %[[SPLIT_RESHAPE]])
  // CHECK: "tf.AssignVariableOp"(%[[VAR]], %[[ADD]])
  %split = "tf.TensorArraySplitV3"(%ta#0, %concat#0, %concat#1, %ta#1) {element_shape_except0 = #tf.shape<*>} : (tensor<!tf.resource>, tensor<*xf32>, tensor<*xi64>, tensor<f32>) -> tensor<f32>
  return
}

// -----

// Test tensor array gather and scatter on contiguous indices.

// CHECK-LABEL: func @main
func @main() -> () {
  %size = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[VAR:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<5x3xf32>>>
  // CHECK: "tf.AssignVariableOp"
  %ta:2 = "tf.TensorArrayV3"(%size) {dtype = f32, element_shape = #tf.shape<3>, dynamic_size = false, clear_after_read = true, identical_element_shapes = true, tensor_array_name = "ta"} : (tensor<i32>) -> (tensor<!tf.resource>, tensor<f32>)
  %indices = "tf.Const"() {value = dense<[0, 1, 2, 3, 4]> : tensor<5xi32>} : () -> tensor<5xi32>
  // CHECK: %[[READ:.*]] = "tf.ReadVariableOp"(%[[VAR]]) : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
  // CHECK: %[[GATHER_SLICE:.*]] = "tf.Slice"(%[[READ]]
  // CHECK-SAME: (tensor<5x3xf32>, tensor<2xi32>, tensor<2xi32>) -> tensor<5x3xf32>
  %gather = "tf.TensorArrayGatherV3"(%ta#0, %indices, %ta#1) {element_shape = #tf.shape<*>} : (tensor<!tf.resource>, tensor<5xi32>, tensor<f32>) -> tensor<*xf32>
  // CHECK: %[[READ2:.*]] = "tf.ReadVariableOp"(%[[VAR]]) : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
  // CHECK: %[[ADD:.*]] = "tf.AddV2"(%[[READ2]], %[[GATHER_SLICE]])
  // CHECK: "tf.AssignVariableOp"(%[[VAR]], %[[ADD]])
  %scatter = "tf.TensorArrayScatterV3"(%ta#0, %indices, %gather, %ta#1) {element_shape_except0 = #tf.shape<*>} : (tensor<!tf.resource>, tensor<5xi32>, tensor<*xf32>, tensor<f32>) -> tensor<f32>
  return
}

// -----

// Test tensor array gather and scatter on non-contiguous indices.

// CHECK-LABEL: func @main
func @main() -> () {
  %size = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[VAR:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<5x3xf32>>>
  // CHECK: "tf.AssignVariableOp"
  %ta:2 = "tf.TensorArrayV3"(%size) {dtype = f32, element_shape = #tf.shape<3>, dynamic_size = false, clear_after_read = true, identical_element_shapes = true, tensor_array_name = "ta"} : (tensor<i32>) -> (tensor<!tf.resource>, tensor<f32>)
  // CHECK: %[[INDS:.*]] = "tf.Const"() {value = dense<[2, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  %indices = "tf.Const"() {value = dense<[2, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK: %[[READ:.*]] = "tf.ReadVariableOp"(%[[VAR]]) : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
  // CHECK: %[[AXIS:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[GATHER:.*]] = "tf.GatherV2"(%[[READ]], %[[INDS]], %[[AXIS]]) : (tensor<5x3xf32>, tensor<2xi32>, tensor<i32>) -> tensor<2x3xf32>
  %gather = "tf.TensorArrayGatherV3"(%ta#0, %indices, %ta#1) {element_shape = #tf.shape<*>} : (tensor<!tf.resource>, tensor<2xi32>, tensor<f32>) -> tensor<*xf32>
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
  %scatter = "tf.TensorArrayScatterV3"(%ta#0, %indices, %gather, %ta#1) {element_shape_except0 = #tf.shape<*>} : (tensor<!tf.resource>, tensor<2xi32>, tensor<*xf32>, tensor<f32>) -> tensor<f32>
  return
}

// -----

// Test tensor list grads.

// CHECK-LABEL: func @main
func @main() {
  %size = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[VAR:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<5x3xf32>>>
  // CHECK: "tf.AssignVariableOp"(%[[VAR]],
  %ta:2 = "tf.TensorArrayV3"(%size) {dtype = f32, element_shape = #tf.shape<3>, dynamic_size = false, clear_after_read = true, identical_element_shapes = true, tensor_array_name = "ta"} : (tensor<i32>) -> (tensor<!tf.resource>, tensor<f32>)
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

// -----

// Tests while loop with access to the tensor array defined outside and its
// gradient defined inside. The gradient creation should be moved outside.

// CHECK-LABEL: func @main
func @main() -> () {
  // CHECK: %[[SIZE:.*]] = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  %size = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  %index = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[VAR:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<5x3xf32>>>
  // CHECK: %[[GVAR:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<5x3xf32>>>
  %ta:2 = "tf.TensorArrayV3"(%size) {dtype = f32, element_shape = #tf.shape<3>, dynamic_size = false, clear_after_read = true, identical_element_shapes = true, tensor_array_name = "ta"} : (tensor<i32>) -> (tensor<!tf.resource>, tensor<f32>)
  // CHECK: "tf.While"(%[[VAR]], %[[SIZE]], %[[GVAR]])
  %1:2 = "tf.While"(%ta#0, %size) {
    body = @while_body, cond = @while_cond, device = "", is_stateless = false}
       : (tensor<!tf.resource>, tensor<i32>) -> (tensor<!tf.resource>, tensor<i32>)
  // CHECK: %[[READ:.*]] = "tf.ReadVariableOp"(%[[VAR]]) : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
  // CHECK: "tf.Slice"(%[[READ]],
  %read = "tf.TensorArrayReadV3"(%1#0, %index, %ta#1) : (tensor<!tf.resource>, tensor<i32>, tensor<f32>) -> tensor<3xf32>
  return
}
// CHECK: func @while_body(%[[BARG0:.*]]: tensor<!tf.resource<tensor<5x3xf32>>>, %[[BARG1:.*]]: tensor<i32>, %[[BARG2:.*]]: tensor<!tf.resource<tensor<5x3xf32>>>)
func @while_body(%arg0: tensor<!tf.resource>, %arg1: tensor<i32>) -> (tensor<!tf.resource>, tensor<i32>) {
  // CHECK: %[[CONST1:.*]] = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %const1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[SUB:.*]] = "tf.Sub"(%[[BARG1]], %[[CONST1]])
  %sub = "tf.Sub"(%arg1, %const1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %elem = "tf._SomeOp"() : () -> tensor<3xf32>
  %flow = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
  // CHECK: %[[READ1:.*]] = "tf.ReadVariableOp"(%[[BARG0]]) : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
  // CHECK: %[[UPDATE1:.*]] = "tf.XlaDynamicUpdateSlice"(%[[READ1]],
  // CHECK: "tf.AssignVariableOp"(%[[BARG0]], %[[UPDATE1]])
  %write = "tf.TensorArrayWriteV3"(%arg0, %sub, %elem, %flow) : (tensor<!tf.resource>, tensor<i32>, tensor<3xf32>, tensor<f32>) -> tensor<f32>
  %grad:2 = "tf.TensorArrayGradV3"(%arg0, %write) {source = "a"} : (tensor<!tf.resource>, tensor<f32>) -> (tensor<!tf.resource>, tensor<f32>)
  // CHECK: %[[READ2:.*]] = "tf.ReadVariableOp"(%[[BARG2]]) : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
  // CHECK: %[[UPDATE2:.*]] = "tf.XlaDynamicUpdateSlice"(%[[READ2]],
  // CHECK: "tf.AssignVariableOp"(%[[BARG2]], %[[UPDATE2]])
  %gwrite = "tf.TensorArrayWriteV3"(%grad#0, %sub, %elem, %grad#1) : (tensor<!tf.resource>, tensor<i32>, tensor<3xf32>, tensor<f32>) -> tensor<f32>
  // CHECK: return %[[BARG0]], %[[SUB]], %[[BARG2]]
  return %arg0, %sub : tensor<!tf.resource>, tensor<i32>
}
// CHECK: func @while_cond(%[[CARG0:.*]]: tensor<!tf.resource<tensor<5x3xf32>>>, %[[CARG1:.*]]: tensor<i32>, %[[CARG2:.*]]: tensor<!tf.resource<tensor<5x3xf32>>>)
func @while_cond(%arg0: tensor<!tf.resource>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK-NEXT: return %[[CARG1]]
  return %arg1 : tensor<i32>
}

// -----

// Tests If op with access to the tensor array defined outside and its gradient
// defined inside. The gradient creation should be moved outside.

// CHECK-LABEL: func @main
func @main() -> () {
  // CHECK: %[[SIZE:.*]] = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  %size = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  %index = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[VAR:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<5x3xf32>>>
  %ta:2 = "tf.TensorArrayV3"(%size) {dtype = f32, element_shape = #tf.shape<3>, dynamic_size = false, clear_after_read = true, identical_element_shapes = true, tensor_array_name = "ta"} : (tensor<i32>) -> (tensor<!tf.resource>, tensor<f32>)
  // CHECK: %[[COND:.*]] = "tf._SomeOp"() : () -> tensor<i1>
  %cond = "tf._SomeOp"() : () -> tensor<i1>
  // CHECK: %[[GVAR1:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<5x3xf32>>>
  // CHECK: %[[GVAR2:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<5x3xf32>>>
  // CHECK: "tf.If"(%[[COND]], %[[VAR]], %[[GVAR1]], %[[GVAR2]])
  %1 = "tf.If"(%cond, %ta#0) {
    then_branch = @then_branch, else_branch = @else_branch, device = "", is_stateless = false}
       : (tensor<i1>, tensor<!tf.resource>) -> tensor<!tf.resource>
  // CHECK: %[[READ:.*]] = "tf.ReadVariableOp"(%[[VAR]]) : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
  // CHECK: "tf.Slice"(%[[READ]],
  %read = "tf.TensorArrayReadV3"(%1, %index, %ta#1) : (tensor<!tf.resource>, tensor<i32>, tensor<f32>) -> tensor<3xf32>
  return
}
// CHECK: func @then_branch(%[[TARG0:.*]]: tensor<!tf.resource<tensor<5x3xf32>>>, %[[TARG1:.*]]: tensor<!tf.resource<tensor<5x3xf32>>>, %[[TARG2:.*]]: tensor<!tf.resource<tensor<5x3xf32>>>)
func @then_branch(%arg0: tensor<!tf.resource>) -> tensor<!tf.resource> {
  %const1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %elem = "tf._SomeOp"() : () -> tensor<3xf32>
  %flow = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
  // CHECK: %[[READ1:.*]] = "tf.ReadVariableOp"(%[[TARG1]]) : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
  // CHECK: %[[UPDATE1:.*]] = "tf.XlaDynamicUpdateSlice"(%[[READ1]],
  // CHECK: "tf.AssignVariableOp"(%[[TARG1]], %[[UPDATE1]])
  %grad:2 = "tf.TensorArrayGradV3"(%arg0, %flow) {source = "a"} : (tensor<!tf.resource>, tensor<f32>) -> (tensor<!tf.resource>, tensor<f32>)
  %gwrite = "tf.TensorArrayWriteV3"(%grad#0, %const1, %elem, %grad#1) : (tensor<!tf.resource>, tensor<i32>, tensor<3xf32>, tensor<f32>) -> tensor<f32>
  // CHECK: return %[[TARG0]]
  return %arg0 : tensor<!tf.resource>
}
// CHECK: func @else_branch(%[[EARG0:.*]]: tensor<!tf.resource<tensor<5x3xf32>>>, %[[EARG1:.*]]: tensor<!tf.resource<tensor<5x3xf32>>>, %[[EARG2:.*]]: tensor<!tf.resource<tensor<5x3xf32>>>)
func @else_branch(%arg0: tensor<!tf.resource>) -> tensor<!tf.resource> {
  // CHECK: %[[CONST1:.*]] = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %const1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %elem = "tf._SomeOp"() : () -> tensor<3xf32>
  %flow = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
  // CHECK: %[[READ2:.*]] = "tf.ReadVariableOp"(%[[EARG2]]) : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
  // CHECK: %[[UPDATE2:.*]] = "tf.XlaDynamicUpdateSlice"(%[[READ2]],
  // CHECK: "tf.AssignVariableOp"(%[[EARG2]], %[[UPDATE2]])
  %grad:2 = "tf.TensorArrayGradV3"(%arg0, %flow) {source = "b"} : (tensor<!tf.resource>, tensor<f32>) -> (tensor<!tf.resource>, tensor<f32>)
  %gwrite = "tf.TensorArrayWriteV3"(%grad#0, %const1, %elem, %grad#1) : (tensor<!tf.resource>, tensor<i32>, tensor<3xf32>, tensor<f32>) -> tensor<f32>
  // CHECK: return %[[EARG0]]
  return %arg0 : tensor<!tf.resource>
}

// -----

// Tests (Stateful)PartitionedCall op with access to the tensor array defined
// outside and its gradient defined inside. The gradient creation should be
// moved outside.

// CHECK-LABEL: func @main
func @main() -> () {
  // CHECK: %[[SIZE:.*]] = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  %size = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  %index = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[VAR:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<5x3xf32>>>
  %ta:2 = "tf.TensorArrayV3"(%size) {dtype = f32, element_shape = #tf.shape<3>, dynamic_size = false, clear_after_read = true, identical_element_shapes = true, tensor_array_name = "ta"} : (tensor<i32>) -> (tensor<!tf.resource>, tensor<f32>)
  // CHECK: %[[COND:.*]] = "tf._SomeOp"() : () -> tensor<i1>
  %cond = "tf._SomeOp"() : () -> tensor<i1>
  // CHECK: %[[GVAR1:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<5x3xf32>>>
  %grad:2 = "tf.TensorArrayGradV3"(%ta#0, %ta#1) {source = "a"} : (tensor<!tf.resource>, tensor<f32>) -> (tensor<!tf.resource>, tensor<f32>)
  // CHECK: %[[GVAR2:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<5x3xf32>>>
  // CHECK: "tf.StatefulPartitionedCall"(%[[VAR]], %[[GVAR1]], %[[GVAR2]])
  // CHECK-SAME: f = @callee_tensorarray_decomposed
  %call = "tf.StatefulPartitionedCall"(%ta#0) {f = @callee, config = "", config_proto = "", executor_type = ""}
    : (tensor<!tf.resource>) -> tensor<!tf.resource>
  // CHECK: "tf.PartitionedCall"(%[[VAR]], %[[GVAR1]], %[[GVAR2]])
  // CHECK-SAME: f = @callee_tensorarray_decomposed
  %call2 = "tf.PartitionedCall"(%call) {f = @callee, config = "", config_proto = "", executor_type = ""}
    : (tensor<!tf.resource>) -> tensor<!tf.resource>
  // CHECK: %[[READ:.*]] = "tf.ReadVariableOp"(%[[VAR]]) : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
  // CHECK: "tf.Slice"(%[[READ]],
  %read = "tf.TensorArrayReadV3"(%call2, %index, %ta#1) : (tensor<!tf.resource>, tensor<i32>, tensor<f32>) -> tensor<3xf32>
  return
}
// CHECK-LABEL: func @callee
// CHECK-SAME: (%[[OCARG0:.*]]: tensor<!tf.resource>) -> tensor<!tf.resource>
func @callee(%arg0: tensor<!tf.resource>) -> tensor<!tf.resource> attributes {sym_visibility = "public"} {
  %const1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %elem = "tf._SomeOp"() : () -> tensor<3xf32>
  %flow = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
  %grad:2 = "tf.TensorArrayGradV3"(%arg0, %flow) {source = "a"} : (tensor<!tf.resource>, tensor<f32>) -> (tensor<!tf.resource>, tensor<f32>)
  %gwrite = "tf.TensorArrayWriteV3"(%grad#0, %const1, %elem, %grad#1) : (tensor<!tf.resource>, tensor<i32>, tensor<3xf32>, tensor<f32>) -> tensor<f32>
  %grad2:2 = "tf.TensorArrayGradV3"(%arg0, %flow) {source = "b"} : (tensor<!tf.resource>, tensor<f32>) -> (tensor<!tf.resource>, tensor<f32>)
  %gwrite2 = "tf.TensorArrayWriteV3"(%grad2#0, %const1, %elem, %grad2#1) : (tensor<!tf.resource>, tensor<i32>, tensor<3xf32>, tensor<f32>) -> tensor<f32>
  return %arg0 : tensor<!tf.resource>
}
// CHECK: func @callee_tensorarray_decomposed(%[[CARG0:.*]]: tensor<!tf.resource<tensor<5x3xf32>>>, %[[CARG1:.*]]: tensor<!tf.resource<tensor<5x3xf32>>>, %[[CARG2:.*]]: tensor<!tf.resource<tensor<5x3xf32>>>)
// CHECK: %[[READ1:.*]] = "tf.ReadVariableOp"(%[[CARG1]]) : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
// CHECK: %[[UPDATE1:.*]] = "tf.XlaDynamicUpdateSlice"(%[[READ1]],
// CHECK: "tf.AssignVariableOp"(%[[CARG1]], %[[UPDATE1]])
// CHECK: %[[READ2:.*]] = "tf.ReadVariableOp"(%[[CARG2]]) : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
// CHECK: %[[UPDATE2:.*]] = "tf.XlaDynamicUpdateSlice"(%[[READ2]],
// CHECK: "tf.AssignVariableOp"(%[[CARG2]], %[[UPDATE2]])
// CHECK: return %[[CARG0]]

// -----

// Tests (Stateful)PartitionedCall op with private callee function.

// CHECK-LABEL: func @main
func @main() -> () {
  // CHECK: %[[SIZE:.*]] = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  %size = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  %index = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[VAR:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<5x3xf32>>>
  %ta:2 = "tf.TensorArrayV3"(%size) {dtype = f32, element_shape = #tf.shape<3>, dynamic_size = false, clear_after_read = true, identical_element_shapes = true, tensor_array_name = "ta"} : (tensor<i32>) -> (tensor<!tf.resource>, tensor<f32>)
  // CHECK: %[[COND:.*]] = "tf._SomeOp"() : () -> tensor<i1>
  %cond = "tf._SomeOp"() : () -> tensor<i1>
  // CHECK: %[[GVAR1:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<5x3xf32>>>
  %grad:2 = "tf.TensorArrayGradV3"(%ta#0, %ta#1) {source = "a"} : (tensor<!tf.resource>, tensor<f32>) -> (tensor<!tf.resource>, tensor<f32>)
  // CHECK: %[[GVAR2:.*]] = "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<5x3xf32>>>
  // CHECK: "tf.StatefulPartitionedCall"(%[[VAR]], %[[GVAR1]], %[[GVAR2]])
  // CHECK-SAME: f = @callee
  %call = "tf.StatefulPartitionedCall"(%ta#0) {f = @callee, config = "", config_proto = "", executor_type = ""}
    : (tensor<!tf.resource>) -> tensor<!tf.resource>
  // CHECK: "tf.PartitionedCall"(%[[VAR]], %[[GVAR1]], %[[GVAR2]])
  // CHECK-SAME: f = @callee
  %call2 = "tf.PartitionedCall"(%call) {f = @callee, config = "", config_proto = "", executor_type = ""}
    : (tensor<!tf.resource>) -> tensor<!tf.resource>
  // CHECK: %[[READ:.*]] = "tf.ReadVariableOp"(%[[VAR]]) : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
  // CHECK: "tf.Slice"(%[[READ]],
  %read = "tf.TensorArrayReadV3"(%call2, %index, %ta#1) : (tensor<!tf.resource>, tensor<i32>, tensor<f32>) -> tensor<3xf32>
  return
}
// CHECK: func @callee(%[[CARG0:.*]]: tensor<!tf.resource<tensor<5x3xf32>>>, %[[CARG1:.*]]: tensor<!tf.resource<tensor<5x3xf32>>>, %[[CARG2:.*]]: tensor<!tf.resource<tensor<5x3xf32>>>)
func @callee(%arg0: tensor<!tf.resource>) -> tensor<!tf.resource> attributes {sym_visibility = "private"} {
  // CHECK: %[[READ1:.*]] = "tf.ReadVariableOp"(%[[CARG1]]) : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
  // CHECK: %[[UPDATE1:.*]] = "tf.XlaDynamicUpdateSlice"(%[[READ1]],
  // CHECK: "tf.AssignVariableOp"(%[[CARG1]], %[[UPDATE1]])
  // CHECK: %[[READ2:.*]] = "tf.ReadVariableOp"(%[[CARG2]]) : (tensor<!tf.resource<tensor<5x3xf32>>>) -> tensor<5x3xf32>
  // CHECK: %[[UPDATE2:.*]] = "tf.XlaDynamicUpdateSlice"(%[[READ2]],
  // CHECK: "tf.AssignVariableOp"(%[[CARG2]], %[[UPDATE2]])
  %const1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %elem = "tf._SomeOp"() : () -> tensor<3xf32>
  %flow = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
  %grad:2 = "tf.TensorArrayGradV3"(%arg0, %flow) {source = "a"} : (tensor<!tf.resource>, tensor<f32>) -> (tensor<!tf.resource>, tensor<f32>)
  %gwrite = "tf.TensorArrayWriteV3"(%grad#0, %const1, %elem, %grad#1) : (tensor<!tf.resource>, tensor<i32>, tensor<3xf32>, tensor<f32>) -> tensor<f32>
  %grad2:2 = "tf.TensorArrayGradV3"(%arg0, %flow) {source = "b"} : (tensor<!tf.resource>, tensor<f32>) -> (tensor<!tf.resource>, tensor<f32>)
  %gwrite2 = "tf.TensorArrayWriteV3"(%grad2#0, %const1, %elem, %grad2#1) : (tensor<!tf.resource>, tensor<i32>, tensor<3xf32>, tensor<f32>) -> tensor<f32>
  // CHECK: return %[[CARG0]]
  return %arg0 : tensor<!tf.resource>
}

// -----

// Tests PartitionedCall op with no signature change on callee.

// CHECK-LABEL: func @main
func @main() -> () {
  %call = "tf.PartitionedCall"() {f = @callee, config = "", config_proto = "", executor_type = ""} : () -> tensor<i32>
  return
}
// CHECK: func @callee() -> tensor<i32>
func @callee() -> tensor<i32> attributes {sym_visibility = "public"} {
  %size = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  // CHECK: "tf.MlirLocalVarOp"() : () -> tensor<!tf.resource<tensor<5xf32>>>
  // CHECK: "tf.AssignVariableOp"
  %ta:2 = "tf.TensorArrayV3"(%size) {dtype = f32, element_shape = #tf.shape<>, dynamic_size = false, clear_after_read = true, identical_element_shapes = true, tensor_array_name = "ta"} : (tensor<i32>) -> (tensor<!tf.resource>, tensor<f32>)
  // CHECK: %[[SIZE:.*]] = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  %size_out = "tf.TensorArraySizeV3"(%ta#0, %ta#1) : (tensor<!tf.resource>, tensor<f32>) -> tensor<i32>
  // CHECK: return %[[SIZE]] : tensor<i32>
  return %size_out : tensor<i32>
}

// -----

// Test the pass reports failure on unknown size.

func @main(%arg0: tensor<i32>) -> () {
  // expected-error @+1 {{unknown max element count}}
  %ta:2 = "tf.TensorArrayV3"(%arg0) {dtype = f32, element_shape = #tf.shape<3>, dynamic_size = false, clear_after_read = true, identical_element_shapes = true, tensor_array_name = "ta"} : (tensor<i32>) -> (tensor<!tf.resource>, tensor<f32>)
  return
}

// -----

// Test the pass reports failure on unknown shape.

func @main(%arg0: tensor<i32>) -> () {
  %size = "tf.Const"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  // expected-error @+1 {{unknown element shape}}
  %ta:2 = "tf.TensorArrayV3"(%size) {dtype = f32, element_shape = #tf.shape<*>, dynamic_size = false, clear_after_read = true, identical_element_shapes = true, tensor_array_name = "ta"} : (tensor<i32>) -> (tensor<!tf.resource>, tensor<f32>)
  return
}

// -----

// Tests that the pass reports error on ambiguous tensor array.

func @main(%arg0: tensor<i1>) -> () {
  %size = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  %ta0:2 = "tf.TensorArrayV3"(%size) {dtype = f32, element_shape = #tf.shape<3>, dynamic_size = false, clear_after_read = true, identical_element_shapes = true, tensor_array_name = "ta"} : (tensor<i32>) -> (tensor<!tf.resource>, tensor<f32>)
  %ta1:2 = "tf.TensorArrayV3"(%size) {dtype = f32, element_shape = #tf.shape<3>, dynamic_size = false, clear_after_read = true, identical_element_shapes = true, tensor_array_name = "ta"} : (tensor<i32>) -> (tensor<!tf.resource>, tensor<f32>)
  %if_op = "tf.If"(%arg0, %ta0#0, %ta1#0) {then_branch = @if_then, else_branch = @if_else, is_stateless = false}
    : (tensor<i1>, tensor<!tf.resource>, tensor<!tf.resource>) -> tensor<!tf.resource>
  %index = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // expected-error @+1 {{unknown tensor array}}
  %read = "tf.TensorArrayReadV3"(%if_op, %index, %ta0#1) : (tensor<!tf.resource>, tensor<i32>, tensor<f32>) -> tensor<3xf32>
  return
}
func @if_then(%arg0: tensor<!tf.resource>, %arg1: tensor<!tf.resource>) -> tensor<!tf.resource> {
  return %arg0 : tensor<!tf.resource>
}
func @if_else(%arg0: tensor<!tf.resource>, %arg1: tensor<!tf.resource>) -> tensor<!tf.resource> {
  return %arg1 : tensor<!tf.resource>
}
