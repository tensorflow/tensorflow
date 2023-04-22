// RUN: tfr-opt %s -tfr-decompose -tfr-raise-to-tf -canonicalize -verify-diagnostics -split-input-file | FileCheck %s

//=================> User models, from GraphDef <====================

// CHECK-LABEL: my_identity
func @my_identity(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "tf.MyIdentity"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>

// CHECK-NEXT: return %arg0 : tensor<2x3xf32>
}

// CHECK-LABEL: my_rsqrt
func @my_rsqrt(%arg0: tensor<2x3xf32>) -> tensor<3x2x3xf32> {
  %0 = "tf.MyRsqrt"(%arg0) : (tensor<2x3xf32>) -> tensor<3x2x3xf32>
  return %0 : tensor<3x2x3xf32>

// CHECK-NEXT: %[[RE:.*]] = "tf.RiscReciprocal"(%arg0) : (tensor<2x3xf32>) -> tensor<*xf32>
// CHECK-NEXT: %[[SQRT:.*]] = "tf.RiscSqrt"(%[[RE]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT: %[[ES:.*]] = "tf.EnsureShape"(%[[SQRT]]) {shape = #tf.shape<3x2x3>} : (tensor<*xf32>) -> tensor<3x2x3xf32>
// CHECK-NEXT: return %[[ES]] : tensor<3x2x3xf32>
}

// CHECK-LABEL: my_leaky_relu
func @my_leaky_relu(%arg0: tensor<2x3xf32>) -> tensor<3x2x3xf32> {
  %0 = "tf.MyLeakyRelu"(%arg0) {alpha=3.0 : f32} : (tensor<2x3xf32>) -> tensor<3x2x3xf32>
  return %0 : tensor<3x2x3xf32>

// CHECK-NEXT: %[[ALPHA:.*]] = "tf.Const"() {value = dense<3.000000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK-NEXT: %[[SHAPE:.*]] = "tf.RiscShape"(%arg0) {T = i32} : (tensor<2x3xf32>) -> tensor<*xi32>
// CHECK-NEXT: %[[ALPHA1:.*]] = "tf.RiscBroadcast"(%[[ALPHA]], %[[SHAPE]]) : (tensor<f32>, tensor<*xi32>) -> tensor<*xf32>
// CHECK-NEXT: %[[MAX:.*]] = "tf.RiscMaximum"(%arg0, %[[ALPHA1]]) : (tensor<2x3xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT: %[[ES:.*]] = "tf.EnsureShape"(%[[MAX]]) {shape = #tf.shape<3x2x3>} : (tensor<*xf32>) -> tensor<3x2x3xf32>
// CHECK-NEXT: return %[[ES]] : tensor<3x2x3xf32>
}

// CHECK-LABEL: my_leaky_relu_with_default
func @my_leaky_relu_with_default(%arg0: tensor<2x3xf32>) -> tensor<3x2x3xf32> {
  %0 = "tf.MyLeakyRelu"(%arg0) : (tensor<2x3xf32>) -> tensor<3x2x3xf32>
  return %0 : tensor<3x2x3xf32>

// CHECK-NEXT: %[[ALPHA:.*]] = "tf.Const"() {value = dense<2.000000e-01> : tensor<f32>} : () -> tensor<f32>
// CHECK-NEXT: %[[SHAPE:.*]] = "tf.RiscShape"(%arg0) {T = i32} : (tensor<2x3xf32>) -> tensor<*xi32>
// CHECK-NEXT: %[[ALPHA1:.*]] = "tf.RiscBroadcast"(%[[ALPHA]], %[[SHAPE]]) : (tensor<f32>, tensor<*xi32>) -> tensor<*xf32>
// CHECK-NEXT: %[[MAX:.*]] = "tf.RiscMaximum"(%arg0, %[[ALPHA1]]) : (tensor<2x3xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT: %[[ES:.*]] = "tf.EnsureShape"(%[[MAX]]) {shape = #tf.shape<3x2x3>} : (tensor<*xf32>) -> tensor<3x2x3xf32>
// CHECK-NEXT: return %[[ES]] : tensor<3x2x3xf32>
}

// CHECK-LABEL: my_cast
func @my_cast(%arg0: tensor<2x3xf32>) -> tensor<2x3xi32> {
  %0 = "tf.MyCast"(%arg0) {Tout=i32} : (tensor<2x3xf32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>

// CHECK-NEXT: %[[CAST:.*]] = "tf.RiscCast"(%arg0) {Tout = i32} : (tensor<2x3xf32>) -> tensor<*xi32>
// CHECK-NEXT: %[[ES:.*]] = "tf.EnsureShape"(%[[CAST]]) {shape = #tf.shape<2x3>} : (tensor<*xi32>) -> tensor<2x3xi32>
// CHECK-NEXT: return %[[ES]] : tensor<2x3xi32>
}

// CHECK-LABEL: my_pack_single_input
func @my_pack_single_input(%arg0: tensor<2x3xf32>) -> tensor<3x2x3xf32> {
  %0 = "tf.MyPack"(%arg0) {N=1:i32, axis=0:i32} : (tensor<2x3xf32>) -> tensor<3x2x3xf32>
  return %0 : tensor<3x2x3xf32>

// CHECK-NEXT: %[[AXIS:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
// CHECK-NEXT: %[[ED:.*]] = "tf.ExpandDims"(%arg0, %[[AXIS]]) : (tensor<2x3xf32>, tensor<i32>) -> tensor<*xf32>
// CHECK-NEXT: %[[ES:.*]] = "tf.EnsureShape"(%[[ED]]) {shape = #tf.shape<3x2x3>} : (tensor<*xf32>) -> tensor<3x2x3xf32>
// CHECK-NEXT: return %[[ES]] : tensor<3x2x3xf32>
}

// CHECK-LABEL: my_pack_multiple_inputs
func @my_pack_multiple_inputs(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<2x3xf32>) -> tensor<3x2x3xf32> {
  %0 = "tf.MyPack"(%arg0, %arg1, %arg2) {N=3:i32, axis=0:i32} : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<3x2x3xf32>
  return %0 : tensor<3x2x3xf32>

// CHECK-NEXT: %[[AXIS:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
// CHECK-NEXT: %[[ED0:.*]] = "tf.ExpandDims"(%arg0, %[[AXIS]]) : (tensor<2x3xf32>, tensor<i32>) -> tensor<*xf32>
// CHECK-NEXT: %[[ED1:.*]] = "tf.ExpandDims"(%arg1, %[[AXIS]]) : (tensor<2x3xf32>, tensor<i32>) -> tensor<*xf32>
// CHECK-NEXT: %[[CC0:.*]] = "tf.RiscConcat"(%[[ED0]], %[[ED1]]) {axis = 0 : i32} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT: %[[ED2:.*]] = "tf.ExpandDims"(%arg2, %[[AXIS]]) : (tensor<2x3xf32>, tensor<i32>) -> tensor<*xf32>
// CHECK-NEXT: %[[CC1:.*]] = "tf.RiscConcat"(%[[CC0]], %[[ED2]]) {axis = 0 : i32} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK-NEXT: %[[ES:.*]] = "tf.EnsureShape"(%[[CC1]]) {shape = #tf.shape<3x2x3>} : (tensor<*xf32>) -> tensor<3x2x3xf32>
// CHECK-NEXT: return %[[ES]] : tensor<3x2x3xf32>
}

// CHECK-LABEL: my_add_n_single_input
func @my_add_n_single_input(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "tf.MyAddN"(%arg0) {N=1:i32} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>

// CHECK-NEXT: return %arg0 : tensor<2x3xf32>
}

// CHECK-LABEL: my_add_n_multiple_inputs
func @my_add_n_multiple_inputs(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>, %arg2: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "tf.MyAddN"(%arg0, %arg1, %arg2) {N=3:i32} : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0 : tensor<2x3xf32>

// CHECK-NEXT: %[[ADD0:.*]] = "tf.RiscAdd"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<*xf32>
// CHECK-NEXT: %[[ADD1:.*]] = "tf.RiscAdd"(%[[ADD0]], %arg2) : (tensor<*xf32>, tensor<2x3xf32>) -> tensor<*xf32>
// CHECK-NEXT: %[[ES:.*]] = "tf.EnsureShape"(%[[ADD1]]) {shape = #tf.shape<2x3>} : (tensor<*xf32>) -> tensor<2x3xf32>
// CHECK-NEXT: return %[[ES]] : tensor<2x3xf32>
}

// CHECK-LABEL: my_map_and_batch_dataset
func @my_map_and_batch_dataset(%input: tensor<*x!tf.variant>,
                               %other1: tensor<*xf32>,
                               %other2: tensor<*xi32>) -> tensor<*x!tf.variant> {
  %0 = "tf.MyMapAndBatchDataset"(%input, %other1, %other2)
    {batch_size=1000 : i64, num_parallel_calls = 8 : i64, drop_remainder = 0 : i1,
     func = @"__some_func", output_types = [f32], output_shapes = [#tf.shape<>], preserve_cardinality = true}
    : (tensor<*x!tf.variant>, tensor<*xf32>, tensor<*xi32>) -> tensor<*x!tf.variant>
  return %0 : tensor<*x!tf.variant>

// CHECK-DAG: %[[BATCH:.*]] = "tf.Const"() {value = dense<1000> : tensor<i64>} : () -> tensor<i64>
// CHECK-DAG: %[[PARAL:.*]] = "tf.Const"() {value = dense<8> : tensor<i64>} : () -> tensor<i64>
// CHECK-DAG: %[[KEEP:.*]] = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
// CHECK: %[[CAST:.*]] = "tf.Cast"(%arg2) {Truncate = false} : (tensor<*xi32>) -> tensor<*xf32>
// CHECK: %[[RET:.*]] = "tf.MapAndBatchDatasetV0"(%arg0, %[[BATCH]], %[[PARAL]], %[[KEEP]], %arg1, %[[CAST]])
// CHECK-SAME: {f = @__some_func, output_shapes = [#tf.shape<>], output_types = [f32], preserve_cardinality = true} : (tensor<*x!tf.variant>, tensor<i64>, tensor<i64>, tensor<i1>, tensor<*xf32>, tensor<*xf32>) -> tensor<*x!tf.variant>
// CHECK: return %[[RET]] : tensor<*x!tf.variant>
}

//=================> decomposition functions, translated from tf.compose api <====================
tfr.func @tf__my_identity(%value: !tfr.tensor) -> !tfr.tensor {
  tfr.return %value : !tfr.tensor
}

tfr.func @tf__my_cast(%value: !tfr.tensor, %tout: !tfr.attr{tfr.name="Tout"}) -> !tfr.tensor {
  %0 = tfr.call @tf__risc_cast(%value, %tout) : (!tfr.tensor, !tfr.attr) -> !tfr.tensor
  tfr.return %0 : !tfr.tensor
}

tfr.func @tf__my_rsqrt(%value: !tfr.tensor) -> !tfr.tensor {
  %1 = tfr.call @tf__risc_reciprocal(%value) : (!tfr.tensor) -> !tfr.tensor
  %2 = tfr.call @tf__risc_sqrt(%1) : (!tfr.tensor) -> !tfr.tensor
  tfr.return %2 : !tfr.tensor
}

tfr.func @tf__my_leaky_relu(%value: !tfr.tensor, %alpha: f32 {tfr.name="alpha", tfr.default=0.2:f32}) -> !tfr.tensor {
  %1 = tfr.call @tf__risc_shape(%value) : (!tfr.tensor) -> !tfr.tensor
  %2 = "tfr.constant_tensor"(%alpha) : (f32) -> tensor<f32>
  %t = "tfr.cast"(%2) : (tensor<f32>) -> !tfr.tensor
  %3 = tfr.call @tf__risc_broadcast(%t, %1) : (!tfr.tensor, !tfr.tensor) -> !tfr.tensor
  %4 = tfr.call @tf__risc_maximum(%value, %3) : (!tfr.tensor, !tfr.tensor) -> !tfr.tensor
  tfr.return %4  : !tfr.tensor
}

// TODO(fengliuai): use shape dialect to manipulate the shape then this can be decomposed further.
tfr.func @tf__my_expand_dims(%value: !tfr.tensor, %axis: i32 {tfr.name="axis"}) -> !tfr.tensor {
  %axis_cst = "tfr.constant_tensor"(%axis) : (i32) -> tensor<i32>
  %dim = "tfr.cast"(%axis_cst) : (tensor<i32>) -> !tfr.tensor
  %0 = tfr.call @tf__expand_dims(%value, %dim) : (!tfr.tensor, !tfr.tensor) -> !tfr.tensor
  tfr.return %0 : !tfr.tensor
}

tfr.func @tf__my_pack(%values: !tfr.tensor_list,
                      %n: i32 {tfr.name="N"},
                      %axis: i32 {tfr.name="axis"}) -> !tfr.tensor {
  %index = constant 0 : index
  %cst = constant 1 : i32
  %eq = cmpi eq, %n, %cst : i32
  %v1 = tfr.get_element %values[%index] : (!tfr.tensor_list, index) -> !tfr.tensor
  %temp = tfr.call @tf__my_expand_dims(%v1, %axis) : (!tfr.tensor, i32) -> !tfr.tensor
  %res = scf.if %eq -> !tfr.tensor {
    scf.yield %temp : !tfr.tensor
  } else {
    %step = index_cast %cst : i32 to index
    %end = index_cast %n : i32 to index
    %reduce = scf.for %i = %step to %end step %step iter_args(%reduce_iter=%temp) -> !tfr.tensor {
      %v = tfr.get_element %values[%i] : (!tfr.tensor_list, index) -> !tfr.tensor
      %temp1 =  tfr.call @tf__my_expand_dims(%v, %axis) : (!tfr.tensor, i32) -> !tfr.tensor
      %reduce_next =  tfr.call @tf__risc_concat(%reduce_iter, %temp1, %axis) : (!tfr.tensor, !tfr.tensor, i32) -> !tfr.tensor
      scf.yield %reduce_next : !tfr.tensor
    }
    scf.yield %reduce : !tfr.tensor
  }
  tfr.return %res : !tfr.tensor
}

tfr.func @tf__my_add_n(%values: !tfr.tensor_list,
                       %n: i32 {tfr.name="N"}) -> !tfr.tensor {
  %index = constant 0 : index
  %cst = constant 1 : i32
  %eq = cmpi eq, %n, %cst : i32
  %v1 = tfr.get_element %values[%index] : (!tfr.tensor_list, index) -> !tfr.tensor
  %res = scf.if %eq -> !tfr.tensor {
    scf.yield %v1 : !tfr.tensor
  } else {
    %step = index_cast %cst : i32 to index
    %end = index_cast %n : i32 to index
    %reduce = scf.for %i = %step to %end step %step iter_args(%reduce_iter=%v1) -> !tfr.tensor {
      %v = tfr.get_element %values[%i] : (!tfr.tensor_list, index) -> !tfr.tensor
      %reduce_next =  tfr.call @tf__risc_add(%reduce_iter, %v) : (!tfr.tensor, !tfr.tensor) -> !tfr.tensor
      scf.yield %reduce_next : !tfr.tensor
    }
    scf.yield %reduce : !tfr.tensor
  }
  tfr.return %res : !tfr.tensor
}

tfr.func @tf__my_map_and_batch_dataset(
    %input_dataset: !tfr.tensor,
    %other_arguments: !tfr.tensor_list,
    %batch_size: i64 {tfr.name="batch_size"},
    %num_parallel_calls: i64 {tfr.name="num_parallel_calls"},
    %drop_remainder: i1 {tfr.name="drop_remainder"},
    %f: !tfr.attr {tfr.name="func"},
    %output_types: !tfr.attr {tfr.name="output_types"},
    %output_shapes: !tfr.attr {tfr.name="output_shapes"},
    %preserve_cardinality: i1 {tfr.name="preserve_cardinality", tfr.default=false}) -> !tfr.tensor {
  %batch = "tfr.constant_tensor"(%batch_size) : (i64) -> tensor<i64>
  %batch1 = "tfr.cast"(%batch) : (tensor<i64>) -> !tfr.tensor
  %calls = "tfr.constant_tensor"(%num_parallel_calls) : (i64) -> tensor<i64>
  %calls1 = "tfr.cast"(%calls) : (tensor<i64>) -> !tfr.tensor
  %drop = "tfr.constant_tensor"(%drop_remainder) : (i1) -> tensor<i1>
  %drop1 = "tfr.cast"(%drop) : (tensor<i1>) -> !tfr.tensor
  %ret = tfr.call @tf__map_and_batch_dataset_v0(%input_dataset, %batch1, %calls1, %drop1, %other_arguments, %f, %output_types, %output_shapes, %preserve_cardinality)
    : (!tfr.tensor, !tfr.tensor, !tfr.tensor, !tfr.tensor, !tfr.tensor_list, !tfr.attr, !tfr.attr, !tfr.attr, i1) -> !tfr.tensor
  tfr.return %ret : !tfr.tensor
}

//=================>  signatures of the primitive ops with kernels, modeled as external TFR function <==
tfr.func @tf__risc_cast_(!tfr.tensor, !tfr.attr{tfr.name="Tout"}) -> !tfr.tensor<Tout> attributes{Tout}
tfr.func @tf__risc_add_(!tfr.tensor<T>, !tfr.tensor<T>) -> !tfr.tensor<T> attributes{T}
tfr.func @tf__risc_concat_(!tfr.tensor<T>, !tfr.tensor<T>, i32{tfr.name="axis"}) -> !tfr.tensor<T> attributes{T}
tfr.func @tf__risc_broadcast_(!tfr.tensor<T>, !tfr.tensor<Tidx>) -> !tfr.tensor<T> attributes{T, Tidx}
tfr.func @tf__risc_reciprocal_(!tfr.tensor<T>) -> !tfr.tensor<T> attributes{T}
tfr.func @tf__risc_sqrt_(!tfr.tensor<T>) -> !tfr.tensor<T> attributes{T}
tfr.func @tf__risc_shape_(!tfr.tensor, !tfr.attr{tfr.name="T", tfr.default=i32}) -> !tfr.tensor<T> attributes{T}
tfr.func @tf__risc_maximum_(!tfr.tensor<T>, !tfr.tensor<T>) -> !tfr.tensor<T> attributes{T}
tfr.func @tf__expand_dims_(!tfr.tensor<T>, !tfr.tensor<Tdim>) -> !tfr.tensor<T> attributes{T, Tdim}
tfr.func @tf__map_and_batch_dataset_v0_(!tfr.tensor<T>, !tfr.tensor, !tfr.tensor, !tfr.tensor, !tfr.tensor_list<Targuments>,
  !tfr.attr{tfr.name="f"}, !tfr.attr{tfr.name="output_types"}, !tfr.attr{tfr.name="output_shapes"}, i1{tfr.name="preserve_cardinality"})
  -> !tfr.tensor<T> attributes{T, Targuments}
