// RUN: tfr-opt %s -verify-diagnostics -split-input-file | tfr-opt | FileCheck %s
// RUN: tfr-opt %s -canonicalize -verify-diagnostics -split-input-file | FileCheck %s -check-prefix=CANON

// Tests for types, ops with custom constraints, verifiers, printer or parser
// methods.

// CHECK-LABEL: tensor_type_noconstraint
func private @tensor_type_noconstraint() -> !tfr.tensor

// -----

// CHECK-LABEL: tensor_type
func private @tensor_type() -> !tfr.tensor<T>

// -----

// CHECK-LABEL: tensor_list_type_noconstraint
func private @tensor_list_type_noconstraint() -> !tfr.tensor_list

// -----

// CHECK-LABEL: tensor_list_type_array_like
func private @tensor_list_type_array_like() -> !tfr.tensor_list<[N, T]>

// -----

// CHECK-LABEL: tensor_list_type_tuple_like
func private @tensor_list_type_tuple_like() -> !tfr.tensor_list<input_T>

// -----

// expected-error@+1 {{unbalanced '>' character in pretty dialect name}}
func private @tensor_invalid_1() -> !tfr.tensor<[N, T>

// -----

// expected-error@+1 {{unexpected nul or EOF in pretty dialect name}}
func @tensor_invalid_2() -> !tfr.tensor<[N, T]

// -----

// CHECK-LABEL: call_op
func @call_op(%arg0: !tfr.tensor<T>, %arg1: !tfr.tensor_list<TL>, %arg2: i32) -> !tfr.tensor<K> {
  %0 = tfr.call @Foo(%arg0, %arg1, %arg2) : (!tfr.tensor<T>, !tfr.tensor_list<TL>, i32) -> !tfr.tensor<K>
  return %0 : !tfr.tensor<K>
}

// -----

// CHECK-LABEL: call_op_arg_attr(%arg0: i32) -> !tfr.tensor<K>
func @call_op_arg_attr(%arg0: i32) -> !tfr.tensor<K> {
  %0 = tfr.call @Bar(%arg0) : (i32) -> !tfr.tensor<K>
  return %0 : !tfr.tensor<K>
}

// -----

func @call_op_invalid_1(%arg0: tensor<?xf32>) -> !tfr.tensor<K> {
  // expected-error@+1 {{got 'tensor<?xf32>'}}
  %0 = tfr.call @Huu(%arg0)  : (tensor<?xf32>) -> !tfr.tensor<K>
  return %0 : !tfr.tensor<K>
}

// -----

// CHECK-LABEL: get_shape
func @get_shape(%arg0: !tfr.tensor) -> (!shape.shape, !shape.shape) {
  %0 = tfr.get_shape %arg0 -> !shape.shape
  %1 = "tfr.get_shape"(%arg0) : (!tfr.tensor) -> !shape.shape
  return %0, %1 : !shape.shape, !shape.shape
}

// -----

// CHECK-LABEL: get_real_shape
// CANON-LABEL: get_real_shape
func @get_real_shape(%arg0: tensor<1x2xf32>) -> tensor<1xindex> {
  %0 = "tfr.cast"(%arg0) : (tensor<1x2xf32>) -> !tfr.tensor
  %1 = tfr.get_shape %0 -> !shape.shape
  %2 = shape.to_extent_tensor %1 : !shape.shape -> tensor<1xindex>
  return %2 : tensor<1xindex>

// CANON-NEXT: %[[s:.*]] = shape.const_shape [1, 2] : tensor<?xindex>
// CANON-NEXT: %[[e:.*]] = shape.to_extent_tensor %[[s]] : tensor<?xindex> -> tensor<1xindex>
// CANON-NEXT: return %[[e]] : tensor<1xindex>
}

// -----

func @get_element_type(%arg0: !tfr.tensor) -> (!tfr.attr, !tfr.attr) {
  %0 = tfr.get_element_type %arg0 -> !tfr.attr
  %1 = "tfr.get_element_type"(%arg0) : (!tfr.tensor) -> !tfr.attr
  return %0, %1 : !tfr.attr, !tfr.attr
}

// -----

// CHECK-LABEL: from_tf_tensor
func @from_tf_tensor(%arg0: tensor<?xf32>) -> !tfr.tensor<K> {
  %0 = "tfr.cast"(%arg0) : (tensor<?xf32>) -> !tfr.tensor<K>
  return %0 : !tfr.tensor<K>
}

// -----

// CHECK-LABEL: to_tf_tensor
func @to_tf_tensor(%arg0: !tfr.tensor<T>) -> tensor<?xi32> {
  %0 = "tfr.cast"(%arg0) : (!tfr.tensor<T>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// -----

// CHECK-LABEL: constant
func @constant() -> (!tfr.attr, !tfr.attr, !tfr.attr, !tfr.attr) {
  %0 = tfr.constant f32 -> !tfr.attr
  %1 = tfr.constant [f32, i32] -> !tfr.attr
  %2 = "tfr.constant"() {value = f32} : () -> !tfr.attr
  %3 = "tfr.constant"() {value = [f32, i32]} : () -> !tfr.attr
  return %0, %1, %2, %3 : !tfr.attr, !tfr.attr, !tfr.attr, !tfr.attr
}

// -----

// CHECK-LABEL: equal
// CANON-LABEL: equal
func @equal() -> (i1, i1, i1, i1) {
  %0 = tfr.constant f32 -> !tfr.attr
  %1 = tfr.constant f32 -> !tfr.attr
  %2 = tfr.constant i32 -> !tfr.attr
  %same_type = tfr.equal %0,%1 -> i1
  %diff_type = tfr.equal %0,%2 -> i1

  %3 = tfr.constant "hello" -> !tfr.attr
  %4 = tfr.constant "hello" -> !tfr.attr
  %5 = tfr.constant "how are you" -> !tfr.attr
  %same_str = tfr.equal %3,%4 -> i1
  %diff_str = tfr.equal %3,%5 -> i1
  return %same_type, %diff_type, %same_str, %diff_str  : i1, i1, i1, i1

// CANON-NEXT: %true = constant true
// CANON-NEXT: %false = constant false
// CANON-NEXT: return %true, %false, %true, %false : i1, i1, i1, i1
}

// -----

// CHECK-LABEL: constant_tensor_scalar
func @constant_tensor_scalar(%arg0: i32) -> tensor<i32> {
  %0 = "tfr.constant_tensor"(%arg0) : (i32) -> tensor<i32>
  return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: constant_tensor_vector
func @constant_tensor_vector(%arg0: vector<1x2xi32>) -> tensor<1x2xi32> {
  %0 = "tfr.constant_tensor"(%arg0) : (vector<1x2xi32>) -> tensor<1x2xi32>
  return %0 : tensor<1x2xi32>
}

// -----

// CHECK-LABEL: constant_tensor_array
// CANON-LABEL: constant_tensor_array
func @constant_tensor_array() -> !tfr.tensor {
  %0 = tfr.constant [1, -1, 3] -> !tfr.attr
  %1 = "tfr.constant_tensor"(%0) : (!tfr.attr) -> !tfr.tensor
  return %1 : !tfr.tensor

// CANON-NEXT: "tf.Const"() {value = dense<[1, -1, 3]> : tensor<3xi64>} : () -> tensor<3xi64>
// CANON-NEXT: "tfr.cast"(%0) : (tensor<3xi64>) -> !tfr.tensor
// CANON-NEXT: return
}

// -----

// CHECK-LABEL: constant_tensor_scalar
// CANON-LABEL: constant_tensor_scalar
func @constant_tensor_scalar() -> !tfr.tensor {
  %0 = "std.constant"() {value = 42 : i32} : () -> i32
  %1 = "tfr.constant_tensor"(%0) : (i32) -> !tfr.tensor
  return %1 : !tfr.tensor

// CANON-NEXT: "tf.Const"() {value = dense<42> : tensor<i32>} : () -> tensor<i32>
// CANON-NEXT: "tfr.cast"(%0) : (tensor<i32>) -> !tfr.tensor
// CANON-NEXT: return
}

// -----

func @constant_tensor_invalid_0(%arg0: i32) -> tensor<f32> {
    // expected-error@+1 {{input and output should have the same scalar types.}}
  %0 = "tfr.constant_tensor"(%arg0) : (i32) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func @constant_tensor_invalid_1(%arg0: vector<1xi32>) -> tensor<?xi32> {
    // expected-error@+1 {{output type should be static and ranked}}
  %0 = "tfr.constant_tensor"(%arg0) : (vector<1xi32>) -> tensor<?xi32>
  return %0 : tensor<?xi32>
}

// -----

func @constant_tensor_invalid_2(%arg0: vector<1xi32>) -> tensor<1xf32> {
    // expected-error@+1 {{input and output should have same shape and element type}}
  %0 = "tfr.constant_tensor"(%arg0) : (vector<1xi32>) -> tensor<1xf32>
  return %0 : tensor<1xf32>
}

// -----

func @constant_tensor_invalid_3(%arg0: vector<1xi32>) -> tensor<1x1xi32> {
    // expected-error@+1 {{input and output should have same shape and element type}}
  %0 = "tfr.constant_tensor"(%arg0) : (vector<1xi32>) -> tensor<1x1xi32>
  return %0 : tensor<1x1xi32>
}

// -----

func @constant_tensor_invalid_4(%arg0: i32) -> tensor<1x1xi32> {
    // expected-error@+1 {{input can not be converted to an output tensor}}
  %0 = "tfr.constant_tensor"(%arg0) : (i32) -> tensor<1x1xi32>
  return %0 : tensor<1x1xi32>
}

// -----

// CHECK-LABEL: get_element
func @get_element(%arg0: !tfr.tensor_list<T>) -> !tfr.tensor {
  %cst = "std.constant"() {value = 1 : index} : () -> index
  %0 = tfr.get_element %arg0[%cst] : (!tfr.tensor_list<T>, index) -> !tfr.tensor
  return %0 : !tfr.tensor
}

// -----

// CHECK-LABEL: build_list
func @build_list(%arg0: !tfr.tensor<A>, %arg1: !tfr.tensor<B>) -> !tfr.tensor_list {
  %0 = "tfr.build_list"(%arg0, %arg1) : (!tfr.tensor<A>, !tfr.tensor<B>) -> !tfr.tensor_list
  return %0 : !tfr.tensor_list
}

// -----

// CHECK-LABEL: build_const_list
// CANON-LABEL: build_const_list
func @build_const_list() -> !tfr.attr {
  %0 = "std.constant"() {value = 42 : i32} : () -> i32
  %1 = "std.constant"() {value = 41 : i32} : () -> i32
  %2 = "tfr.build_list"(%0, %1) : (i32, i32) -> !tfr.attr
  return %2 : !tfr.attr

// CANON-NEXT: %[[c:.*]] = tfr.constant [42 : i32, 41 : i32] -> !tfr.attr
// CANON-NEXT: return %[[c]] : !tfr.attr
}

// -----

// CHECK-LABEL: build_high_dim_const_list
// CANON-LABEL: build_high_dim_const_list
func @build_high_dim_const_list() -> !tfr.attr {
  %0 = "std.constant"() {value = 42 : i32} : () -> i32
  %1 = "std.constant"() {value = 41 : i32} : () -> i32
  %2 = "tfr.build_list"(%0, %1) : (i32, i32) -> !tfr.attr
  %3 = "tfr.build_list"(%0, %1) : (i32, i32) -> !tfr.attr
  %4 = "tfr.build_list"(%2, %3) : (!tfr.attr, !tfr.attr) -> !tfr.attr
  return %4 : !tfr.attr

// CANON-NEXT: %[[c:.*]] = tfr.constant {{\[}}[42 : i32, 41 : i32], [42 : i32, 41 : i32]] -> !tfr.attr
// CANON-NEXT: return %[[c]] : !tfr.attr
}

// -----

// CHECK-LABEL: get_length
// CANON-LABEL: get_length
func @get_length(%arg0: !tfr.tensor<A>, %arg1: !tfr.tensor<B>) -> index {
  %0 = "tfr.build_list"(%arg0, %arg1) : (!tfr.tensor<A>, !tfr.tensor<B>) -> !tfr.tensor_list
  %1 = "tfr.get_length"(%0) : (!tfr.tensor_list) -> index
  return %1 : index

// CANON-NEXT: %[[c:.*]] = constant 2 : index
// CANON-NEXT: return %[[c]] : index
}

// -----

// CHECK-LABEL: tfr.func
tfr.func @External(%arg0: !tfr.tensor<A>,
              %arg1: !tfr.tensor_list<C>,
              %arg2: i32 {tfr.name = "A"},
              %arg3: !tfr.attr {tfr.name = "T"})
  -> (!tfr.tensor<A>, !tfr.tensor_list<C>)
  attributes {A, C}

// -----

// CHECK-LABEL: tfr.func
tfr.func @Foo(%arg0: !tfr.tensor<A>,
              %arg1: !tfr.tensor_list<C>,
              %arg2: i32 {tfr.name = "A"},
              %arg3: vector<1xi32> {tfr.name = "C"})
  -> (!tfr.tensor<A>, !tfr.tensor_list<C>)
  attributes {A, C} {
  tfr.return %arg0, %arg1 : !tfr.tensor<A>, !tfr.tensor_list<C>
}

// -----

// CHECK-LABEL: tfr.func
tfr.func @Bar(%arg0: !tfr.tensor<A>,
              %arg2: i32 {tfr.name = "B"},
              %arg3: vector<1xi32> {tfr.name = "C"})
  -> (!tfr.tensor<A>, !tfr.tensor<A>)
  attributes {A} {
  tfr.return %arg0, %arg0 : !tfr.tensor<A>, !tfr.tensor<A>
}

// -----

// expected-error@+1 {{Undefined attributes are used: A}}
tfr.func @Foo_undefined_attr(%arg0: !tfr.tensor<A>,
              %arg1: !tfr.tensor_list<A>,
              %arg2: i32 {tfr.name = "A"},
              %arg3: vector<1xi32> {tfr.name = "C"}) ->
    (!tfr.tensor<A>, !tfr.tensor_list<A>) {
  tfr.return %arg0, %arg1 : !tfr.tensor<A>, !tfr.tensor_list<A>
}

// -----

// expected-error@+1 {{3 attribute argument doesn't have a tfr.name attribute}}
tfr.func @Foo_unnamed_attr(%arg0: !tfr.tensor<A>,
              %arg1: !tfr.tensor_list<A>,
              %arg2: i32 {tfr.name = "A"},
              %arg3: vector<1xi32>) ->
    (!tfr.tensor<A>, !tfr.tensor_list<A>) {
  tfr.return %arg0, %arg1 : !tfr.tensor<A>, !tfr.tensor_list<A>
}

// -----

// expected-error@+1 {{tfr.tensor/tfr.tensor_list argument should be before non tensor arguments}}
tfr.func @Foo_invalid_arg_order(%arg0: !tfr.tensor<A>,
              %arg2: i32 {tfr.name = "A"},
              %arg1: !tfr.tensor_list<A>,
              %arg3: vector<1xi32> {tfr.name = "C"}) ->
    (!tfr.tensor<A>, !tfr.tensor_list<A>) {
  tfr.return %arg0, %arg1 : !tfr.tensor<A>, !tfr.tensor_list<A>
}

// -----

tfr.func @Foo_valid_arg_order0(
              %arg1: !tfr.tensor_list,
              %arg0: !tfr.tensor<T>,
              %arg2: i32 {tfr.name = "A"},
              %arg3: vector<1xi32> {tfr.name = "C"}) ->
    (!tfr.tensor, !tfr.tensor_list) attributes {T}{
  tfr.return %arg0, %arg1 : !tfr.tensor<T>, !tfr.tensor_list
}

// -----

// expected-error@+1 {{tfr.tensor argument should be before tfr.tensor_list argument.}}
tfr.func @Foo_invalid_arg_order0(
              %arg1: !tfr.tensor_list,
              %arg0: !tfr.tensor<T>,
              %arg2: i32 {tfr.name = "A"},
              %arg3: vector<1xi32> {tfr.name = "C"}) ->
    (!tfr.tensor, !tfr.tensor_list) {
  tfr.return %arg0, %arg1 : !tfr.tensor<T>, !tfr.tensor_list
}

// -----

// expected-error@+1 {{tfr.tensor result should be before tfr.tensor_list result}}
tfr.func @Foo_invalid_result_order(%arg0: !tfr.tensor<A>,
              %arg1: !tfr.tensor_list<A>,
              %arg2: i32 {tfr.name = "A"},
              %arg3: vector<1xi32> {tfr.name = "C"}) ->
    (!tfr.tensor_list<A>, !tfr.tensor<A>) {
  tfr.return %arg1, %arg0 : !tfr.tensor_list<A>, !tfr.tensor<A>
}

// -----

// expected-error@+1 {{More than one tfr.tensor_list argument isn't allowed}}
tfr.func @Foo_multiple_tensor_list_args(%arg0: !tfr.tensor<A>,
              %arg1: !tfr.tensor_list<A>,
              %arg2: !tfr.tensor_list<A>,
              %arg3: i32 {tfr.name = "A"},
              %arg4: vector<1xi32> {tfr.name = "C"}) ->
    (!tfr.tensor<A>, !tfr.tensor_list<A>) {
  tfr.return %arg0, %arg1 : !tfr.tensor<A>, !tfr.tensor_list<A>
}

// -----

// expected-error@+1 {{More than one tfr.tensor_list result isn't allowed}}
tfr.func @Foo_multiple_tensor_list_results(%arg0: !tfr.tensor<C>,
              %arg1: !tfr.tensor_list<A>,
              %arg2: i32 {tfr.name = "A"},
              %arg3: vector<1xi32> {tfr.name = "C"}) ->
    (!tfr.tensor_list<A>, !tfr.tensor_list<A>) {
  tfr.return %arg1, %arg1 : !tfr.tensor_list<A>, !tfr.tensor_list<A>
}

// -----

// expected-error@+1 {{None tfr.tensor/tfr.tensor_list results aren't allowed as a result}}
tfr.func @Foo_return_attr(%arg0: !tfr.tensor<C>,
              %arg1: !tfr.tensor_list<A>,
              %arg2: i32 {tfr.name = "A"},
              %arg3: vector<1xi32> {tfr.name = "C"}) -> i32 {
  tfr.return %arg2 : i32
}
