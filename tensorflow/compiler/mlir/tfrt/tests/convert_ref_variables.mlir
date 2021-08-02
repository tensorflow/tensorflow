// RUN: tf-tfrt-opt -tfrt-convert-ref-variables -split-input-file -verify-diagnostics %s | FileCheck %s

// Test the basic cases where all uses of a ref variable can be converted.

// CHECK-LABEL: @init
func @init() {
  // CHECK-NOT: tf.VariableV2
  // CHECK-NOT: tf.Assign

  // CHECK: [[handle:%.*]] = "tf.VarHandleOp"
  // CHECK-SAME: shared_name = "x"
  // CHECK: "tf.AssignVariableOp"([[handle]], {{%.*}})
  %0 = "tf.VariableV2"() {container = "", shape = #tf_type.shape<>, shared_name = "x"} : () -> tensor<!tf_type.int32ref>
  %1 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Assign"(%0, %1) {T = i32, device = "", use_locking = true, validate_shape = true} : (tensor<!tf_type.int32ref>, tensor<i32>) -> tensor<!tf_type.int32ref>
  return
}

// CHECK-LABEL: @inference
func @inference() -> tensor<i32> {
  // CHECK-NOT: tf.VariableV2

  // CHECK: [[handle:%.*]] = "tf.VarHandleOp"
  // CHECK-SAME: shared_name = "x"
  // CHECK: "tf.ReadVariableOp"([[handle]])
  %0 = "tf.VariableV2"() {container = "", shape = #tf_type.shape<>, shared_name = "x"} : () -> tensor<!tf_type.int32ref>
  %1 = "tf.Identity"(%0) : (tensor<!tf_type.int32ref>) -> tensor<i32>
  return %1 : tensor<i32>
}

// -----

// Test the cases when there are both reads and writes, the order of the reads and writes are preserved.

// CHECK-LABEL: @init
func @init() -> tensor<i32> {
  // CHECK-NOT: tf.VariableV2

  // CHECK: [[zero:%.*]] = "tf.Const"
  // CHECK-SAME: dense<0>
  %0 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>

  // CHECK: [[handle:%.*]] = "tf.VarHandleOp"
  // CHECK-SAME: shared_name = "x"
  // CHECK-NEXT: "tf.AssignVariableOp"([[handle]], [[zero]])
  // CHECK-NEXT: "tf.ReadVariableOp"([[handle]])
  %1 = "tf.VariableV2"() {container = "", shape = #tf_type.shape<>, shared_name = "x"} : () -> tensor<!tf_type.int32ref>
  %2 = "tf.Assign"(%1, %0) {T = i32, device = "", use_locking = true, validate_shape = true} : (tensor<!tf_type.int32ref>, tensor<i32>) -> tensor<!tf_type.int32ref>
  %3 = "tf.Identity"(%1) : (tensor<!tf_type.int32ref>) -> tensor<i32>

  // CHECK: [[one:%.*]] = "tf.Const"
  // CHECK-SAME: dense<1>
  // CHECK-NEXT: "tf.AssignVariableOp"([[handle]], [[one]])
  // CHECK-NEXT: "tf.ReadVariableOp"([[handle]])
  %4 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %5 = "tf.Assign"(%1, %4) {T = i32, device = "", use_locking = true, validate_shape = true} : (tensor<!tf_type.int32ref>, tensor<i32>) -> tensor<!tf_type.int32ref>
  %6 = "tf.Identity"(%1) : (tensor<!tf_type.int32ref>) -> tensor<i32>

  return %6 : tensor<i32>
}

// CHECK-LABEL: @inference
func @inference() -> (tensor<i32>, tensor<i32>, tensor<i32>) {
  // CHECK-NOT: tf.VariableV2

  // CHECK: [[handle:%.*]] = "tf.VarHandleOp"
  // CHECK-SAME: shared_name = "x"
  // CHECK: "tf.ReadVariableOp"([[handle]])
  // CHECK: "tf.ReadVariableOp"([[handle]])
  // CHECK: "tf.ReadVariableOp"([[handle]])
  %0 = "tf.VariableV2"() {container = "", shape = #tf_type.shape<>, shared_name = "x"} : () -> tensor<!tf_type.int32ref>
  %1 = "tf.Identity"(%0) : (tensor<!tf_type.int32ref>) -> tensor<i32>
  %2 = "tf.Identity"(%0) : (tensor<!tf_type.int32ref>) -> tensor<i32>
  %3 = "tf.Identity"(%0) : (tensor<!tf_type.int32ref>) -> tensor<i32>
  return %1, %2, %3 : tensor<i32>, tensor<i32>, tensor<i32>
}

// -----

// Test report error when the shared_name of the tf.VariableV2 op is empty.

// CHECK-LABEL: @inference
func @inference() -> tensor<i32> {
  // expected-error @+1 {{unable to convert reference variables with empty shared_names.}}
  %0 = "tf.VariableV2"() {container = "", shape = #tf_type.shape<>, shared_name = ""} : () -> tensor<!tf_type.int32ref>
  %1 = "tf.Identity"(%0) : (tensor<!tf_type.int32ref>) -> tensor<i32>
  return %1 : tensor<i32>
}

// -----

// Test conversion when the user is a side-effect-free op.

// CHECK-LABEL: @side_effect_free_user
func @side_effect_free_user() -> tensor<2xi32> {
  // CHECK: [[handle:%.*]] = "tf.VarHandleOp"
  // CHECK-SAME: shared_name = "x"
  // CHECK: [[value0:%.*]] = "tf.ReadVariableOp"([[handle]])
  // CHECK: [[value1:%.*]] = "tf.ReadVariableOp"([[handle]])
  // CHECK: "tf.ConcatV2"([[value1]], [[value0]]
  // CHECK-SAME: (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %axis = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.VariableV2"() {container = "", shape = #tf_type.shape<>, shared_name = "x"} : () -> tensor<!tf_type.int32ref>
  %1 = "tf.ConcatV2"(%0, %0, %axis) : (tensor<!tf_type.int32ref>, tensor<!tf_type.int32ref>, tensor<i32>) -> tensor<2xi32>
  return %1 : tensor<2xi32>
}
