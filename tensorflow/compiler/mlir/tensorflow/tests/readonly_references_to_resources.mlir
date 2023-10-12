// RUN: tf-opt -verify-diagnostics -tf-readonly-references-to-resources -split-input-file %s | FileCheck %s

// Test case: Basic converting.

func.func @f() {
  // CHECK: "tf.VarHandleOp"
  // CHECK: "tf.ReadVariableOp"
  %val0 = "tf.VariableV2"() {_class = ["loc:@v"], container = "", device = "", shape = #tf_type.shape<96>, shared_name = ""} : () -> tensor<96x!tf_type.f32ref>
  %val1 = "tf.Identity"(%val0) : (tensor<96x!tf_type.f32ref>) -> tensor<96xf32>
  func.return
}

// -----

// Test case: Basic converting. '_class' attribute is at IdentityOp.

func.func @f() {
  // CHECK: "tf.VarHandleOp"
  // CHECK: "tf.ReadVariableOp"
  %val0 = "tf.VariableV2"() {container = "", device = "", shape = #tf_type.shape<96>, shared_name = ""} : () -> tensor<96x!tf_type.f32ref>
  %val1 = "tf.Identity"(%val0) {_class = ["loc:@v"]} : (tensor<96x!tf_type.f32ref>) -> tensor<96xf32>
  func.return
}

// -----

// Test case: Two ReadVariable ops.

func.func @f() {
  // CHECK: "tf.VarHandleOp"

  // During lowering to resource variables, this pass will preserve the
  // locations of the ReadVariableOps as Identity ops to keep the original graph
  // composition and order.

  // CHECK: "tf.ReadVariableOp"
  // CHECK: "tf.ReadVariableOp"
  %val0 = "tf.VariableV2"() {_class = ["loc:@v"], container = "", device = "", shape = #tf_type.shape<96>, shared_name = ""} : () -> tensor<96x!tf_type.f32ref>
  %val1 = "tf.Identity"(%val0) : (tensor<96x!tf_type.f32ref>) -> tensor<96xf32>
  %val2 = "tf.Identity"(%val0) : (tensor<96x!tf_type.f32ref>) -> tensor<96xf32>
  func.return
}

// -----

// Test case: No follow-up ReadVariable case.

func.func @f() {
  // CHECK-NOT: "tf.VariableV2"
  // CHECK-NOT: "tf.VarHandleOp"
  %val0 = "tf.VariableV2"() {_class = ["loc:@v"], container = "", device = "", shape = #tf_type.shape<96>, shared_name = ""} : () -> tensor<96x!tf_type.f32ref>
  func.return
}

// -----

// Test case: No converting when there is another use case.

func.func @f() {
  // expected-error @+1 {{'tf.VariableV2' op expects all users to be 'tf.Identity', but got user tf.CustomIdentity}}
  %val0 = "tf.VariableV2"() {_class = ["loc:@v"], container = "", device = "", shape = #tf_type.shape<96>, shared_name = ""} : () -> tensor<96x!tf_type.f32ref>
  %val1 = "tf.CustomIdentity"(%val0) : (tensor<96x!tf_type.f32ref>) -> tensor<96xf32>
  func.return
}

// -----

// Test case: Get variable name from the shared_name attribute from VariableV2 op.

func.func @f() {
  // CHECK: "tf.VarHandleOp"
  // CHECK: "tf.ReadVariableOp"
  %val0 = "tf.VariableV2"() {container = "", device = "", shape = #tf_type.shape<96>, shared_name = "test"} : () -> tensor<96x!tf_type.f32ref>
  %val1 = "tf.Identity"(%val0) : (tensor<96x!tf_type.f32ref>) -> tensor<96xf32>
  func.return
}

// -----

// Test case: No class and shared_name attributes on VariableV2 op.

func.func @f() {
  // expected-error @+1 {{'tf.VariableV2' op has no '_class' and 'shared_name' attributes}}
  %val0 = "tf.VariableV2"() {container = "", device = "", shape = #tf_type.shape<96>, shared_name = ""} : () -> tensor<96x!tf_type.f32ref>
  %val1 = "tf.Identity"(%val0) : (tensor<96x!tf_type.f32ref>) -> tensor<96xf32>
  func.return
}

// -----

// Test case: No named location found on VariableV2 op.

func.func @f() {
  // expected-error @+1 {{'tf.VariableV2' op expects variable name in '_class' attribute, but got ["unrelated_class"]}}
  %val0 = "tf.VariableV2"() {_class = ["unrelated_class"], container = "", device = "", shape = #tf_type.shape<96>, shared_name = ""} : () -> tensor<96x!tf_type.f32ref>
  %val1 = "tf.Identity"(%val0) : (tensor<96x!tf_type.f32ref>) -> tensor<96xf32>
  func.return
}

// -----

// Test case: Invalid multiple location information in a class attribute on VariableV2 op.

func.func @f() {
  // expected-error @+1 {{'tf.VariableV2' op expects only one named location in '_class' attribute, but got ["loc:@v1", "loc:@v2"]}}
  %val0 = "tf.VariableV2"() {_class = ["loc:@v1", "loc:@v2"], container = "", device = "", shape = #tf_type.shape<96>, shared_name = ""} : () -> tensor<96x!tf_type.f32ref>
  %val1 = "tf.Identity"(%val0) : (tensor<96x!tf_type.f32ref>) -> tensor<96xf32>
  func.return
}
