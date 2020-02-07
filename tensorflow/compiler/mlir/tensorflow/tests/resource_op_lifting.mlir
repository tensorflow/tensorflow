// RUN: tf-opt %s -split-input-file -verify-diagnostics -tf-resource-op-lifting | FileCheck %s -dump-input-on-failure

// Tests that resource load operations are hoisted.

// CHECK-LABEL: func @only_resource_load
func @only_resource_load() -> tensor<*xi32> {

  // CHECK: %[[RES_HANDLE:[0-9]*]] = "tf.VarHandleOp"
  %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource>

  // CHECK: %[[RES_READ_VAL:[0-9]*]] = "tf.ReadVariableOp"(%[[RES_HANDLE]]) {dtype = i32}
  // CHECK: "tf_device.launch"
  // CHECK: %[[COMPUTE_RES:[0-9]*]] = "tf.SomeComputation"(%[[RES_READ_VAL]])
  // CHECK: tf_device.return %[[COMPUTE_RES]]
  // CHECK: {device = "tpu0", launch_attr = "launch_attr"}
  // CHECK-SAME: () -> tensor<*xi32>

  %1 = "tf_device.launch"() ( {
    %2 = "tf.ReadVariableOp"(%0) {dtype = i32} : (tensor<*x!tf.resource>) -> tensor<*xi32>
    %3 = "tf.SomeComputation"(%2) : (tensor<*xi32>) -> (tensor<*xi32>)
    tf_device.return %3 : tensor<*xi32>
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> tensor<*xi32>

  return %1 : tensor<*xi32>
}

// -----

// Tests that resource store operations are hoisted.

// CHECK-LABEL: func @only_resource_store
func @only_resource_store() -> tensor<*xi32> {

  // CHECK: %[[RES_HANDLE:[0-9]*]] = "tf.VarHandleOp"
  %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource>

  // CHECK: %[[LAUNCH_RES:[0-9]*]]:2 = "tf_device.launch"
  // CHECK: %[[COMPUTE_RES:[0-9]*]] = "tf.SomeComputation"()
  // CHECK: tf_device.return %[[COMPUTE_RES]], %[[COMPUTE_RES]]
  // CHECK: {device = "tpu0", launch_attr = "launch_attr"}
  // CHECK-SAME: () -> (tensor<*xi32>, tensor<*xi32>)
  // CHECK: "tf.AssignVariableOp"(%[[RES_HANDLE]], %[[LAUNCH_RES]]#1) {dtype = i32}

  %1 = "tf_device.launch"() ( {
    %2 = "tf.SomeComputation"() : () -> (tensor<*xi32>)
    "tf.AssignVariableOp"(%0, %2) {dtype = i32} : (tensor<*x!tf.resource>, tensor<*xi32>) -> ()
    tf_device.return %2 : tensor<*xi32>
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> tensor<*xi32>

  // CHECK: return %[[LAUNCH_RES]]#0
  return %1 : tensor<*xi32>
}

// -----

// Tests that a resource ops with both load and store are hoisted.

// CHECK-LABEL: func @same_resource_load_and_store
func @same_resource_load_and_store() -> tensor<*xi32> {

  // CHECK: %[[RES_HANDLE:[0-9]*]] = "tf.VarHandleOp"
  %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource>

  // CHECK: %[[RES_READ_VAL:[0-9]*]] = "tf.ReadVariableOp"(%[[RES_HANDLE]]) {dtype = i32}
  // CHECK: %[[LAUNCH_RES:[0-9]*]]:2 = "tf_device.launch"
  // CHECK: %[[COMPUTE_RES:[0-9]*]] = "tf.SomeComputation"(%[[RES_READ_VAL]])
  // CHECK: tf_device.return %[[COMPUTE_RES]], %[[COMPUTE_RES]]
  // CHECK: {device = "tpu0", launch_attr = "launch_attr"}
  // CHECK-SAME: () -> (tensor<*xi32>, tensor<*xi32>)
  // CHECK: "tf.AssignVariableOp"(%[[RES_HANDLE]], %[[LAUNCH_RES]]#1) {dtype = i32}

  %1 = "tf_device.launch"() ( {
    %2 = "tf.ReadVariableOp"(%0) {dtype = i32} : (tensor<*x!tf.resource>) -> tensor<*xi32>
    %3 = "tf.SomeComputation"(%2) : (tensor<*xi32>) -> (tensor<*xi32>)
    "tf.AssignVariableOp"(%0, %3) {dtype = i32} : (tensor<*x!tf.resource>, tensor<*xi32>) -> ()
    tf_device.return %3 : tensor<*xi32>
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> tensor<*xi32>

  // CHECK: return %[[LAUNCH_RES]]#0
  return %1 : tensor<*xi32>
}

// -----

// Tests that internal resource operations are not hoisted.

// CHECK-LABEL: func @internal_resource
func @internal_resource() -> tensor<*xi32> {

  // CHECK: %[[LAUNCH_RES:[0-9]*]] = "tf_device.launch"
  %0 = "tf_device.launch"() ( {

    // CHECK: %[[RES_HANDLE:[0-9]*]] = "tf.VarHandleOp"
    %1 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource>

    // CHECK: %[[RES_READ_VAL:[0-9]*]] = "tf.ReadVariableOp"(%[[RES_HANDLE]])
    %2 = "tf.ReadVariableOp"(%1) {dtype = i32} : (tensor<*x!tf.resource>) -> tensor<*xi32>

    // CHECK: %[[COMPUTE_RES:[0-9]*]] = "tf.SomeComputation"(%[[RES_READ_VAL]])
    %3 = "tf.SomeComputation"(%2) : (tensor<*xi32>) -> (tensor<*xi32>)

    // CHECK: "tf.AssignVariableOp"(%[[RES_HANDLE]], %[[COMPUTE_RES]])
    "tf.AssignVariableOp"(%1, %3) {dtype = i32} : (tensor<*x!tf.resource>, tensor<*xi32>) -> ()

    // CHECK: tf_device.return %[[COMPUTE_RES]]
    tf_device.return %3 : tensor<*xi32>
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> tensor<*xi32>

  // CHECK: return %[[LAUNCH_RES]]
  return %0 : tensor<*xi32>
}

// -----

// Tests that pass fails when there are remaining resource operationss that can
// not be lifted.

func @lifting_failure() -> tensor<*xi32> {

  %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource>

  // expected-error @+1 {{has remaining resource inputs that can not be lifted}}
  %1 = "tf_device.launch"() ( {
    %2 = "tf.ReadVariableOp"(%0) {dtype = i32} : (tensor<*x!tf.resource>) -> tensor<*xi32>
		%3 = "tf.SomeResourceOp"(%0, %2) : (tensor<*x!tf.resource>, tensor<*xi32>) -> tensor<*xi32>
    "tf.AssignVariableOp"(%0, %3) {dtype = i32} : (tensor<*x!tf.resource>, tensor<*xi32>) -> ()
    tf_device.return %3 : tensor<*xi32>
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> tensor<*xi32>

  return %1 : tensor<*xi32>
}

// -----

// Tests that pass lifts resource reads/writes from a loop, and removed unused
// resources.

// CHECK-LABEL: func @launch_with_loop
func @launch_with_loop() -> () {
  // CHECK: %[[COUNT:.*]] = "tf.Const"() {value = dense<10> : tensor<i32>}
  %0 = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[VH:.*]] = "tf.VarHandleOp"()
  %1 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource<tensor<f32>>>
  %unused = "tf.VarHandleOp"() {container = "c", shared_name = "v2"} : () -> tensor<*x!tf.resource<tensor<f32>>>
  // CHECK: %[[READ:.*]] = "tf.ReadVariableOp"(%[[VH]])
  // CHECK: %[[LAUNCH:.*]] = "tf_device.launch"()
  "tf_device.launch"() ( {
    // CHECK: %[[WHILE:.*]]:2 = "tf.While"(%[[COUNT]], %[[READ]])
    %2:3 = "tf.While"(%0, %1, %unused)
               {body = @while_body, cond = @while_cond, device = "", is_stateless = false,
                output_shapes = ["tfshape$", "tfshape$"]}
         : (tensor<i32>, tensor<*x!tf.resource<tensor<f32>>>, tensor<*x!tf.resource<tensor<f32>>>)
         -> (tensor<i32>, tensor<*x!tf.resource<tensor<f32>>>, tensor<*x!tf.resource<tensor<f32>>>)
    // CHECK: tf_device.return %[[WHILE]]#1 : tensor<f32>
    tf_device.return
  // CHECK: {device = "tpu0", launch_attr = "launch_attr"} : () -> tensor<f32>
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> ()
  // CHECK: "tf.AssignVariableOp"(%[[VH]], %[[LAUNCH]])
  // CHECK: return
  return
}
// CHECK: func @while_body(%[[BARG0:.*]]: tensor<i32>, %[[BARG1:.*]]: tensor<f32>)
func @while_body(%arg0: tensor<i32>, %arg1: tensor<*x!tf.resource<tensor<f32>>>, %arg2: tensor<*x!tf.resource<tensor<f32>>>)
    -> (tensor<i32>, tensor<*x!tf.resource<tensor<f32>>>, tensor<*x!tf.resource<tensor<f32>>>) {
  %read0 = "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf.resource<tensor<f32>>>) -> tensor<f32>
  // CHECK-NEXT: %[[ADD0:.*]] = "tf.AddV2"(%[[BARG1]], %[[BARG1]])
  %add0 = "tf.AddV2"(%read0, %read0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  "tf.AssignVariableOp"(%arg1, %add0) : (tensor<*x!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
  %read1 = "tf.ReadVariableOp"(%arg1) : (tensor<*x!tf.resource<tensor<f32>>>) -> tensor<f32>
  // CHECK-NEXT: %[[ADD1:.*]] = "tf.AddV2"(%[[ADD0]], %[[ADD0]])
  %add1 = "tf.AddV2"(%read1, %read1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  "tf.AssignVariableOp"(%arg1, %add1) : (tensor<*x!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
  // CHECK-NEXT: %[[DELTA:.*]] = "tf.Const"() {value = dense<-1> : tensor<i32>}
  %constant = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  // CHECK-NEXT: %[[ADD2:.*]] = "tf.AddV2"(%[[BARG0]], %[[DELTA]])
  %add2 = "tf.AddV2"(%arg0, %constant) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK-NEXT: return %[[ADD2]], %[[ADD1]]
  %id = "tf.Identity"(%arg2) : (tensor<*x!tf.resource<tensor<f32>>>) -> tensor<*x!tf.resource<tensor<f32>>>
  return %add2, %arg1, %id : tensor<i32>, tensor<*x!tf.resource<tensor<f32>>>, tensor<*x!tf.resource<tensor<f32>>>
}
// CHECK: func @while_cond(%[[CARG0:.*]]: tensor<i32>, %[[CARG1:.*]]: tensor<f32>)
func @while_cond(%arg0: tensor<i32>, %arg1: tensor<*x!tf.resource<tensor<f32>>>, %arg2: tensor<*x!tf.resource<tensor<f32>>>) -> tensor<i32> {
  // CHECK-NEXT: return %[[CARG0]]
  return %arg0 : tensor<i32>
}

// -----

// Tests that pass lifts resource reads from loop condition.

// CHECK-LABEL: func @launch_with_loop
func @launch_with_loop() -> () {
  // CHECK: %[[VH:.*]] = "tf.VarHandleOp"()
  %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource<tensor<f32>>>
  // CHECK: %[[READ:.*]] = "tf.ReadVariableOp"(%[[VH]])
  // CHECK: %[[LAUNCH:.*]] = "tf_device.launch"()
  "tf_device.launch"() ( {
    // CHECK: %[[WHILE:.*]] = "tf.While"(%[[READ]])
    %1 = "tf.While"(%0) {
      body = @while_body, cond = @while_cond, device = "", is_stateless = false,
      output_shapes = ["tfshape$"]}
         : (tensor<*x!tf.resource<tensor<f32>>>)
         -> (tensor<*x!tf.resource<tensor<f32>>>)
    // CHECK: tf_device.return %[[WHILE]] : tensor<f32>
    tf_device.return
  // CHECK: {device = "tpu0", launch_attr = "launch_attr"} : () -> tensor<f32>
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> ()
  // CHECK: "tf.AssignVariableOp"(%[[VH]], %[[LAUNCH]])
  // CHECK: return
  return
}
// CHECK: func @while_body(%[[BARG0:.*]]: tensor<f32>)
func @while_body(%arg0: tensor<*x!tf.resource<tensor<f32>>>) -> (tensor<*x!tf.resource<tensor<f32>>>) {
  // CHECK-NEXT: %[[CONST:.*]] = "tf.Const"()
  %constant = "tf.Const"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  "tf.AssignVariableOp"(%arg0, %constant) : (tensor<*x!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
  // CHECK-NEXT: return %[[CONST]]
  return %arg0 : tensor<*x!tf.resource<tensor<f32>>>
}
// CHECK: func @while_cond(%arg0: tensor<f32>)
func @while_cond(%arg0: tensor<*x!tf.resource<tensor<f32>>>) -> tensor<f32> {
  %id = "tf.Identity"(%arg0) : (tensor<*x!tf.resource<tensor<f32>>>) -> tensor<*x!tf.resource<tensor<f32>>>
  %read = "tf.ReadVariableOp"(%id) : (tensor<*x!tf.resource<tensor<f32>>>) -> tensor<f32>
  // CHECK-NEXT: return %[[CARG0]]
  return %read : tensor<f32>
}

// -----

// Tests that pass lifts read-only resource reads from loop, but does not add
// assign after the loop.

// CHECK-LABEL: func @launch_with_loop
func @launch_with_loop() -> () {
  // CHECK: %[[VH:.*]] = "tf.VarHandleOp"()
  %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource<tensor<f32>>>
  // CHECK: %[[READ:.*]] = "tf.ReadVariableOp"(%[[VH]])
  // CHECK: "tf_device.launch"()
  "tf_device.launch"() ( {
    // CHECK: %[[WHILE:.*]] = "tf.While"(%[[READ]])
    %1 = "tf.While"(%0) {
      body = @while_body, cond = @while_cond, device = "", is_stateless = false,
      output_shapes = ["tfshape$"]}
         : (tensor<*x!tf.resource<tensor<f32>>>)
         -> (tensor<*x!tf.resource<tensor<f32>>>)
    // CHECK: tf_device.return
    tf_device.return
  // CHECK: {device = "tpu0", launch_attr = "launch_attr"} : () -> ()
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> ()
  // CHECK-NOT: "tf.AssignVariableOp"
  // CHECK: return
  return
}
// CHECK: func @while_body(%[[BARG0:.*]]: tensor<f32>)
func @while_body(%arg0: tensor<*x!tf.resource<tensor<f32>>>) -> (tensor<*x!tf.resource<tensor<f32>>>) {
  // CHECK-NEXT: return %[[BARG0]]
  return %arg0 : tensor<*x!tf.resource<tensor<f32>>>
}
// CHECK: func @while_cond(%arg0: tensor<f32>)
func @while_cond(%arg0: tensor<*x!tf.resource<tensor<f32>>>) -> tensor<f32> {
  %read = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf.resource<tensor<f32>>>) -> tensor<f32>
  // CHECK-NEXT: return %[[CARG0]]
  return %read : tensor<f32>
}

// -----

// Tests that pass lifts resource reads from nested loops.

// CHECK-LABEL: func @launch_with_nested_loop
func @launch_with_nested_loop() -> () {
  // CHECK: %[[VH:.*]] = "tf.VarHandleOp"()
  %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource<tensor<f32>>>
  // CHECK: %[[VH_UNUSED:.*]] = "tf.VarHandleOp"()
  %1 = "tf.VarHandleOp"() {container = "c", shared_name = "v2"} : () -> tensor<*x!tf.resource<tensor<f32>>>
  // CHECK: %[[READ:.*]] = "tf.ReadVariableOp"(%[[VH]])
  // CHECK: %[[LAUNCH:.*]] = "tf_device.launch"()
  "tf_device.launch"() ( {
    // CHECK: %[[WHILE:.*]] = "tf.While"(%[[READ]])
    %2:2 = "tf.While"(%0, %1) {
      body = @while_body, cond = @while_cond, device = "", is_stateless = false,
      output_shapes = ["tfshape$", "tfshape$"]}
         : (tensor<*x!tf.resource<tensor<f32>>>, tensor<*x!tf.resource<tensor<f32>>>)
         -> (tensor<*x!tf.resource<tensor<f32>>>, tensor<*x!tf.resource<tensor<f32>>>)
    // CHECK: tf_device.return %[[WHILE]] : tensor<f32>
    tf_device.return
  // CHECK: {device = "tpu0", launch_attr = "launch_attr"} : () -> tensor<f32>
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> ()
  // CHECK: "tf.AssignVariableOp"(%[[VH]], %[[LAUNCH]])
  // CHECK: return
  return
}
// CHECK: func @while_body(%[[BARG0:.*]]: tensor<f32>)
func @while_body(%arg0: tensor<*x!tf.resource<tensor<f32>>>, %arg1: tensor<*x!tf.resource<tensor<f32>>>)
    -> (tensor<*x!tf.resource<tensor<f32>>>, tensor<*x!tf.resource<tensor<f32>>>) {
  // CHECK: %[[WHILE:.*]] = "tf.While"(%[[BARG0]])
  %0:2 = "tf.While"(%arg0, %arg1) {
    body = @while_body1, cond = @while_cond1, device = "", is_stateless = false,
    output_shapes = ["tfshape$", "tfshape$"]}
       : (tensor<*x!tf.resource<tensor<f32>>>, tensor<*x!tf.resource<tensor<f32>>>)
       -> (tensor<*x!tf.resource<tensor<f32>>>, tensor<*x!tf.resource<tensor<f32>>>)
  // CHECK-NEXT: return %[[WHILE]]
  return %0#0, %0#1 : tensor<*x!tf.resource<tensor<f32>>>, tensor<*x!tf.resource<tensor<f32>>>
}
// CHECK: func @while_cond(%arg0: tensor<f32>)
func @while_cond(%arg0: tensor<*x!tf.resource<tensor<f32>>>, %arg1: tensor<*x!tf.resource<tensor<f32>>>) -> tensor<f32> {
  %id = "tf.Identity"(%arg0) : (tensor<*x!tf.resource<tensor<f32>>>) -> tensor<*x!tf.resource<tensor<f32>>>
  %read = "tf.ReadVariableOp"(%id) : (tensor<*x!tf.resource<tensor<f32>>>) -> tensor<f32>
  // CHECK-NEXT: return %[[CARG0]]
  return %read : tensor<f32>
}
// CHECK: func @while_body1(%[[BARG0:.*]]: tensor<f32>)
func @while_body1(%arg0: tensor<*x!tf.resource<tensor<f32>>>, %arg1: tensor<*x!tf.resource<tensor<f32>>>)
    -> (tensor<*x!tf.resource<tensor<f32>>>, tensor<*x!tf.resource<tensor<f32>>>) {
  // CHECK-NEXT: %[[CONST:.*]] = "tf.Const"()
  %constant = "tf.Const"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  "tf.AssignVariableOp"(%arg0, %constant) : (tensor<*x!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
  // CHECK-NEXT: return %[[CONST]]
  return %arg0, %arg1 : tensor<*x!tf.resource<tensor<f32>>>, tensor<*x!tf.resource<tensor<f32>>>
}
// CHECK: func @while_cond1(%arg0: tensor<f32>)
func @while_cond1(%arg0: tensor<*x!tf.resource<tensor<f32>>>, %arg1: tensor<*x!tf.resource<tensor<f32>>>) -> tensor<f32> {
  %id = "tf.Identity"(%arg0) : (tensor<*x!tf.resource<tensor<f32>>>) -> tensor<*x!tf.resource<tensor<f32>>>
  %read = "tf.ReadVariableOp"(%id) : (tensor<*x!tf.resource<tensor<f32>>>) -> tensor<f32>
  // CHECK-NEXT: return %[[CARG0]]
  return %read : tensor<f32>
}

// -----

// Tests that pass reports error on non-aliasing while input/output resources.

func @launch_with_loop() -> () {
  %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource<tensor<f32>>>
  "tf_device.launch"() ( {
    %1 = "tf.While"(%0) {
      body = @while_body, cond = @while_cond, device = "", is_stateless = false,
      output_shapes = ["tfshape$"]}
         : (tensor<*x!tf.resource<tensor<f32>>>) -> (tensor<*x!tf.resource<tensor<f32>>>)
    tf_device.return
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> ()
  return
}
func @while_body(%arg0: tensor<*x!tf.resource<tensor<f32>>>) -> (tensor<*x!tf.resource<tensor<f32>>>) {
  %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v2"} : () -> tensor<*x!tf.resource<tensor<f32>>>
  // expected-error @+1 {{Resource used in while loop is only supported when the resource input and output alias each other in the loop body}}
  return %0 : tensor<*x!tf.resource<tensor<f32>>>
}
func @while_cond(%arg0: tensor<*x!tf.resource<tensor<f32>>>) -> tensor<f32> {
  %read = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf.resource<tensor<f32>>>) -> tensor<f32>
  return %read : tensor<f32>
}

// -----

// Tests that pass reports error on unsupported ops in loop body.

func @launch_with_loop() -> () {
  %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource<tensor<f32>>>
  "tf_device.launch"() ( {
    %1 = "tf.While"(%0) {
      body = @while_body, cond = @while_cond, device = "", is_stateless = false,
      output_shapes = ["tfshape$"]}
         : (tensor<*x!tf.resource<tensor<f32>>>) -> (tensor<*x!tf.resource<tensor<f32>>>)
    tf_device.return
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> ()
  return
}
func @while_body(%arg0: tensor<*x!tf.resource<tensor<f32>>>) -> (tensor<*x!tf.resource<tensor<f32>>>) {
  // expected-error @+1 {{Found unsupported operations on resource.}}
  "tf._UnknownOp"(%arg0) : (tensor<*x!tf.resource<tensor<f32>>>) -> ()
  return %arg0 : tensor<*x!tf.resource<tensor<f32>>>
}
func @while_cond(%arg0: tensor<*x!tf.resource<tensor<f32>>>) -> tensor<f32> {
  %read = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf.resource<tensor<f32>>>) -> tensor<f32>
  return %read : tensor<f32>
}

// -----

// Tests that pass reports error on unsupported ops in loop cond.

func @launch_with_loop() -> () {
  %0 = "tf.VarHandleOp"() {container = "c", shared_name = "v"} : () -> tensor<*x!tf.resource<tensor<f32>>>
  "tf_device.launch"() ( {
    %1 = "tf.While"(%0) {
      body = @while_body, cond = @while_cond, device = "", is_stateless = false,
      output_shapes = ["tfshape$"]}
         : (tensor<*x!tf.resource<tensor<f32>>>) -> (tensor<*x!tf.resource<tensor<f32>>>)
    tf_device.return
  }) {device = "tpu0", launch_attr = "launch_attr"} : () -> ()
  return
}
func @while_body(%arg0: tensor<*x!tf.resource<tensor<f32>>>) -> (tensor<*x!tf.resource<tensor<f32>>>) {
  %constant = "tf.Const"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  "tf.AssignVariableOp"(%arg0, %constant) : (tensor<*x!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
  return %arg0 : tensor<*x!tf.resource<tensor<f32>>>
}
func @while_cond(%arg0: tensor<*x!tf.resource<tensor<f32>>>) -> tensor<f32> {
  %read = "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf.resource<tensor<f32>>>) -> tensor<f32>
  %constant = "tf.Const"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  // expected-error @+1 {{Found resource write in loop condition.}}
  "tf.AssignVariableOp"(%arg0, %constant) : (tensor<*x!tf.resource<tensor<f32>>>, tensor<f32>) -> ()
  return %read : tensor<f32>
}
