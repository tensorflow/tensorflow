// RUN: tf-tfrt-opt -split-input-file -tf-mlrt-parallelization %s | FileCheck %s --dump-input=fail --dump-input-filter=all

// CHECK-LABEL: func private @main_stream_{{[0-9]*}}
// CHECK-SAME: ({{%.*}}: tensor<i32>, [[PROMISE:%.*]]: !mlrt.promise)
// CHECK: tf.Sub
// CHECK: tf.Sub
// CHECK: tf.Sub
// CHECK: [[RES:%.*]] = "tf.Sub"
// CHECK: "tf_mlrt.tf_promise"([[PROMISE]], [[RES]])
// CHECK: return

// CHECK-LABEL: func @main
// CHECK: [[PROMISE:%.*]], [[FUTURE:%.*]] = "tf_mlrt.allocate_futures"
// CHECK: [[HANDLE:%.*]] = mlrt.async({{%.*}}, [[PROMISE]])
// CHECK-SAME: callee = @main_stream_{{[0-9]*}}
// CHECK: tf.AddV2
// CHECK: tf.AddV2
// CHECK: tf.AddV2
// CHECK: [[x:%.*]] = "tf.AddV2"
// CHECK: [[y:%.*]] = "tf_mlrt.tf_await"([[FUTURE]])
// CHECK: [[RES:%.*]] = "tf.AddV2"([[x]], [[y]])
// CHECK: mlrt.await_handle [[HANDLE]]
// CHECK: return [[RES]]

func.func @main(%a: tensor<i32>, %b: tensor<i32>) -> tensor<i32> {

  %a0 = "tf.AddV2"(%a, %a) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %a1 = "tf.AddV2"(%a0, %a) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %a2 = "tf.AddV2"(%a1, %a) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %a3 = "tf.AddV2"(%a2, %a) : (tensor<i32>, tensor<i32>) -> tensor<i32>

  %b0 = "tf.Sub"(%b, %b) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %b1 = "tf.Sub"(%b0, %b) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %b2 = "tf.Sub"(%b1, %b) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %b3 = "tf.Sub"(%b2, %b) : (tensor<i32>, tensor<i32>) -> tensor<i32>

  %c = "tf.AddV2"(%a3, %b3) : (tensor<i32>, tensor<i32>) -> tensor<i32>

  func.return %c : tensor<i32>
}

// -----

// Test merging child streams

// CHECK-LABEL: func private @main_stream_{{[0-9]*}}
// CHECK-SAME: ({{%.*}}: tensor<i32>, {{%.*}}: tensor<i32>, [[PROMISE:%.*]]: !mlrt.promise)
// CHECK: tf.Sub
// CHECK: tf.Sub
// CHECK: tf.Sub
// CHECK: [[RES:%.*]] = "tf.Sub"
// CHECK: "tf_mlrt.tf_promise"([[PROMISE]], [[RES]])
// CHECK: return

// CHECK-LABEL: func @main
// CHECK: [[PROMISE:%.*]], [[FUTURE:%.*]] = "tf_mlrt.allocate_futures"
// CHECK: tf.AddV2
// CHECK: tf.AddV2
// CHECK: tf.AddV2
// CHECK: [[VALUE:%.*]] = "tf.AddV2"
// CHECK: [[HANDLE:%.*]] = mlrt.async([[VALUE]], {{%.*}}, [[PROMISE]])
// CHECK-SAME: callee = @main_stream_{{[0-9]*}}
// CHECK: tf.AddV2
// CHECK: tf.AddV2
// CHECK: tf.AddV2
// CHECK: [[x:%.*]] = "tf.AddV2"
// CHECK: [[y:%.*]] = "tf_mlrt.tf_await"([[FUTURE]])
// CHECK: [[RES:%.*]] = "tf.AddV2"([[x]], [[y]])
// CHECK: mlrt.await_handle [[HANDLE]]
// CHECK: return [[RES]]

func.func @main(%a: tensor<i32>, %b: tensor<i32>) -> tensor<i32> {

  %a0 = "tf.AddV2"(%a, %a) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %a1 = "tf.AddV2"(%a0, %a) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %a2 = "tf.AddV2"(%a1, %a) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %a3 = "tf.AddV2"(%a2, %a) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %a4 = "tf.AddV2"(%a3, %a) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %a5 = "tf.AddV2"(%a4, %a) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %a6 = "tf.AddV2"(%a5, %a) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %a7 = "tf.AddV2"(%a6, %a) : (tensor<i32>, tensor<i32>) -> tensor<i32>

  %b0 = "tf.Sub"(%a3, %b) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %b1 = "tf.Sub"(%b0, %b) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %b2 = "tf.Sub"(%b1, %b) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %b3 = "tf.Sub"(%b2, %b) : (tensor<i32>, tensor<i32>) -> tensor<i32>

  %c = "tf.AddV2"(%a7, %b3) : (tensor<i32>, tensor<i32>) -> tensor<i32>

  func.return %c : tensor<i32>
}

// -----

// Test side-effecting ops

// CHECK-LABEL: func private @main_stream_{{[0-9]*}}
// CHECK-SAME: ([[ARG:%.*]]: tensor<i32>, [[FUTURE:%.*]]: !mlrt.future, [[CONTROL_PROMISE:%.*]]: !mlrt.promise)
// CHECK: [[HANDLE:%.*]] = "tf_mlrt.tf_await"([[FUTURE]])
// CHECK: "tf.AssignVariableOp"([[HANDLE]], [[ARG]])
// CHECK-NEXT: mlrt.promise_control [[CONTROL_PROMISE]]

// CHECK-LABEL: func private @main_stream_{{[0-9]*}}
// CHECK-SAME: ({{%.*}}: tensor<i32>, {{%.*}}: tensor<i32>, [[FUTURE:%.*]]: !mlrt.future, [[PROMISE:%.*]]: !mlrt.promise)
// CHECK: tf.Sub
// CHECK: tf.Sub
// CHECK: tf.Sub
// CHECK: [[V:%.*]] = "tf_mlrt.tf_await"([[FUTURE]])
// CHECK-NEXT: [[RES:%.*]] = "tf.Sub"({{%.*}}, [[V]])
// CHECK: "tf_mlrt.tf_promise"([[PROMISE]], [[RES]])
// CHECK: return

// CHECK-LABEL: func private @main_stream_{{[0-9]*}}
// CHECK-SAME: ([[CONTROL_FUTURE:%.*]]: !mlrt.future, [[PROMISE:%.*]]: !mlrt.promise, [[PROMISE_HANDLE:%.*]]: !mlrt.promise)
// CHECK: [[HANDLE:%.*]] = "tf.VarHandleOp"
// CHECK-NEXT: "tf_mlrt.tf_promise"([[PROMISE_HANDLE]], [[HANDLE]])
// CHECK: mlrt.await_control [[CONTROL_FUTURE]]
// CHECK-NEXT: [[V:%.*]] = "tf.ReadVariableOp"([[HANDLE]])
// CHECK: "tf_mlrt.tf_promise"([[PROMISE]], [[V]])

// CHECK-LABEL: func @main
// CHECK: [[PROMISE:%.*]]:3, [[FUTURE:%.*]]:3 = "tf_mlrt.allocate_futures"
// CHECK: [[CONTROL_PROMISE:%.*]], [[CONTROL_FUTURE:%.*]] = "mlrt.allocate_control_futures"
// CHECK: [[ASYNC_HANDLE_0:%.*]] = mlrt.async([[CONTROL_FUTURE]], [[PROMISE]]#0, [[PROMISE]]#1)
// CHECK-SAME: callee = @main_stream_{{[0-9]*}}
// CHECK: [[ASYNC_HANDLE_1:%.*]] = mlrt.async({{%.*}}, {{%.*}}, [[FUTURE]]#0, [[PROMISE]]#2)
// CHECK-SAME: callee = @main_stream_{{[0-9]*}}
// CHECK: tf.AddV2
// CHECK: tf.AddV2
// CHECK: tf.AddV2
// CHECK: [[x:%.*]] = "tf.AddV2"
// CHECK: [[ASYNC_HANDLE_2:%.*]] = mlrt.async([[x]], [[FUTURE]]#1, [[CONTROL_PROMISE]])
// CHECK-SAME: callee = @main_stream_{{[0-9]*}}
// CHECK: [[y:%.*]] = "tf_mlrt.tf_await"([[FUTURE]]#2)
// CHECK: [[RES:%.*]] = "tf.AddV2"([[x]], [[y]])
// CHECK: mlrt.await_handle [[ASYNC_HANDLE_0]]
// CHECK-NEXT: mlrt.await_handle [[ASYNC_HANDLE_1]]
// CHECK-NEXT: mlrt.await_handle [[ASYNC_HANDLE_2]]
// CHECK-NEXT: return [[RES]]

func.func @main(%a: tensor<i32>, %b: tensor<i32>) -> tensor<i32> {
  %handle = "tf.VarHandleOp"() {container = "", shared_name = "var"} : () -> tensor<!tf_type.resource_handle<tensor<i32>>>

  %a0 = "tf.AddV2"(%a, %a) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %a1 = "tf.AddV2"(%a0, %a) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %a2 = "tf.AddV2"(%a1, %a) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %a3 = "tf.AddV2"(%a2, %a) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  "tf.AssignVariableOp"(%handle, %a3) : (tensor<!tf_type.resource<tensor<i32>>>, tensor<i32>) -> ()

  %b0 = "tf.Sub"(%a, %b) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %b1 = "tf.Sub"(%b0, %b) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %b2 = "tf.Sub"(%b1, %b) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %var = "tf.ReadVariableOp"(%handle) : (tensor<!tf_type.resource<tensor<i32>>>) -> tensor<i32>
  %b3 = "tf.Sub"(%b2, %var) : (tensor<i32>, tensor<i32>) -> tensor<i32>

  %c = "tf.AddV2"(%a3, %b3) : (tensor<i32>, tensor<i32>) -> tensor<i32>

  func.return %c : tensor<i32>
}

// -----

// Test multiple promises and futures

// CHECK-LABEL: func private @main_stream_1
// CHECK: mlrt.await_control
// CHECK: "tf.DummySideEffecting"() {id = 4
// CHECK: return

// CHECK-LABEL: func private @main_stream_2
// CHECK: mlrt.await_control
// CHECK: "tf.DummySideEffecting"() {id = 3
// CHECK: mlrt.promise_control
// CHECK: return

// CHECK-LABEL: func private @main_stream_3
// CHECK: mlrt.await_control
// CHECK: "tf.DummySideEffecting"() {id = 2
// CHECK: mlrt.promise_control
// CHECK: return

// CHECK-LABEL: func private @main_stream_4
// CHECK: "tf.DummySideEffecting"() {id = 1
// CHECK: mlrt.promise_control
// CHECK: return

// CHECK-LABEL: func @main()
// CHECK: [[PROMISES:%.*]]:3, [[FUTURES:%.*]]:3 = "mlrt.allocate_control_futures"
// CHECK: mlrt.async([[PROMISES]]#2) {callee = @main_stream_4
// CHECK: mlrt.async([[FUTURES]]#2, [[PROMISES]]#1) {callee = @main_stream_3
// CHECK: mlrt.async([[FUTURES]]#1, [[PROMISES]]#0) {callee = @main_stream_2
// CHECK: mlrt.async([[FUTURES]]#0) {callee = @main_stream_1
// CHECK: mlrt.await_handle
// CHECK: mlrt.await_handle
// CHECK: mlrt.await_handle
// CHECK: mlrt.await_handle

func.func @main() {
  "tf.DummySideEffecting"() {id = 1} : () -> ()
  "tf.DummySideEffecting"() {id = 2} : () -> ()
  "tf.DummySideEffecting"() {id = 3} : () -> ()
  "tf.DummySideEffecting"() {id = 4} : () -> ()
  func.return
}

// -----

// Test correctness when there are both data and control promises in a stream function.

// CHECK-LABEL: func private @main_stream_1
// CHECK-SAME: ([[PROMISE:%.*]]: !mlrt.promise, [[CONTROL_PROMISE:%.*]]: !mlrt.promise)
// CHECK: tf.DummySideEffecting
// CHECK: "tf_mlrt.tf_promise"([[PROMISE]]
// CHECK: mlrt.promise_control [[CONTROL_PROMISE]]

func.func @main() -> tensor<i32> {
  %v = "tf.DummySideEffecting"() {id = 1} : () -> tensor<i32>

  %w = "tf.DummySideEffecting"() {id = 2} : () -> tensor<i32>
  %r = "tf.AddV2"(%w, %v) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %r : tensor<i32>
}

// -----

// Test inputs to the child streams are merged to the parent streams

// CHECK-LABEL: func private @main_stream_1
// CHECK-SAME: ([[INPUT0:%.*]]: tensor<i32>, [[INPUT1:%.*]]: tensor<i32>
// CHECK: tf.Sub
// CHECK: tf.Sub
// CHECK: mlrt.async({{%.*}}, [[INPUT1]]

// CHECK-LABEL: func @main
func.func @main(%a: tensor<i32>, %b: tensor<i32>) -> tensor<i32> {

  %a0 = "tf.AddV2"(%a, %a) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %a1 = "tf.AddV2"(%a0, %a) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %a2 = "tf.AddV2"(%a1, %a) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %a3 = "tf.AddV2"(%a2, %a) : (tensor<i32>, tensor<i32>) -> tensor<i32>

  %b0 = "tf.Sub"(%b, %b) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %b1 = "tf.Sub"(%b0, %b) : (tensor<i32>, tensor<i32>) -> tensor<i32>

  %c = "tf.AddV2"(%b1, %a) : (tensor<i32>, tensor<i32>) -> tensor<i32>

  %b2 = "tf.Sub"(%b1, %b) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %b3 = "tf.Sub"(%b2, %b) : (tensor<i32>, tensor<i32>) -> tensor<i32>

  %d = "tf.AddN"(%a3, %b3, %c) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %d : tensor<i32>
}

// -----

// Test that constants are copied instead of using promise/await.

// CHECK-LABEL: func private @main_stream_1
// CHECK-SAME: ({{%.*}}: tensor<i32>, [[PROMISE:%.*]]: !mlrt.promise)
// CHECK: tf._TfrtGetResource
// CHECK: tf.Sub
// CHECK: tf.Sub
// CHECK: tf.Sub
// CHECK: [[RES:%.*]] = "tf.Sub"
// CHECK: "tf_mlrt.tf_promise"([[PROMISE]], [[RES]])
// CHECK: return

// CHECK-NOT: func private @main_stream

// CHECK-LABEL: func @main
// CHECK: [[PROMISE:%.*]], [[FUTURE:%.*]] = "tf_mlrt.allocate_futures"
// CHECK-NEXT: [[HANDLE:%.*]] = mlrt.async({{%.*}}, [[PROMISE]])
// CHECK-SAME: callee = @main_stream_1
// CHECK: tf._TfrtGetResource
// CHECK: tf.AddV2
// CHECK: tf.AddV2
// CHECK: tf.AddV2
// CHECK: [[x:%.*]] = "tf.AddV2"
// CHECK: [[y:%.*]] = "tf_mlrt.tf_await"([[FUTURE]])
// CHECK: [[RES:%.*]] = "tf.AddV2"([[x]], [[y]])
// CHECK: mlrt.await_handle [[HANDLE]]
// CHECK: return [[RES]]

func.func @main(%a: tensor<i32>, %b: tensor<i32>) -> tensor<i32> {

  %c0 = "tf._TfrtGetResource"() {indices = [0], shared_name = [""], container = [""]} : () -> (tensor<i32>)

  %a0 = "tf.AddV2"(%a, %c0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %a1 = "tf.AddV2"(%a0, %c0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %a2 = "tf.AddV2"(%a1, %c0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %a3 = "tf.AddV2"(%a2, %c0) : (tensor<i32>, tensor<i32>) -> tensor<i32>

  %b0 = "tf.Sub"(%b, %c0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %b1 = "tf.Sub"(%b0, %c0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %b2 = "tf.Sub"(%b1, %c0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %b3 = "tf.Sub"(%b2, %c0) : (tensor<i32>, tensor<i32>) -> tensor<i32>

  %c = "tf.AddV2"(%a3, %b3) : (tensor<i32>, tensor<i32>) -> tensor<i32>

  func.return %c : tensor<i32>
}

// -----

// Test that constants private to a stream are still handled properly when we are copying shared constants.

// CHECK-LABEL: func private @main_stream_1
// CHECK: [[r:%.*]] = "tf._TfrtGetResource"
// CHECK-SAME: indices = [1]
// CHECK: "tf.DummySideEffecting"([[r]])

// CHECK-LABEL: func private @main_stream_2
// CHECK: [[r:%.*]] = "tf._TfrtGetResource"
// CHECK-SAME: indices = [0]
// CHECK: "tf.DummySideEffecting"([[r]])

// CHECK-LABEL: func @main

func.func @main(%a: tensor<i32>, %b: tensor<i32>) -> () {

  %c0 = "tf._TfrtGetResource"() {indices = [0], shared_name = [""], container = [""]} : () -> (tensor<i32>)
  "tf.DummySideEffecting"(%c0) : (tensor<i32>) -> ()

  %c1 = "tf._TfrtGetResource"() {indices = [1], shared_name = [""], container = [""]} : () -> (tensor<i32>)
  "tf.DummySideEffecting"(%c1) : (tensor<i32>) -> ()

  func.return
}

// -----

// Test that streams with no args but side-effecting ops are still created properly

// CHECK-LABEL: func private @main_stream_1()
// CHECK: [[r:%.*]] = "tf._TfrtGetResource"
// CHECK-SAME: indices = [0]
// CHECK: "tf.DummySideEffecting"([[r]])

// CHECK-LABEL: func @main

func.func @main(%a: tensor<i32>, %b: tensor<i32>) -> () {
  %c0 = "tf._TfrtGetResource"() {indices = [0], shared_name = [""], container = [""]} : () -> (tensor<i32>)
  "tf.DummySideEffecting"(%c0) : (tensor<i32>) -> ()
  func.return
}

// -----

// Test control deps of tf.Assert is skipped.

// CHECK-LABEL: func.func private @skip_assert_stream_3(
// CHECK-NOT: mlrt.await_control
// CHECK: tf.Assert
// CHECK-NOT: mlrt.promise_control
// CHECK: return

// CHECK-LABEL: func.func private @skip_assert_stream_2(
// CHECK-NOT: mlrt.await_control
// CHECK: tf.Assert
// CHECK-NOT: mlrt.promise_control
// CHECK: return

func.func @skip_assert(%key: tensor<!tf_type.string>) -> (tensor<i64>, tensor<i64>) {
  %error_message = "tf.Const"() {value = dense<"error"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %default = "tf.Const"() {value = dense<-1> : tensor<i64>} : () -> tensor<i64>
  %handle = "tf.HashTableV2"() {container = "", device = "/job:localhost/replica:0/task:0/device:CPU:0", key_dtype = !tf_type.string, shared_name = "hash_table", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>


  %keys = "tf.Const"() {value = dense<["a", "b", "c", "d"]> : tensor<4x!tf_type.string>} : () -> tensor<4x!tf_type.string>
  %values = "tf.Const"() {value = dense<[1, 2, 3, 4]> : tensor<4xi64>} : () -> tensor<4xi64>
  "tf.LookupTableImportV2"(%handle, %keys, %values) {device = ""} : (tensor<!tf_type.resource>, tensor<4x!tf_type.string>, tensor<4xi64>) -> ()
  %value0 = "tf.LookupTableFindV2"(%handle, %key, %default) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<!tf_type.resource>, tensor<!tf_type.string>, tensor<i64>) -> tensor<i64>
  %cond = "tf.Equal"(%value0, %default) {device = "/job:localhost/replica:0/task:0/device:CPU:0", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  "tf.Assert"(%cond, %error_message) {device = "/job:localhost/replica:0/task:0/device:CPU:0", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>) -> ()
  "tf.Assert"(%cond, %error_message) {device = "/job:localhost/replica:0/task:0/device:CPU:0", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>) -> ()
  %value1 = "tf.LookupTableFindV2"(%handle, %key, %default) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<!tf_type.resource>, tensor<!tf_type.string>, tensor<i64>) -> tensor<i64>
  func.return %value0, %value1 : tensor<i64>, tensor<i64>
}
