// RUN: tf-opt -tfe-legalize-tfg %s | FileCheck %s

module  {
  tfg.graph #tf_type.version<producer = 27, min_consumer = 0> {
    // CHECK: tf_executor.Enter
    // CHECK: {{%.*}}, %[[TOKEN:.*]], {{%.*}} = tf_executor.NextIteration.Source
    // CHECK: {{%.*}}, {{%.*}}, %[[CONTROL:.*]] = tf_executor.Merge
    // CHECK: tf_executor.island(%[[CONTROL]]) wraps "tf.Const"()
    // CHECK: tf_executor.LoopCond
    // CHECK: tf_executor.Switch
    // CHECK: tf_executor.NextIteration.Sink[%[[TOKEN]]]
    // CHECK: tf_executor.Exit
    %Const, %ctl = Const name("Const") {dtype = i32, value = dense<0> : tensor<i32>} : () -> (tensor<i32>)
    %Enter, %ctl_0 = Enter(%Const) name("while/Enter") {T = i32, frame_name = "while/while_context", is_constant = false, parallel_iterations = 10 : i64} : (tensor<i32>) -> (tensor<*xi32>)
    %NextIteration, %ctl_1 = NextIteration(%Add) name("while/NextIteration") {T = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
    %Merge:2, %ctl_2 = Merge(%Enter, %NextIteration) name("while/Merge") {N = 2 : i64, T = i32} : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>)
    %Const_3, %ctl_4 = Const [%ctl_2] name("while/Less/y") {dtype = i32, value = dense<10> : tensor<i32>} : () -> (tensor<i32>)
    %Less, %ctl_5 = Less(%Merge#0, %Const_3) name("while/Less") {T = i32} : (tensor<*xi32>, tensor<i32>) -> (tensor<*xi1>)
    %LoopCond, %ctl_6 = LoopCond(%Less) name("while/LoopCond") : (tensor<*xi1>) -> (tensor<*xi1>)
    %Switch:2, %ctl_7 = Switch(%Merge#0, %LoopCond) name("while/Switch") {T = i32, _class = ["loc:@while/Merge"]} : (tensor<*xi32>, tensor<*xi1>) -> (tensor<*xi32>, tensor<*xi32>)
    %Identity, %ctl_8 = Identity(%Switch#1) name("while/Identity") {T = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
    %Const_9, %ctl_10 = Const [%ctl_8] name("while/Add/y") {dtype = i32, value = dense<1> : tensor<i32>} : () -> (tensor<i32>)
    %Add, %ctl_11 = Add(%Identity, %Const_9) name("while/Add") {T = i32} : (tensor<*xi32>, tensor<i32>) -> (tensor<*xi32>)
    %Exit, %ctl_12 = Exit(%Switch#0) name("while/Exit") {T = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
  }
}
