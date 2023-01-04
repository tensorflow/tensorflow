// RUN: tf-opt -tfe-legalize-tfg %s | FileCheck %s

module  {
  tfg.graph #tf_type.version<producer = 62, min_consumer = 12> {
    %Const, %ctl = Const name("Constant") {dtype = i32, value = dense<0> : tensor<i32>} : () -> (tensor<i32>)
    %foo, %ctl_0 = foo(%Const) name("_tf.foo") : (tensor<i32>) -> (tensor<*xi32>)
  }
  tfg.func @foo(%arg: tensor<*xi32> {tfg.name = "arg"})
       -> (tensor<*xi32> {tfg.dtype = i32, tfg.name = "return_value"})
   {
    // CHECK-NOT: ^bb0{{.*}}, %arg2: !tf_type.control
    %Const, %ctl = Const [%arg.ctl] name("test") {dtype = i32, value = dense<0> : tensor<i32>} : () -> (tensor<i32>)
    %Enter, %ctl_0 = Enter(%Const) [%arg.ctl] name("while/Enter") {T = i32, frame_name = "while/while_context", is_constant = false, parallel_iterations = 10 : i64} : (tensor<i32>) -> (tensor<*xi32>)
    %NextIteration, %ctl_1 = NextIteration(%Const) [%arg.ctl] name("while/NextIteration") {T = i32} : (tensor<i32>) -> (tensor<*xi32>)
    %Merge:2, %ctl_2 = Merge(%Enter, %NextIteration) [%arg.ctl] name("while/Merge") {N = 2 : i64, T = i32} : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<i32>)
    %Less, %ctl_5 = Less(%Merge#0, %Const) [%arg.ctl] name("while/Less") {T = i32} : (tensor<*xi32>, tensor<i32>) -> (tensor<*xi1>)
    %LoopCond, %ctl_6 = LoopCond(%Less) [%arg.ctl] name("while/LoopCond") : (tensor<*xi1>) -> (tensor<*xi1>)
    %Switch:2, %ctl_7 = Switch(%Merge#0, %LoopCond) [%arg.ctl] name("while/Switch") {T = i32, _class = ["loc:@while/Merge"]} : (tensor<*xi32>, tensor<*xi1>) -> (tensor<*xi32>, tensor<*xi32>)
    %Exit, %ctl_12 = Exit(%Switch#0) [%arg.ctl] name("while/Exit") {T = i32} : (tensor<*xi32>) -> (tensor<*xi32>)
    return(%arg) : tensor<*xi32>
  }
}
