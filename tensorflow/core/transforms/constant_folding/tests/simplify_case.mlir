// RUN: tfg-transforms-opt -constant-folding %s | FileCheck %s

module  {
  tfg.graph #tf_type.version<producer = 1010, min_consumer = 0> {
    %Const, %ctl = Const device("/job:localhost/replica:0/task:0/device:CPU:0") name("one") {dtype = i32, value = dense<1> : tensor<i32>} : () -> (tensor<i32>)
    // CHECK: %[[PLACEHOLDER:.*]], {{.*}} = Placeholder {{.*}} name("x")
    %Placeholder, %ctl_0 = Placeholder device("/job:localhost/replica:0/task:0/device:CPU:0") name("x") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    // CHECK: PartitionedCall(%[[PLACEHOLDER]]) {{.*}} name("case")
    %Case, %ctl_1 = Case(%Const, %Placeholder) device("/job:localhost/replica:0/task:0/device:CPU:0") name("case") {Tin = [f32], Tout = [f32], branches = [#tf_type.func<@XTimesTwo, {T = f32}>, #tf_type.func<@NonZero, {T = f32}>], output_shapes = [#tf_type.shape<>, #tf_type.shape<*>]} : (tensor<i32>, tensor<*xf32>) -> (tensor<*xf32>)
    %Identity, %ctl_2 = Identity(%Case) device("/job:localhost/replica:0/task:0/device:CPU:0") name("y") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
  }
  tfg.func generic @XTimesTwo(%x: !tf_type.tensor {tfg.name = "x", tfg.type_attr = "T"})
       -> (!tf_type.tensor {tfg.name = "y", tfg.type_attr = "T"})
   attributes {tfg.func_attrs = {T = {allowed_values = [f32, f64, i32, i64], type = "type"}}} {
    %Const, %ctl = Const name("two") {dtype = i64, value = dense<2> : tensor<i64>} : () -> (!tf_type.tensor)
    %0 = get_result(%Const) "output" : 0
    %Cast, %ctl_0 = Cast(%0) name("scale") {DstT = #tf_type.placeholder<"T">, SrcT = i64} : (!tf_type.tensor) -> (!tf_type.tensor)
    %1 = get_result(%Cast) "y" : 0
    %Mul, %ctl_1 = Mul(%x, %1) name("y") {T = #tf_type.placeholder<"T">} : (!tf_type.tensor, !tf_type.tensor) -> (!tf_type.tensor)
    %2 = get_result(%Mul) "z" : 0
    return(%2) : !tf_type.tensor
  }
  tfg.func generic @NonZero(%x: !tf_type.tensor {tfg.name = "x", tfg.type_attr = "T"})
       -> (!tf_type.tensor {tfg.name = "y", tfg.type_attr = "T"})
   attributes {tfg.func_attrs = {T = {allowed_values = [f32, f64, i32, i64, !tf_type.string], type = "type"}}} {
    %Identity, %ctl = Identity(%x) name("y") {T = #tf_type.placeholder<"T">} : (!tf_type.tensor) -> (!tf_type.tensor)
    %0 = get_result(%Identity) "output" : 0
    return(%0) : !tf_type.tensor
  }
}
