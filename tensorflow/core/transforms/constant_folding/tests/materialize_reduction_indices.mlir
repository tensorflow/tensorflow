// RUN: tfg-transforms-opt -tfg-constant-folding %s | FileCheck %s

module {
  tfg.func @test() {
    %Placeholder, %ctl = Placeholder name("input") {dtype = f32, shape = #tf_type.shape<?x?>} : () -> (tensor<?x?xf32>)
    // CHECK: {{.*}}, %[[CTRL:.*]] = {{.*}} name("indices")
    %Placeholder_0, %ctl_1 = Placeholder name("indices") {dtype = i32, shape = #tf_type.shape<2>} : () -> (tensor<2xi32>)
    // CHECK: %[[CONST:.*]], {{.*}} Const [%[[CTRL]]]
    // CHECK: Sum({{.*}}, %[[CONST]]) name("sum")
    %Sum, %ctl_2 = Sum(%Placeholder, %Placeholder_0) name("sum") {T = f32, Tidx = i32, keep_dims = false} : (tensor<?x?xf32>, tensor<2xi32>) -> (tensor<*xf32>)
    return
  }
}
