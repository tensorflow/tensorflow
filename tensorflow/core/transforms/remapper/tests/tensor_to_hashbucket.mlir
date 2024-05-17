// RUN: tfg-transforms-opt --tfg-remapper %s | FileCheck %s

// -----

// CHECK-LABEL: tfg.func @tensor_to_hashbucket_test
tfg.func @tensor_to_hashbucket_test() {
  // CHECK: %[[PLACEHOLDER:.*]], {{.*}} name("input")
  %Placeholder, %ctl = Placeholder device("/device:CPU:0") name("input") {dtype = i8, shape = #tf_type.shape<8x32x32x3>} : () -> (tensor<*xi8>)
  // CHECK: %[[ASSTRING:.*]], {{.*}} name("to_string")
  %AsString, %ctl_0 = AsString(%Placeholder) device("/device:CPU:0") name("to_string") {T = i8, fill = "", precision = -1 : i64, scientific = false, shortest = false, width = -1 : i64} : (tensor<*xi8>) -> (tensor<*x!tf_type.string>)
  // CHECK: _TensorToHashBucketFast(%[[PLACEHOLDER:.*]]) {{.*}} name("to_bucket") {T = {{.*}}, num_buckets = {{.*}}}
  %StringToHashBucketFast, %ctl_1 = StringToHashBucketFast(%AsString) device("/device:CPU:0") name("to_bucket") {num_buckets = 100 : i64} : (tensor<*x!tf_type.string>) -> (tensor<*xi64>)
  return
}
