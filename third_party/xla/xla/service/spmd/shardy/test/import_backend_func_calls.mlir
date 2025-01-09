// RUN: sdy_opt %s -xla-sdy-import-backend-func-calls 2>&1 | FileCheck %s

sdy.mesh @mesh = #sdy.mesh<["x"=2, "y"=2]>

// CHECK-LABEL: func @no_out_shardings
func.func @no_out_shardings(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %[[NC:.*]] = sdy.named_computation<"foo">(%arg0) (%arg1: tensor<8x2xi32>) {
  // CHECK-NEXT:   %[[MULT:.*]] = stablehlo.multiply %arg1, %arg1 {mhlo.frontend_attributes = {_xla_compute_type = "host"}} : tensor<8x2xi32>
  // CHECK-NEXT:   sdy.return %[[MULT]] : tensor<8x2xi32>
  // CHECK-NEXT: } {mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"},
  // CHECK-SAME:    random_attr = "random_value"}
  // CHECK-SAME: (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[MOVE_TO_HOST:.*]] = stablehlo.custom_call @MoveToHost(%[[NC]]) {backend_config = ""} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[MOVE_TO_HOST]] : tensor<8x2xi32>
  %0 = call @foo(%arg0) {random_attr = "random_value", mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"}} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = stablehlo.custom_call @MoveToHost(%0) {backend_config = ""} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

func.func private @foo(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// CHECK-LABEL: func @out_shardings
func.func @out_shardings(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %[[NC:.*]] = sdy.named_computation<"bar">(%arg0) out_shardings=[<@mesh, [{"x"}, {"y"}]>] (%arg1: tensor<8x2xi32>) {
  // CHECK-NEXT:   %[[MULT:.*]] = stablehlo.multiply %arg1, %arg1 {mhlo.frontend_attributes = {_xla_compute_type = "host"}} : tensor<8x2xi32>
  // CHECK-NEXT:   sdy.return %[[MULT]] : tensor<8x2xi32>
  // CHECK-NEXT: } {mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"},
  // CHECK-SAME:    random_attr = "random_value"}
  // CHECK-SAME: (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[MOVE_TO_HOST:.*]] = stablehlo.custom_call @MoveToHost(%[[NC]]) {backend_config = ""} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[MOVE_TO_HOST]] : tensor<8x2xi32>
  %0 = call @bar(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>, random_attr = "random_value", mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"}} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = stablehlo.custom_call @MoveToHost(%0) {backend_config = ""} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// NOTE: we ignore any arg/result shardings on the function.
func.func private @bar(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// Don't import if there is no backend_config.
// CHECK-LABEL: func @no_backend_config
func.func @no_backend_config(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %[[CALL:.*]] = call @baz(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL]] : tensor<8x2xi32>
  %0 = call @baz(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

func.func private @baz(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  %0 = stablehlo.multiply %arg0, %arg0 : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
