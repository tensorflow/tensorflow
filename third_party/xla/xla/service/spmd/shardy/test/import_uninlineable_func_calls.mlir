// RUN: sdy_opt --split-input-file %s -xla-sdy-import-uninlineable-func-calls  2>&1 | FileCheck %s
// RUN: sdy_opt %s -split-input-file -xla-sdy-import-uninlineable-func-calls -verify-diagnostics

sdy.mesh @mesh = #sdy.mesh<["x"=2, "y"=2]>

// CHECK-LABEL: func @backend_config_no_out_shardings
func.func @backend_config_no_out_shardings(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
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

// CHECK-NOT: func private @foo
func.func private @foo(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// CHECK-LABEL: func @backend_config_out_shardings
func.func @backend_config_out_shardings(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
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
// CHECK-NOT: func private @bar
func.func private @bar(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// CHECK-LABEL: func @inlineable_false
func.func @inlineable_false(%arg0: tensor<8x2xi32>, %arg1: tensor<8x2xi32>) -> (tensor<8x2xi32>) {
  // CHECK-NEXT: %[[NC:.*]]:2 = sdy.named_computation<"baz">(%arg0, %arg1) out_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>] (%arg2: tensor<8x2xi32>, %arg3: tensor<8x2xi32>) {
  // CHECK-NEXT:   %[[MULT:.*]] = stablehlo.multiply %arg2, %arg3 : tensor<8x2xi32>
  // CHECK-NEXT:   sdy.return %[[MULT]], %arg3 : tensor<8x2xi32>, tensor<8x2xi32>
  // CHECK-NEXT: } {mhlo.frontend_attributes = {inlineable = "false"}}
  // CHECK-SAME: (tensor<8x2xi32>, tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>)
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[NC]]#0, %[[NC]]#1 : tensor<8x2xi32>
  // CHECK-NEXT: return %[[ADD]] : tensor<8x2xi32>
  %0:2 = call @baz(%arg0, %arg1) {mhlo.frontend_attributes = {inlineable = "false"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>]>} : (tensor<8x2xi32>, tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>)
  %1 = stablehlo.add %0#0, %0#1 : tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK-NOT: func private @baz
func.func private @baz(%arg0: tensor<8x2xi32>, %arg1: tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>) {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<8x2xi32>
  return %0, %arg1 : tensor<8x2xi32>, tensor<8x2xi32>
}

// CHECK-LABEL: func @inlineable_true
func.func @inlineable_true(%arg0: tensor<8x2xi32>, %arg1: tensor<8x2xi32>) -> (tensor<8x2xi32>) {
  // CHECK-NEXT: %[[CALL:.*]]:2 =  call @qux(%arg0, %arg1) {mhlo.frontend_attributes = {inlineable = "true"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>]>}
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[CALL]]#0, %[[CALL]]#1 : tensor<8x2xi32>
  // CHECK-NEXT: return %[[ADD]] : tensor<8x2xi32>
  %0:2 = call @qux(%arg0, %arg1) {mhlo.frontend_attributes = {inlineable = "true"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>]>} : (tensor<8x2xi32>, tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>)
  %1 = stablehlo.add %0#0, %0#1 : tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK: func private @qux
func.func private @qux(%arg0: tensor<8x2xi32>, %arg1: tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>) {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<8x2xi32>
  return %0, %arg1 : tensor<8x2xi32>, tensor<8x2xi32>
}

// Don't import if there is no backend_config or inlineable attr.
// CHECK-LABEL: func @no_backend_config_or_inlineable_attr
func.func @no_backend_config_or_inlineable_attr(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %[[CALL:.*]] = call @quux(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[CALL]] : tensor<8x2xi32>
  %0 = call @quux(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// CHECK: func private @quux
func.func private @quux(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  %0 = stablehlo.multiply %arg0, %arg0 : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = #sdy.mesh<["x"=2, "y"=2]>

// CHECK-LABEL: func @multiple_call_ops_same_name
func.func @multiple_call_ops_same_name(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %[[NC_0:.*]] = sdy.named_computation<"foobar">(%arg0) out_shardings=[<@mesh, [{"x"}, {"y"}]>] (%arg1: tensor<8x2xi32>) {
  // CHECK-NEXT:   %[[MULT_0:.*]] = stablehlo.multiply %arg1, %arg1 {mhlo.frontend_attributes = {_xla_compute_type = "host"}} : tensor<8x2xi32>
  // CHECK-NEXT:   sdy.return %[[MULT_0]] : tensor<8x2xi32>
  // CHECK-NEXT: } {mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"},
  // CHECK-SAME:    random_attr = "random_value"}
  // CHECK-SAME: (tensor<8x2xi32>) -> tensor<8x2xi32>

  // CHECK-NEXT: %[[NC_1:.*]] = sdy.named_computation<"foobar">(%[[NC_0]]) out_shardings=[<@mesh, [{"x"}, {"y"}]>] (%arg1: tensor<8x2xi32>) {
  // CHECK-NEXT:   %[[MULT_1:.*]] = stablehlo.multiply %arg1, %arg1 {mhlo.frontend_attributes = {_xla_compute_type = "host"}} : tensor<8x2xi32>
  // CHECK-NEXT:   sdy.return %[[MULT_1]] : tensor<8x2xi32>
  // CHECK-NEXT: } {mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"},
  // CHECK-SAME:    random_attr = "random_value"}
  // CHECK-SAME: (tensor<8x2xi32>) -> tensor<8x2xi32>

  // CHECK-NEXT: %[[MOVE_TO_HOST:.*]] = stablehlo.custom_call @MoveToHost(%[[NC_1]]) {backend_config = ""} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[MOVE_TO_HOST]] : tensor<8x2xi32>
  %0 = call @foobar(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>, random_attr = "random_value", mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"}} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // expected-warning @+1 {{uninlineable function @foobar has multiple call ops, we need to clone the function body for each call}}
  %1 = call @foobar(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>, random_attr = "random_value", mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"}} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = stablehlo.custom_call @MoveToHost(%1) {backend_config = ""} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %2 : tensor<8x2xi32>
}

// CHECK-NOT: func private @foobar
func.func private @foobar(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}
