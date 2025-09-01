// RUN: sdy_opt --split-input-file %s -xla-sdy-import-func-calls | FileCheck %s
// RUN: sdy_opt %s -split-input-file -xla-sdy-import-func-calls -verify-diagnostics

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
  // CHECK-NEXT: %[[NC:.*]] = sdy.named_computation<"bar">(%arg0) in_shardings=[<@mesh, [{"x"}, {}]>] out_shardings=[<@mesh, [{"x"}, {"y"}]>] (%arg1: tensor<8x2xi32>) {
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
  // CHECK-NEXT: %[[NC:.*]]:2 = sdy.named_computation<"qux">(%arg0, %arg1) out_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>] (%arg2: tensor<8x2xi32>, %arg3: tensor<8x2xi32>) {
  // CHECK-NEXT:   %[[MULT:.*]] = stablehlo.multiply %arg2, %arg3 : tensor<8x2xi32>
  // CHECK-NEXT:   sdy.return %[[MULT]], %arg3 : tensor<8x2xi32>, tensor<8x2xi32>
  // CHECK-NEXT: } {mhlo.frontend_attributes = {inlineable = "true"}}
  // CHECK-SAME: (tensor<8x2xi32>, tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>)
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[NC]]#0, %[[NC]]#1 : tensor<8x2xi32>
  // CHECK-NEXT: return %[[ADD]] : tensor<8x2xi32>
  %0:2 = call @qux(%arg0, %arg1) {mhlo.frontend_attributes = {inlineable = "true"}, sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>]>} : (tensor<8x2xi32>, tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>)
  %1 = stablehlo.add %0#0, %0#1 : tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK-NOT: func private @qux
func.func private @qux(%arg0: tensor<8x2xi32>, %arg1: tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>) {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<8x2xi32>
  return %0, %arg1 : tensor<8x2xi32>, tensor<8x2xi32>
}

// CHECK-LABEL: func @no_backend_config_or_inlineable_attr
func.func @no_backend_config_or_inlineable_attr(%arg0: tensor<8x2xi32>, %arg1: tensor<8x2xi32>) -> (tensor<8x2xi32>) {
  // CHECK-NEXT: %[[NC:.*]]:2 = sdy.named_computation<"quux">(%arg0, %arg1) out_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>] (%arg2: tensor<8x2xi32>, %arg3: tensor<8x2xi32>) {
  // CHECK-NEXT:   %[[MULT:.*]] = stablehlo.multiply %arg2, %arg3 : tensor<8x2xi32>
  // CHECK-NEXT:   sdy.return %[[MULT]], %arg3 : tensor<8x2xi32>, tensor<8x2xi32>
  // CHECK-NEXT: }
  // CHECK-SAME: (tensor<8x2xi32>, tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>)
  // CHECK-NEXT: %[[ADD:.*]] = stablehlo.add %[[NC]]#0, %[[NC]]#1 : tensor<8x2xi32>
  // CHECK-NEXT: return %[[ADD]] : tensor<8x2xi32>
  %0:2 = call @quux(%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>]>} : (tensor<8x2xi32>, tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>)
  %1 = stablehlo.add %0#0, %0#1 : tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK-NOT: func private @quux
func.func private @quux(%arg0: tensor<8x2xi32>, %arg1: tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>) {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<8x2xi32>
  return %0, %arg1 : tensor<8x2xi32>, tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = #sdy.mesh<["x"=2, "y"=2]>

// CHECK-LABEL: func @multiple_call_ops_same_name
func.func @multiple_call_ops_same_name(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
  // CHECK-NEXT: %[[NC_0:.*]] = sdy.named_computation<"foobar">(%arg0) in_shardings=[<@mesh, [{"x"}, {}]>] out_shardings=[<@mesh, [{"x"}, {"y"}]>] (%arg1: tensor<8x2xi32>) {
  // CHECK-NEXT:   %[[MULT_0:.*]] = stablehlo.multiply %arg1, %arg1 {mhlo.frontend_attributes = {_xla_compute_type = "host"}} : tensor<8x2xi32>
  // CHECK-NEXT:   sdy.return %[[MULT_0]] : tensor<8x2xi32>
  // CHECK-NEXT: } {mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"},
  // CHECK-SAME:    random_attr = "random_value"}
  // CHECK-SAME: (tensor<8x2xi32>) -> tensor<8x2xi32>

  // CHECK-NEXT: %[[NC_1:.*]] = sdy.named_computation<"foobar">(%[[NC_0]]) in_shardings=[<@mesh, [{"x"}, {}]>] out_shardings=[<@mesh, [{"x"}, {"y"}]>] (%arg1: tensor<8x2xi32>) {
  // CHECK-NEXT:   %[[MULT_1:.*]] = stablehlo.multiply %arg1, %arg1 {mhlo.frontend_attributes = {_xla_compute_type = "host"}} : tensor<8x2xi32>
  // CHECK-NEXT:   sdy.return %[[MULT_1]] : tensor<8x2xi32>
  // CHECK-NEXT: } {mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"},
  // CHECK-SAME:    random_attr = "random_value"}
  // CHECK-SAME: (tensor<8x2xi32>) -> tensor<8x2xi32>

  // CHECK-NEXT: %[[MOVE_TO_HOST:.*]] = stablehlo.custom_call @MoveToHost(%[[NC_1]]) {backend_config = ""} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[MOVE_TO_HOST]] : tensor<8x2xi32>
  %0 = call @foobar(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>, random_attr = "random_value", mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"}} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  // expected-warning @+1 {{function @foobar has multiple call ops, we need to clone the function body for each call}}
  %1 = call @foobar(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>, random_attr = "random_value", mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"}} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = stablehlo.custom_call @MoveToHost(%1) {backend_config = ""} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %2 : tensor<8x2xi32>
}

// CHECK-NOT: func private @foobar
func.func private @foobar(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = #sdy.mesh<["x"=2, "y"=2]>

// CHECK-LABEL: func @multiple_call_ops_same_name_func_no_input_output_shardings
func.func @multiple_call_ops_same_name_func_no_input_output_shardings(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
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
  %1 = call @foobar(%0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}, {"y"}]>]>, random_attr = "random_value", mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"}} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  %2 = stablehlo.custom_call @MoveToHost(%1) {backend_config = ""} : (tensor<8x2xi32>) -> tensor<8x2xi32>
  return %2 : tensor<8x2xi32>
}

// CHECK-NOT: func private @foobar
func.func private @foobar(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
  %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}} : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

// CHECK-LABEL: func @non_flat_call_graph_all_uninlineable
func.func @non_flat_call_graph_all_uninlineable(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[NC1:.*]] = sdy.named_computation<"foo">(%arg0) (%arg1: tensor<8xf32>) {
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg1 : tensor<8xf32>
  // CHECK-NEXT:   %[[NC2:.*]] = sdy.named_computation<"bar">(%[[ADD]]) (%arg2: tensor<8xf32>) {
  // CHECK-NEXT:     %[[ABS1:.*]] = stablehlo.abs %arg2 : tensor<8xf32>
  // CHECK-NEXT:     sdy.return %[[ABS1]] : tensor<8xf32>
  // CHECK-NEXT:   }
  // CHECK-SAME:   {mhlo.frontend_attributes = {inlineable = "false"}}
  // CHECK-SAME:   (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT:   sdy.return %[[NC2]] : tensor<8xf32>
  // CHECK-NEXT: }
  // CHECK-SAME: {mhlo.frontend_attributes = {inlineable = "false"}}
  // CHECK-SAME: (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[NC1]] : tensor<8xf32>
  // CHECK-NEXT: %[[NC3:.*]] = sdy.named_computation<"baz">(%[[NEGATE]]) (%arg1: tensor<8xf32>) {
  // CHECK-NEXT:   %[[ABS2:.*]] = stablehlo.abs %arg1 : tensor<8xf32>
  // CHECK-NEXT:   sdy.return %[[ABS2]] : tensor<8xf32>
  // CHECK-NEXT: }
  // CHECK-SAME: {mhlo.frontend_attributes = {inlineable = "false"}}
  // CHECK-SAME: (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[NC4:.*]] = sdy.named_computation<"bar">(%[[NC3]]) (%arg1: tensor<8xf32>) {
  // CHECK-NEXT:   %[[ABS3:.*]] = stablehlo.abs %arg1 : tensor<8xf32>
  // CHECK-NEXT:   sdy.return %[[ABS3]] : tensor<8xf32>
  // CHECK-NEXT: }
  // CHECK-SAME: {mhlo.frontend_attributes = {inlineable = "false"}}
  // CHECK-SAME: (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: return %[[NC4]] : tensor<8xf32>
  %0 = call @foo(%arg0) {mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8xf32>) -> tensor<8xf32>
  %1 = stablehlo.negate %0 : tensor<8xf32>
  %2 = call @baz(%1) {mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8xf32>) -> tensor<8xf32>
  %3 = call @bar(%2) {mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8xf32>) -> tensor<8xf32>
  return %3 : tensor<8xf32>
}

// CHECK-NOT: func private @foo
func.func private @foo(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<8xf32>
  %1 = call @bar(%0) {mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8xf32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>
}

// CHECK-NOT: func private @bar
func.func private @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-NOT: func private @baz
func.func private @baz(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

// CHECK-LABEL: func @non_flat_call_graph_all_inlineable
func.func @non_flat_call_graph_all_inlineable(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  // CHECK-NEXT: %[[NC1:.*]] = sdy.named_computation<"foo">(%arg0) (%arg1: tensor<8xf32>) {
  // CHECK-NEXT:   %[[ADD:.*]] = stablehlo.add %arg1, %arg1 : tensor<8xf32>
  // CHECK-NEXT:   %[[NC2:.*]] = sdy.named_computation<"bar">(%[[ADD]]) (%arg2: tensor<8xf32>) {
  // CHECK-NEXT:     %[[ABS1:.*]] = stablehlo.abs %arg2 : tensor<8xf32>
  // CHECK-NEXT:     sdy.return %[[ABS1]] : tensor<8xf32>
  // CHECK-NEXT:   }
  // CHECK-SAME:   {mhlo.frontend_attributes = {inlineable = "true"}}
  // CHECK-SAME:   (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT:   sdy.return %[[NC2]] : tensor<8xf32>
  // CHECK-NEXT: }
  // CHECK-SAME: {mhlo.frontend_attributes = {inlineable = "true"}}
  // CHECK-SAME: (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[NC1]] : tensor<8xf32>
  // CHECK-NEXT: %[[NC3:.*]] = sdy.named_computation<"baz">(%[[NEGATE]]) (%arg1: tensor<8xf32>) {
  // CHECK-NEXT:   %[[ABS2:.*]] = stablehlo.abs %arg1 : tensor<8xf32>
  // CHECK-NEXT:   sdy.return %[[ABS2]] : tensor<8xf32>
  // CHECK-NEXT: }
  // CHECK-SAME: {mhlo.frontend_attributes = {inlineable = "true"}}
  // CHECK-SAME: (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: %[[NC4:.*]] = sdy.named_computation<"bar">(%[[NC3]]) (%arg1: tensor<8xf32>) {
  // CHECK-NEXT:   %[[ABS3:.*]] = stablehlo.abs %arg1 : tensor<8xf32>
  // CHECK-NEXT:   sdy.return %[[ABS3]] : tensor<8xf32>
  // CHECK-NEXT: }
  // CHECK-SAME: {mhlo.frontend_attributes = {inlineable = "true"}}
  // CHECK-SAME: (tensor<8xf32>) -> tensor<8xf32>
  // CHECK-NEXT: return %[[NC4]] : tensor<8xf32>
  %0 = call @foo(%arg0) {mhlo.frontend_attributes = {inlineable = "true"}} : (tensor<8xf32>) -> tensor<8xf32>
  %1 = stablehlo.negate %0 : tensor<8xf32>
  %2 = call @baz(%1) {mhlo.frontend_attributes = {inlineable = "true"}} : (tensor<8xf32>) -> tensor<8xf32>
  %3 = call @bar(%2) {mhlo.frontend_attributes = {inlineable = "true"}} : (tensor<8xf32>) -> tensor<8xf32>
  return %3 : tensor<8xf32>
}

// CHECK-NOT: func private @foo
func.func private @foo(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<8xf32>
  %1 = call @bar(%0) {mhlo.frontend_attributes = {inlineable = "true"}} : (tensor<8xf32>) -> tensor<8xf32>
  return %1 : tensor<8xf32>
}

// CHECK-NOT: func private @bar
func.func private @bar(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK-NOT: func private @baz
func.func private @baz(%arg0: tensor<8xf32>) -> tensor<8xf32> {
  %0 = stablehlo.abs %arg0 : tensor<8xf32>
  return %0 : tensor<8xf32>
}

// -----

sdy.mesh @mesh = #sdy.mesh<["x"=2, "y"=2]>

// CHECK-LABEL: func @single_call_multiple_args_func_no_input_sharding
func.func @single_call_multiple_args_func_no_input_sharding(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[NC:.*]] = sdy.named_computation<"foo">(%arg0, %arg0) out_shardings=[<@mesh, [{"y"}, {"x"}]>] (%arg1: tensor<8x2xi32>, %arg2: tensor<8x2xi32>) {
  // CHECK-NEXT:   %[[MULTIPLY:.*]] = stablehlo.multiply %arg1, %arg2 : tensor<8x2xi32>
  // CHECK-NEXT:   sdy.return %[[MULTIPLY]] : tensor<8x2xi32>
  // CHECK-NEXT: } {mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8x2xi32>, tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[NC]] : tensor<8x2xi32>
  // CHECK-NEXT: return %[[NEGATE]] : tensor<8x2xi32>
  %0 = call @foo(%arg0, %arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>, mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8x2xi32>, tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = stablehlo.negate %0 : tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK-NOT: func private @foo
func.func private @foo(%arg0: tensor<8x2xi32>, %arg1: tensor<8x2xi32>) -> tensor<8x2xi32> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = #sdy.mesh<["x"=2, "y"=2]>

// CHECK-LABEL: func @single_call_multiple_args_func_all_arguments_with_input_sharding
func.func @single_call_multiple_args_func_all_arguments_with_input_sharding(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[NC:.*]] = sdy.named_computation<"foo">(%arg0, %arg0) in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {"y"}]>] out_shardings=[<@mesh, [{"y"}, {"x"}]>] (%arg1: tensor<8x2xi32>, %arg2: tensor<8x2xi32>) {
  // CHECK-NEXT:   %[[MULTIPLY:.*]] = stablehlo.multiply %arg1, %arg2 : tensor<8x2xi32>
  // CHECK-NEXT:   sdy.return %[[MULTIPLY]] : tensor<8x2xi32>
  // CHECK-NEXT: } {mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8x2xi32>, tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[NC]] : tensor<8x2xi32>
  // CHECK-NEXT: return %[[NEGATE]] : tensor<8x2xi32>
  %0 = call @foo(%arg0, %arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>, mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8x2xi32>, tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = stablehlo.negate %0 : tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK-NOT: func private @foo
func.func private @foo(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = #sdy.mesh<["x"=2, "y"=2]>

// CHECK-LABEL: func @single_call_multiple_args_func_some_arguments_with_input_sharding_some_arguments_without
func.func @single_call_multiple_args_func_some_arguments_with_input_sharding_some_arguments_without(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[NC:.*]] = sdy.named_computation<"foo">(%arg0, %arg0) in_shardings=[<@mesh, [{?}, {?}]>, <@mesh, [{}, {"y"}]>] out_shardings=[<@mesh, [{"y"}, {"x"}]>] (%arg1: tensor<8x2xi32>, %arg2: tensor<8x2xi32>) {
  // CHECK-NEXT:   %[[MULTIPLY:.*]] = stablehlo.multiply %arg1, %arg2 : tensor<8x2xi32>
  // CHECK-NEXT:   sdy.return %[[MULTIPLY]] : tensor<8x2xi32>
  // CHECK-NEXT: } {mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8x2xi32>, tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[NC]] : tensor<8x2xi32>
  // CHECK-NEXT: return %[[NEGATE]] : tensor<8x2xi32>
  %0 = call @foo(%arg0, %arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>, mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8x2xi32>, tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = stablehlo.negate %0 : tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK-NOT: func private @foo
func.func private @foo(%arg0: tensor<8x2xi32>, %arg1: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = #sdy.mesh<["x"=2, "y"=2]>

// CHECK-LABEL: func @single_call_multiple_args_with_different_ranks_func_some_arguments_with_input_sharding_some_arguments_without
func.func @single_call_multiple_args_with_different_ranks_func_some_arguments_with_input_sharding_some_arguments_without(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}, %arg1: tensor<8xi32>) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[NC:.*]] = sdy.named_computation<"foo">(%arg0, %arg1) in_shardings=[<@mesh, [{?}, {?}]>, <@mesh, [{"y"}]>] out_shardings=[<@mesh, [{"y"}, {"x"}]>] (%arg2: tensor<8x2xi32>, %arg3: tensor<8xi32>) {
  // CHECK-NEXT:   %[[MULTIPLY:.*]] = stablehlo.multiply %arg2, %arg2 : tensor<8x2xi32>
  // CHECK-NEXT:   sdy.return %[[MULTIPLY]] : tensor<8x2xi32>
  // CHECK-NEXT: } {mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8x2xi32>, tensor<8xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[NC]] : tensor<8x2xi32>
  // CHECK-NEXT: return %[[NEGATE]] : tensor<8x2xi32>
  %0 = call @foo(%arg0, %arg1) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>, mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8x2xi32>, tensor<8xi32>) -> tensor<8x2xi32>
  %1 = stablehlo.negate %0 : tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK-NOT: func private @foo
func.func private @foo(%arg0: tensor<8x2xi32>, %arg1: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}) -> tensor<8x2xi32> {
  %0 = stablehlo.multiply %arg0, %arg0 : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = #sdy.mesh<["x"=2, "y"=2]>
sdy.mesh @mesh_maximal = #sdy.mesh<[], device_ids=[0]>

// CHECK-LABEL: func @single_call_multiple_args_func_some_arguments_with_input_sharding_on_maximal_mesh_some_arguments_without
func.func @single_call_multiple_args_func_some_arguments_with_input_sharding_on_maximal_mesh_some_arguments_without(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[NC:.*]] = sdy.named_computation<"foo">(%arg0, %arg0) out_shardings=[<@mesh, [{"y"}, {"x"}]>] (%arg1: tensor<8x2xi32>, %arg2: tensor<8x2xi32>) {
  // CHECK-NEXT:   %[[MULTIPLY:.*]] = stablehlo.multiply %arg1, %arg2 : tensor<8x2xi32>
  // CHECK-NEXT:   sdy.return %[[MULTIPLY]] : tensor<8x2xi32>
  // CHECK-NEXT: } {mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8x2xi32>, tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[NC]] : tensor<8x2xi32>
  // CHECK-NEXT: return %[[NEGATE]] : tensor<8x2xi32>
  %0 = call @foo(%arg0, %arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>, mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8x2xi32>, tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = stablehlo.negate %0 : tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK-NOT: func private @foo
func.func private @foo(%arg0: tensor<8x2xi32>, %arg1: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh_maximal, []>}) -> tensor<8x2xi32> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = #sdy.mesh<["x"=2, "y"=2]>
sdy.mesh @mesh_maximal = #sdy.mesh<[], device_ids=[0]>

// CHECK-LABEL: func @single_call_multiple_args_func_some_arguments_with_input_sharding_on_maximal_and_on_non_maximal_mesh_some_arguments_without
func.func @single_call_multiple_args_func_some_arguments_with_input_sharding_on_maximal_and_on_non_maximal_mesh_some_arguments_without(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[NC:.*]] = sdy.named_computation<"foo">(%arg0, %arg0, %arg0) in_shardings=[<@mesh, [{?}, {?}]>, <@mesh_maximal, []>, <@mesh, [{}, {"y"}]>] out_shardings=[<@mesh, [{"y"}, {"x"}]>] (%arg1: tensor<8x2xi32>, %arg2: tensor<8x2xi32>, %arg3: tensor<8x2xi32>) {
  // CHECK-NEXT:   %[[MULTIPLY:.*]] = stablehlo.multiply %arg1, %arg2 : tensor<8x2xi32>
  // CHECK-NEXT:   sdy.return %[[MULTIPLY]] : tensor<8x2xi32>
  // CHECK-NEXT: } {mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8x2xi32>, tensor<8x2xi32>, tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[NC]] : tensor<8x2xi32>
  // CHECK-NEXT: return %[[NEGATE]] : tensor<8x2xi32>
  %0 = call @foo(%arg0, %arg0, %arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>, mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8x2xi32>, tensor<8x2xi32>, tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = stablehlo.negate %0 : tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK-NOT: func private @foo
func.func private @foo(%arg0: tensor<8x2xi32>, %arg1: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh_maximal, []>}, %arg2: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"y"}]>}) -> tensor<8x2xi32> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = #sdy.mesh<["x"=2, "y"=2]>

// CHECK-LABEL: func @func_has_out_sharding_call_no_out_sharding
func.func @func_has_out_sharding_call_no_out_sharding(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[NC:.*]] = sdy.named_computation<"foo">(%arg0, %arg0) out_shardings=[<@mesh, [{"x", "y"}, {}]>] (%arg1: tensor<8x2xi32>, %arg2: tensor<8x2xi32>) {
  // CHECK-NEXT:   %[[MULTIPLY:.*]] = stablehlo.multiply %arg1, %arg2 : tensor<8x2xi32>
  // CHECK-NEXT:   sdy.return %[[MULTIPLY]] : tensor<8x2xi32>
  // CHECK-NEXT: } {mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8x2xi32>, tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[NC]] : tensor<8x2xi32>
  // CHECK-NEXT: return %[[NEGATE]] : tensor<8x2xi32>
  %0 = call @foo(%arg0, %arg0) {mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8x2xi32>, tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = stablehlo.negate %0 : tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK-NOT: func private @foo
func.func private @foo(%arg0: tensor<8x2xi32>, %arg1: tensor<8x2xi32>) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = #sdy.mesh<["x"=2, "y"=2]>

// CHECK-LABEL: func @func_has_out_sharding_call_has_different_out_sharding
func.func @func_has_out_sharding_call_has_different_out_sharding(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[NC:.*]] = sdy.named_computation<"foo">(%arg0, %arg0) out_shardings=[<@mesh, [{"y"}, {"x"}]>] (%arg1: tensor<8x2xi32>, %arg2: tensor<8x2xi32>) {
  // CHECK-NEXT:   %[[MULTIPLY:.*]] = stablehlo.multiply %arg1, %arg2 : tensor<8x2xi32>
  // CHECK-NEXT:   sdy.return %[[MULTIPLY]] : tensor<8x2xi32>
  // CHECK-NEXT: } {mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8x2xi32>, tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[NC]] : tensor<8x2xi32>
  // CHECK-NEXT: return %[[NEGATE]] : tensor<8x2xi32>
  %0 = call @foo(%arg0, %arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>, mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8x2xi32>, tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = stablehlo.negate %0 : tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK-NOT: func private @foo
func.func private @foo(%arg0: tensor<8x2xi32>, %arg1: tensor<8x2xi32>) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x", "y"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = #sdy.mesh<["x"=2, "y"=2]>

// CHECK-LABEL: func @func_no_out_sharding_call_has_out_sharding
func.func @func_no_out_sharding_call_has_out_sharding(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[NC:.*]] = sdy.named_computation<"foo">(%arg0, %arg0) out_shardings=[<@mesh, [{"y"}, {"x"}]>] (%arg1: tensor<8x2xi32>, %arg2: tensor<8x2xi32>) {
  // CHECK-NEXT:   %[[MULTIPLY:.*]] = stablehlo.multiply %arg1, %arg2 : tensor<8x2xi32>
  // CHECK-NEXT:   sdy.return %[[MULTIPLY]] : tensor<8x2xi32>
  // CHECK-NEXT: } {mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8x2xi32>, tensor<8x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: %[[NEGATE:.*]] = stablehlo.negate %[[NC]] : tensor<8x2xi32>
  // CHECK-NEXT: return %[[NEGATE]] : tensor<8x2xi32>
  %0 = call @foo(%arg0, %arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>]>, mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8x2xi32>, tensor<8x2xi32>) -> tensor<8x2xi32>
  %1 = stablehlo.negate %0 : tensor<8x2xi32>
  return %1 : tensor<8x2xi32>
}

// CHECK-NOT: func private @foo
func.func private @foo(%arg0: tensor<8x2xi32>, %arg1: tensor<8x2xi32>) -> tensor<8x2xi32> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<8x2xi32>
  return %0 : tensor<8x2xi32>
}

// -----

sdy.mesh @mesh = #sdy.mesh<["x"=2, "y"=2]>

// CHECK-LABEL: func @func_has_out_sharding_on_one_result_call_has_out_sharding_on_both_results
func.func @func_has_out_sharding_on_one_result_call_has_out_sharding_on_both_results(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[NC:.*]]:2 = sdy.named_computation<"foo">(%arg0, %arg0) out_shardings=[<@mesh, [{"y"}, {"x"}]>, <@mesh, [{"x"}, {}]>] (%arg1: tensor<8x2xi32>, %arg2: tensor<8x2xi32>) {
  // CHECK-NEXT:   %[[MULTIPLY:.*]] = stablehlo.multiply %arg1, %arg2 : tensor<8x2xi32>
  // CHECK-NEXT:   %[[TRANSPOSE:.*]] = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<8x2xi32>) -> tensor<2x8xi32>
  // CHECK-NEXT:   %[[DOT_1:.*]] = stablehlo.dot %[[TRANSPOSE]], %arg1 : (tensor<2x8xi32>, tensor<8x2xi32>) -> tensor<2x2xi32>
  // CHECK-NEXT:   sdy.return %[[MULTIPLY]], %[[DOT_1]] : tensor<8x2xi32>, tensor<2x2xi32>
  // CHECK-NEXT: } {mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8x2xi32>, tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<2x2xi32>)
  // CHECK-NEXT: %[[DOT_2:.*]] = stablehlo.dot %[[NC]]#0, %[[NC]]#1 : (tensor<8x2xi32>, tensor<2x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[DOT_2]] : tensor<8x2xi32>
  %0:2 = call @foo(%arg0, %arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"y"}, {"x"}]>, <@mesh, [{"x"}, {}]>]>, mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8x2xi32>, tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<2x2xi32>)
  %1 = stablehlo.dot %0#0, %0#1 : (tensor<8x2xi32>, tensor<2x2xi32>) -> (tensor<8x2xi32>)
  return %1 : tensor<8x2xi32>
}

// CHECK-NOT: func private @foo
func.func private @foo(%arg0: tensor<8x2xi32>, %arg1: tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<2x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<8x2xi32>
  %1 = stablehlo.transpose %arg0, dims=[1, 0] : (tensor<8x2xi32>) -> tensor<2x8xi32>
  %2 = stablehlo.dot %1, %arg0 : (tensor<2x8xi32>, tensor<8x2xi32>) -> tensor<2x2xi32>
  return %0, %2 : tensor<8x2xi32>, tensor<2x2xi32>
}

// -----

sdy.mesh @mesh = #sdy.mesh<["x"=2, "y"=2]>

// CHECK-LABEL: func @func_has_out_sharding_on_one_result_call_has_no_out_sharding
func.func @func_has_out_sharding_on_one_result_call_has_no_out_sharding(%arg0: tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> tensor<8x2xi32> {
  // CHECK-NEXT: %[[NC:.*]]:2 = sdy.named_computation<"foo">(%arg0, %arg0) out_shardings=[<@mesh, [{?}, {?}]>, <@mesh, [{"y"}, {}]>] (%arg1: tensor<8x2xi32>, %arg2: tensor<8x2xi32>) {
  // CHECK-NEXT:   %[[MULTIPLY:.*]] = stablehlo.multiply %arg1, %arg2 : tensor<8x2xi32>
  // CHECK-NEXT:   %[[TRANSPOSE:.*]] = stablehlo.transpose %arg1, dims = [1, 0] : (tensor<8x2xi32>) -> tensor<2x8xi32>
  // CHECK-NEXT:   %[[DOT_1:.*]] = stablehlo.dot %[[TRANSPOSE]], %arg1 : (tensor<2x8xi32>, tensor<8x2xi32>) -> tensor<2x2xi32>
  // CHECK-NEXT:   sdy.return %[[MULTIPLY]], %[[DOT_1]] : tensor<8x2xi32>, tensor<2x2xi32>
  // CHECK-NEXT: } {mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8x2xi32>, tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<2x2xi32>)
  // CHECK-NEXT: %[[DOT_2:.*]] = stablehlo.dot %[[NC]]#0, %[[NC]]#1 : (tensor<8x2xi32>, tensor<2x2xi32>) -> tensor<8x2xi32>
  // CHECK-NEXT: return %[[DOT_2]] : tensor<8x2xi32>
  %0:2 = call @foo(%arg0, %arg0) {mhlo.frontend_attributes = {inlineable = "false"}} : (tensor<8x2xi32>, tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<2x2xi32>)
  %1 = stablehlo.dot %0#0, %0#1 : (tensor<8x2xi32>, tensor<2x2xi32>) -> (tensor<8x2xi32>)
  return %1 : tensor<8x2xi32>
}

// CHECK-NOT: func private @foo
func.func private @foo(%arg0: tensor<8x2xi32>, %arg1: tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<2x2xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<8x2xi32>
  %1 = stablehlo.transpose %arg0, dims=[1, 0] : (tensor<8x2xi32>) -> tensor<2x8xi32>
  %2 = stablehlo.dot %1, %arg0 : (tensor<2x8xi32>, tensor<8x2xi32>) -> tensor<2x2xi32>
  return %0, %2 : tensor<8x2xi32>, tensor<2x2xi32>
}
