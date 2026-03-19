// RUN: sdy_opt %s -split-input-file -xla-sdy-flatten-call-graph | FileCheck %s

// CHECK-LABEL: func @singleton(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    return
func.func @singleton(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  return %arg0 : tensor<8xi32>
}

// -----

// CHECK-LABEL: func @simple_call_graph(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    call @foo(
// CHECK-NEXT:    return
func.func @simple_call_graph(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    return
func.func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  return %arg0 : tensor<8xi32>
}

// -----

// CHECK-LABEL: func @main_calls_foo_twice(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
// CHECK-NEXT:    call @foo(
// CHECK-NEXT:    call @foo_0(
// CHECK-NEXT:    return
func.func @main_calls_foo_twice(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
  %0 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %1 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  return %0, %1 : tensor<8xi32>, tensor<8xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    return
func.func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  return %arg0 : tensor<8xi32>
}

// CHECK-LABEL: func private @foo_0(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME:  attributes {xla.sdy.original_func_name = "foo"} {
// CHECK-NEXT:    return


// -----

// CHECK-LABEL: func @main_calls_foo_calls_bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    call @foo(
// CHECK-NEXT:    return
func.func @main_calls_foo_calls_bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    call @bar(
// CHECK-NEXT:    return
func.func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = call @bar(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

// CHECK-LABEL: func private @bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    return
func.func private @bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  return %arg0 : tensor<8xi32>
}

// -----

// CHECK-LABEL: func @main_calls_foo_calls_bar_twice(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    call @foo(
// CHECK-NEXT:    return
func.func @main_calls_foo_calls_bar_twice(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    call @bar(
// CHECK-NEXT:    call @bar_0(
// CHECK-NEXT:    stablehlo.add
func.func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = call @bar(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %1 = call @bar(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %2 = stablehlo.add %0, %1 : tensor<8xi32>
  return %2 : tensor<8xi32>
}

// CHECK-LABEL: func private @bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    return
func.func private @bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  return %arg0 : tensor<8xi32>
}

// CHECK-LABEL: func private @bar_0(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME:  attributes {xla.sdy.original_func_name = "bar"} {
// CHECK-NEXT:    return

// -----

// CHECK-LABEL: func @simple_non_flat(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
// CHECK-NEXT:    call @foo(
// CHECK-NEXT:    call @bar_0(
// CHECK-NEXT:    return
func.func @simple_non_flat(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
  %0 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %1 = call @bar(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  return %0, %1 : tensor<8xi32>, tensor<8xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    call @bar(
// CHECK-NEXT:    return
func.func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = call @bar(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  return %0 : tensor<8xi32>
}

// CHECK-LABEL: func private @bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    return
func.func private @bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  return %arg0 : tensor<8xi32>
}

// CHECK-LABEL: func private @bar_0(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME:  attributes {xla.sdy.original_func_name = "bar"} {
// CHECK-NEXT:    return

// -----

// CHECK-LABEL: func @main_calls_foo_twice_and_foo_calls_bar_twice(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
// CHECK-NEXT:    call @foo(
// CHECK-NEXT:    call @foo_3(
// CHECK-NEXT:    return
func.func @main_calls_foo_twice_and_foo_calls_bar_twice(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
  %0 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %1 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  return %0, %1 : tensor<8xi32>, tensor<8xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    call @bar(
// CHECK-NEXT:    call @bar_0(
// CHECK-NEXT:    stablehlo.add
func.func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = call @bar(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %1 = call @bar(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %2 = stablehlo.add %0, %1 : tensor<8xi32>
  return %2 : tensor<8xi32>
}

// CHECK-LABEL: func private @bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    return
func.func private @bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  return %arg0 : tensor<8xi32>
}

// CHECK-LABEL: func private @bar_0(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME:  attributes {xla.sdy.original_func_name = "bar"} {
// CHECK-NEXT:    return

// CHECK-LABEL: func private @bar_1(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME:  attributes {xla.sdy.original_func_name = "bar"} {
// CHECK-NEXT:    return

// CHECK-LABEL: func private @bar_2(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME:  attributes {xla.sdy.original_func_name = "bar"} {
// CHECK-NEXT:    return

// CHECK-LABEL: func private @foo_3(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME:  attributes {xla.sdy.original_func_name = "foo"} {
// CHECK-NEXT:    call @bar_1(
// CHECK-NEXT:    call @bar_2(
// CHECK-NEXT:    stablehlo.add

// -----

// CHECK-LABEL: func @main_calls_foo_twice_and_foo_calls_bar_once(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
// CHECK-NEXT:    call @foo(
// CHECK-NEXT:    call @foo_1(
// CHECK-NEXT:    return
func.func @main_calls_foo_twice_and_foo_calls_bar_once(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
  %0 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %1 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  return %0, %1 : tensor<8xi32>, tensor<8xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    call @bar(
// CHECK-NEXT:    return
func.func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = call @bar(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  return %0: tensor<8xi32>
}

// CHECK-LABEL: func private @bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:    return
func.func private @bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  return %arg0 : tensor<8xi32>
}

// CHECK-LABEL: func private @bar_0(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME:  attributes {xla.sdy.original_func_name = "bar"} {
// CHECK-NEXT:    return

// CHECK-LABEL: func private @foo_1(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME:  attributes {xla.sdy.original_func_name = "foo"} {
// CHECK-NEXT:    call @bar_0(
// CHECK-NEXT:    return

// -----

// CHECK-LABEL:func.func @complex_non_flat(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
// CHECK-NEXT:   %0 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
// CHECK-NEXT:   %1 = call @bar_4(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
// CHECK-NEXT:   return %0, %1 : tensor<8xi32>, tensor<8xi32>
// CHECK-NEXT: }
// CHECK-LABEL:func.func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:   %0 = call @bar(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
// CHECK-NEXT:   %1 = call @bar_1(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
// CHECK-NEXT:   %2 = stablehlo.add %0, %1 : tensor<8xi32>
// CHECK-NEXT:   %3 = call @baz_2(%2) : (tensor<8xi32>) -> tensor<8xi32>
// CHECK-NEXT:   return %3 : tensor<8xi32>
// CHECK-NEXT: }
// CHECK-LABEL:func.func private @bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:   %0 = call @baz(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
// CHECK-NEXT:   return %0 : tensor<8xi32>
// CHECK-NEXT: }
// CHECK-LABEL:func.func private @baz(%arg0: tensor<8xi32>) -> tensor<8xi32> {
// CHECK-NEXT:   return %arg0 : tensor<8xi32>
// CHECK-NEXT: }
// CHECK-LABEL:func.func private @baz_0(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME: attributes {xla.sdy.original_func_name = "baz"} {
// CHECK-NEXT:   return %arg0 : tensor<8xi32>
// CHECK-NEXT: }
// CHECK-LABEL:func.func private @bar_1(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME: attributes {xla.sdy.original_func_name = "bar"} {
// CHECK-NEXT:   %0 = call @baz_0(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
// CHECK-NEXT:   return %0 : tensor<8xi32>
// CHECK-NEXT: }
// CHECK-LABEL:func.func private @baz_2(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME: attributes {xla.sdy.original_func_name = "baz"} {
// CHECK-NEXT:   return %arg0 : tensor<8xi32>
// CHECK-NEXT: }
// CHECK-LABEL:func.func private @baz_3(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME: attributes {xla.sdy.original_func_name = "baz"} {
// CHECK-NEXT:   return %arg0 : tensor<8xi32>
// CHECK-NEXT: }
// CHECK-LABEL:func.func private @bar_4(%arg0: tensor<8xi32>) -> tensor<8xi32>
// CHECK-SAME: attributes {xla.sdy.original_func_name = "bar"} {
// CHECK-NEXT:   %0 = call @baz_3(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
// CHECK-NEXT:   return %0 : tensor<8xi32>
// CHECK-NEXT: }
func.func @complex_non_flat(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
  %0 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %1 = call @bar(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  return %0, %1 : tensor<8xi32>, tensor<8xi32>
}
func.func private @foo(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = call @bar(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %1 = call @bar(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %2 = stablehlo.add %0, %1 : tensor<8xi32>
  %3 = call @baz(%2) : (tensor<8xi32>) -> tensor<8xi32>
  return %3 : tensor<8xi32>
}

func.func private @bar(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  %0 = call @baz(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  return %0: tensor<8xi32>
}

func.func private @baz(%arg0: tensor<8xi32>) -> tensor<8xi32> {
  return %arg0 : tensor<8xi32>
}

// -----

sdy.mesh @mesh = #sdy.mesh<["a"=2, "b"=2]>

// CHECK-LABEL: func @simple_non_flat_sharding_on_func_arguments(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
// CHECK-NEXT:    call @foo(%arg0)
// CHECK-NEXT:    call @foo_0(%arg0)
// CHECK-NEXT:    return
func.func @simple_non_flat_sharding_on_func_arguments(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
  %0 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %1 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  return %0, %1 : tensor<8xi32>, tensor<8xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) -> tensor<8xi32> {
// CHECK-NEXT:    return
func.func private @foo(%arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) -> tensor<8xi32> {
  return %arg0 : tensor<8xi32>
}

// CHECK-LABEL: func private @foo_0(%arg0: tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) -> tensor<8xi32>
// CHECK-SAME:  attributes {xla.sdy.original_func_name = "foo"} {
// CHECK-NEXT:    return

// -----

sdy.mesh @mesh = #sdy.mesh<["a"=2, "b"=2]>

// CHECK-LABEL: func @simple_non_flat_sharding_on_func_results(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
// CHECK-NEXT:    call @foo(%arg0)
// CHECK-NEXT:    call @foo_0(%arg0)
// CHECK-NEXT:    return
func.func @simple_non_flat_sharding_on_func_results(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
  %0 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %1 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  return %0, %1 : tensor<8xi32>, tensor<8xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8xi32>) -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) {
// CHECK-NEXT:    return
func.func private @foo(%arg0: tensor<8xi32>) -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) {
  return %arg0 : tensor<8xi32>
}

// CHECK-LABEL: func private @foo_0(%arg0: tensor<8xi32>) -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>})
// CHECK-SAME:  attributes {xla.sdy.original_func_name = "foo"} {
// CHECK-NEXT:    return

// -----

sdy.mesh @mesh = #sdy.mesh<["a"=2, "b"=2]>

// CHECK-LABEL: func @simple_non_flat_non_matching_sharding_on_func_results_and_second_call_results(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
// CHECK-NEXT:    call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
// CHECK-NEXT:    call @foo_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}]>]>} : (tensor<8xi32>) -> tensor<8xi32>
// CHECK-NEXT:    return
func.func @simple_non_flat_non_matching_sharding_on_func_results_and_second_call_results(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
  %0 = call @foo(%arg0) : (tensor<8xi32>) -> tensor<8xi32>
  %1 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}]>]>} : (tensor<8xi32>) -> tensor<8xi32>
  return %0, %1 : tensor<8xi32>, tensor<8xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8xi32>) -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) {
// CHECK-NEXT:    return
func.func private @foo(%arg0: tensor<8xi32>) -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) {
  return %arg0 : tensor<8xi32>
}

// CHECK-LABEL: func private @foo_0(%arg0: tensor<8xi32>) -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}]>})
// CHECK-SAME:  attributes {xla.sdy.original_func_name = "foo"} {
// CHECK-NEXT:    return

// -----

sdy.mesh @mesh = #sdy.mesh<["a"=2, "b"=2, "c"=2]>

// CHECK-LABEL: func @simple_non_flat_non_matching_sharding_on_func_results_and_both_call_results(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
// CHECK-NEXT:    call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}]>]>} : (tensor<8xi32>) -> tensor<8xi32>
// CHECK-NEXT:    call @foo_0(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"c"}]>]>} : (tensor<8xi32>) -> tensor<8xi32>
// CHECK-NEXT:    return
func.func @simple_non_flat_non_matching_sharding_on_func_results_and_both_call_results(%arg0: tensor<8xi32>) -> (tensor<8xi32>, tensor<8xi32>) {
  %0 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"b"}]>]>} : (tensor<8xi32>) -> tensor<8xi32>
  %1 = call @foo(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"c"}]>]>} : (tensor<8xi32>) -> tensor<8xi32>
  return %0, %1 : tensor<8xi32>, tensor<8xi32>
}

// CHECK-LABEL: func private @foo(%arg0: tensor<8xi32>) -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"b"}]>}) {
// CHECK-NEXT:    return
func.func private @foo(%arg0: tensor<8xi32>) -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}) {
  return %arg0 : tensor<8xi32>
}

// CHECK-LABEL: func private @foo_0(%arg0: tensor<8xi32>) -> (tensor<8xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"c"}]>})
// CHECK-SAME:  attributes {xla.sdy.original_func_name = "foo"} {
// CHECK-NEXT:    return

