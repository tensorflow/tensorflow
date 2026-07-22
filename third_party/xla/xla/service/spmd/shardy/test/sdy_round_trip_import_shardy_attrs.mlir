// RUN: sdy_opt %s --split-input-file -xla-sdy-round-trip-import-shardy-attrs 2>&1 | FileCheck %s

// CHECK-LABEL: module @sparse_offload_module
// CHECK:         sdy.mesh @mesh = <["a"=2, "b"=2]>
module @sparse_offload_module attributes {
  mhlo.frontend_attributes = {
    xla.sdy.meshes = "{mesh = #sdy.mesh<[\"a\"=2, \"b\"=2]>}"
  }
} {

  // CHECK-LABEL: func private @sparse_offload_callee(
  // CHECK-SAME:      %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT:    %[[BARRIER:.*]] = sdy.propagation_barrier %arg0 allowed_direction=FORWARD : tensor<8x8xf32>
  // CHECK-NEXT:    %[[SHARDING:.*]] = stablehlo.custom_call @Sharding(%[[BARRIER]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:    return %[[SHARDING]] : tensor<8x8xf32>
  // CHECK-NEXT:  }
  func.func private @sparse_offload_callee(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = stablehlo.custom_call @Sharding(%arg0) {
      mhlo.frontend_attributes = {
        xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{}, {}]>]>"
      }
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }

  // CHECK-LABEL: func @main(
  // CHECK-SAME:      %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT:    %[[CALL:.*]] = call @sparse_offload_callee(%arg0) {mhlo.frontend_attributes = {_xla_compute_type = "sparseoffload"}} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:    return %[[CALL]] : tensor<8x8xf32>
  // CHECK-NEXT:  }
  func.func @main(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = call @sparse_offload_callee(%arg0) {
      mhlo.frontend_attributes = {_xla_compute_type = "sparseoffload"}
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

// CHECK-LABEL: module @sparse_offload_sharded_module
// CHECK:         sdy.mesh @mesh = <["a"=2, "b"=2]>
module @sparse_offload_sharded_module attributes {
  mhlo.frontend_attributes = {
    xla.sdy.meshes = "{mesh = #sdy.mesh<[\"a\"=2, \"b\"=2]>}"
  }
} {

  // CHECK-LABEL: func private @sparse_offload_callee_sharded(
  // CHECK-SAME:      %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT:    %[[SHARDING:.*]] = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:    return %[[SHARDING]] : tensor<8x8xf32>
  // CHECK-NEXT:  }
  func.func private @sparse_offload_callee_sharded(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = stablehlo.custom_call @Sharding(%arg0) {
      mhlo.frontend_attributes = {
        xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\"a\"}, {}]>]>"
      }
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }

  // CHECK-LABEL: func @main(
  // CHECK-SAME:      %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT:    %[[CALL:.*]] = call @sparse_offload_callee_sharded(%arg0) {mhlo.frontend_attributes = {_xla_compute_type = "sparseoffload"}} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:    return %[[CALL]] : tensor<8x8xf32>
  // CHECK-NEXT:  }
  func.func @main(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = call @sparse_offload_callee_sharded(%arg0) {
      mhlo.frontend_attributes = {_xla_compute_type = "sparseoffload"}
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

// CHECK-LABEL: module @sparse_offload_sharded_caller_module
// CHECK:         sdy.mesh @mesh = <["a"=2, "b"=2]>
module @sparse_offload_sharded_caller_module attributes {
  mhlo.frontend_attributes = {
    xla.sdy.meshes = "{mesh = #sdy.mesh<[\"a\"=2, \"b\"=2]>}"
  }
} {

  // CHECK-LABEL: func private @sparse_offload_callee_sharded(
  // CHECK-SAME:      %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT:    %[[BARRIER:.*]] = sdy.propagation_barrier %arg0 allowed_direction=FORWARD : tensor<8x8xf32>
  // CHECK-NEXT:    %[[SHARDING:.*]] = stablehlo.custom_call @Sharding(%[[BARRIER]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"a"}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:    return %[[SHARDING]] : tensor<8x8xf32>
  // CHECK-NEXT:  }
  func.func private @sparse_offload_callee_sharded(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = stablehlo.custom_call @Sharding(%arg0) {
      mhlo.frontend_attributes = {
        xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{\"a\"}, {}]>]>"
      }
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }

  // CHECK-LABEL: func @main(
  // CHECK-SAME:      %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}) -> tensor<8x8xf32> {
  // CHECK-NEXT:    %[[CALL:.*]] = call @sparse_offload_callee_sharded(%arg0) {mhlo.frontend_attributes = {_xla_compute_type = "sparseoffload"}} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:    return %[[CALL]] : tensor<8x8xf32>
  // CHECK-NEXT:  }
  func.func @main(%arg0: tensor<8x8xf32> {
    mhlo.frontend_attributes = {
      xla.sdy.sharding = "#sdy.sharding<@mesh, [{\"a\"}, {}]>"
    }
  }) -> tensor<8x8xf32> {
    %0 = call @sparse_offload_callee_sharded(%arg0) {
      mhlo.frontend_attributes = {_xla_compute_type = "sparseoffload"}
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

// CHECK-LABEL: module @sparse_offload_multi_args
module @sparse_offload_multi_args attributes {
  mhlo.frontend_attributes = {
    xla.sdy.meshes = "{mesh = #sdy.mesh<[\"a\"=2, \"b\"=2]>}"
  }
} {
  // CHECK-LABEL: func private @sparse_offload_callee_multi_args(
  // CHECK-SAME:      %arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT:    %[[BARRIER:.*]] = sdy.propagation_barrier %arg0 allowed_direction=FORWARD : tensor<8x8xf32>
  // CHECK-NEXT:    %[[SHARDING:.*]] = stablehlo.custom_call @Sharding(%[[BARRIER]]) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:    return %[[SHARDING]] : tensor<8x8xf32>
  // CHECK-NEXT:  }
  func.func private @sparse_offload_callee_multi_args(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = stablehlo.custom_call @Sharding(%arg0) {
      mhlo.frontend_attributes = {
        xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{}, {}]>]>"
      }
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }

  func.func @main(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = call @sparse_offload_callee_multi_args(%arg0, %arg1) {
      mhlo.frontend_attributes = {_xla_compute_type = "sparseoffload"}
    } : (tensor<8x8xf32>, tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

// CHECK-LABEL: module @non_offloaded_call
module @non_offloaded_call attributes {
  mhlo.frontend_attributes = {
    xla.sdy.meshes = "{mesh = #sdy.mesh<[\"a\"=2, \"b\"=2]>}"
  }
} {
  // CHECK-LABEL: func private @non_offloaded_callee(
  // CHECK-SAME:      %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT:    %[[SHARDING:.*]] = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}]>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:    return %[[SHARDING]] : tensor<8x8xf32>
  // CHECK-NEXT:  }
  func.func private @non_offloaded_callee(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = stablehlo.custom_call @Sharding(%arg0) {
      mhlo.frontend_attributes = {
        xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{}, {}]>]>"
      }
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }

  func.func @main(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = call @non_offloaded_callee(%arg0) : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

// CHECK-LABEL: module @external_offloaded_call
module @external_offloaded_call {
  // CHECK-LABEL: func private @external_callee(tensor<8x8xf32>) -> tensor<8x8xf32>
  func.func private @external_callee(tensor<8x8xf32>) -> tensor<8x8xf32>

  func.func @main(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = call @external_callee(%arg0) {
      mhlo.frontend_attributes = {_xla_compute_type = "sparseoffload"}
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

// CHECK-LABEL: module @sparse_offload_unreduced_module
// CHECK:         sdy.mesh @mesh = <["a"=2, "b"=2]>
module @sparse_offload_unreduced_module attributes {
  mhlo.frontend_attributes = {
    xla.sdy.meshes = "{mesh = #sdy.mesh<[\"a\"=2, \"b\"=2]>}"
  }
} {

  // CHECK-LABEL: func private @sparse_offload_callee_unreduced(
  // CHECK-SAME:      %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NOT:     sdy.propagation_barrier
  // CHECK:         %[[SHARDING:.*]] = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={"a"}>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:    return %[[SHARDING]] : tensor<8x8xf32>
  // CHECK-NEXT:  }
  func.func private @sparse_offload_callee_unreduced(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = stablehlo.custom_call @Sharding(%arg0) {
      mhlo.frontend_attributes = {
        xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced={\"a\"}>]>"
      }
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }

  // CHECK-LABEL: func @main(
  // CHECK-SAME:      %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK-NEXT:    %[[CALL:.*]] = call @sparse_offload_callee_unreduced(%arg0) {mhlo.frontend_attributes = {_xla_compute_type = "sparseoffload"}} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:    return %[[CALL]] : tensor<8x8xf32>
  // CHECK-NEXT:  }
  func.func @main(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = call @sparse_offload_callee_unreduced(%arg0) {
      mhlo.frontend_attributes = {_xla_compute_type = "sparseoffload"}
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

// -----

// CHECK-LABEL: module @sparse_offload_unreduced_max_module
// CHECK:         sdy.mesh @mesh = <["a"=2, "b"=2]>
module @sparse_offload_unreduced_max_module attributes {
  mhlo.frontend_attributes = {
    xla.sdy.meshes = "{mesh = #sdy.mesh<[\"a\"=2, \"b\"=2]>}"
  }
} {

  // CHECK-LABEL: func private @sparse_offload_callee_unreduced_max(
  // CHECK-SAME:      %arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
  // CHECK:         %[[SHARDING:.*]] = stablehlo.custom_call @Sharding(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced=max{"a"}>]>} : (tensor<8x8xf32>) -> tensor<8x8xf32>
  // CHECK-NEXT:    return %[[SHARDING]] : tensor<8x8xf32>
  // CHECK-NEXT:  }
  func.func private @sparse_offload_callee_unreduced_max(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %0 = stablehlo.custom_call @Sharding(%arg0) {
      mhlo.frontend_attributes = {
        xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh, [{}, {}], unreduced=max{\"a\"}>]>"
      }
    } : (tensor<8x8xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}

