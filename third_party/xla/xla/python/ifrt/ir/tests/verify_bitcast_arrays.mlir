// RUN: ifrt-opt %s -split-input-file -verify-diagnostics

!array0 = !ifrt.array<tensor<1x2x2xi32>,
                      #ifrt.sharding_param<1x1x1 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
func.func @bitcast_drop_and_add_a_dimension(%arg0: !array0, %arg1: !array1)
    attributes {ifrt.function} {
  %0, %1, %ctrl_0 = ifrt.BitcastArrays(%arg0, %arg1) : (!array0, !array1) -> (!array1, !array0)
  return
}

// -----

!array0 = !ifrt.array<tensor<1x2x2xi32>,
                      #ifrt.sharding_param<1x1x1 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x2xf32>,
                      #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
func.func @bitcast_drop_one_dimension_different_dtype(%arg0: !array0)
    attributes {ifrt.function} {
  %0, %ctrl_0 = ifrt.BitcastArrays(%arg0) : (!array0) -> (!array1)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<1x2x2xi32>,
                      #ifrt.sharding_param<1x1x1 to [0] on 2>, [0,1]>
func.func @bitcast_add_one_dimension(%arg0: !array0)
    attributes {ifrt.function} {
  %0, %ctrl_0 = ifrt.BitcastArrays(%arg0) : (!array0) -> (!array1)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<1x2x2xi32>,
                      #ifrt.sharding_param<1x1x1 to [0] on 2>, [0,1]>
func.func @error_different_num_inputs_and_outputs(%arg0: !array0)
    attributes {ifrt.function} {
  // expected-error@+1 {{op requires the same number of input and output arrays}}
  %0, %1, %ctrl_0 = ifrt.BitcastArrays(%arg0) : (!array0) -> (!array0, !array1)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 2>, [0,1],
                      memory_kind = "device">
!array1 = !ifrt.array<tensor<1x2x2xi32>,
                      #ifrt.sharding_param<1x1x1 to [0] on 2>, [0,1],
                      memory_kind = "pinned_host">
func.func @error_different_memory_kind(%arg0: !array0)
    attributes {ifrt.function} {
  // expected-error@+1 {{op requires input array #0 and output array #0 to have the same memory kind}}
  %0, %ctrl_0 = ifrt.BitcastArrays(%arg0) : (!array0) -> (!array1)
  return
}

// -----

!array0 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!array1 = !ifrt.array<tensor<2x2xi32>,
                      #ifrt.sharding_param<1x1 to [0] on 3>, [0,1,2]>
func.func @error_different_num_devices(%arg0: !array0)
    attributes {ifrt.function} {
  // expected-error@+1 {{op requires input array #0 and output array #0 to have the same number of devices}}
  %0, %ctrl_0 = ifrt.BitcastArrays(%arg0) : (!array0) -> (!array1)
  return
}
