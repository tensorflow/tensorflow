// RUN: ifrt-opt --ifrt-legalize-to-vifrt --symbol-dce --split-input-file -verify-diagnostics %s

!array_t0 = !ifrt.array<tensor<2x4xi32>,
                        #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
!array_t1 = !ifrt.array<tensor<2x4xi32>,
                        #ifrt.sharding_param<1x1 to [0] on 2>, [2,3]>
func.func @error_on_op_attr_from_another_dialect(%arg0: !array_t0)
     -> (!array_t1) attributes {ifrt.function} {
  // expected-error@+1 {{failed to legalize operation 'ifrt.CopyArrays' that was explicitly marked illegal}}
  %0, %ctrl = ifrt.CopyArrays(%arg0) {invalid_attr = #vifrt<devices_v1[0,1]>}
      : (!array_t0) -> !array_t1
  return %0 : !array_t1
}

// -----

!array_t0 = !ifrt.array<tensor<2x4xi32>,
                        #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
// expected-error@+1 {{'func.func' op arguments may only have dialect attributes}}
func.func @error_on_arg_attr_from_another_dialect(
     %arg0: !array_t0 {invalid_attr = #vifrt<devices_v1[0,1]>}) -> (!array_t0)
     attributes {ifrt.function} {
  return %arg0 : !array_t0
}

// -----

!array_t0 = !ifrt.array<tensor<2x4xi32>,
                        #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
// expected-error@+1 {{'func.func' op results may only have dialect attributes}}
func.func @error_on_res_attr_from_another_dialect(%arg0: !array_t0)
     -> (!array_t0 {invalid_attr = #vifrt<devices_v1[0,1]>})
     attributes {ifrt.function} {
  return %arg0 : !array_t0
}

// -----

!array_t0 = !ifrt.array<tensor<2x4xi32>,
                        #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>
// expected-error@+1 {{failed to legalize operation 'func.func' that was explicitly marked illegal}}
func.func @error_on_func_attr_from_another_dialect(
     %arg0: !array_t0) -> (!array_t0)
     attributes {ifrt.function, invalid_attr = #vifrt<devices_v1[0,1]>} {
  return %arg0 : !array_t0
}
