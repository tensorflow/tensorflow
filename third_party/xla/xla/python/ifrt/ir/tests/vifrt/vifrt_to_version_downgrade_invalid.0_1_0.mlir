// RUN: ifrt-opt %s --ifrt-legalize-to-vifrt --vifrt-to-version='target_version=0.1.0' --symbol-dce --verify-diagnostics --split-input-file

!array = !ifrt.array<tensor<6xi32>,
                     #ifrt.sharding_param<2 to [0,1] on 2x2 unreduced [1]>,
                     [0, 1, 2, 3]>

// expected-error@-7 {{failed to convert to VIFRT version 0.1.0}}
// expected-error@+1 {{failed to legalize operation 'vifrt.FuncV1' that was explicitly marked illegal}}
func.func @array_with_unreduced_axes(%arg0: !array) attributes {ifrt.function} {
  %0, %ctrl = ifrt.CopyArrays(%arg0) : (!array) -> !array
  return
}
