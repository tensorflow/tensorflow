// RUN: ifrt-opt %s --ifrt-legalize-to-vifrt --vifrt-to-version='target_version=0.3.0' --symbol-dce --verify-diagnostics --split-input-file

!array = !ifrt.array<tensor<2x4xi32>,
                     #ifrt.sharding_param<1x1 to [0] on 2>, [0,1]>

// expected-error@-6 {{failed to convert to VIFRT version 0.3.0}}
// expected-error@+2 {{failed to legalize operation 'vifrt.CopyArraysV2' that was explicitly marked illegal}}
func.func @copy_array_with_reuse(%arg0: !array) attributes {ifrt.function} {
  %0, %ctrl = ifrt.CopyArrays(%arg0) {reuse = true} : (!array) -> !array
  return
}
