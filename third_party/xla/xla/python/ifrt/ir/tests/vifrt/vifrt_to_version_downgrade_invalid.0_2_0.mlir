// RUN: ifrt-opt %s --ifrt-legalize-to-vifrt --vifrt-to-version='target_version=0.2.0' --symbol-dce --verify-diagnostics --split-input-file

!token = !ifrt.array<tensor<!vifrt.token_v1>,
                     #ifrt.sharding_param< to [0] on 2>, [0, 1]>

// expected-error@-6 {{failed to convert to VIFRT version 0.2.0}}
// expected-error@+1 {{failed to legalize operation 'vifrt.FuncV1' that was explicitly marked illegal}}
func.func @token_array(%arg0: !token) attributes {ifrt.function} {
  %0, %ctrl = ifrt.CopyArrays(%arg0) : (!token) -> !token
  return
}
