// RUN: sdy_opt %s -xla-sdy-round-trip-import-shardy-attrs -split-input-file -verify-diagnostics
// // expected-error @below {{JAX does not support multiple meshes with different axis sizes.}}
module @multiple_func_result_shardings attributes {mhlo.frontend_attributes = {xla.sdy.meshes =
    "{mesh = #sdy.mesh<[\\\22a\\\22=8, \\\22b\\\22=8]>, mesh2 = #sdy.mesh<[\\\22a\\\22=8, \\\22b\\\22=4]>}"}} {
}
