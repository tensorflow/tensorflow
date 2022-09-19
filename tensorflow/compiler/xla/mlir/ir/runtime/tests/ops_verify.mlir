// RUN: xla-runtime-opt -verify-diagnostics -split-input-file %s

// -----
// expected-error @+1 {{'func.func' op requires "rt.entrypoint" to be a unit attribute}}
func.func private @verify_rt_entrypoint(%arg0: memref<?xf32>)
  attributes { rt.entrypoint = 1 } {
  call @custom_call(%arg0) : (memref<?xf32>) -> ()
  return
}

// -----
func.func private @verify_entrypoint_non_func(%arg0: memref<?xf32>) {
  // expected-error @+1 {{"rt.entrypoint" can only be applied to a function}}
  call @custom_call(%arg0) {rt.entrypoint}: (memref<?xf32>) -> ()
  return
}

// -----
func.func private @verify_entrypoint_non_func(%arg0: memref<?xf32>) {
  // expected-error @+1 {{"rt.dynamic" can only be applied to a custom call declaration}}
  call @custom_call(%arg0) {rt.dynamic}: (memref<?xf32>) -> ()
  return
}

// -----
// expected-error @+1 {{'func.func' op requires non-empty body for function with attribute "rt.entrypoint"}}
func.func private @verify_rt_entrypoint(%arg0: memref<?xf32>)
  attributes { rt.entrypoint}


// -----
// expected-error @+1 {{'func.func' op requires "rt.custom_call" to only accept string value}}
func.func private @custom_call(%arg0: memref<?xf32>) -> memref<?xf32>
  attributes { rt.custom_call = 1, attr0 = 1 : i32, attr1 = 1.0 : f32 }


// -----
// expected-error @+1 {{'func.func' op requires "rt.custom_call" to only accept string value}}
func.func private @custom_call(%arg0: memref<?xf32>) -> memref<?xf32>
  attributes { rt.custom_call = 1, attr0 = 1 : i32, attr1 = 1.0 : f32 }


// -----
// expected-error @+1 {{'func.func' op requires "rt.custom_call" to only apply to a function declaration}}
func.func private @custom_call(%arg0: memref<?xf32>) -> memref<?xf32>
  attributes { rt.custom_call = "target", attr0 = 1 : i32, attr1 = 1.0 : f32 }{
  call @custom_call(%arg0) : (memref<?xf32>) -> ()
  return
}

// -----
func.func private @verify_custom_call_non_func(%arg0: memref<?xf32>)
  attributes { rt.entrypoint } {
  // expected-error @+1 {{"rt.custom_call" can only be applied to a function}}
  call @custom_call(%arg0) {rt.custom_call = "target"}: (memref<?xf32>) -> ()
  return
}

// -----
// expected-error @+1 {{'func.func' op has illegal attribute value of rt.constraint for argument 0}}
func.func private @constraint(
      %input0: memref<*xf32>   { rt.constraint = "test"  },
      %input1: memref<?x?xf32> { rt.constraint = "shape" },
      %perm: memref<4xi32>     { rt.constraint = "value" }
) attributes {rt.custom_call = "target"}