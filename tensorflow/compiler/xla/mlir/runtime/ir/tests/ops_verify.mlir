// RUN: xla-runtime-opt -verify-diagnostics -split-input-file %s

// -----
// expected-error @+1 {{func op named 'foo' not found for export}}
rt.export @foo

// -----
// expected-error @+1 {{'func.func' op requires "rt.exported" to be an integer attribute}}
func.func private @verify_rt_exported(%arg0: memref<?xf32>)
  attributes { rt.exported } {
  call @custom_call(%arg0) : (memref<?xf32>) -> ()
  return
}

// -----
func.func private @verify_exported_non_func(%arg0: memref<?xf32>) {
  // expected-error @+1 {{"rt.exported" can only be applied to a function}}
  call @custom_call(%arg0) { rt.exported = 0 : i32}: (memref<?xf32>) -> ()
  return
}

// -----
func.func private @verify_exported_non_func(%arg0: memref<?xf32>) {
  // expected-error @+1 {{"rt.dynamic" can only be applied to a custom call declaration}}
  call @custom_call(%arg0) {rt.dynamic}: (memref<?xf32>) -> ()
  return
}

// -----
// expected-error @+1 {{'func.func' op requires non-empty body for function with attribute "rt.exported"}}
func.func private @verify_rt_exported(%arg0: memref<?xf32>)
  attributes { rt.exported = 0 : i32 }


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
  attributes { rt.exported = 0 : i32 } {
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

// -----
func.func @trace(%ctx: !rt.execution_context) {
  // expected-error @+1 {{'rt.trace' invalid kind of attribute specified}}
  rt.trace "string attribute", %ctx {}
  return
}

// -----
func.func @trace_attribute(%ctx: !rt.execution_context) {
  // expected-error @+1 {{"rt.trace" to be a trace annotation attribute}}
  call @custom_call() { rt.trace = "foo" } : () -> ()
  return
}
