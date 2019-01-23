// RUN: mlir-opt %s -split-input-file -verify

// -----

func @location_missing_l_paren() {
^bb:
  return loc) // expected-error {{expected '(' in inline location}}
}

// -----

func @location_missing_r_paren() {
^bb:
  return loc(unknown // expected-error@+1 {{expected ')' in inline location}}
}

// -----

func @location_invalid_instance() {
^bb:
  return loc() // expected-error {{expected location instance}}
}

// -----

func @location_callsite_missing_l_paren() {
^bb:
  return loc(callsite unknown  // expected-error {{expected '(' in callsite location}}
}

// -----

func @location_callsite_missing_callee() {
^bb:
  return loc(callsite( at )  // expected-error {{expected location instance}}
}

// -----

func @location_callsite_missing_at() {
^bb:
  return loc(callsite(unknown unknown) // expected-error {{expected 'at' in callsite location}}
}

// -----

func @location_callsite_missing_caller() {
^bb:
  return loc(callsite(unknown at )  // expected-error {{expected location instance}}
}

// -----

func @location_callsite_missing_r_paren() {
^bb:
  return loc(callsite( unknown at unknown  // expected-error@+1 {{expected ')' in callsite location}}
}

// -----

func @location_fused_missing_greater() {
^bb:
  return loc(fused<true [unknown]) // expected-error {{expected '>' after fused location metadata}}
}

// -----

func @location_fused_missing_metadata() {
^bb:
  // expected-error@+1 {{expected type}}
  return loc(fused<) // expected-error {{expected valid attribute metadata}}
}

// -----

func @location_fused_missing_l_square() {
^bb:
  return loc(fused<true>unknown]) // expected-error {{expected '[' in fused location}}
}

// -----

func @location_fused_missing_r_square() {
^bb:
  return loc(fused[unknown) // expected-error {{expected ']' in fused location}}
}
