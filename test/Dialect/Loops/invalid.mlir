// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func @loop_for_lb(%arg0: f32, %arg1: index) {
  // expected-error@+1 {{operand #0 must be index}}
  "loop.for"(%arg0, %arg1, %arg1) : (f32, index, index) -> ()
  return
}

// -----

func @loop_for_ub(%arg0: f32, %arg1: index) {
  // expected-error@+1 {{operand #1 must be index}}
  "loop.for"(%arg1, %arg0, %arg1) : (index, f32, index) -> ()
  return
}

// -----

func @loop_for_step(%arg0: f32, %arg1: index) {
  // expected-error@+1 {{operand #2 must be index}}
  "loop.for"(%arg1, %arg1, %arg0) : (index, index, f32) -> ()
  return
}

// -----

func @loop_for_step_positive(%arg0: index) {
  // expected-error@+2 {{constant step operand must be positive}}
  %c0 = constant 0 : index
  "loop.for"(%arg0, %arg0, %c0) ({
    ^bb0(%arg1: index):
      "loop.terminator"() : () -> ()
  }) : (index, index, index) -> ()
  return
}

// -----

func @loop_for_one_region(%arg0: index) {
  // expected-error@+1 {{incorrect number of regions: expected 1 but found 2}}
  "loop.for"(%arg0, %arg0, %arg0) (
    {"loop.terminator"() : () -> ()},
    {"loop.terminator"() : () -> ()}
  ) : (index, index, index) -> ()
  return
}

// -----

func @loop_for_single_block(%arg0: index) {
  // expected-error@+1 {{expects region #0 to have 0 or 1 blocks}}
  "loop.for"(%arg0, %arg0, %arg0) (
    {
    ^bb1:
      "loop.terminator"() : () -> ()
    ^bb2:
      "loop.terminator"() : () -> ()
    }
  ) : (index, index, index) -> ()
  return
}

// -----

func @loop_for_single_index_argument(%arg0: index) {
  // expected-error@+1 {{expected body to have a single index argument for the induction variable}}
  "loop.for"(%arg0, %arg0, %arg0) (
    {
    ^bb0(%i0 : f32):
      "loop.terminator"() : () -> ()
    }
  ) : (index, index, index) -> ()
  return
}

// -----

func @loop_if_not_i1(%arg0: index) {
  // expected-error@+1 {{operand #0 must be 1-bit integer}}
  "loop.if"(%arg0) : (index) -> ()
  return
}

// -----

func @loop_if_more_than_2_regions(%arg0: i1) {
  // expected-error@+1 {{op has incorrect number of regions: expected 2}}
  "loop.if"(%arg0) ({}, {}, {}): (i1) -> ()
  return
}

// -----

func @loop_if_not_one_block_per_region(%arg0: i1) {
  // expected-error@+1 {{expects region #0 to have 0 or 1 blocks}}
  "loop.if"(%arg0) ({
    ^bb0:
      "loop.terminator"() : () -> ()
    ^bb1:
      "loop.terminator"() : () -> ()
  }, {}): (i1) -> ()
  return
}

// -----

func @loop_if_illegal_block_argument(%arg0: i1) {
  // expected-error@+1 {{requires that child entry blocks have no arguments}}
  "loop.if"(%arg0) ({
    ^bb0(%0 : index):
      "loop.terminator"() : () -> ()
  }, {}): (i1) -> ()
  return
}

