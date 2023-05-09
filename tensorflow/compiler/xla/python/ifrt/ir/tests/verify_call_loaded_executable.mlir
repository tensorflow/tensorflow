// RUN: ifrt-opt %s -split-input-file -verify-diagnostics

func.func @good(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>) {
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0)
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, 1x2 to [0] on 2, [0,1]>
  return
}

ifrt.LoadedExecutable @callee {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, 1x2 to [0] on 2, [0,1]>

// -----

func.func @good_with_control_dep(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>,
    %arg1: !ifrt.control) {
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0) after %arg1
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, 1x2 to [0] on 2, [0,1]>
  return
}

ifrt.LoadedExecutable @callee {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, 1x2 to [0] on 2, [0,1]>

// -----

func.func @requires_valid_reference() {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op requires '@missing_reference' to reference a valid LoadedExecutable}}
  %ctrl_0 = ifrt.CallLoadedExecutable @missing_reference() : () -> ()
  return
}

// -----

func.func @requires_loaded_executable_callee(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>) {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op requires '@wrong_reference' to reference a valid LoadedExecutable}}
  %ctrl_0 = ifrt.CallLoadedExecutable @wrong_reference() : () -> ()
  return
}

func.func @wrong_reference() {
  return
}

// -----

func.func @requires_matching_signature(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>) {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op requires callee signature matching '(!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0, 1]>) -> !ifrt.array<tensor<4x3xi32>, 1x2 to [0] on 2, [0, 1]>'. Actual '(!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0, 1]>) -> !ifrt.array<tensor<4x4xi32>, 1x2 to [0] on 2, [0, 1]>'}}
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0)
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<4x3xi32>, 1x2 to [0] on 2, [0,1]>
  return
}

ifrt.LoadedExecutable @callee {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<4x4xi32>, 1x2 to [0] on 2, [0,1]>

// -----

func.func @io_aliases_should_be_pairs(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>) {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op attribute 'io_aliases' failed to satisfy constraint: Array of pairs of aliased input/output indices}}
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0)
    {io_aliases=[array<i32: 0>]}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>
  return
}

ifrt.LoadedExecutable @callee {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>

// -----

func.func @io_aliases_should_have_valid_input_index(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>) {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op can't alias input #1 to output #0 as only having 1 inputs}}
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0)
    {io_aliases=[array<i32: 1, 0>]}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>
  return
}

ifrt.LoadedExecutable @callee {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>

// -----

func.func @io_aliases_should_only_alias_input_once(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>) {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op can't alias input #0 more than once}}
  %0, %1, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0)
    {io_aliases=[array<i32: 0, 0>, array<i32: 0, 1>]}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>,
        !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
  return
}

ifrt.LoadedExecutable @callee {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>

// -----

func.func @io_aliases_should_have_valid_output_index(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>) {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op can't alias input #0 to output #1 as only having 1 outputs}}
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0)
    {io_aliases=[array<i32: 0, 1>]}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>
  return
}

ifrt.LoadedExecutable @callee {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>

// -----

func.func @io_aliases_should_only_alias_output_once(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>) {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op can't alias output #0 more than once}}
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0, %arg0)
    {io_aliases=[array<i32: 0, 0>, array<i32: 1, 0>]}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>,
       !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>
  return
}

ifrt.LoadedExecutable @callee {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>

// -----

func.func @io_aliases_should_have_same_type(
    %arg0: !ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>) {
  // expected-error@+1 {{'ifrt.CallLoadedExecutable' op can't alias input #0 to output #0 with different types: '!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0, 1]>' vs '!ifrt.array<tensor<2x2xi32>, 2x1 to [0] on 2, [0, 1]>'}}
  %0, %ctrl_0 = ifrt.CallLoadedExecutable @callee(%arg0)
    {io_aliases=[array<i32: 0, 0>]}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, 2x1 to [0] on 2, [0,1]>
  return
}

ifrt.LoadedExecutable @callee {devices=array<i64: 0, 1>}
    : (!ifrt.array<tensor<2x2xi32>, 1x1 to [0] on 2, [0,1]>)
    -> !ifrt.array<tensor<2x2xi32>, 2x1 to [0] on 2, [0,1]>
