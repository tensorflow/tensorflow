// RUN: ifrt-opt %s -split-input-file -verify-diagnostics

func.func @good_function_attr() attributes {ifrt.function} {
  return
}

func.func @good_donated_attr(
    %arg0: !ifrt.array<tensor<4x4xi32>,
                       #ifrt.sharding_param<1x1 to [0] on 1>,
                       [0]> {ifrt.donated})
    attributes {ifrt.function} {
  return
}

// -----

// expected-error@+1 {{'func.func' op has `ifrt.function` attr that is not a UnitAttr}}
func.func @func_attr_should_be_unit() attributes {ifrt.function = "1"} {
  return
}

// -----

// expected-error@+1 {{'builtin.module' op has `ifrt.function` attr but is not a function}}
module @func_attr_should_be_on_func_op attributes {ifrt.function} {}

// -----

// expected-error@+1 {{'func.func' op has `ifrt.donated` arg attr that is not a UnitAttr}}
func.func @donated_attr_should_be_unit(
    %arg0: !ifrt.array<tensor<4x4xi32>,
                       #ifrt.sharding_param<1x1 to [0] on 1>, [0]>
        {ifrt.donated = "1"})
    attributes {ifrt.function} {
  return
}

// -----

// expected-error@+1 {{'func.func' op has `ifrt.donated` arg attr but not has `ifrt.function` attr}}
func.func @donated_attr_should_be_with_func_attr(
    %arg0: !ifrt.array<tensor<4x4xi32>,
                       #ifrt.sharding_param<1x1 to [0] on 1>,
                       [0]> {ifrt.donated}) {
  return
}
