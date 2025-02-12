// RUN: ifrt-opt %s -split-input-file -spmd-expandable-interface-verification='excluded-dialects=arith' -verify-diagnostics

module @good_return_only {
  func.func @main(
      %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                         [0,1]>)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @simple_return(%arg0) on devices [0,1]
      : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                     [0,1]>)
      -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                     [0,1]>
    return
  }

  func.func @simple_return(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }
}

module @good_non_expandable_on_one_device{
  func.func @main(
      %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 1>,
                         [0]>)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @math_absi(%arg0) on devices [0]
      : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 1>,
                     [0]>)
      -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 1>,
                     [0]>
    return
  }

  func.func @math_absi(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = math.absi %arg0 : tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
}

module @good_excluded_dialect_on_two_devices {
  func.func @main(
      %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                         [0,1]>)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @arith_self_add(%arg0) on devices [0,1]
      : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                     [0,1]>)
      -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                     [0,1]>
    return
  }

  func.func @arith_self_add(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = arith.addi %arg0, %arg0 : tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
}

// -----

module @unexpandable_on_two_devices {
  func.func @main(
      %arg0: !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                         [0,1]>)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @math_absi(%arg0) on devices [0,1]
      : (!ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                     [0,1]>)
      -> !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 2>,
                     [0,1]>
    return
  }

  func.func @math_absi(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    // expected-error@+1 {{requires op to have `IfrtSpmdExpandable` OpInterface implemented}}
    %0 = math.absi %arg0 : tensor<2x2xi32>
    return %0 : tensor<2x2xi32>
  }
}
