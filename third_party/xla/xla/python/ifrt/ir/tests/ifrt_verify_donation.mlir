// RUN: ifrt-opt %s -ifrt-verify-donation -split-input-file -verify-diagnostics | FileCheck %s

!array0 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [2, 3]>
// CHECK-LABEL: @donate_call_output_to_call_and_reshard
module @donate_call_output_to_call_and_reshard {
  func.func @main(%arg0: !array0 {ifrt.donated}) -> !array1
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0,1]
        : (!array0) -> !array0
    %1, %ctrl_1 = ifrt.Call @identity(%0) on devices [0,1]
        {io_aliases=[array<i32: 0, 0>]} : (!array0) -> !array0
    %2, %ctrl_2 = ifrt.Reshard(%1) {donated=true} : (!array0) -> !array1
    return %2 : !array1
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array0 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [2]>
!array2 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<1 to [0] on 1>, [3]>
module @donate_to_reshard_duplicated_arg {
  func.func @main(%arg0: !array0 {ifrt.donated}) -> (!array1, !array2)
        attributes {ifrt.function} {
    %0, %1, %ctrl_1 = ifrt.Reshard(%arg0, %arg0) {donated=true}
        : (!array0, !array0) -> (!array1, !array2)
    return %0, %1 : !array1, !array2
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @alias_to_two_calls_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!array, !array)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0,1]
        {io_aliases=[array<i32: 0, 0>]} : (!array) -> !array
    // expected-error @+1 {{'ifrt.Call' op input #0 of @identity was already donated}}
    %1, %ctrl_1 = ifrt.Call @identity(%arg0) on devices [0,1]
        {io_aliases=[array<i32: 0, 0>]} : (!array) -> !array
    return %0, %1 : !array, !array
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @donate_to_two_calls_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!array, !array)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0,1]
        {donated_input_indices=array<i32: 0>} : (!array) -> !array
    // expected-error @+1 {{'ifrt.Call' op input #0 of @identity was already donated}}
    %1, %ctrl_1 = ifrt.Call @identity(%arg0) on devices [0,1]
        {donated_input_indices=array<i32: 0>} : (!array) -> !array
    return %0, %1 : !array, !array
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @arg_donated_to_call_not_donated_to_program {
  func.func @main(%arg0: !array) -> (!array)
      attributes {ifrt.function} {
    // expected-error @+1 {{'ifrt.Call' op input #0 has not been donated to the program.}}
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0,1]
        {donated_input_indices=array<i32: 0>} : (!array) -> !array
    return %0 : !array
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array0 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [2, 3]>
module @program_arg_not_donated_error {
  func.func @main(%arg0: !array0) -> (!array1) attributes {ifrt.function} {
    // expected-error @+1 {{'ifrt.Reshard' op input #0 has not been donated to the program.}}
    %0, %ctrl_0 = ifrt.Reshard(%arg0) {donated=true} : (!array0) -> !array1
    return %0 : !array1
  }
}

// -----

!array0 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @arg_both_donated_and_not_donated_error {
  func.func @main(%arg0: !array0 {ifrt.donated}) -> !array0
      attributes {ifrt.function} {
    // expected-error @+1 {{'ifrt.Call' op input #0 of @add_two_args was already donated}}
    %0, %ctrl_0 = ifrt.Call @add_two_args(%arg0, %arg0) on devices [0,1]
        {io_aliases=[array<i32: 1, 0>]} : (!array0, !array0) -> !array0
    return %0 : !array0
  }

  func.func private @add_two_args(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>)
        -> tensor<2xi32> {
    %0 = mhlo.add %arg0, %arg1 : tensor<2xi32>
    return %0 : tensor<2xi32>
  }
}

// -----

!array0 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [2, 3]>
module @donate_to_two_reshards_error {
  func.func @main(%arg0: !array0 {ifrt.donated}) -> (!array1, !array1)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Reshard(%arg0) {donated=true} : (!array0) -> !array1
    // expected-error @+1 {{'ifrt.Reshard' op input #0 of op}}
    %1, %ctrl_1 = ifrt.Reshard(%arg0) {donated=true} : (!array0) -> !array1
    return %0, %1 : !array1, !array1
  }
}

// -----

!array0 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [2, 3]>
module @donate_to_two_reshards_error {
  func.func @main(%arg0: !array0 {ifrt.donated}) -> (!array1, !array1)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Reshard(%arg0) {donated=true} : (!array0) -> !array1
    // expected-error @+1 {{'ifrt.Reshard' op input #0 of op}}
    %1, %ctrl_1 = ifrt.Reshard(%arg0) {donated=true} : (!array0) -> !array1
    return %0, %1 : !array1, !array1
  }
}

// -----

!array0 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [2, 3]>
module @donate_to_reshard_and_call_error {
  func.func @main(%arg0: !array0 {ifrt.donated}) -> (!array0, !array1)
        attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0,1]
        {io_aliases=[array<i32: 0, 0>]} : (!array0) -> !array0
    // expected-error @+1 {{'ifrt.Reshard' op input #0 of op}}
    %1, %ctrl_1 = ifrt.Reshard(%arg0) {donated=true} : (!array0) -> !array1
    return %0, %1 : !array0, !array1
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array0 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [2, 3]>
module @donate_to_two_copy_arrays_error {
  func.func @main(%arg0: !array0 {ifrt.donated}) -> (!array1, !array1)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CopyArrays(%arg0) {donated=true} : (!array0) -> !array1
    // expected-error @+1 {{'ifrt.CopyArrays' op input #0 of op}}
    %1, %ctrl_1 = ifrt.CopyArrays(%arg0) {donated=true} : (!array0) -> !array1
    return %0, %1 : !array1, !array1
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @program_arg_not_donated_to_remap_error {
  func.func @main(%arg0: !array {ifrt.donated}, %arg1: !array) -> (!array)
      attributes {ifrt.function} {
    // expected-error @+1 {{'ifrt.RemapArrays' op input #1 has not been donated to the program.}}
    %0 = ifrt.RemapArrays(%arg0, %arg1)
      mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>,
                #ifrt.array_mapping<1, 0, [#ifrt.mapping<[0:1:1] to [1:2:1]>]>]
      {donated=true} : (!array, !array) -> !array
    return %0 : !array
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @donate_to_reshard_and_call_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!array)
        attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0,1]
        {io_aliases=[array<i32: 0, 0>]} : (!array) -> !array
    // expected-error @+1 {{'ifrt.RemapArrays' op input #1 of op}}
    %1 = ifrt.RemapArrays(%0, %arg0)
      mappings=[#ifrt.array_mapping<0, 0, [#ifrt.mapping<[0:1:1] to [0:1:1]>]>,
                #ifrt.array_mapping<1, 0, [#ifrt.mapping<[0:1:1] to [1:2:1]>]>]
      {donated=true} : (!array, !array) -> !array
    return %1 : !array
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>, #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @call_after_donation_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!array, !array)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0,1]
        {io_aliases=[array<i32: 0, 0>]} : (!array) -> !array
    // expected-error @+1 {{'ifrt.Call' op input #0 of @identity was already donated}}
    %1, %ctrl_1 = ifrt.Call @identity(%arg0) on devices [0,1]
        : (!array) -> !array
    return %0, %1 : !array, !array
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array0 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [2, 3]>
module @reshard_with_already_donated_array_error {
  func.func @main(%arg0: !array0 {ifrt.donated}) -> (!array0, !array1)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0,1]
        {io_aliases=[array<i32: 0, 0>]} : (!array0) -> !array0
    // expected-error @+1 {{'ifrt.Reshard' op input #0 of op}}
    %1, %ctrl_1 = ifrt.Reshard(%arg0) : (!array0) -> !array1
    return %0, %1 : !array0, !array1
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array0 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
!array1 = !ifrt.array<tensor<2xi32>,
                      #ifrt.sharding_param<2 to [0] on 2>, [2, 3]>
module @copy_arrays_with_already_donated_array_error {
  func.func @main(%arg0: !array0 {ifrt.donated}) -> (!array0, !array1)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0,1]
        {io_aliases=[array<i32: 0, 0>]} : (!array0) -> !array0
    // expected-error @+1 {{'ifrt.CopyArrays' op input #0 of op}}
    %1, %ctrl_1 = ifrt.CopyArrays(%arg0) : (!array0) -> !array1
    return %0, %1 : !array0, !array1
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}

// -----

!array = !ifrt.array<tensor<2xi32>,
                     #ifrt.sharding_param<2 to [0] on 2>, [0, 1]>
module @copy_arrays_with_already_donated_array_error {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!array, !array)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0,1]
        {io_aliases=[array<i32: 0, 0>]} : (!array) -> !array
    // expected-error @+1 {{'func.return' op result #1 of op at}}
    return %0, %arg0 : !array, !array
  }

  func.func private @identity(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    return %arg0 : tensor<2xi32>
  }
}
