// RUN: xla-cpu-opt %s -split-input-file -xla-legalize-abi \
// RUN:     -allow-unregistered-dialect \
// RUN:   | FileCheck %s

func.func @all_custom(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x3x4xf32>)
    -> tensor<2x3x4xf32> attributes {
      xla_entry_computation_parameter_layouts = [
        dense<[0, 1, 2]> : tensor<3xindex>,
        dense<[1, 2, 0]> : tensor<3xindex>
      ],
      xla_entry_computation_result_layout = dense<[2, 0, 1]> : tensor<3xindex>
    } {
  %add = mhlo.add %arg0, %arg1 : tensor<2x3x4xf32>
  func.return %add : tensor<2x3x4xf32>
}

// CHECK-LABEL: @all_custom
//  CHECK-SAME:   %[[ARG0:.*]]: tensor{{.*}}, %[[ARG1:.*]]: tensor{{.*}}
//   CHECK-NOT:   attributes
//       CHECK: %[[R0:.*]] = mhlo.reshape %[[ARG0]] {{.*}} -> tensor<4x3x2xf32>
//       CHECK: %[[T0:.*]] = "mhlo.transpose"(%[[R0]]) {{.*}} -> tensor<2x3x4xf32>
//       CHECK: %[[R1:.*]] = mhlo.reshape %[[ARG1]] {{.*}} -> tensor<2x4x3xf32>
//       CHECK: %[[T1:.*]] = "mhlo.transpose"(%[[R1]]) {{.*}} -> tensor<2x3x4xf32>
//       CHECK: %[[ADD:.*]] = mhlo.add %[[T0]], %[[T1]]
//       CHECK: %[[TR:.*]] = "mhlo.transpose"(%[[ADD]]) {{.*}} -> tensor<3x2x4xf32>
//       CHECK: %[[RR:.*]] = mhlo.reshape %[[TR]] {{.*}} -> tensor<2x3x4xf32>
//       CHECK: return %[[RR]]

// -----

func.func @scalar_and_default_args(%arg0: tensor<f32>, %arg1: tensor<2x3xf32>,
    %arg2: tensor<2x3xf32>) -> tensor<f32> attributes {
      xla_entry_computation_parameter_layouts = [
        dense<> : tensor<0xindex>,
        dense<[0, 1]> : tensor<2xindex>,
        dense<[1, 0]> : tensor<2xindex>
      ],
      xla_entry_computation_result_layout = dense<> : tensor<0xindex>
    } {
  %result = "test.dummy"(%arg0, %arg1, %arg2) :
    (tensor<f32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<f32>
  func.return %result : tensor<f32>
}

// CHECK-LABEL: @scalar_and_default_args
//  CHECK-SAME:   %[[ARG0:.*]]: tensor{{.*}}, %[[ARG1:.*]]: tensor{{.*}}, %[[ARG2:.*]]: tensor{{.*}}
//       CHECK: %[[R1:.*]] = mhlo.reshape %[[ARG1]] {{.*}} -> tensor<3x2xf32>
//       CHECK: %[[T1:.*]] = "mhlo.transpose"(%[[R1]]) {{.*}} -> tensor<2x3xf32>
//       CHECK: %[[R:.*]] = "test.dummy"(%[[ARG0]], %[[T1]], %[[ARG2]])
//       CHECK: return %[[R]]

// -----

func.func @two_scalar_return_values() -> (tensor<f32>, tensor<f32>) attributes {
      xla_entry_computation_result_layout = [
        dense<> : tensor<0xindex>,
        dense<> : tensor<0xindex>
      ]
    } {
  %result:2 = "test.dummy"() : () -> (tensor<f32>, tensor<f32>)
  func.return %result#0, %result#1 : tensor<f32>, tensor<f32>
}

// CHECK-LABEL: @two_scalar_return_values

// -----

func.func @return_i1() -> (tensor<i1>) {
  %result = "test.dummy"() : () -> tensor<i1>
  func.return %result : tensor<i1>
}

// CHECK-LABEL: @return_i1() -> tensor<ui8>
//       CHECK: %[[I1:.*]] = "test.dummy"() : () -> tensor<i1>
//       CHECK: %[[U8:.*]] = mhlo.convert %[[I1]] {{.*}} -> tensor<ui8>
//       CHECK: return %[[U8]]

// -----

func.func @custom_call(%arg0: tensor<f32>, %arg1: tensor<2x3xf32>) -> (tensor<6x3xf32>, tensor<3xf32>) {
  %result:2 = "mhlo.custom_call"(%arg0, %arg1) {
    call_target_name = "yolo",
    operand_layouts = [dense<> : tensor<0xindex>, dense<[0, 1]> : tensor<2xindex>],
    result_layouts = [dense<[0, 1]> : tensor<2xindex>, dense<0> : tensor<1xindex>]
  } : (tensor<f32>, tensor<2x3xf32>) -> (tensor<6x3xf32>, tensor<3xf32>)
  return %result#0, %result#1 : tensor<6x3xf32>, tensor<3xf32>
}

// CHECK-LABEL: @custom_call
//  CHECK-SAME:   %[[ARG0:.*]]: tensor{{.*}}, %[[ARG1:.*]]: tensor{{.*}}
//   CHECK-NOT: operand_layouts
//   CHECK-NOT: result_layouts
//       CHECK: %[[T1:.*]] = "mhlo.transpose"(%[[ARG1]]) {{.*}} -> tensor<3x2xf32>
//       CHECK: %[[R1:.*]] = mhlo.reshape %[[T1]] {{.*}} -> tensor<2x3xf32>
//       CHECK: %[[CC:.*]]:2 = mhlo.custom_call @yolo(%[[ARG0]], %[[R1]])
//       CHECK: %[[RR:.*]] = mhlo.reshape %[[CC]]#0 {{.*}} -> tensor<3x6xf32>
//       CHECK: %[[TR:.*]] = "mhlo.transpose"(%[[RR]]) {{.*}} -> tensor<6x3xf32>
//       CHECK: return %[[TR]], %[[CC]]#1

// -----

func.func @custom_call_i1_input(%arg0: tensor<42xi1>) {
  "mhlo.custom_call"(%arg0) { call_target_name = "yolo" }
      : (tensor<42xi1>) -> ()
  return
}

// CHECK-LABEL: @custom_call_i1_input
// CHECK: %[[CONVERTED:.*]] = mhlo.convert {{.*}} : (tensor<42xi1>) -> tensor<42xui8>
// CHECK: mhlo.custom_call @yolo(%[[CONVERTED]])

// -----

func.func @constant_with_layout() -> tensor<2x3xf32> {
  %c = "mhlo.constant"() {
    value = dense<[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]> : tensor<2x3xf32>,
    result_layout = dense<[0, 1]> : tensor<2xindex>
  } : () -> tensor<2x3xf32>
  return %c : tensor<2x3xf32>
}

// CHECK-LABEL: @constant_with_layout
//       CHECK: %[[CST:.*]] = mhlo.constant {{.*}} : tensor<3x2xf32>
//       CHECK: %[[TR:.*]] = "mhlo.transpose"(%[[CST]]) {{.*}} -> tensor<2x3xf32>
//       CHECK: return %[[TR]]

// -----

func.func @non_tensor_inouts() -> !mhlo.token {
  %0 = mhlo.create_token : !mhlo.token
  %1 = "mhlo.custom_call"(%0) {
      call_target_name = "yolo",
      operand_layouts = [dense<> : tensor<0xindex>],
      result_layouts = [dense<> : tensor<0xindex>]
  } : (!mhlo.token) -> (!mhlo.token)
  return %1 : !mhlo.token
}

// CHECK-LABEL: @non_tensor_inouts
