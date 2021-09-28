// RUN: mlir-hlo-opt -mhlo-flatten-tuple %s | FileCheck %s

// CHECK-LABEL:  func @while_with_variadic(
// CHECK-SAME:                             %arg0: tensor<3xf32>) -> tensor<3xf32> {
// CHECK-DAG:      %[[CST_0:.*]] = constant dense<0> : tensor<1xi32>
// CHECK-DAG:      %[[CST_1:.*]] = constant dense<100> : tensor<2xi32>
// CHECK-DAG:      %[[CST_2:.*]] = constant dense<1.000000e+00> : tensor<1xf32>
// CHECK:          %[[WHILE_0:.*]]:4 = "mhlo.while"(%[[CST_0]], %[[CST_1]], %[[CST_2]], %arg0) ( {
// CHECK:          ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):  // no predecessors
// CHECK:            %[[SLICE_0:.*]] = "mhlo.slice"(%arg2) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
// CHECK:            %[[COMPARE_0:.*]] = "mhlo.compare"(%arg1, %[[SLICE_0]]) {comparison_direction = "LT"} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
// CHECK:            "mhlo.return"(%[[COMPARE_0]]) : (tensor<1xi1>) -> ()
// CHECK:          },  {
// CHECK:          ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):  // no predecessors
// CHECK:            %[[BROADCAST_IN_DIM_0:.*]] = "mhlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<3xf32>
// CHECK:            %[[ADD_0:.*]] = mhlo.add %[[BROADCAST_IN_DIM_0]], %arg4 : tensor<3xf32>
// CHECK:            "mhlo.return"(%arg1, %arg2, %arg3, %[[ADD_0]]) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
// CHECK:          }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
// CHECK:          return %[[WHILE_0]]#3 : tensor<3xf32>
// CHECK:        }
func @while_with_variadic(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = constant dense<0> : tensor<1xi32>
  %cst_1 = constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = constant dense<1.00> : tensor<1xf32>
  %1:4 = "mhlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ( {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):  // no predecessors
    %2 = constant dense<0> : tensor<i32>
    %3 = "mhlo.slice"(%arg2) {limit_indices = dense<[1]> : tensor<1xi64>, start_indices = dense<[0]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "mhlo.compare"(%arg1, %3) {comparison_direction = "LT"} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    "mhlo.return"(%4) : (tensor<1xi1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):  // no predecessors
    %3 = "mhlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<3xf32>
    %4 = mhlo.add %3, %arg4 : tensor<3xf32>
    "mhlo.return"(%arg1, %arg2, %arg3, %4) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  return %1#3: tensor<3xf32>
}

// CHECK-LABEL:  func @while_with_tuple(%arg0: tensor<3xf32>) -> tensor<3xf32> {
// CHECK-DAG:      %[[CST_0:.*]] = constant dense<0> : tensor<1xi32>
// CHECK-DAG:      %[[CST_1:.*]] = constant dense<100> : tensor<2xi32>
// CHECK-DAG:      %[[CST_2:.*]] = constant dense<1.000000e+00> : tensor<1xf32>
// CHECK:          %[[WHILE_0:.*]]:4 = "mhlo.while"(%[[CST_0]], %[[CST_1]], %[[CST_2]], %arg0) ( {
// CHECK:          ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):  // no predecessors
// CHECK:            %[[SLICE_0:.*]] = "mhlo.slice"(%arg2) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
// CHECK:            %[[COMPARE_0:.*]] = "mhlo.compare"(%arg1, %[[SLICE_0]]) {comparison_direction = "LT"} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
// CHECK:            "mhlo.return"(%[[COMPARE_0]]) : (tensor<1xi1>) -> ()
// CHECK:          },  {
// CHECK:          ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):  // no predecessors
// CHECK:            %[[BROADCAST_IN_DIM_0:.*]] = "mhlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<3xf32>
// CHECK:            %[[ADD_0:.*]] = mhlo.add %[[BROADCAST_IN_DIM_0]], %arg4 : tensor<3xf32>
// CHECK:            "mhlo.return"(%arg1, %arg2, %arg3, %[[ADD_0]]) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
// CHECK:          }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
// CHECK:          return %[[WHILE_0]]#3 : tensor<3xf32>
// CHECK:        }
func @while_with_tuple(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = constant dense<0> : tensor<1xi32>
  %cst_1 = constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = constant dense<1.00> : tensor<1xf32>
  %0 = "mhlo.tuple"(%cst_0, %cst_1, %cst_2, %arg0) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> tuple<tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>>
  %1 = "mhlo.while"(%0) ( {
  ^bb0(%arg2: tuple<tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>>):  // no predecessors
    %t0 = "mhlo.get_tuple_element"(%arg2) {index = 0 : i32} : (tuple<tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>>) -> tensor<1xi32>
    %t1 = "mhlo.get_tuple_element"(%arg2) {index = 1 : i32} : (tuple<tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>>) -> tensor<2xi32>
    %2 = constant dense<0> : tensor<i32>
    %3 = "mhlo.slice"(%t1) {limit_indices = dense<[1]> : tensor<1xi64>, start_indices = dense<[0]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "mhlo.compare"(%t0, %3) {comparison_direction = "LT"} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    "mhlo.return"(%4) : (tensor<1xi1>) -> ()
  },  {
  ^bb0(%arg2: tuple<tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>>):  // no predecessors
    %t0 = "mhlo.get_tuple_element"(%arg2) {index = 0 : i32} : (tuple<tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>>) -> tensor<1xi32>
    %t1 = "mhlo.get_tuple_element"(%arg2) {index = 1 : i32} : (tuple<tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>>) -> tensor<2xi32>
    %t2 = "mhlo.get_tuple_element"(%arg2) {index = 2 : i32} : (tuple<tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>>) -> tensor<1xf32>
    %t3 = "mhlo.get_tuple_element"(%arg2) {index = 3 : i32} : (tuple<tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>>) -> tensor<3xf32>
    %2 = "mhlo.broadcast_in_dim"(%t2) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<3xf32>
    %3 = mhlo.add %2, %t3 : tensor<3xf32>
    %4 = "mhlo.tuple"(%t0, %t1, %t2, %3) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> tuple<tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>>
    "mhlo.return"(%4) : (tuple<tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>>) -> ()
  }) : (tuple<tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>>) -> (tuple<tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>>)
  %5 = "mhlo.get_tuple_element"(%1) {index = 3 : i32} : (tuple<tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>>) -> tensor<3xf32>
  return %5: tensor<3xf32>
}

// CHECK-LABEL:  func @while_with_mix_types(%arg0: tensor<3xf32>) -> tensor<3xf32> {
// CHECK-DAG:      %[[CST_0:.*]] = constant dense<0> : tensor<1xi32>
// CHECK-DAG:      %[[CST_1:.*]] = constant dense<100> : tensor<2xi32>
// CHECK-DAG:      %[[CST_2:.*]] = constant dense<1.000000e+00> : tensor<1xf32>
// CHECK:          %[[WHILE_0:.*]]:4 = "mhlo.while"(%[[CST_0]], %[[CST_1]], %[[CST_2]], %arg0) ( {
// CHECK:          ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):  // no predecessors
// CHECK:            %[[SLICE_0:.*]] = "mhlo.slice"(%arg2) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
// CHECK:            %[[COMPARE_0:.*]] = "mhlo.compare"(%arg1, %[[SLICE_0]]) {comparison_direction = "LT"} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
// CHECK:            "mhlo.return"(%[[COMPARE_0]]) : (tensor<1xi1>) -> ()
// CHECK:          },  {
// CHECK:          ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):  // no predecessors
// CHECK:            %[[BROADCAST_IN_DIM_0:.*]] = "mhlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<3xf32>
// CHECK:            %[[ADD_0:.*]] = mhlo.add %[[BROADCAST_IN_DIM_0]], %arg4 : tensor<3xf32>
// CHECK:            "mhlo.return"(%arg1, %arg2, %arg3, %[[ADD_0]]) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
// CHECK:          }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
// CHECK:          return %[[WHILE_0]]#3 : tensor<3xf32>
// CHECK:        }
func @while_with_mix_types(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = constant dense<0> : tensor<1xi32>
  %cst_1 = constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = constant dense<1.00> : tensor<1xf32>
  %0 = "mhlo.tuple"(%cst_0, %cst_1) : (tensor<1xi32>, tensor<2xi32>) -> tuple<tensor<1xi32>, tensor<2xi32>>
  %1 = "mhlo.tuple"(%cst_2, %arg0) : (tensor<1xf32>, tensor<3xf32>) -> tuple<tensor<1xf32>, tensor<3xf32>>
  %2 = "mhlo.tuple"(%0, %1) : (tuple<tensor<1xi32>, tensor<2xi32>>, tuple<tensor<1xf32>, tensor<3xf32>>) -> tuple<tuple<tensor<1xi32>, tensor<2xi32>>, tuple<tensor<1xf32>, tensor<3xf32>>>
  %3 = "mhlo.while"(%2) ( {
  ^bb0( %arg1: tuple<tuple<tensor<1xi32>, tensor<2xi32>>, tuple<tensor<1xf32>, tensor<3xf32>>>):  // no predecessors
    %t0_1 = "mhlo.get_tuple_element"(%arg1) {index = 0 : i32} : (tuple<tuple<tensor<1xi32>, tensor<2xi32>>, tuple<tensor<1xf32>, tensor<3xf32>>>) -> tuple<tensor<1xi32>, tensor<2xi32>>
    %t0 = "mhlo.get_tuple_element"(%t0_1) {index = 0 : i32} : (tuple<tensor<1xi32>, tensor<2xi32>>) -> tensor<1xi32>
    %t1 = "mhlo.get_tuple_element"(%t0_1) {index = 1 : i32} : (tuple<tensor<1xi32>, tensor<2xi32>>) -> tensor<2xi32>
    %3 = constant dense<0> : tensor<i32>
    %4 = "mhlo.slice"(%t1) {limit_indices = dense<[1]> : tensor<1xi64>, start_indices = dense<[0]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
    %5 = "mhlo.compare"(%t0, %4) {comparison_direction = "LT"} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    "mhlo.return"(%5) : (tensor<1xi1>) -> ()
  },  {
  ^bb0(%arg1: tuple<tuple<tensor<1xi32>, tensor<2xi32>>, tuple<tensor<1xf32>, tensor<3xf32>>>):  // no predecessors
    %t0_1 = "mhlo.get_tuple_element"(%arg1) {index = 0 : i32} : (tuple<tuple<tensor<1xi32>, tensor<2xi32>>, tuple<tensor<1xf32>, tensor<3xf32>>>) -> tuple<tensor<1xi32>, tensor<2xi32>>
    %t2_3 = "mhlo.get_tuple_element"(%arg1) {index = 1 : i32} : (tuple<tuple<tensor<1xi32>, tensor<2xi32>>, tuple<tensor<1xf32>, tensor<3xf32>>>) -> tuple<tensor<1xf32>, tensor<3xf32>>
    %t2 = "mhlo.get_tuple_element"(%t2_3) {index = 0 : i32} : (tuple<tensor<1xf32>, tensor<3xf32>>) -> tensor<1xf32>
    %t3 = "mhlo.get_tuple_element"(%t2_3) {index = 1 : i32} : (tuple<tensor<1xf32>, tensor<3xf32>>) -> tensor<3xf32>
    %4 = "mhlo.broadcast_in_dim"(%t2) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<3xf32>
    %5 = mhlo.add %4, %t3 : tensor<3xf32>
    %6 = "mhlo.tuple"(%t2, %5) : (tensor<1xf32>, tensor<3xf32>) -> tuple<tensor<1xf32>, tensor<3xf32>>
    %7 = "mhlo.tuple"(%t0_1, %6) : (tuple<tensor<1xi32>, tensor<2xi32>>, tuple<tensor<1xf32>, tensor<3xf32>>) -> tuple<tuple<tensor<1xi32>, tensor<2xi32>>, tuple<tensor<1xf32>, tensor<3xf32>>>
    "mhlo.return"(%7) : (tuple<tuple<tensor<1xi32>, tensor<2xi32>>, tuple<tensor<1xf32>, tensor<3xf32>>>) -> ()
  }) : (tuple<tuple<tensor<1xi32>, tensor<2xi32>>, tuple<tensor<1xf32>, tensor<3xf32>>>) -> tuple<tuple<tensor<1xi32>, tensor<2xi32>>, tuple<tensor<1xf32>, tensor<3xf32>>>
  %4 = "mhlo.get_tuple_element"(%3) {index = 1 : i32} : (tuple<tuple<tensor<1xi32>, tensor<2xi32>>, tuple<tensor<1xf32>, tensor<3xf32>>>) -> tuple<tensor<1xf32>, tensor<3xf32>>
  %5 = "mhlo.get_tuple_element"(%4) {index = 1 : i32} : (tuple<tensor<1xf32>, tensor<3xf32>>) -> tensor<3xf32>
  return %5 : tensor<3xf32>
}

// CHECK-LABEL:  func @while_with_nested_tuple(%arg0: tensor<3xf32>) -> tensor<3xf32> {
// CHECK-DAG:      %[[CST_0:.*]] = constant dense<0> : tensor<1xi32>
// CHECK-DAG:      %[[CST_1:.*]] = constant dense<100> : tensor<2xi32>
// CHECK-DAG:      %[[CST_2:.*]] = constant dense<1.000000e+00> : tensor<1xf32>
// CHECK:          %[[WHILE_0:.*]]:4 = "mhlo.while"(%[[CST_0]], %[[CST_1]], %[[CST_2]], %arg0) ( {
// CHECK:          ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):  // no predecessors
// CHECK:            %[[SLICE_0:.*]] = "mhlo.slice"(%arg2) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
// CHECK:            %[[COMPARE_0:.*]] = "mhlo.compare"(%arg1, %[[SLICE_0]]) {comparison_direction = "LT"} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
// CHECK:            "mhlo.return"(%[[COMPARE_0]]) : (tensor<1xi1>) -> ()
// CHECK:          },  {
// CHECK:          ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):  // no predecessors
// CHECK:            %[[BROADCAST_IN_DIM_0:.*]] = "mhlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<3xf32>
// CHECK:            %[[ADD_0:.*]] = mhlo.add %[[BROADCAST_IN_DIM_0]], %arg4 : tensor<3xf32>
// CHECK:            "mhlo.return"(%arg1, %arg2, %arg3, %[[ADD_0]]) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
// CHECK:          }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
// CHECK:          return %[[WHILE_0]]#3 : tensor<3xf32>
// CHECK:        }
func @while_with_nested_tuple(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = constant dense<0> : tensor<1xi32>
  %cst_1 = constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = constant dense<1.00> : tensor<1xf32>
  %0 = "mhlo.tuple"(%arg0) : (tensor<3xf32>) -> tuple<tensor<3xf32>>
  %1 = "mhlo.tuple"(%cst_2, %0) : (tensor<1xf32>, tuple<tensor<3xf32>>) -> tuple<tensor<1xf32>, tuple<tensor<3xf32>>>
  %2 = "mhlo.tuple"(%cst_1, %1) : (tensor<2xi32>, tuple<tensor<1xf32>, tuple<tensor<3xf32>>>) -> tuple<tensor<2xi32>, tuple<tensor<1xf32>, tuple<tensor<3xf32>>>>
  %3:2 = "mhlo.while"(%cst_0, %2) ( {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tuple<tensor<2xi32>, tuple<tensor<1xf32>, tuple<tensor<3xf32>>>>):  // no predecessors
    %t0 = "mhlo.get_tuple_element"(%arg2) {index = 0 : i32} : (tuple<tensor<2xi32>, tuple<tensor<1xf32>, tuple<tensor<3xf32>>>>) -> tensor<2xi32>
    %3 = constant dense<0> : tensor<i32>
    %4 = "mhlo.slice"(%t0) {limit_indices = dense<[1]> : tensor<1xi64>, start_indices = dense<[0]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
    %5 = "mhlo.compare"(%arg1, %4) {comparison_direction = "LT"} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    "mhlo.return"(%5) : (tensor<1xi1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tuple<tensor<2xi32>, tuple<tensor<1xf32>, tuple<tensor<3xf32>>>>):  // no predecessors
    %t0 = "mhlo.get_tuple_element"(%arg2) {index = 0 : i32} : (tuple<tensor<2xi32>, tuple<tensor<1xf32>, tuple<tensor<3xf32>>>>) -> tensor<2xi32>
    %t1_2 = "mhlo.get_tuple_element"(%arg2) {index = 1 : i32} : (tuple<tensor<2xi32>, tuple<tensor<1xf32>, tuple<tensor<3xf32>>>>) -> tuple<tensor<1xf32>, tuple<tensor<3xf32>>>
    %t1 = "mhlo.get_tuple_element"(%t1_2) {index = 0 : i32} : (tuple<tensor<1xf32>, tuple<tensor<3xf32>>>) -> tensor<1xf32>
    %t2_2 = "mhlo.get_tuple_element"(%t1_2) {index = 1 : i32} : (tuple<tensor<1xf32>, tuple<tensor<3xf32>>>) -> tuple<tensor<3xf32>>
    %t2 = "mhlo.get_tuple_element"(%t2_2) {index = 0 : i32} : (tuple<tensor<3xf32>>) -> tensor<3xf32>
    %3 = "mhlo.broadcast_in_dim"(%t1) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<3xf32>
    %4 = mhlo.add %3, %t2 : tensor<3xf32>
    %5 = "mhlo.tuple"(%4) : (tensor<3xf32>) -> tuple<tensor<3xf32>>
    %6 = "mhlo.tuple"(%t1, %5) : (tensor<1xf32>, tuple<tensor<3xf32>>) -> tuple<tensor<1xf32>, tuple<tensor<3xf32>>>
    %7 = "mhlo.tuple"(%t0, %6) : (tensor<2xi32>, tuple<tensor<1xf32>, tuple<tensor<3xf32>>>) -> tuple<tensor<2xi32>, tuple<tensor<1xf32>, tuple<tensor<3xf32>>>>
    "mhlo.return"(%arg1, %7) : (tensor<1xi32>, tuple<tensor<2xi32>, tuple<tensor<1xf32>, tuple<tensor<3xf32>>>>) -> ()
  }) : (tensor<1xi32>, tuple<tensor<2xi32>, tuple<tensor<1xf32>, tuple<tensor<3xf32>>>>) -> (tensor<1xi32>, tuple<tensor<2xi32>, tuple<tensor<1xf32>, tuple<tensor<3xf32>>>>)
  %4 = "mhlo.get_tuple_element"(%3#1) {index = 1 : i32} : (tuple<tensor<2xi32>, tuple<tensor<1xf32>, tuple<tensor<3xf32>>>>) -> tuple<tensor<1xf32>, tuple<tensor<3xf32>>>
  %5 = "mhlo.get_tuple_element"(%4) {index = 1 : i32} : (tuple<tensor<1xf32>, tuple<tensor<3xf32>>>) -> tuple<tensor<3xf32>>
  %6 = "mhlo.get_tuple_element"(%5) {index = 0 : i32} : (tuple<tensor<3xf32>>) -> tensor<3xf32>
  return %6: tensor<3xf32>
}

// CHECK-LABEL:  func @while_generate_tuple_for_operands(
// CHECK-SAME:                                           %arg0: tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>) -> tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>> {
// CHECK:          %[[GET_TUPLE_ELEMENT_0:.*]] = "mhlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>) -> tensor<i32>
// CHECK:          %[[GET_TUPLE_ELEMENT_1:.*]] = "mhlo.get_tuple_element"(%arg0) {index = 1 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>) -> tuple<tensor<i32>, tuple<tensor<i32>>>
// CHECK:          %[[GET_TUPLE_ELEMENT_2:.*]] = "mhlo.get_tuple_element"(%[[GET_TUPLE_ELEMENT_1]]) {index = 0 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>>>) -> tensor<i32>
// CHECK:          %[[GET_TUPLE_ELEMENT_3:.*]] = "mhlo.get_tuple_element"(%[[GET_TUPLE_ELEMENT_1]]) {index = 1 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>>>) -> tuple<tensor<i32>>
// CHECK:          %[[GET_TUPLE_ELEMENT_4:.*]] = "mhlo.get_tuple_element"(%[[GET_TUPLE_ELEMENT_3]]) {index = 0 : i32} : (tuple<tensor<i32>>) -> tensor<i32>
// CHECK:          %[[WHILE_0:.*]]:3 = "mhlo.while"(%[[GET_TUPLE_ELEMENT_0]], %[[GET_TUPLE_ELEMENT_2]], %[[GET_TUPLE_ELEMENT_4]]) ( {
// CHECK:          ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>):  // no predecessors
// CHECK:            %[[COMPARE_0:.*]] = "mhlo.compare"(%arg1, %arg3) {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK:            "mhlo.return"(%[[COMPARE_0]]) : (tensor<i1>) -> ()
// CHECK:          },  {
// CHECK:          ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>):  // no predecessors
// CHECK:            %[[ADD_0:.*]] = mhlo.add %arg1, %arg2 : tensor<i32>
// CHECK:            "mhlo.return"(%[[ADD_0]], %arg2, %arg3) : (tensor<i32>, tensor<i32>, tensor<i32>) -> ()
// CHECK:          }) : (tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
// CHECK:          %[[TUPLE_0:.*]] = "mhlo.tuple"(%[[WHILE_0]]#2) : (tensor<i32>) -> tuple<tensor<i32>>
// CHECK:          %[[TUPLE_1:.*]] = "mhlo.tuple"(%[[WHILE_0]]#1, %[[TUPLE_0]]) : (tensor<i32>, tuple<tensor<i32>>) -> tuple<tensor<i32>, tuple<tensor<i32>>>
// CHECK:          %[[TUPLE_2:.*]] = "mhlo.tuple"(%[[WHILE_0]]#0, %[[TUPLE_1]]) : (tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>) -> tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>
// CHECK:          return %[[TUPLE_2]] : tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>
// CHECK:        }
func @while_generate_tuple_for_operands(%arg0: tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>) -> tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>> {
  %3 = "mhlo.while"(%arg0) ( {
  ^bb0(%arg9: tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>):  // no predecessors
    %t0 = "mhlo.get_tuple_element"(%arg9) {index = 0 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>) -> tensor<i32>
    %t_1_2 = "mhlo.get_tuple_element"(%arg9) {index = 1 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>) -> tuple<tensor<i32>, tuple<tensor<i32>>>
    %t_2_2 = "mhlo.get_tuple_element"(%t_1_2) {index = 1 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>>>) -> tuple<tensor<i32>>
    %t2 = "mhlo.get_tuple_element"(%t_2_2) {index = 0 : i32} : (tuple<tensor<i32>>) -> tensor<i32>
    %4 = "mhlo.compare"(%t0, %t2) {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%4) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg9: tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>):  // no predecessors
    %t0 = "mhlo.get_tuple_element"(%arg9) {index = 0 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>) -> tensor<i32>
    %t_1_2 = "mhlo.get_tuple_element"(%arg9) {index = 1 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>) -> tuple<tensor<i32>, tuple<tensor<i32>>>
    %t1 = "mhlo.get_tuple_element"(%t_1_2) {index = 0 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>>>) -> tensor<i32>
    %t_2_2 = "mhlo.get_tuple_element"(%t_1_2) {index = 1 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>>>) -> tuple<tensor<i32>>
    %t2 = "mhlo.get_tuple_element"(%t_2_2) {index = 0 : i32} : (tuple<tensor<i32>>) -> tensor<i32>
    %3 = mhlo.add %t0, %t1 : tensor<i32>
    %4 = "mhlo.tuple"(%3, %t_1_2) : (tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>) -> tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>
    "mhlo.return"(%4) : (tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>) -> ()
  }) : (tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>) -> (tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>)
  return %3: tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>
}

// CHECK-LABEL:  func @while_with_nonlocal_constants_in_regions() -> (tensor<i32>, tensor<i32>, tensor<i32>) {
// CHECK-DAG:      %[[CONSTANT_0:.*]] = mhlo.constant dense<1> : tensor<i32>
// CHECK-DAG:      %[[CONSTANT_1:.*]] = mhlo.constant dense<0> : tensor<i32>
// CHECK-DAG:      %[[CONSTANT_2:.*]] = mhlo.constant dense<1000> : tensor<i32>
// CHECK-DAG:      %[[CONSTANT_3:.*]] = mhlo.constant dense<100> : tensor<i32>
// CHECK:          %[[WHILE_0:.*]]:3 = "mhlo.while"(%[[CONSTANT_1]], %[[CONSTANT_0]], %[[CONSTANT_2]]) ( {
// CHECK:          ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>):  // no predecessors
// CHECK:            %[[COMPARE_0:.*]] = "mhlo.compare"(%arg0, %[[CONSTANT_3]]) {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK:            "mhlo.return"(%[[COMPARE_0]]) : (tensor<i1>) -> ()
// CHECK:          },  {
// CHECK:          ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>):  // no predecessors
// CHECK:            %[[ADD_0:.*]] = mhlo.add %arg0, %arg1 : tensor<i32>
// CHECK:            "mhlo.return"(%[[ADD_0]], %arg1, %arg2) : (tensor<i32>, tensor<i32>, tensor<i32>) -> ()
// CHECK:          }) : (tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
// CHECK:          return %[[WHILE_0]]#0, %[[WHILE_0]]#1, %[[WHILE_0]]#2 : tensor<i32>, tensor<i32>, tensor<i32>
// CHECK:        }
func @while_with_nonlocal_constants_in_regions() -> (tensor<i32>, tensor<i32>, tensor<i32>) {
  %cst_0 = mhlo.constant dense<1> : tensor<i32>
  %cst_1 = mhlo.constant dense<0> : tensor<i32>
  %cst_2 = mhlo.constant dense<1000> : tensor<i32>
  %cst_3 = mhlo.constant dense<100> : tensor<i32>
  %0 = "mhlo.tuple"(%cst_2) : (tensor<i32>) -> tuple<tensor<i32>>
  %1 = "mhlo.tuple"(%cst_0, %0) : (tensor<i32>, tuple<tensor<i32>>) -> tuple<tensor<i32>, tuple<tensor<i32>>>
  %2 = "mhlo.tuple"(%cst_1, %1) : (tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>) -> tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>
  %3 = "mhlo.while"(%2) ( {
  ^bb0(%arg9: tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>):  // no predecessors
    %t0 = "mhlo.get_tuple_element"(%arg9) {index = 0 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>) -> tensor<i32>
    %4 = "mhlo.compare"(%t0, %cst_3) {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%4) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg9: tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>):  // no predecessors
    %t0 = "mhlo.get_tuple_element"(%arg9) {index = 0 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>) -> tensor<i32>
    %t_1_2 = "mhlo.get_tuple_element"(%arg9) {index = 1 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>) -> tuple<tensor<i32>, tuple<tensor<i32>>>
    %t1 = "mhlo.get_tuple_element"(%t_1_2) {index = 0 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>>>) -> tensor<i32>
    %t_2_2 = "mhlo.get_tuple_element"(%t_1_2) {index = 1 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>>>) -> tuple<tensor<i32>>
    %t2 = "mhlo.get_tuple_element"(%t_2_2) {index = 0 : i32} : (tuple<tensor<i32>>) -> tensor<i32>
    %3 = mhlo.add %t0, %t1 : tensor<i32>
    %4 = "mhlo.tuple"(%3, %t_1_2) : (tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>) -> tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>
    "mhlo.return"(%4) : (tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>) -> ()
  }) : (tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>) -> (tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>)
  %t0 = "mhlo.get_tuple_element"(%3) {index = 0 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>) -> tensor<i32>
    %t_1_2 = "mhlo.get_tuple_element"(%3) {index = 1 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>) -> tuple<tensor<i32>, tuple<tensor<i32>>>
    %t1 = "mhlo.get_tuple_element"(%t_1_2) {index = 0 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>>>) -> tensor<i32>
    %t_2_2 = "mhlo.get_tuple_element"(%t_1_2) {index = 1 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>>>) -> tuple<tensor<i32>>
    %t2 = "mhlo.get_tuple_element"(%t_2_2) {index = 0 : i32} : (tuple<tensor<i32>>) -> tensor<i32>
  return %t0, %t1, %t2: tensor<i32>, tensor<i32>, tensor<i32>
}
