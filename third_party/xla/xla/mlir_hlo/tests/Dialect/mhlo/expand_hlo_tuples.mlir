// RUN: mlir-hlo-opt %s -split-input-file -expand-hlo-tuples='entry-function=main' -allow-unregistered-dialect | FileCheck %s
// Check if the `expand-hlo-tuples` pass adds the right variable to return_op and function return type.

func.func @main(%arg0: tensor<1x1xf32>, %arg1: tensor<1x8x8x16xf32>) -> tuple<tensor<1024xf32>, tensor<1xf32>> {
  %1 = "mhlo.reshape"(%arg0) : (tensor<1x1xf32>) -> tensor<1xf32>
  %2 = "mhlo.reshape"(%arg1) : (tensor<1x8x8x16xf32>) -> tensor<1024xf32>
  %3 = "mhlo.tuple"(%2, %1) {name = "tuple.374"} : (tensor<1024xf32>, tensor<1xf32>) -> tuple<tensor<1024xf32>, tensor<1xf32>>
  func.return %3 : tuple<tensor<1024xf32>, tensor<1xf32>>
  // CHECK: %[[RES0:.*]] = mhlo.reshape %arg0 : (tensor<1x1xf32>) -> tensor<1xf32>
  // CHECK: %[[RES1:.*]] = mhlo.reshape %arg1 : (tensor<1x8x8x16xf32>) -> tensor<1024xf32>
  // CHECK: return %[[RES1]], %[[RES0]] : tensor<1024xf32>, tensor<1xf32>
}

// -----
func.func @main(%arg0: tensor<1x224x224x3xf16>, %arg1: tensor<f32>) -> tensor<1x224x224x3xf16> {
  func.return %arg0 : tensor<1x224x224x3xf16>
}

// -----

func.func @main(%arg0: tuple<tensor<1024xf32>, tensor<1xf32>>) -> tuple<tensor<1024xf32>, tensor<1xf32>> {
  func.return %arg0 : tuple<tensor<1024xf32>, tensor<1xf32>>
}

// CHECK: func @main(%[[VAL_0:.*]]: tensor<1024xf32>, %[[VAL_1:.*]]: tensor<1xf32>) -> (tensor<1024xf32>, tensor<1xf32>) {
// CHECK:   %[[VAL_2:.*]] = mhlo.tuple %[[VAL_0]], %[[VAL_1]] : tuple<tensor<1024xf32>, tensor<1xf32>>
// CHECK:   return %[[VAL_0]], %[[VAL_1]] : tensor<1024xf32>, tensor<1xf32>
// CHECK: }

// -----

func.func @main() -> tuple<> {
  %0 = "mhlo.tuple"() {xla_shape = "()"} : () -> tuple<>
  func.return %0 : tuple<>
}

// CHECK-LABEL: func @main() {
//       CHECK:   return{{$}}
//       CHECK: }

// -----

func.func @main() -> tuple<tensor<1xf32>, tensor<1xi32>> {
  %0 = "test.dummy"() : () -> tuple<tensor<1xf32>, tensor<1xi32>>
  func.return %0 : tuple<tensor<1xf32>, tensor<1xi32>>
}

// CHECK-LABEL: func @main()
//       CHECK: %[[TUPLE:.*]] = "test.dummy"()
//       CHECK: %[[T0:.*]] = mhlo.get_tuple_element %[[TUPLE]][0]
//       CHECK: %[[T1:.*]] = mhlo.get_tuple_element %[[TUPLE]][1]
//       CHECK: return %[[T0]], %[[T1]]


// -----

func.func @main(%arg0: tuple<tuple<tensor<1xi8>>>) -> tuple<tuple<tensor<1xf32>>, tensor<1xi32>> {
  %0 = "test.dummy"(%arg0) : (tuple<tuple<tensor<1xi8>>>) -> tuple<tuple<tensor<1xf32>>, tensor<1xi32>>
  func.return %0 : tuple<tuple<tensor<1xf32>>, tensor<1xi32>>
}

// CHECK-LABEL: func @main
//  CHECK-SAME: %[[ARG:.+]]: tensor<1xi8>
//       CHECK: %[[T0:.*]] = mhlo.tuple %[[ARG]] : tuple<tensor<1xi8>>
//       CHECK: %[[T1:.*]] = mhlo.tuple %[[T0]] : tuple<tuple<tensor<1xi8>>>
//       CHECK: %[[T:.*]] = "test.dummy"(%[[T1]]) : (tuple<tuple<tensor<1xi8>>>) -> tuple<tuple<tensor<1xf32>>, tensor<1xi32>>
//       CHECK: %[[GTE0:.*]] = mhlo.get_tuple_element %[[T]][0] : (tuple<tuple<tensor<1xf32>>, tensor<1xi32>>) -> tuple<tensor<1xf32>>
//       CHECK: %[[GTE1:.*]] = mhlo.get_tuple_element %[[T]][1] : (tuple<tuple<tensor<1xf32>>, tensor<1xi32>>) -> tensor<1xi32>
//       CHECK: %[[GTE2:.*]] = mhlo.get_tuple_element %[[GTE0]][0] : (tuple<tensor<1xf32>>) -> tensor<1xf32>
//       CHECK: return %[[GTE2]], %[[GTE1]] : tensor<1xf32>, tensor<1xi32>

// -----

// Check that sharding attributes are also flattened

func.func @main(%arg0: tuple<tensor<1x1xf32>, tensor<1x2x512xf32>> {mhlo.sharding = "{{replicated}, {devices=[1,2,4]<=[4,2]T(1,0)}}"},
                %arg1: tensor<2x2xf32>,
                %arg2: tensor<2048xf32> {mhlo.sharding = "{devices=[8]<=[8]}"})
    -> (tensor<2048xf32> {mhlo.sharding = "{devices=[8]<=[8]}"}, tensor<2x2xf32>,
        tuple<tensor<1024xf32>, tensor<1xf32>> {mhlo.sharding = "{{devices=[8]<=[4,2]T(1,0)}, {replicated}}"}) {
  %0 = mhlo.get_tuple_element %arg0[0] : (tuple<tensor<1x1xf32>, tensor<1x2x512xf32>>) -> tensor<1x1xf32>
  %1 = mhlo.get_tuple_element %arg0[1] : (tuple<tensor<1x1xf32>, tensor<1x2x512xf32>>) -> tensor<1x2x512xf32>
  %2 = "mhlo.reshape"(%0) : (tensor<1x1xf32>) -> tensor<1xf32>
  %3 = "mhlo.reshape"(%1) : (tensor<1x2x512xf32>) -> tensor<1024xf32>
  %4 = "mhlo.tuple"(%3, %2) {name = "tuple.374"} : (tensor<1024xf32>, tensor<1xf32>) -> tuple<tensor<1024xf32>, tensor<1xf32>>
  func.return %arg2, %arg1, %4 : tensor<2048xf32>, tensor<2x2xf32>, tuple<tensor<1024xf32>, tensor<1xf32>>
}

// CHECK-LABEL: func @main
//  CHECK-SAME:   %arg0: tensor<1x1xf32> {mhlo.sharding = "{replicated}"},
//  CHECK-SAME:   %arg1: tensor<1x2x512xf32> {mhlo.sharding = "{devices=[1,2,4]<=[4,2]T(1,0)}"},
//  CHECK-SAME:   %arg2: tensor<2x2xf32>,
//  CHECK-SAME:   %arg3: tensor<2048xf32> {mhlo.sharding = "{devices=[8]<=[8]}"})
//  CHECK-SAME:     -> (tensor<2048xf32> {mhlo.sharding = "{devices=[8]<=[8]}"},
//  CHECK-SAME:         tensor<2x2xf32>,
//  CHECK-SAME:         tensor<1024xf32> {mhlo.sharding = "{devices=[8]<=[4,2]T(1,0)}"},
//  CHECK-SAME:         tensor<1xf32> {mhlo.sharding = "{replicated}"})

// -----

// Check that invalid sharding attributes are handled gracefully

func.func @main(%arg0: tuple<tensor<1024xf32>, tensor<1024xf32>> {mhlo.sharding = "{{devices=[8]<=[8]}, {replicated}, {devices=[8]<=[8]}}"},
                %arg1: tuple<tensor<1024xf32>, tensor<1024xf32>> {mhlo.sharding = "{replicated}"})
    -> (tuple<tensor<1024xf32>, tensor<1024xf32>> {mhlo.sharding = "{{devices=[8]<=[8]}}"},
        tuple<tensor<1024xf32>, tensor<1024xf32>> {mhlo.sharding = "{not-a-valid-sharding}"}) {
  func.return %arg0, %arg1 : tuple<tensor<1024xf32>, tensor<1024xf32>>, tuple<tensor<1024xf32>, tensor<1024xf32>>
}

// CHECK-LABEL: func @main
//  CHECK-SAME:   %arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>,
//  CHECK-SAME:   %arg2: tensor<1024xf32>, %arg3: tensor<1024xf32>)
//  CHECK-SAME:     -> (tensor<1024xf32>, tensor<1024xf32>,
//  CHECK-SAME:         tensor<1024xf32>, tensor<1024xf32>)
