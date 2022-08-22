// RUN: mlir-hlo-opt --sparse-chlo-legalize-to-linalg %s | FileCheck %s

#CSR = #sparse_tensor.encoding<{
  dimLevelType = ["dense", "compressed"]
}>

// CHECK-LABEL: @asinh_scalar(
// CHECK-SAME: %[[ARG:.*]]: tensor<f32>) -> tensor<f32> {
// CHECK:         %[[RESULT:.*]] = chlo.asinh %[[ARG]] : tensor<f32> -> tensor<f32>
// CHECK:         return %[[RESULT]] : tensor<f32>
func.func @asinh_scalar(%arg : tensor<f32>) -> tensor<f32> {
  %result = "chlo.asinh"(%arg) : (tensor<f32>) -> tensor<f32>
  func.return %result : tensor<f32>
}

// CHECK-LABEL: @asinh_tensor(
// CHECK-SAME: %[[ARG:.*]]: tensor<10x20xf32, #{{.*}}>) ->
// CHECK-SAME:   tensor<10x20xf32, #{{.*}}> {
// CHECK:         %[[OUT:.*]] = bufferization.alloc_tensor() :
// CHECK-SAME:      tensor<10x20xf32, #{{.*}}>
// CHECK:         %[[VAL:.*]] = linalg.generic
// CHECK-SAME:        ins(%[[ARG]] : tensor<10x20xf32,
// CHECK-SAME:        #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>
// CHECK-SAME:        outs(%[[OUT]]
// CHECK:           sparse_tensor.unary %{{.*}} : f32 to f32
// CHECK:           present = {
// CHECK:             tensor.from_elements
// CHECK:             chlo.asinh
// CHECK:             tensor.extract
// CHECK:             sparse_tensor.yield %{{.*}} : f32
// CHECK:           }
// CHECK:           absent = {
// CHECK:           }
// CHECK:         }
func.func @asinh_tensor(%arg : tensor<10x20xf32, #CSR>)
                        -> tensor<10x20xf32, #CSR> {
  %result = "chlo.asinh"(%arg) : (tensor<10x20xf32, #CSR>)
                                 -> tensor<10x20xf32, #CSR>
  func.return %result : tensor<10x20xf32, #CSR>
}


// CHECK-LABEL:  func.func @tan_tensor(
// CHECK-SAME:   %[[TMP_arg0:.*]]: tensor<10x20xf32,
// CHECK:          %[[TMP_0:.*]] = bufferization.alloc_tensor() : tensor<10x20xf32,
// CHECK:          %[[TMP_1:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:     ins(%[[TMP_arg0]] : tensor<10x20xf32,
// CHECK-SAME:     outs(%[[TMP_0]] : tensor<10x20xf32,
// CHECK:           ^bb0(%[[TMP_arg1:.*]]: f32, %[[TMP_arg2:.*]]: f32):
// CHECK:             %[[TMP_2:.*]] = sparse_tensor.unary %[[TMP_arg1]] : f32 to f32
// CHECK:              present = {
// CHECK:              ^bb0(%[[TMP_arg3:.*]]: f32):
// CHECK:                %[[TMP_3:.*]] = tensor.from_elements %[[TMP_arg3]] : tensor<f32>
// CHECK:                %[[TMP_4:.*]] = chlo.tan %[[TMP_3]] : tensor<f32> -> tensor<f32>
// CHECK:                %[[TMP_5:.*]] = tensor.extract %[[TMP_4]][] : tensor<f32>
// CHECK:                sparse_tensor.yield %[[TMP_5]] : f32
// CHECK:              }
// CHECK:              absent = {
// CHECK:              }
// CHECK:             linalg.yield %[[TMP_2]] : f32
// CHECK:           } -> tensor<10x20xf32,
// CHECK:           return %[[TMP_1]] : tensor<10x20xf32,
func.func @tan_tensor(%arg : tensor<10x20xf32, #CSR>)
                        -> tensor<10x20xf32, #CSR> {
  %result = "chlo.tan"(%arg) : (tensor<10x20xf32, #CSR>)
                                 -> tensor<10x20xf32, #CSR>
  func.return %result : tensor<10x20xf32, #CSR>
}

// CHECK-LABEL:  func.func @sinh_tensor(
// CHECK-SAME:   %[[TMP_arg0:.*]]: tensor<10x20xf32,
// CHECK:          %[[TMP_0:.*]] = bufferization.alloc_tensor() : tensor<10x20xf32,
// CHECK:          %[[TMP_1:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:     ins(%[[TMP_arg0]] : tensor<10x20xf32,
// CHECK-SAME:     outs(%[[TMP_0]] : tensor<10x20xf32,
// CHECK:          ^bb0(%[[TMP_arg1:.*]]: f32, %[[TMP_arg2:.*]]: f32):
// CHECK:            %[[TMP_2:.*]] = sparse_tensor.unary %[[TMP_arg1]] : f32 to f32
// CHECK:             present = {
// CHECK:            ^bb0(%[[TMP_arg3:.*]]: f32):
// CHECK:              %[[TMP_3:.*]] = tensor.from_elements %[[TMP_arg3]] : tensor<f32>
// CHECK:              %[[TMP_4:.*]] = chlo.sinh %[[TMP_3]] : tensor<f32> -> tensor<f32>
// CHECK:              %[[TMP_5:.*]] = tensor.extract %[[TMP_4]][] : tensor<f32>
// CHECK:              sparse_tensor.yield %[[TMP_5]] : f32
// CHECK:            }
// CHECK:             absent = {
// CHECK:            }
// CHECK:            linalg.yield %[[TMP_2]] : f32
// CHECK:          } -> tensor<10x20xf32,
// CHECK:          return %[[TMP_1]] : tensor<10x20xf32,
func.func @sinh_tensor(%arg : tensor<10x20xf32, #CSR>)
                        -> tensor<10x20xf32, #CSR> {
  %result = "chlo.sinh"(%arg) : (tensor<10x20xf32, #CSR>)
                                 -> tensor<10x20xf32, #CSR>
  func.return %result : tensor<10x20xf32, #CSR>
}
