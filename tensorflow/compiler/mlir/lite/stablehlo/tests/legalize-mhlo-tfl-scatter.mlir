// RUN: tf-mhlo-tfl-opt %s -mhlo-tfl | FileCheck %s

module {
func.func @main(%arg0: tensor<3xi32>, %arg1: tensor<1x1xi32>, %arg2: tensor<1xi32>) -> tensor<3xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    "mhlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1>,
    indices_are_sorted = false,
    unique_indices = false} :
       (tensor<3xi32>, tensor<1x1xi32>, tensor<1xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}
}

// CHECK:      module {
// CHECK-NEXT:   func.func @main(%arg0: tensor<3xi32>, %arg1: tensor<1x1xi32>, %arg2: tensor<1xi32>) -> tensor<3xi32> {
// CHECK-NEXT:     %0 = "tfl.custom"(%arg0, %arg1, %arg2) {custom_code = "mhlo.scatter", custom_option = #tfl<const_bytes : "0x696E64696365735F6172655F736F7274656400736361747465725F64696D656E73696F6E5F6E756D626572730000010004010004040707050128282804756E697175655F696E646963657300034D3B12030103001F00042804062401">} : (tensor<3xi32>, tensor<1x1xi32>, tensor<1xi32>) -> tensor<3xi32>
// CHECK-NEXT:     return %0 : tensor<3xi32>
// CHECK-NEXT:   }
// CHECK-NEXT: }
