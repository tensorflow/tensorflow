// RUN: mlir-hlo-opt %s --gml-compose-extract-insert-slice | FileCheck %s

func.func @fold_tensor_cast_into_parallel(
    %in: tensor<2xi32>, %out: tensor<2xi32>) -> tensor<2xi32> {
  %cst = arith.constant dense<[100500]> : tensor<1xi32>


  %out_cast = tensor.cast %out : tensor<2xi32> to tensor<?xi32>
  %result = scf.forall (%i) = (0) to (2) step (1)
      shared_outs (%out_ = %out_cast) -> tensor<?xi32> {

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %cst into %out_[%i] [1] [1]
        : tensor<1xi32> into tensor<?xi32>
    }
  }
  %result_cast = tensor.cast %result : tensor<?xi32> to tensor<2xi32>
  func.return %result_cast : tensor<2xi32>
}
// CHECK-LABEL: @fold_tensor_cast_into_parallel
// CHECK-NOT:     tensor.cast
// CHECK:         parallel_insert_slice
// CHECK-SAME:      : tensor<1xi32> into tensor<2xi32>
// CHECK-NOT:     tensor.cast

// -----

func.func @do_not_fold_tensor_cast_from_dynamic_to_static_type_into_parallel(
    %in: tensor<?xi32>, %out: tensor<?xi32>) -> tensor<?xi32> {
  %cst = arith.constant dense<[100500]> : tensor<1xi32>


  %out_cast = tensor.cast %out : tensor<?xi32> to tensor<2xi32>
  %result = scf.forall (%i) = (0) to (2) step (1)
      shared_outs (%out_ = %out_cast) -> tensor<2xi32> {

    scf.forall.in_parallel {
      tensor.parallel_insert_slice %cst into %out_[%i] [1] [1]
        : tensor<1xi32> into tensor<2xi32>
    }
  }
  %result_cast = tensor.cast %result : tensor<2xi32> to tensor<?xi32>
  func.return %result_cast : tensor<?xi32>
}
// CHECK-LABEL: @do_not_fold_tensor_cast_
// CHECK:         tensor.cast
// CHECK:         parallel_insert_slice
// CHECK-SAME:      : tensor<1xi32> into tensor<2xi32>
// CHECK:         tensor.cast
