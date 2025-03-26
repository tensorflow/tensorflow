// RUN: mlir-bisect %s --debug-strategy=ReduceScfForallBounds | FileCheck %s

func.func @main() -> tensor<8xindex> {
  %init = tensor.empty() : tensor<8xindex>
  %iota = scf.forall (%i) = (0) to (8) step (1)
      shared_outs (%init_ = %init) -> (tensor<8xindex>) {
    %tensor = tensor.from_elements %i : tensor<1xindex>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %tensor into %init_[%i] [1] [1]
        : tensor<1xindex> into tensor<8xindex>
    }
  }
  func.return %iota : tensor<8xindex>
}
// CHECK: func @main()
// CHECK:   scf.forall ({{.*}}) in (7)
