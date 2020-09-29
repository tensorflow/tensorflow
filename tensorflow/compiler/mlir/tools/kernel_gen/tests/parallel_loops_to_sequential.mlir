// RUN: kernel-gen-opt %s --parallel-loops-to-sequential | FileCheck %s

// CHECK-LABEL: @parallel_loop
func @parallel_loop(%lb_0 : index, %lb_1 : index,
                     %ub_0 : index, %ub_1 : index,
                     %s_0 : index, %s_1 : index,
                     %buf: memref<?x?xindex>) {
  scf.parallel (%i0, %i1) = (%lb_0, %lb_1) to (%ub_0, %ub_1) step (%s_0, %s_1) {
    %sum_elem = addi %i0, %i1 : index
    store %sum_elem, %buf[%i0, %i1] : memref<?x?xindex>
  }
  return
}
// CHECK: scf.for [[I_0:%.*]] = [[LB_0:%.*]] to [[UB_0:%.*]] step [[S_0:%.*]]
// CHECK:   scf.for [[I_1:%.*]] = [[LB_1:%.*]] to [[UB_1:%.*]] step [[S_1:%.*]]
// CHECK:     [[SUM:%.*]] = addi [[I_0]], [[I_1]] : index
// CHECK:     store [[SUM]], {{%.*}}{{\[}}[[I_0]], [[I_1]]] : memref<?x?xindex>
