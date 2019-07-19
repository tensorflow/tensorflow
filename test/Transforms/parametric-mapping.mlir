// RUN: mlir-opt -test-mapping-to-processing-elements %s | FileCheck %s

// CHECK-LABEL: @map1d
//       CHECK: (%[[lb:.*]]: index, %[[ub:.*]]: index, %[[step:.*]]: index) {
func @map1d(%lb: index, %ub: index, %step: index) {
  // CHECK: %[[threads:.*]]:2 = "new_processor_id_and_range"() : () -> (index, index)
  %0:2 = "new_processor_id_and_range"() : () -> (index, index)

  // CHECK: %[[new_lb:.*]] = addi %[[lb]], %[[threads]]#0
  // CHECK: loop.for %{{.*}} = %[[new_lb]] to %[[ub]] step %[[threads]]#1 {
  loop.for %i = %lb to %ub step %step {}
  return
}

// CHECK-LABEL: @map2d
//       CHECK: (%[[lb:.*]]: index, %[[ub:.*]]: index, %[[step:.*]]: index) {
func @map2d(%lb : index, %ub : index, %step : index) {
  // CHECK: %[[blocks:.*]]:2 = "new_processor_id_and_range"() : () -> (index, index)
  %0:2 = "new_processor_id_and_range"() : () -> (index, index)

  // CHECK: %[[threads:.*]]:2 = "new_processor_id_and_range"() : () -> (index, index)
  %1:2 = "new_processor_id_and_range"() : () -> (index, index)

  // blockIdx.x * blockDim.x
  // CHECK: %[[bidxXbdimx:.*]] = muli %[[blocks]]#0, %[[threads]]#1 : index
  //
  // threadIdx.x + blockIdx.x * blockDim.x
  // CHECK: %[[tidxpbidxXbdimx:.*]] = addi %[[bidxXbdimx]], %[[threads]]#0 : index
  //
  // new_lb = lb + threadIdx.x + blockIdx.x * blockDim.x
  // CHECK: %[[new_lb:.*]] = addi %[[lb]], %[[tidxpbidxXbdimx]] : index
  //
  // new_step = gridDim.x * blockDim.x
  // CHECK: %[[new_step:.*]] = muli %[[blocks]]#1, %[[threads]]#1 : index
  //
  // CHECK: loop.for %{{.*}} = %[[new_lb]] to %[[ub]] step %[[new_step]] {
  loop.for %i = %lb to %ub step %step {}
  return
}
