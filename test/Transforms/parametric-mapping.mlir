// RUN: mlir-opt -test-mapping-to-processing-elements %s | FileCheck %s

// CHECK-LABEL: @map1d
//       CHECK: (%[[lb:.*]]: index, %[[ub:.*]]: index, %[[step:.*]]: index) {
func @map1d(%lb: index, %ub: index, %step: index) {
  // CHECK: %[[threads:.*]]:2 = "new_processor_id_and_range"() : () -> (index, index)
  %0:2 = "new_processor_id_and_range"() : () -> (index, index)

  // CHECK: %[[thread_offset:.*]] = muli %[[step]], %[[threads]]#0
  // CHECK: %[[new_lb:.*]] = addi %[[lb]], %[[thread_offset]]
  // CHECK: %[[new_step:.*]] = muli %[[step]], %[[threads]]#1
  // CHECK: loop.for %{{.*}} = %[[new_lb]] to %[[ub]] step %[[new_step]] {
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
  // thread_offset = step * (threadIdx.x + blockIdx.x * blockDim.x)
  // CHECK: %[[thread_offset:.*]] = muli %[[step]], %[[tidxpbidxXbdimx]] : index
  //
  // new_lb = lb + thread_offset
  // CHECK: %[[new_lb:.*]] = addi %[[lb]], %[[thread_offset]] : index
  //
  // stepXgdimx = step * gridDim.x
  // CHECK: %[[stepXgdimx:.*]] = muli %[[step]], %[[blocks]]#1 : index
  //
  // new_step = step * gridDim.x * blockDim.x
  // CHECK: %[[new_step:.*]] = muli %[[stepXgdimx]], %[[threads]]#1 : index
  //
  // CHECK: loop.for %{{.*}} = %[[new_lb]] to %[[ub]] step %[[new_step]] {
  loop.for %i = %lb to %ub step %step {}
  return
}
