// RUN: tfg-opt-no-passes %s | tfg-opt-no-passes | FileCheck %s

// CHECK-LABEL: @test
tfg.func @test_region_preserved_attrs(%arg0: tensor<i32>) -> (tensor<i32>) {
  %CaseRegion, %ctl = CaseRegion %arg0 {
    yield(%arg0) : tensor<i32>
  } {Tout = [i32], branch_attrs = [{}], output_shapes = [#tf_type.shape<>],
     // CHECK: #tfg.region_attrs<{tf._some_attr} [] [{tf._some_ret_attr}]>
     region_attrs = [#tfg.region_attrs<{tf._some_attr} [] [{tf._some_ret_attr}]>]}
  : (tensor<i32>) -> (tensor<i32>)
  return(%CaseRegion) : tensor<i32>
}
