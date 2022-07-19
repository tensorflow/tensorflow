// RUN: tfg-transforms-opt --tfg-region-to-functional %s | FileCheck %s
// RUN: tfg-transforms-opt --tfg-region-to-functional='force-control-capture=true' %s | FileCheck %s --check-prefix=CAPTURE

// Check that chain constants are inserted to force control capture (when
// enabled).

// CAPTURE: @test(%[[INDEX:.*]]: tensor<i32>
tfg.func @test(%index: tensor<i32> {tfg.name = "index"}) -> (tensor<i32>) {
  %ctlA = NoOp [%index.ctl]
  %ctlB = NoOp
  // CHECK: CaseRegion
  // CAPTURE: %[[UNUSED_0:.*]], %{{.*}} = Const [%{{.*}}] device("/foo") assigned_device("/bar") name("Case_mlir_const_capture_0")
  // CAPTURE-SAME: _tpu_replicate = "cluster"
  // CAPTURE: %[[UNUSED_1:.*]], %{{.*}} = Const [%{{.*}}] device("/foo") assigned_device("/bar") name("Case_mlir_const_capture_1")
  // CAPTURE-SAME: _tpu_replicate = "cluster"
  // CAPTURE: Case(%[[INDEX]], %[[UNUSED_0]], %[[UNUSED_1]]) device("/foo") assigned_device("/bar") name("Case")
  // CAPTURE-SAME: _tpu_replicate = "cluster"
  %Case, %ctl_0 = CaseRegion %index {
    %A, %ctl_1 = A [%ctlA, %ctlB] : () -> (tensor<i32>)
    yield(%A) : tensor<i32>
  } {_tpu_replicate = "cluster", _mlir_name = "Case",
     _mlir_device = "/foo", _mlir_assigned_device = "/bar"}
  : (tensor<i32>) -> (tensor<i32>)
  return(%Case) : tensor<i32>
}
