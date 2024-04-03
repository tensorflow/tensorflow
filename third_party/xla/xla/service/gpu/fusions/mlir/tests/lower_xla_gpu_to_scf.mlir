// RUN: mlir_fusions_opt %s -xla-gpu-lower-xla-gpu-to-scf | FileCheck %s

module {
  func.func @reducer(%a: f32, %b: i32, %c: f32, %d: i32) -> (f32, i32) {
    return %a, %b : f32, i32
  }

  func.func @shuffler(%a: f32, %b: i32) -> (f32, i32) {
    %ret:2 = xla_gpu.shuffle_reduce @reducer(%a, %b) to 4 : f32, i32
    return %ret#0, %ret#1 : f32, i32
  }
}

// CHECK: @shuffler(%[[A:.*]]: f32, %[[B:.*]]: i32)
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK-DAG: %[[C2:.*]] = arith.constant 2
// CHECK-DAG: %[[C4:.*]] = arith.constant 4
// CHECK-DAG: %[[C32:.*]] = arith.constant 32
// CHECK: %[[A4H:.*]], {{.*}} = gpu.shuffle down %[[A]], %[[C4]], %[[C32]]
// CHECK: %[[B4H:.*]], {{.*}} = gpu.shuffle down %[[B]], %[[C4]], %[[C32]]
// CHECK: %[[AB4:.*]]:2 = call @reducer(%[[A]], %[[B]], %[[A4H]], %[[B4H]])
// CHECK: %[[A2H:.*]], {{.*}} = gpu.shuffle down %[[AB4]]#0, %[[C2]], %[[C32]]
// CHECK: %[[B2H:.*]], {{.*}} = gpu.shuffle down %[[AB4]]#1, %[[C2]], %[[C32]]
// CHECK: %[[AB2:.*]]:2 = call @reducer(%[[AB4]]#0, %[[AB4]]#1, %[[A2H]], %[[B2H]])
// CHECK: %[[A1H:.*]], {{.*}} = gpu.shuffle down %[[AB2]]#0, %[[C1]], %[[C32]]
// CHECK: %[[B1H:.*]], {{.*}} = gpu.shuffle down %[[AB2]]#1, %[[C1]], %[[C32]]
// CHECK: %[[AB1:.*]]:2 = call @reducer(%[[AB2]]#0, %[[AB2]]#1, %[[A1H]], %[[B1H]])
// CHECK: return %[[AB1]]#0, %[[AB1]]#1

// -----

module {
  func.func @reducer(%a: f64, %b: f64) -> f64 {
    return %a : f64
  }

  func.func @shuffler(%a: f64) -> f64 {
    %ret = xla_gpu.shuffle_reduce @reducer(%a) to 1 : f64
    return %ret : f64
  }
}

// CHECK: @shuffler(%[[A:.*]]: f64
// CHECK: gpu.shuffle down {{.*}}, %[[C1]]
// CHECK: gpu.shuffle down {{.*}}, %[[C1]]

// -----

module {
  func.func @predicated_insert(
      %v: i32, %tensor: tensor<2xi32>, %index: index,
      %cond: i1) -> tensor<2xi32> {
    %ret = xla_gpu.predicated_insert %v into %tensor[%index] if %cond
      : tensor<2xi32>
    return %ret : tensor<2xi32>
  }
}

// CHECK: @predicated_insert
// CHECK-SAME: %[[V:.*]]: i32, %[[TENSOR:.*]]: tensor<2xi32>,
// CHECK-SAME: %[[INDEX:.*]]: index, %[[COND:.*]]: i1
// CHECK-NEXT: %[[RET:.*]] = scf.if %[[COND]]
// CHECK-NEXT:   %[[UPD:.*]] = tensor.insert %[[V]] into %[[TENSOR]][%[[INDEX]]]
// CHECK-NEXT:   yield %[[UPD]]
// CHECK-NEXT: else
// CHECK-NEXT:   yield %[[TENSOR]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[RET]]

// -----

module {
  func.func @predicated_extract(
      %v: i32, %tensor: tensor<2xi32>, %index: index,
      %cond: i1) -> i32 {
    %ret = xla_gpu.predicated_extract %tensor[%index] if %cond else %v
      : tensor<2xi32>
    return %ret : i32
  }
}

// CHECK: @predicated_extract
// CHECK-SAME: %[[V:.*]]: i32, %[[TENSOR:.*]]: tensor<2xi32>,
// CHECK-SAME: %[[INDEX:.*]]: index, %[[COND:.*]]: i1
// CHECK-NEXT: %[[RET:.*]] = scf.if %[[COND]]
// CHECK-NEXT:   %[[VAL:.*]] = tensor.extract  %[[TENSOR]][%[[INDEX]]]
// CHECK-NEXT:   yield %[[VAL]]
// CHECK-NEXT: else
// CHECK-NEXT:   yield %[[V]]
// CHECK-NEXT: }
// CHECK-NEXT: return %[[RET]]
