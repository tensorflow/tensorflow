// RUN: tf-opt -test-buffer-assignment -split-input-file %s | FileCheck %s -dump-input-on-failure

// CHECK-LABEL: Testing : condBranch
func @condBranch(%cond : i1, %arg0 : tensor<2xf32>) -> tensor<2xf32>{
  // CHECK: Alloc: cond_br
    cond_br %cond, ^bb1, ^bb2
  ^bb1:
    br ^exit(%arg0 : tensor<2xf32>)
  ^bb2:
    %1 = "xla_hlo.exponential"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
    br ^exit(%1 : tensor<2xf32>)
  ^exit(%arg1: tensor<2xf32>):
    return %arg1 : tensor<2xf32>
  // CHECK-NEXT: Dealloc: return
}

// -----

// CHECK-LABEL: Testing : criticalEdge
func @criticalEdge(%cond : i1, %arg0 : tensor<2xf32>) -> tensor<2xf32>{
  // CHECK: Alloc: cond_br
    cond_br %cond, ^bb1, ^exit(%arg0 : tensor<2xf32>)
  ^bb1:
    %0 = "xla_hlo.exponential"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
    br ^exit(%0 : tensor<2xf32>)
  ^exit(%arg1: tensor<2xf32>):
    return %arg1 : tensor<2xf32>
  // CHECK-NEXT: Dealloc: return
}

// -----

// CHECK-LABEL: Testing : invCriticalEdge
func @invCriticalEdge(%cond : i1, %arg0 : tensor<2xf32>) -> tensor<2xf32>{
  // CHECK: Alloc: %{{.*}} = "xla_hlo.exponential"
    %0 = "xla_hlo.exponential"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
    cond_br %cond, ^bb1, ^exit(%arg0 : tensor<2xf32>)
  ^bb1:
    br ^exit(%0 : tensor<2xf32>)
  ^exit(%arg1: tensor<2xf32>):
    return %arg1 : tensor<2xf32>
  // CHECK-NEXT: Dealloc: return
}

// -----

// CHECK-LABEL: Testing : ifElse
func @ifElse(%cond : i1, %arg0 : tensor<2xf32>) -> tensor<2xf32>{
  // CHECK: Alloc: %{{.*}} = "xla_hlo.exponential"(%{{.*}})
    %0 = "xla_hlo.exponential"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
    cond_br %cond, ^bb1(%arg0, %0: tensor<2xf32>, tensor<2xf32>),
                   ^bb2(%0, %arg0: tensor<2xf32>, tensor<2xf32>)
  ^bb1(%arg1 : tensor<2xf32>, %arg2 : tensor<2xf32>):
    br ^exit(%arg1, %arg2 : tensor<2xf32>, tensor<2xf32>)
  ^bb2(%arg3 : tensor<2xf32>, %arg4 : tensor<2xf32>):
    br ^exit(%arg3, %arg4 : tensor<2xf32>, tensor<2xf32>)
  ^exit(%arg5 : tensor<2xf32>, %arg6 : tensor<2xf32>):
  // CHECK-NEXT: Dealloc: %[[EXP_RES:.*]] = "xla_hlo.exponential"(%[[EXP_INPUT:.*]])
  // CHECK: Alloc: %[[EXP_RES]] = "xla_hlo.exponential"(%[[EXP_INPUT]])
  // CHECK-NEXT: Dealloc: return
    %1 = "xla_hlo.exponential"(%arg5) : (tensor<2xf32>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
}

// -----

// CHECK-LABEL: Testing : ifElseNoUsers
func @ifElseNoUsers(%cond : i1, %arg0 : tensor<2xf32>) -> tensor<2xf32>{
  // CHECK: Alloc: %{{.*}} = "xla_hlo.exponential"(%{{.*}})
    %0 = "xla_hlo.exponential"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
    cond_br %cond, ^bb1(%arg0, %0: tensor<2xf32>, tensor<2xf32>),
                   ^bb2(%0, %arg0: tensor<2xf32>, tensor<2xf32>)
  ^bb1(%arg1 : tensor<2xf32>, %arg2 : tensor<2xf32>):
    br ^exit(%arg1, %arg2 : tensor<2xf32>, tensor<2xf32>)
  ^bb2(%arg3 : tensor<2xf32>, %arg4 : tensor<2xf32>):
    br ^exit(%arg3, %arg4 : tensor<2xf32>, tensor<2xf32>)
  ^exit(%arg5 : tensor<2xf32>, %arg6 : tensor<2xf32>):
  // CHECK-NEXT: return
    return %arg0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: Testing : ifElseNested
func @ifElseNested(%cond : i1, %arg0 : tensor<2xf32>) -> tensor<2xf32>{
  // CHECK: Alloc: %{{.*}} = "xla_hlo.exponential"(%{{.*}})
    %0 = "xla_hlo.exponential"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
    cond_br %cond, ^bb1(%arg0, %0: tensor<2xf32>, tensor<2xf32>),
                   ^bb2(%0, %arg0: tensor<2xf32>, tensor<2xf32>)
  ^bb1(%arg1 : tensor<2xf32>, %arg2 : tensor<2xf32>):
    br ^exit(%arg1, %arg2 : tensor<2xf32>, tensor<2xf32>)
  ^bb2(%arg3 : tensor<2xf32>, %arg4 : tensor<2xf32>):
    cond_br %cond, ^bb3(%arg3 : tensor<2xf32>), ^bb4(%arg4 : tensor<2xf32>)
  ^bb3(%arg7 : tensor<2xf32>):
    br ^exit(%arg7, %arg3 : tensor<2xf32>, tensor<2xf32>)
  ^bb4(%arg8 : tensor<2xf32>):
    br ^exit(%arg3, %arg8 : tensor<2xf32>, tensor<2xf32>)
  ^exit(%arg5 : tensor<2xf32>, %arg6 : tensor<2xf32>):
  // CHECK-NEXT: Dealloc: %[[EXP_RES:.*]] = "xla_hlo.exponential"(%[[EXP_INPUT:.*]])
  // CHECK: Alloc: %[[EXP_RES]] = "xla_hlo.exponential"(%[[EXP_INPUT]])
  // CHECK-NEXT: Dealloc: return
    %1 = "xla_hlo.exponential"(%arg5) : (tensor<2xf32>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
}

// -----

// CHECK-LABEL: Testing : redundantOperations
func @redundantOperations(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) {
  // CHECK: Alloc: %{{.*}} = xla_hlo.maximum
  // CHECK-NEXT: Dealloc: %[[ADD_RES:.*]] = xla_hlo.add
  %1 = "xla_hlo.maximum"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // CHECK: Alloc: %[[ADD_RES]] = xla_hlo.add
  // CHECK-NEXT: Dealloc: %[[ADD_RES]] = xla_hlo.add
  %2 = "xla_hlo.add"(%arg0, %1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return
}

// -----

// CHECK-LABEL: Testing : reduce
func @reduce(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  // CHECK: Alloc: %[[CONST_RES:.*]] = xla_hlo.constant
  // CHECK-NEXT: Dealloc: %[[REDUCE_RES:.*]] = "xla_hlo.reduce"(%arg{{.*}}, %[[CONST_RES]])
  %0 = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: Alloc: %[[REDUCE_RES]] = "xla_hlo.reduce"(%arg{{.*}}, %[[CONST_RES]])
  // CHECK: Dealloc: return
  %2 = "xla_hlo.reduce"(%arg0, %0) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %4 = xla_hlo.add %arg1, %arg2 : tensor<f32>
    "xla_hlo.return"(%4) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
  return %2 : tensor<4x8xf32>
}
