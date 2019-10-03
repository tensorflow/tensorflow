// RUN: mlir-opt -convert-loop-to-std %s | FileCheck %s

// CHECK-LABEL: func @simple_std_for_loop(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
//  CHECK-NEXT:  br ^bb1(%{{.*}} : index)
//  CHECK-NEXT:  ^bb1(%{{.*}}: index):    // 2 preds: ^bb0, ^bb2
//  CHECK-NEXT:    %{{.*}} = cmpi "slt", %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:    cond_br %{{.*}}, ^bb2, ^bb3
//  CHECK-NEXT:  ^bb2:   // pred: ^bb1
//  CHECK-NEXT:    %{{.*}} = constant 1 : index
//  CHECK-NEXT:    %[[iv:.*]] = addi %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:    br ^bb1(%[[iv]] : index)
//  CHECK-NEXT:  ^bb3:   // pred: ^bb1
//  CHECK-NEXT:    return
func @simple_std_for_loop(%arg0 : index, %arg1 : index, %arg2 : index) {
  loop.for %i0 = %arg0 to %arg1 step %arg2 {
    %c1 = constant 1 : index
  }
  return
}

// CHECK-LABEL: func @simple_std_2_for_loops(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
//  CHECK-NEXT:    br ^bb1(%{{.*}} : index)
//  CHECK-NEXT:  ^bb1(%[[ub0:.*]]: index):    // 2 preds: ^bb0, ^bb5
//  CHECK-NEXT:    %[[cond0:.*]] = cmpi "slt", %[[ub0]], %{{.*}} : index
//  CHECK-NEXT:    cond_br %[[cond0]], ^bb2, ^bb6
//  CHECK-NEXT:  ^bb2:   // pred: ^bb1
//  CHECK-NEXT:    %{{.*}} = constant 1 : index
//  CHECK-NEXT:    br ^bb3(%{{.*}} : index)
//  CHECK-NEXT:  ^bb3(%[[ub1:.*]]: index):    // 2 preds: ^bb2, ^bb4
//  CHECK-NEXT:    %[[cond1:.*]] = cmpi "slt", %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:    cond_br %[[cond1]], ^bb4, ^bb5
//  CHECK-NEXT:  ^bb4:   // pred: ^bb3
//  CHECK-NEXT:    %{{.*}} = constant 1 : index
//  CHECK-NEXT:    %[[iv1:.*]] = addi %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:    br ^bb3(%[[iv1]] : index)
//  CHECK-NEXT:  ^bb5:   // pred: ^bb3
//  CHECK-NEXT:    %[[iv0:.*]] = addi %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:    br ^bb1(%[[iv0]] : index)
//  CHECK-NEXT:  ^bb6:   // pred: ^bb1
//  CHECK-NEXT:    return
func @simple_std_2_for_loops(%arg0 : index, %arg1 : index, %arg2 : index) {
  loop.for %i0 = %arg0 to %arg1 step %arg2 {
    %c1 = constant 1 : index
    loop.for %i1 = %arg0 to %arg1 step %arg2 {
      %c1_0 = constant 1 : index
    }
  }
  return
}

// CHECK-LABEL: func @simple_std_if(%{{.*}}: i1) {
//  CHECK-NEXT:   cond_br %{{.*}}, ^bb1, ^bb2
//  CHECK-NEXT:   ^bb1:   // pred: ^bb0
//  CHECK-NEXT:     %{{.*}} = constant 1 : index
//  CHECK-NEXT:     br ^bb2
//  CHECK-NEXT:   ^bb2:   // 2 preds: ^bb0, ^bb1
//  CHECK-NEXT:     return
func @simple_std_if(%arg0: i1) {
  loop.if %arg0 {
    %c1 = constant 1 : index
  }
  return
}

// CHECK-LABEL: func @simple_std_if_else(%{{.*}}: i1) {
//  CHECK-NEXT:   cond_br %{{.*}}, ^bb1, ^bb2
//  CHECK-NEXT:   ^bb1:   // pred: ^bb0
//  CHECK-NEXT:     %{{.*}} = constant 1 : index
//  CHECK-NEXT:     br ^bb3
//  CHECK-NEXT:   ^bb2:   // pred: ^bb0
//  CHECK-NEXT:     %{{.*}} = constant 1 : index
//  CHECK-NEXT:     br ^bb3
//  CHECK-NEXT:   ^bb3:   // 2 preds: ^bb1, ^bb2
//  CHECK-NEXT:     return
func @simple_std_if_else(%arg0: i1) {
  loop.if %arg0 {
    %c1 = constant 1 : index
  } else {
    %c1_0 = constant 1 : index
  }
  return
}

// CHECK-LABEL: func @simple_std_2_ifs(%{{.*}}: i1) {
//  CHECK-NEXT:   cond_br %{{.*}}, ^bb1, ^bb5
//  CHECK-NEXT: ^bb1:   // pred: ^bb0
//  CHECK-NEXT:   %{{.*}} = constant 1 : index
//  CHECK-NEXT:   cond_br %{{.*}}, ^bb2, ^bb3
//  CHECK-NEXT: ^bb2:   // pred: ^bb1
//  CHECK-NEXT:   %{{.*}} = constant 1 : index
//  CHECK-NEXT:   br ^bb4
//  CHECK-NEXT: ^bb3:   // pred: ^bb1
//  CHECK-NEXT:   %{{.*}} = constant 1 : index
//  CHECK-NEXT:   br ^bb4
//  CHECK-NEXT: ^bb4:   // 2 preds: ^bb2, ^bb3
//  CHECK-NEXT:   br ^bb5
//  CHECK-NEXT: ^bb5:   // 2 preds: ^bb0, ^bb4
//  CHECK-NEXT:   return
func @simple_std_2_ifs(%arg0: i1) {
  loop.if %arg0 {
    %c1 = constant 1 : index
    loop.if %arg0 {
      %c1_0 = constant 1 : index
    } else {
      %c1_1 = constant 1 : index
    }
  }
  return
}

// CHECK-LABEL: func @simple_std_for_loop_with_2_ifs(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: i1) {
//  CHECK-NEXT:   br ^bb1(%{{.*}} : index)
//  CHECK-NEXT:   ^bb1(%{{.*}}: index):    // 2 preds: ^bb0, ^bb7
//  CHECK-NEXT:     %{{.*}} = cmpi "slt", %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:     cond_br %{{.*}}, ^bb2, ^bb8
//  CHECK-NEXT:   ^bb2:   // pred: ^bb1
//  CHECK-NEXT:     %{{.*}} = constant 1 : index
//  CHECK-NEXT:     cond_br %{{.*}}, ^bb3, ^bb7
//  CHECK-NEXT:   ^bb3:   // pred: ^bb2
//  CHECK-NEXT:     %{{.*}} = constant 1 : index
//  CHECK-NEXT:     cond_br %{{.*}}, ^bb4, ^bb5
//  CHECK-NEXT:   ^bb4:   // pred: ^bb3
//  CHECK-NEXT:     %{{.*}} = constant 1 : index
//  CHECK-NEXT:     br ^bb6
//  CHECK-NEXT:   ^bb5:   // pred: ^bb3
//  CHECK-NEXT:     %{{.*}} = constant 1 : index
//  CHECK-NEXT:     br ^bb6
//  CHECK-NEXT:   ^bb6:   // 2 preds: ^bb4, ^bb5
//  CHECK-NEXT:     br ^bb7
//  CHECK-NEXT:   ^bb7:   // 2 preds: ^bb2, ^bb6
//  CHECK-NEXT:     %[[iv0:.*]] = addi %{{.*}}, %{{.*}} : index
//  CHECK-NEXT:     br ^bb1(%[[iv0]] : index)
//  CHECK-NEXT:   ^bb8:   // pred: ^bb1
//  CHECK-NEXT:     return
//  CHECK-NEXT: }
func @simple_std_for_loop_with_2_ifs(%arg0 : index, %arg1 : index, %arg2 : index, %arg3 : i1) {
  loop.for %i0 = %arg0 to %arg1 step %arg2 {
    %c1 = constant 1 : index
    loop.if %arg3 {
      %c1_0 = constant 1 : index
      loop.if %arg3 {
        %c1_1 = constant 1 : index
      } else {
        %c1_2 = constant 1 : index
      }
    }
  }
  return
}
