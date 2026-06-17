// RUN: tfg-transforms-opt --tfg-eliminate-passthrough-iter-args %s | FileCheck %s

// CHECK-LABEL: @test_uncapture_all
// CHECK: %[[INDEX:.*]]: tensor
// CHECK-NEXT: %[[A0:.*]]: tensor
// CHECK-NEXT: %[[A1:.*]]: tensor
// CHECK-NEXT: %[[A2:.*]]: tensor
// CHECK-NEXT: %[[A3:.*]]: tensor
tfg.func @test_uncapture_all(%index: tensor<i32> {tfg.name = "index"},
                             %a0: tensor<i8> {tfg.name = "a0"},
                             %a1: tensor<i16> {tfg.name = "a1"},
                             %a2: tensor<i32> {tfg.name = "a2"},
                             %a3: tensor<i64> {tfg.name = "a3"})
    -> (tensor<i8>, tensor<i16>, tensor<i32>, tensor<i64>) {
  // CHECK: %{{.*}} = ForRegion from %[[INDEX]]
  // CHECK: ^bb0(%{{.*}}: tensor<i32>, %{{.*}}: !tf_type.control):
  // CHECK:   %[[USE:.*]]:2, {{.*}} = Use(%[[A0]], %[[A1]], %[[A2]], %[[A3]])
  // CHECK:   yield
  // CHECK: _some_attr
  // CHECK: return(%[[A0]], %[[A1]], %[[A2]], %[[A3]])
  %For:4, %ctl = ForRegion(%a0, %a1, %a2, %a3) from %index to %index by %index {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i8>, %arg2: tensor<i16>, %arg3: tensor<i32>, %arg4: tensor<i64>,
       %ctl0: !tf_type.control, %ctl1: !tf_type.control, %ctl2: !tf_type.control, %ctl3: !tf_type.control, %ctl4: !tf_type.control):
    %Use:2, %ctl_0 = Use(%arg1, %arg2, %arg3, %arg4) : (tensor<i8>, tensor<i16>, tensor<i32>, tensor<i64>) -> (tensor<i8>, tensor<i32>)
    yield(%arg1, %arg2, %arg3, %arg4) : tensor<i8>, tensor<i16>, tensor<i32>, tensor<i64>
  } {_some_attr}
  : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i8>, tensor<i16>, tensor<i32>, tensor<i64>)
  -> (tensor<i8>, tensor<i16>, tensor<i32>, tensor<i64>)
  return(%For#0, %For#1, %For#2, %For#3) : tensor<i8>, tensor<i16>, tensor<i32>, tensor<i64>
}

// CHECK-LABEL: @test_uncapture_some
// CHECK: %[[INDEX:.*]]: tensor
// CHECK-NEXT: %[[A0:.*]]: tensor
// CHECK-NEXT: %[[A1:.*]]: tensor
// CHECK-NEXT: %[[A2:.*]]: tensor
// CHECK-NEXT: %[[A3:.*]]: tensor
tfg.func @test_uncapture_some(%index: tensor<i32> {tfg.name = "index"},
                              %a0: tensor<i8> {tfg.name = "a0"},
                              %a1: tensor<i16> {tfg.name = "a1"},
                              %a2: tensor<i32> {tfg.name = "a2"},
                              %a3: tensor<i64> {tfg.name = "a3"})
    -> (tensor<i8>, tensor<i16>, tensor<i32>, tensor<i64>) {
  // CHECK: %[[FOR:.*]]:2, %{{.*}} = ForRegion(%[[A0]], %[[A2]]) from %[[INDEX]]
  // CHECK: ^bb0(%{{.*}}: tensor<i32>, %[[ARG0:.*]]: tensor<i8>, %[[ARG1:.*]]: tensor<i32>
  // CHECK:   %[[USE:.*]]:2, {{.*}} = Use(%[[ARG0]], %[[A1]], %[[ARG1]], %[[A3]])
  // CHECK:   yield(%[[USE]]#0, %[[USE]]#1)
  // CHECK: return(%[[FOR]]#0, %[[A1]], %[[FOR]]#1, %[[A3]])
  %For:4, %ctl = ForRegion(%a0, %a1, %a2, %a3) from %index to %index by %index {
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i8>, %arg2: tensor<i16>, %arg3: tensor<i32>, %arg4: tensor<i64>,
       %ctl0: !tf_type.control, %ctl1: !tf_type.control, %ctl2: !tf_type.control, %ctl3: !tf_type.control, %ctl4: !tf_type.control):
    %Use:2, %ctl_0 = Use(%arg1, %arg2, %arg3, %arg4) : (tensor<i8>, tensor<i16>, tensor<i32>, tensor<i64>) -> (tensor<i8>, tensor<i32>)
    yield(%Use#0, %arg2, %Use#1, %arg4) : tensor<i8>, tensor<i16>, tensor<i32>, tensor<i64>
  } : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i8>, tensor<i16>, tensor<i32>, tensor<i64>)
  -> (tensor<i8>, tensor<i16>, tensor<i32>, tensor<i64>)
  return(%For#0, %For#1, %For#2, %For#3) : tensor<i8>, tensor<i16>, tensor<i32>, tensor<i64>
}

// CHECK-LABEL: @test_uncapture_while
// CHECK: %[[A0:.*]]: tensor
// CHECK: %[[A1:.*]]: tensor
tfg.func @test_uncapture_while(%a0: tensor<i8> {tfg.name = "a0"},
                               %a1: tensor<i16> {tfg.name = "a1"})
    -> (tensor<i8>, tensor<i16>) {
  // CHECK: %[[WHILE:.*]], {{.*}} = WhileRegion(%[[A1]])
  // CHECK: ^bb0(%[[ARG0:.*]]: tensor<i16>
  // CHECK:  %[[COND:.*]], %{{.*}} = Cond(%[[A0]], %[[ARG0]])
  // CHECK:  %[[CTL:.*]] = NoOp [%[[A0]].ctl]
  // CHECK:  condition %[[COND]] : tensor<i1> (%[[ARG0]])
  // CHECK: ^bb0(%[[ARG0:.*]]: tensor<i16>, %[[CTL0:.*]]: !tf_type.control
  // CHECK:   %[[THING:.*]], %{{.*}} = Thing(%[[A0]], %[[ARG0]]) [%[[CTL0]]]
  // CHECK:   yield(%[[THING]])
  // CHECK: cond_region_attrs = #tfg.region_attrs<{tf._a} [{}] [{}]>
  %While:2, %ctl = WhileRegion(%a0, %a1) {
  ^bb0(%arg0: tensor<i8>, %arg1: tensor<i16>, %ctl0: !tf_type.control, %ctl1: !tf_type.control):
    %Cond, %ctl_Cond = Cond(%arg0, %arg1) : (tensor<i8>, tensor<i16>) -> (tensor<i1>)
    %ctl_NoOp = NoOp [%ctl0] : () -> ()
    condition %Cond : tensor<i1> (%arg0, %arg1) [%ctl_NoOp] : tensor<i8>, tensor<i16>
  } do {
  ^bb0(%arg0: tensor<i8>, %arg1: tensor<i16>, %ctl0: !tf_type.control, %ctl1: !tf_type.control):
    %Thing, %ctl_Thing = Thing(%arg0, %arg1) [%ctl1] : (tensor<i8>, tensor<i16>) -> (tensor<i16>)
    yield(%arg0, %Thing) : tensor<i8>, tensor<i16>
  } {parallel_iterations = 10 : i64,
     cond_region_attrs = #tfg.region_attrs<{tf._a} [{}, {}] [{}]>}
  : (tensor<i8>, tensor<i16>) -> (tensor<i8>, tensor<i16>)
  // CHECK: return(%[[A0]], %[[WHILE]])
  return(%While#0, %While#1) : tensor<i8>, tensor<i16>
}
