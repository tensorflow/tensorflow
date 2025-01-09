// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text %s -o - | FileCheck %s

// CHECK-LABEL: HloModule main
module {
  func.func @main(%arg0: tensor<i64>) -> tensor<i64> {
    %0 = "mhlo.while"(%arg0) ({
    // CHECK: [[R0:%.+]] ([[A0:.+]]: s64[]) -> s64[] {
    // CHECK:   %[[A0]] = s64[] parameter(0)
    // CHECK:   ROOT %add.{{.*}} = s64[] add(s64[] %[[A0]], s64[] %[[A0]])
    // CHECK: [[R1:%.+]] ([[A0:.+]]: s64[]) -> pred[] {
    // CHECK:   %[[A0]] = s64[] parameter(0)
    // CHECK:   ROOT %compare.{{.*}} = pred[] compare(s64[] %[[A0]], s64[] %[[A0]]), direction=LT
    ^bb0(%arg1: tensor<i64>):
      %1 = "mhlo.compare"(%arg1, %arg1) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "mhlo.return"(%1) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg1: tensor<i64>):
      %1 = mhlo.add %arg1, %arg1 : tensor<i64>
      "mhlo.return"(%1) : (tensor<i64>) -> ()
    }) : (tensor<i64>) -> tensor<i64>

    // CHECK: ENTRY %main.{{.*}} ([[A0:.+]]: s64[]) -> s64[] {
    // CHECK:   %[[A0]] = s64[] parameter(0)
    // CHECK:   ROOT %while.{{.*}} = s64[] while(s64[] %[[A0]]), condition=[[R1]], body=[[R0]]
    func.return %0 : tensor<i64>
  }
}

// -----

// CHECK-LABEL: HloModule main

// CHECK: [[BODY:%.+]] ([[TUPLE:.+]]: (s32[], s32[], f32[], f32[])) -> (s32[], s32[], f32[], f32[]) {
// CHECK-NEXT:  %[[TUPLE]] = (s32[], s32[], f32[], f32[]) parameter(0)
// CHECK-NEXT:  %[[GTE_0:.*]] = s32[] get-tuple-element((s32[], s32[], f32[], f32[]) %[[TUPLE]]), index=0
// CHECK-NEXT:  %[[GTE_1:.*]] = s32[] get-tuple-element((s32[], s32[], f32[], f32[]) %[[TUPLE]]), index=1
// CHECK-NEXT:  %[[GTE_2:.*]] = f32[] get-tuple-element((s32[], s32[], f32[], f32[]) %[[TUPLE]]), index=2
// CHECK-NEXT:  %[[GTE_3:.*]] = f32[] get-tuple-element((s32[], s32[], f32[], f32[]) %[[TUPLE]]), index=3
// CHECK-NEXT:  %[[ADD:.*]] = f32[] add(f32[] %[[GTE_2]], f32[] %[[GTE_3]])
// CHECK-NEXT:  ROOT %[[TUPLE_RES:.*]] = (s32[], s32[], f32[], f32[]) tuple(s32[] %[[GTE_0]], s32[] %[[GTE_1]], f32[] %[[GTE_2]], f32[] %[[ADD]])
// CHECK: }

// CHECK: [[COND:%.+]] ([[TUPLE:.+]]: (s32[], s32[], f32[], f32[])) -> pred[] {
// CHECK-NEXT:  %[[TUPLE]] = (s32[], s32[], f32[], f32[]) parameter(0)
// CHECK-NEXT:  %[[GTE_0:.*]] = f32[] get-tuple-element((s32[], s32[], f32[], f32[]) %[[TUPLE]]), index=2
// CHECK-NEXT:  %[[GTE_1:.*]] = f32[] get-tuple-element((s32[], s32[], f32[], f32[]) %[[TUPLE]]), index=3
// CHECK-NEXT:  %[[CST_0:.*]] = s32[] constant(0)
// CHECK-NEXT:  %[[GTE_2:.*]] = s32[] get-tuple-element((s32[], s32[], f32[], f32[]) %[[TUPLE]]), index=0
// CHECK-NEXT:  %[[GTE_3:.*]] = s32[] get-tuple-element((s32[], s32[], f32[], f32[]) %[[TUPLE]]), index=1
// CHECK-NEXT:  ROOT %[[CMP:.*]] = pred[] compare(s32[] %[[GTE_2]], s32[] %[[GTE_3]]), direction=LT
// CHECK: }


// CHECK: ENTRY
// CHECK-NEXT:  %[[CST_0:.*]] = s32[] constant(0)
// CHECK-NEXT:  %[[CST_1:.*]] = s32[] constant(100)
// CHECK-NEXT:  %[[CST_2:.*]] = f32[] constant(1)
// CHECK-NEXT:  %[[ARG_0:.*]] = f32[] parameter(0)
// CHECK-NEXT:  %[[TUPLE:.*]] = (s32[], s32[], f32[], f32[]) tuple(s32[] %[[CST_0]], s32[] %[[CST_1]], f32[] %[[CST_2]], f32[] %[[ARG_0]])
// CHECK-NEXT:  %[[WHILE:.*]] = (s32[], s32[], f32[], f32[]) while((s32[], s32[], f32[], f32[]) %[[TUPLE]]), condition=[[COND]], body=[[BODY]]
// CHECK-NEXT:  %[[GTE_0:.*]] = s32[] get-tuple-element((s32[], s32[], f32[], f32[]) %[[WHILE]]), index=0
// CHECK-NEXT:  %[[GTE_1:.*]] = s32[] get-tuple-element((s32[], s32[], f32[], f32[]) %[[WHILE]]), index=1
// CHECK-NEXT:  %[[GTE_2:.*]] = f32[] get-tuple-element((s32[], s32[], f32[], f32[]) %[[WHILE]]), index=2
// CHECK-NEXT:  ROOT %[[GTE_3:.*]] = f32[] get-tuple-element((s32[], s32[], f32[], f32[]) %[[WHILE]]), index=3

func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.constant dense<100> : tensor<i32>
  %2 = mhlo.constant dense<1.000000e+00> : tensor<f32>
  %3:4 = "mhlo.while"(%0, %1, %2, %arg0) ({
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<f32>, %arg4: tensor<f32>):
    %4 = mhlo.constant dense<0> : tensor<i32>
    %5 = "mhlo.compare"(%arg1, %arg2) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%5) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<f32>, %arg4: tensor<f32>):
    %4 = mhlo.add %arg3, %arg4 : tensor<f32>
    "mhlo.return"(%arg1, %arg2, %arg3, %4) : (tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>) -> ()
  }) : (tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>) -> (tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>)
  func.return %3#3 : tensor<f32>
}

// -----

// CHECK-LABEL: HloModule main

// CHECK: [[BODY:%.+]] ([[TUPLE_0:.+]]: (s32[1], s32[2], f32[1], f32[3])) -> (s32[1], s32[2], f32[1], f32[3]) {
// CHECK-NEXT:   %[[TUPLE_0:.*]] = (s32[1], s32[2], f32[1], f32[3]) parameter(0)
// CHECK-NEXT:   %[[GTE_0:.*]] = s32[1]  get-tuple-element((s32[1], s32[2], f32[1], f32[3]) %[[TUPLE_0]]), index=0
// CHECK-NEXT:   %[[GTE_1:.*]] = s32[2] get-tuple-element((s32[1], s32[2], f32[1], f32[3]) %[[TUPLE_0]]), index=1
// CHECK-NEXT:   %[[GTE_2:.*]] = f32[1] get-tuple-element((s32[1], s32[2], f32[1], f32[3]) %[[TUPLE_0]]), index=2
// CHECK-NEXT:   %[[GTE_3:.*]] = f32[3] get-tuple-element((s32[1], s32[2], f32[1], f32[3]) %[[TUPLE_0]]), index=3
// CHECK-NEXT:   %[[BDCAST_0:.*]] = f32[1] broadcast(f32[1] %[[GTE_2]]), dimensions={0}
// CHECK-NEXT:   %[[RESHAPE_0:.*]] = f32[] reshape(f32[1] %[[BDCAST_0]])
// CHECK-NEXT:   %[[BDCAST_1:.*]] = f32[3] broadcast(f32[] %[[RESHAPE_0]]), dimensions={}
// CHECK-NEXT:   %[[ADD:.*]] = f32[3] add(f32[3] %[[GTE_3]], f32[3] %[[BDCAST_1]])
// CHECK-NEXT:   ROOT %[[TUPLE_0:.*]] = (s32[1], s32[2], f32[1], f32[3]) tuple(s32[1] %[[GTE_0]], s32[2] %[[GTE_1]], f32[1] %[[GTE_2]], f32[3] %[[ADD]])
// CHECK: }

// CHECK: [[COND:%.+]] ([[TUPLE_0:.+]]: (s32[1], s32[2], f32[1], f32[3])) -> pred[] {
// CHECK-NEXT:   %[[TUPLE_0:.*]] = (s32[1], s32[2], f32[1], f32[3]) parameter(0)
// CHECK-NEXT:   %[[GTE_0:.*]] = f32[1] get-tuple-element((s32[1], s32[2], f32[1], f32[3]) %[[TUPLE_0]]), index=2
// CHECK-NEXT:   %[[GTE_1:.*]] = f32[3] get-tuple-element((s32[1], s32[2], f32[1], f32[3]) %[[TUPLE_0]]), index=3
// CHECK-NEXT:   %[[GTE_2:.*]] = s32[1] get-tuple-element((s32[1], s32[2], f32[1], f32[3]) %[[TUPLE_0]]), index=0
// CHECK-NEXT:   %[[CST_0:.*]] = s32[] constant(0)
// CHECK-NEXT:   %[[RED_0:.*]] = s32[] reduce(s32[1] %[[GTE_2]], s32[] %[[CST_0]]), dimensions={0}, to_apply=
// CHECK-NEXT:   %[[GTE_3:.*]] = s32[2] get-tuple-element((s32[1], s32[2], f32[1], f32[3]) %[[TUPLE_0]]), index=1
// CHECK-NEXT:   %[[RED_1:.*]] = s32[] reduce(s32[2] %[[GTE_3]], s32[] %[[CST_0]]), dimensions={0}, to_apply=
// CHECK-NEXT:   ROOT %[[CMP:.*]] = pred[] compare(s32[] %[[RED_0]], s32[] %[[RED_1]]), direction=LT
// CHECK: }

// CHECK: ENTRY
// CHECK-NEXT:  %[[CST_0:.*]] = s32[1] constant({0})
// CHECK-NEXT:  %[[CST_1:.*]] = s32[] constant(100)
// CHECK-NEXT:  %[[BDCAST_0:.*]] = s32[2] broadcast(s32[] %[[CST_1]]), dimensions={}
// CHECK-NEXT:  %[[CST_2:.*]] = f32[1] constant({1})
// CHECK-NEXT:  %[[ARG_0:.*]] = f32[3] parameter(0)
// CHECK-NEXT:  %[[TUPLE:.*]] = (s32[1], s32[2], f32[1], f32[3]) tuple(s32[1] %[[CST_0]], s32[2] %[[BDCAST_0]], f32[1] %[[CST_2]], f32[3] %[[ARG_0]])
// CHECK-NEXT:  %[[WHILE:.*]] = (s32[1], s32[2], f32[1], f32[3]) while((s32[1], s32[2], f32[1], f32[3]) %[[TUPLE]]), condition=[[COND]], body=[[BODY]]
// CHECK-NEXT:  %[[GTE_0:.*]] = s32[1] get-tuple-element((s32[1], s32[2], f32[1], f32[3]) %[[WHILE]]), index=0
// CHECK-NEXT:  %[[GTE_1:.*]] = s32[2] get-tuple-element((s32[1], s32[2], f32[1], f32[3]) %[[WHILE]]), index=1
// CHECK-NEXT:  %[[GTE_2:.*]] = f32[1] get-tuple-element((s32[1], s32[2], f32[1], f32[3]) %[[WHILE]]), index=2
// CHECK-NEXT:  ROOT %[[GTE_3:.*]] = f32[3] get-tuple-element((s32[1], s32[2], f32[1], f32[3]) %[[WHILE]]), index=3

func.func @main(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = mhlo.constant dense<0> : tensor<1xi32>
  %1 = mhlo.constant dense<100> : tensor<2xi32>
  %2 = mhlo.constant dense<1.000000e+00> : tensor<1xf32>
  %3:4 = "mhlo.while"(%0, %1, %2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %4 = mhlo.constant dense<0> : tensor<i32>
    %5 = "mhlo.reduce"(%arg1, %4) ({
    ^bb0(%arg5: tensor<i32>, %arg6: tensor<i32>):
      %8 = mhlo.add %arg5, %arg6 : tensor<i32>
      "mhlo.return"(%8) : (tensor<i32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<1xi32>, tensor<i32>) -> tensor<i32>
    %6 = "mhlo.reduce"(%arg2, %4) ({
    ^bb0(%arg5: tensor<i32>, %arg6: tensor<i32>):
      %8 = mhlo.add %arg5, %arg6 : tensor<i32>
      "mhlo.return"(%8) : (tensor<i32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<2xi32>, tensor<i32>) -> tensor<i32>
    %7 = "mhlo.compare"(%5, %6) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %4 = "mhlo.broadcast_in_dim"(%arg3) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<1xf32>) -> tensor<3xf32>
    %5 = mhlo.add %arg4, %4 : tensor<3xf32>
    "mhlo.return"(%arg1, %arg2, %arg3, %5) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  func.return %3#3 : tensor<3xf32>
}

// -----

// CHECK-LABEL: HloModule main

// CHECK: [[BODY:%.+]] ([[TUPLE:.+]]: (s32[], s32[], s32[])) -> (s32[], s32[], s32[]) {
// CHECK-NEXT: %[[TUPLE:.*]] = (s32[], s32[], s32[]) parameter(0)
// CHECK-NEXT: %[[GTE_0:.*]] = s32[] get-tuple-element((s32[], s32[], s32[]) %[[TUPLE]]), index=0
// CHECK-NEXT: %[[GTE_1:.*]] = s32[] get-tuple-element((s32[], s32[], s32[]) %[[TUPLE]]), index=1
// CHECK-NEXT: %[[ADD:.*]] = s32[] add(s32[] %[[GTE_0]], s32[] %[[GTE_1]])
// CHECK-NEXT: %[[GTE_2:.*]] = s32[] get-tuple-element((s32[], s32[], s32[]) %[[TUPLE]]), index=2
// CHECK-NEXT: ROOT %[[TUPLE_RES:.*]] = (s32[], s32[], s32[]) tuple(s32[] %[[ADD]], s32[] %[[GTE_1]], s32[] %[[GTE_2]])
// CHECK: }

// CHECK: [[COND:%.+]] ([[TUPLE:.+]]: (s32[], s32[], s32[])) -> pred[] {
// CHECK-NEXT:  %[[TUPLE:.*]] = (s32[], s32[], s32[]) parameter(0)
// CHECK-NEXT:  %[[GTE_0:.*]] = s32[] get-tuple-element((s32[], s32[], s32[]) %[[TUPLE]]), index=1
// CHECK-NEXT:  %[[GTE_1:.*]] = s32[] get-tuple-element((s32[], s32[], s32[]) %[[TUPLE]]), index=0
// CHECK-NEXT:  %[[GTE_2:.*]] = s32[] get-tuple-element((s32[], s32[], s32[]) %[[TUPLE]]), index=2
// CHECK-NEXT:  ROOT %[[CMP:.*]] = pred[] compare(s32[] %[[GTE_1]], s32[] %[[GTE_2]]), direction=LT
// CHECK: }

// CHECK: ENTRY
// CHECK-NEXT:  %[[ARG_0:.*]] = (s32[], (s32[], (s32[]))) parameter(0)
// CHECK-NEXT:  %[[GTE_0:.*]] = s32[] get-tuple-element((s32[], (s32[], (s32[]))) %[[ARG_0]]), index=0
// CHECK-NEXT:  %[[GTE_1:.*]] = (s32[], (s32[])) get-tuple-element((s32[], (s32[], (s32[]))) %[[ARG_0]]), index=1
// CHECK-NEXT:  %[[GTE_2:.*]] = s32[] get-tuple-element((s32[], (s32[])) %[[GTE_1]]), index=0
// CHECK-NEXT:  %[[GTE_3:.*]] = (s32[]) get-tuple-element((s32[], (s32[])) %[[GTE_1]]), index=1
// CHECK-NEXT:  %[[GTE_4:.*]] = s32[] get-tuple-element((s32[]) %[[GTE_3]]), index=0
// CHECK-NEXT:  %[[TUPLE_0:.*]] = (s32[], s32[], s32[]) tuple(s32[] %[[GTE_0]], s32[] %[[GTE_2]], s32[] %[[GTE_4]])
// CHECK-NEXT:  %[[WHILE:.*]] = (s32[], s32[], s32[]) while((s32[], s32[], s32[]) %[[TUPLE_0]]), condition=[[COND]], body=[[BODY]]
// CHECK-NEXT:  %[[GTE_5:.*]] = s32[] get-tuple-element((s32[], s32[], s32[]) %[[WHILE]]), index=0
// CHECK-NEXT:  %[[GTE_6:.*]] = s32[] get-tuple-element((s32[], s32[], s32[]) %[[WHILE]]), index=1
// CHECK-NEXT:  %[[GTE_7:.*]] = s32[] get-tuple-element((s32[], s32[], s32[]) %[[WHILE]]), index=2
// CHECK-NEXT:  %[[TUPLE_1:.*]] = (s32[]) tuple(s32[] %[[GTE_7]])
// CHECK-NEXT:  %[[TUPLE_2:.*]] = (s32[], (s32[])) tuple(s32[] %[[GTE_6]], (s32[]) %[[TUPLE_1]])
// CHECK-NEXT:  ROOT %[[TUPLE_3:.*]] = (s32[], (s32[], (s32[]))) tuple(s32[] %[[GTE_5]], (s32[], (s32[])) %[[TUPLE_2]])

 func.func  @main(%arg0: tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>) -> tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>> {
    %0 = "mhlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>) -> tensor<i32>
    %1 = "mhlo.get_tuple_element"(%arg0) {index = 1 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>) -> tuple<tensor<i32>, tuple<tensor<i32>>>
    %2 = "mhlo.get_tuple_element"(%1) {index = 0 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>>>) -> tensor<i32>
    %3 = "mhlo.get_tuple_element"(%1) {index = 1 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>>>) -> tuple<tensor<i32>>
    %4 = "mhlo.get_tuple_element"(%3) {index = 0 : i32} : (tuple<tensor<i32>>) -> tensor<i32>
    %5:3 = "mhlo.while"(%0, %2, %4) ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
      %9 = "mhlo.compare"(%arg1, %arg3) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "mhlo.return"(%9) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
      %9 = mhlo.add %arg1, %arg2 : tensor<i32>
      "mhlo.return"(%9, %arg2, %arg3) : (tensor<i32>, tensor<i32>, tensor<i32>) -> ()
    }) : (tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
    %6 = "mhlo.tuple"(%5#2) : (tensor<i32>) -> tuple<tensor<i32>>
    %7 = "mhlo.tuple"(%5#1, %6) : (tensor<i32>, tuple<tensor<i32>>) -> tuple<tensor<i32>, tuple<tensor<i32>>>
    %8 = "mhlo.tuple"(%5#0, %7) {xla_shape = "(s32[], (s32[], (s32[])))"} : (tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>) -> tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>
    func.return %8 : tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>
}

// -----

// CHECK-LABEL: HloModule main

// CHECK: [[BODY:%.+]] ([[ARG_0:.+]]: f32[3,3]) -> f32[3,3] {
// CHECK-NEXT:  pred[] constant(false)
// CHECK-NEXT:  %[[ARG_0:.*]] = f32[3,3] parameter(0)
// CHECK-NEXT:  %[[CST_1:.*]] = f32[] constant(2)
// CHECK-NEXT:  %[[BDCAST:.*]] = f32[3,3] broadcast(f32[] %[[CST_1]]), dimensions={}
// CHECK-NEXT:  ROOT %[[ADD:.*]] = f32[3,3] add(f32[3,3] %[[ARG_0]], f32[3,3] %[[BDCAST]])
// CHECK: }

// CHECK: [[REDUCER:%.+]] ([[ARG_0:.+]]: f32[], [[ARG_1:.+]]: f32[]) -> f32[] {
// CHECK-NEXT:   constant(false)
// CHECK-NEXT:   %[[ARG_0:.*]] = f32[] parameter(0)
// CHECK-NEXT:   %[[ARG_1:.*]] = f32[] parameter(1)
// CHECK-NEXT:   ROOT %[[ADD:.*]] = f32[] add(f32[] %[[ARG_0]], f32[] %[[ARG_1]])
// CHECK: }

// CHECK: [[COND:%.+]] ([[ARG_0:.+]]: f32[3,3]) -> pred[] {
// CHECK-NEXT:   pred[] constant(false)
// CHECK-NEXT:   %[[ARG_0:.*]] = f32[3,3] parameter(0)
// CHECK-NEXT:   %[[CST_0:.*]] = f32[] constant(0)
// CHECK-NEXT:   %[[REDUCE:.*]] = f32[] reduce(f32[3,3] %[[ARG_0]], f32[] %[[CST_0]]), dimensions={0,1}, to_apply=[[REDUCER]]
// CHECK-NEXT:   %[[CST_1:.*]] = f32[] constant(100)
// CHECK-NEXT:   ROOT %[[CMP:.*]] = pred[] compare(f32[] %[[REDUCE]], f32[] %[[CST_1]]), direction=LT

// CHECK: ENTRY
// CHECK-NEXT:  %[[CST_0:.*]] = pred[] constant(false)
// CHECK-NEXT:  %[[ARG_0:.*]] = f32[3,3] parameter(0)
// CHECK-NEXT:  ROOT %[[WHILE:.*]] = f32[3,3] while(f32[3,3] %[[ARG_0]]), condition=[[COND]], body=[[BODY]]

func.func @main(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %0 = mhlo.constant dense<false> : tensor<i1>
  %1 = "mhlo.while"(%arg0) ({
  ^bb0(%arg1: tensor<3x3xf32>):
    %2 = mhlo.constant dense<false> : tensor<i1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = "mhlo.reduce"(%arg1, %3) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %7 = mhlo.constant dense<false> : tensor<i1>
      %8 = mhlo.add %arg2, %arg3 : tensor<f32>
      "mhlo.return"(%8) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<3x3xf32>, tensor<f32>) -> tensor<f32>
    %5 = mhlo.constant dense<1.000000e+02> : tensor<f32>
    %6 = "mhlo.compare"(%4, %5) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%6) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<3x3xf32>):
    %2 = mhlo.constant dense<false> : tensor<i1>
    %3 = mhlo.constant dense<2.000000e+00> : tensor<f32>
    %4 = "mhlo.broadcast_in_dim"(%3) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<3x3xf32>
    %5 = mhlo.add %arg1, %4 : tensor<3x3xf32>
    "mhlo.return"(%5) : (tensor<3x3xf32>) -> ()
  }) : (tensor<3x3xf32>) -> tensor<3x3xf32>
  func.return %1 : tensor<3x3xf32>
}

// -----

// CHECK-LABEL: HloModule main

// CHECK: [[BODY:%.+]] ([[ARG_TUPLE:.+]]: (s32[], s32[])) -> (s32[], s32[]) {
// CHECK-NEXT:  %[[ARG_TUPLE:.*]] = (s32[], s32[]) parameter(0)
// CHECK-NEXT:  %[[GTE_0:.*]] = s32[] get-tuple-element((s32[], s32[]) %[[ARG_TUPLE]]), index=0
// CHECK-NEXT:  %[[GTE_1:.*]] = s32[] get-tuple-element((s32[], s32[]) %[[ARG_TUPLE]]), index=1
// CHECK-NEXT:  %[[TUPLE_0:.*]] = (s32[], s32[]) tuple(s32[] %[[GTE_0]], s32[] %[[GTE_1]])
// CHECK-NEXT:  %[[CC:.*]] = (s32[], s32[]) custom-call(s32[] %[[GTE_0]], (s32[], s32[]) %[[TUPLE_0]])
// CHECK-NEXT:  %[[GTE_2:.*]] = s32[] get-tuple-element((s32[], s32[]) %[[CC]]), index=0
// CHECK-NEXT:  %[[GTE_3:.*]] = s32[] get-tuple-element((s32[], s32[]) %[[CC]]), index=1
// CHECK-NEXT:  ROOT %[[TUPLE_1:.*]] = (s32[], s32[]) tuple(s32[] %[[GTE_2]], s32[] %[[GTE_3]])
// CHECK: }

// CHECK: [[COND:%.+]] ([[ARG_TUPLE:.+]]: (s32[], s32[])) -> pred[] {
// CHECK-NEXT:  %[[ARG_TUPLE:.*]] = (s32[], s32[]) parameter(0)
// CHECK-NEXT:  %[[GTE_0:.*]] = s32[] get-tuple-element((s32[], s32[]) %[[ARG_TUPLE]]), index=0
// CHECK-NEXT:  %[[GTE_1:.*]] = s32[] get-tuple-element((s32[], s32[]) %[[ARG_TUPLE]]), index=1
// CHECK-NEXT:  ROOT %compare.{{.*}} = pred[] compare(s32[] %[[GTE_0]], s32[] %[[GTE_1]]), direction=LT
// CHECK: }

// CHECK: ENTRY
// CHECK-NEXT:  %[[CST_0:.*]] = s32[] constant(0)
// CHECK-NEXT:  %[[ARG_0:.*]] = s32[] parameter(0)
// CHECK-NEXT:  %[[TUPLE:.*]] = (s32[], s32[]) tuple(s32[] %[[CST_0]], s32[] %[[ARG_0]])
// CHECK-NEXT:  %[[WHILE:.*]] = (s32[], s32[]) while((s32[], s32[]) %[[TUPLE]]), condition=[[COND]], body=[[BODY]]
// CHECK-NEXT:  ROOT %[[GTE_0:.*]] = s32[] get-tuple-element((s32[], s32[]) %[[WHILE]]), index=0
// CHECK-NEXT:  %[[GTE_1:.*]] = s32[] get-tuple-element((s32[], s32[]) %[[WHILE]]), index=1

func.func @main(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1:2 = "mhlo.while"(%0, %arg0) ({
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %2 = "mhlo.compare"(%arg1, %arg2) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%2) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
    %2 = "mhlo.tuple"(%arg1, %arg2) : (tensor<i32>, tensor<i32>) -> tuple<tensor<i32>, tensor<i32>>
    %3 = "mhlo.custom_call"(%arg1, %2) {api_version = 1 : i32, backend_config = "bar", call_target_name = "foo", has_side_effect = false, xla_shape = "(s32[], s32[])"} : (tensor<i32>, tuple<tensor<i32>, tensor<i32>>) -> tuple<tensor<i32>, tensor<i32>>
    %4 = "mhlo.get_tuple_element"(%3) {index = 0 : i32} : (tuple<tensor<i32>, tensor<i32>>) -> tensor<i32>
    %5 = "mhlo.get_tuple_element"(%3) {index = 1 : i32} : (tuple<tensor<i32>, tensor<i32>>) -> tensor<i32>
    "mhlo.return"(%4, %5) : (tensor<i32>, tensor<i32>) -> ()
  }) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  func.return %1#0 : tensor<i32>
}

