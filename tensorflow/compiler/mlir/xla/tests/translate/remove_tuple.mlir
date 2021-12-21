// RUN: tf-mlir-translate -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s

// CHECK: HloModule

// CHECK: ENTRY
// CHECK-NEXT:  %[[ARG_0:.*]] = f32[4] parameter(0)
// CHECK-NEXT:  %[[ARG_1:.*]] = f32[4] parameter(1)
// CHECK-NEXT:  %[[ADD:.*]] = f32[4] add(f32[4] %[[ARG_0]], f32[4] %[[ARG_1]])
// CHECK-NEXT:  %[[TUPLE:.*]] = (f32[4], f32[4]) tuple(f32[4] %[[ADD]], f32[4] %[[ADD]])
// CHECK-NEXT:  %[[GTE:.*]] = f32[4] get-tuple-element((f32[4], f32[4]) %[[TUPLE]]), index=1
// CHECK-NEXT:  ROOT %[[DOT:.*]] = f32[] dot(f32[4] %[[GTE]], f32[4] %[[ARG_1]])

func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<f32> {
  %0 = mhlo.add %arg0, %arg1 : tensor<4xf32>
  %1 = "mhlo.tuple"(%0, %0) {xla_shape = "(f32[4]{0}, f32[4]{0})"} : (tensor<4xf32>, tensor<4xf32>) -> tuple<tensor<4xf32>, tensor<4xf32>>
  %2 = "mhlo.get_tuple_element"(%1) {index = 1 : i32} : (tuple<tensor<4xf32>, tensor<4xf32>>) -> tensor<4xf32>
  %3 = "mhlo.dot"(%2, %arg1) {precision_config = ["DEFAULT", "DEFAULT"]} : (tensor<4xf32>, tensor<4xf32>) -> tensor<f32>
  return %3 : tensor<f32>
}

// -----

// CHECK: HloModule

// CHECK-LABEL: %region
// CHECK-NEXT:  %[[TUPLE:.*]] = (s32[], s32[], f32[], f32[]) parameter(0)
// CHECK-NEXT:  %[[GTE_0:.*]] = s32[] get-tuple-element((s32[], s32[], f32[], f32[]) %[[TUPLE]]), index=0
// CHECK-NEXT:  %[[GTE_1:.*]] = s32[] get-tuple-element((s32[], s32[], f32[], f32[]) %[[TUPLE]]), index=1
// CHECK-NEXT:  %[[GTE_2:.*]] = f32[] get-tuple-element((s32[], s32[], f32[], f32[]) %[[TUPLE]]), index=2
// CHECK-NEXT:  %[[GTE_3:.*]] = f32[] get-tuple-element((s32[], s32[], f32[], f32[]) %[[TUPLE]]), index=3
// CHECK-NEXT:  %[[ADD:.*]] = f32[] add(f32[] %[[GTE_2]], f32[] %[[GTE_3]])
// CHECK-NEXT:  ROOT %[[TUPLE_RES:.*]] = (s32[], s32[], f32[], f32[]) tuple(s32[] %[[GTE_0]], s32[] %[[GTE_1]], f32[] %[[GTE_2]], f32[] %[[ADD]])

func private @REG_BODY(%arg0: tuple<tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>>) -> tuple<tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>> {
  %0 = "mhlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>>) -> tensor<i32>
  %1 = "mhlo.get_tuple_element"(%arg0) {index = 1 : i32} : (tuple<tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>>) -> tensor<i32>
  %2 = "mhlo.get_tuple_element"(%arg0) {index = 2 : i32} : (tuple<tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>>) -> tensor<f32>
  %3 = "mhlo.get_tuple_element"(%arg0) {index = 3 : i32} : (tuple<tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>>) -> tensor<f32>
  %4 = mhlo.add %2, %3 : tensor<f32>
  %5 = "mhlo.tuple"(%0, %1, %2, %4) {xla_shape = "(s32[], s32[], f32[], f32[])"} : (tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tuple<tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>>
  return %5 : tuple<tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>>
}

// CHECK-LABEL: %region
// CHECK-NEXT:  %[[TUPLE:.*]] = (s32[], s32[], f32[], f32[]) parameter(0)
// CHECK-NEXT:  %[[GTE_0:.*]] = f32[] get-tuple-element((s32[], s32[], f32[], f32[]) %[[TUPLE]]), index=2
// CHECK-NEXT:  %[[GTE_1:.*]] = f32[] get-tuple-element((s32[], s32[], f32[], f32[]) %[[TUPLE]]), index=3
// CHECK-NEXT:  %[[CST_0:.*]] = s32[] constant(0)
// CHECK-NEXT:  %[[GTE_2:.*]] = s32[] get-tuple-element((s32[], s32[], f32[], f32[]) %[[TUPLE]]), index=0
// CHECK-NEXT:  %[[GTE_3:.*]] = s32[] get-tuple-element((s32[], s32[], f32[], f32[]) %[[TUPLE]]), index=1
// CHECK-NEXT:  ROOT %[[CMP:.*]] = pred[] compare(s32[] %[[GTE_2]], s32[] %[[GTE_3]]), direction=LT

func private @REG_COND(%arg0: tuple<tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>>) -> tensor<i1> {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = "mhlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>>) -> tensor<i32>
  %2 = "mhlo.get_tuple_element"(%arg0) {index = 1 : i32} : (tuple<tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>>) -> tensor<i32>
  %3 = "mhlo.compare"(%1, %2) {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %3 : tensor<i1>
}

// CHECK: ENTRY
// CHECK-NEXT:  %[[CST_0:.*]] = s32[] constant(0)
// CHECK-NEXT:  %[[CST_1:.*]] = s32[] constant(100)
// CHECK-NEXT:  %[[CST_2:.*]] = f32[] constant(1)
// CHECK-NEXT:  %[[ARG_0:.*]] = f32[] parameter(0)
// CHECK-NEXT:  %[[TUPLE:.*]] = (s32[], s32[], f32[], f32[]) tuple(s32[] %[[CST_0]], s32[] %[[CST_1]], f32[] %[[CST_2]], f32[] %[[ARG_0]])
// CHECK-NEXT:  %[[WHILE:.*]] = (s32[], s32[], f32[], f32[]) while((s32[], s32[], f32[], f32[]) %[[TUPLE]]), condition=[[COND:.*]], body=[[BODY:.*]]
// CHECK-NEXT:  %[[GTE_0:.*]] = s32[] get-tuple-element((s32[], s32[], f32[], f32[]) %[[WHILE]]), index=0
// CHECK-NEXT:  %[[GTE_1:.*]] = s32[] get-tuple-element((s32[], s32[], f32[], f32[]) %[[WHILE]]), index=1
// CHECK-NEXT:  %[[GTE_2:.*]] = f32[] get-tuple-element((s32[], s32[], f32[], f32[]) %[[WHILE]]), index=2
// CHECK-NEXT:  ROOT %[[GTE_3:.*]] = f32[] get-tuple-element((s32[], s32[], f32[], f32[]) %[[WHILE]]), index=3

func @main(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.constant dense<100> : tensor<i32>
  %2 = mhlo.constant dense<1.000000e+00> : tensor<f32>
  %3:4 = "mhlo.while"(%0, %1, %2, %arg0) ( {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
    %4 = mhlo.constant dense<0> : tensor<i32>
    %5 = "mhlo.compare"(%arg1, %arg2) {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%5) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<f32>, %arg4: tensor<f32>):  // no predecessors
    %4 = mhlo.add %arg3, %arg4 : tensor<f32>
    "mhlo.return"(%arg1, %arg2, %arg3, %4) : (tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>) -> ()
  }) : (tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>) -> (tensor<i32>, tensor<i32>, tensor<f32>, tensor<f32>)
  return %3#3 : tensor<f32>
}

// -----

// CHECK: HloModule

// CHECK-LABEL: %region_0.7
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

// CHECK-LABEL: %region_1.26
// CHECK-NEXT:   %[[TUPLE_0:.*]] = (s32[1], s32[2], f32[1], f32[3]) parameter(0)
// CHECK-NEXT:   %[[GTE_0:.*]] = f32[1] get-tuple-element((s32[1], s32[2], f32[1], f32[3]) %[[TUPLE_0]]), index=2
// CHECK-NEXT:   %[[GTE_1:.*]] = f32[3] get-tuple-element((s32[1], s32[2], f32[1], f32[3]) %[[TUPLE_0]]), index=3
// CHECK-NEXT:   %[[GTE_2:.*]] = s32[1] get-tuple-element((s32[1], s32[2], f32[1], f32[3]) %[[TUPLE_0]]), index=0
// CHECK-NEXT:   %[[CST_0:.*]] = s32[] constant(0)
// CHECK-NEXT:   %[[RED_0:.*]] = s32[] reduce(s32[1] %[[GTE_2]], s32[] %constant.32), dimensions={0}, to_apply=
// CHECK-NEXT:   %[[GTE_3:.*]] = s32[2] get-tuple-element((s32[1], s32[2], f32[1], f32[3]) %[[TUPLE_0]]), index=1
// CHECK-NEXT:   %[[RED_1:.*]] = s32[] reduce(s32[2] %[[GTE_3]], s32[] %[[CST_0]]), dimensions={0}, to_apply=
// CHECK-NEXT:   ROOT %[[CMP:.*]] = pred[] compare(s32[] %[[RED_0]], s32[] %[[RED_1]]), direction=LT

// CHECK: ENTRY
// CHECK-NEXT:  %[[CST_0:.*]] = s32[1] constant({0})
// CHECK-NEXT:  %[[CST_1:.*]].3 = s32[] constant(100)
// CHECK-NEXT:  %[[BDCAST_0:.*]] = s32[2] broadcast(s32[] %constant.3), dimensions={}
// CHECK-NEXT:  %[[CST_2:.*]] = f32[1] constant({1})
// CHECK-NEXT:  %[[ARG_0:.*]] = f32[3] parameter(0)
// CHECK-NEXT:  %[[TUPLE:.*]] = (s32[1], s32[2], f32[1], f32[3]) tuple(s32[1] %[[CST_0]], s32[2] %[[BDCAST_0]], f32[1] %[[CST_2]], f32[3] %[[ARG_0]])
// CHECK-NEXT:  %[[WHILE:.*]] = (s32[1], s32[2], f32[1], f32[3]) while((s32[1], s32[2], f32[1], f32[3]) %[[TUPLE]]), condition=[[COND:.*]], body=[[BODY:.*]]
// CHECK-NEXT:  %[[GTE_0:.*]] = s32[1] get-tuple-element((s32[1], s32[2], f32[1], f32[3]) %[[WHILE]]), index=0
// CHECK-NEXT:  %[[GTE_1:.*]] = s32[2] get-tuple-element((s32[1], s32[2], f32[1], f32[3]) %[[WHILE]]), index=1
// CHECK-NEXT:  %[[GTE_2:.*]] = f32[1] get-tuple-element((s32[1], s32[2], f32[1], f32[3]) %[[WHILE]]), index=2
// CHECK-NEXT:  ROOT %[[GTE_3:.*]] = f32[3] get-tuple-element((s32[1], s32[2], f32[1], f32[3]) %[[WHILE]]), index=3

func @main(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %0 = mhlo.constant dense<0> : tensor<1xi32>
  %1 = mhlo.constant dense<100> : tensor<2xi32>
  %2 = mhlo.constant dense<1.000000e+00> : tensor<1xf32>
  %3:4 = "mhlo.while"(%0, %1, %2, %arg0) ( {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):  // no predecessors
    %4 = mhlo.constant dense<0> : tensor<i32>
    %5 = mhlo.reduce %arg1, %4 ( {
    ^bb0(%arg5: tensor<i32>, %arg6: tensor<i32>):  // no predecessors
      %8 = mhlo.add %arg5, %arg6 : tensor<i32>
      "mhlo.return"(%8) : (tensor<i32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<1xi32>, tensor<i32>) -> tensor<i32>
    %6 = mhlo.reduce %arg2, %4 ( {
    ^bb0(%arg5: tensor<i32>, %arg6: tensor<i32>):  // no predecessors
      %8 = mhlo.add %arg5, %arg6 : tensor<i32>
      "mhlo.return"(%8) : (tensor<i32>) -> ()
    }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<2xi32>, tensor<i32>) -> tensor<i32>
    %7 = "mhlo.compare"(%5, %6) {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):  // no predecessors
    %4 = "mhlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<3xf32>
    %5 = mhlo.add %arg4, %4 : tensor<3xf32>
    "mhlo.return"(%arg1, %arg2, %arg3, %5) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  return %3#3 : tensor<3xf32>
}

// -----

// CHECK: HloModule

// CHECK-LABEL: %region_0.8
// CHECK_NEXT: %[[TUPLE:.*]] = (s32[], s32[], s32[]) parameter(0)
// CHECK_NEXT: %[[GTE_0:.*]] = s32[] get-tuple-element((s32[], s32[], s32[]) %[[TU{LE]]), index=0
// CHECK_NEXT: %[[GTE_1:.*]] = s32[] get-tuple-element((s32[], s32[], s32[]) %[[TU{LE]]), index=1
// CHECK_NEXT: %[[ADD:.*]] = s32[] add(s32[] %[[GTE_0]], s32[] %[[GTE_1]])
// CHECK_NEXT: %[[GTE_2:.*]] = s32[] get-tuple-element((s32[], s32[], s32[]) %[[TUPLE]]), index=2
// CHECK_NEXT: ROOT %[[TUPLE_RES:.*]] = (s32[], s32[], s32[]) tuple(s32[] %[[ADD]], s32[] %[[GTE_1]], s32[] %[[GTE_2]])

// CHECK-LABEL: %region_1.15
// CHECK-NEXT:  %[[TUPLE:.*]] = (s32[], s32[], s32[]) parameter(0)
// CHECK-NEXT:  %[[GTE_0:.*]] = s32[] get-tuple-element((s32[], s32[], s32[]) %[[TUPLE]]), index=1
// CHECK-NEXT:  %[[GTE_1:.*]] = s32[] get-tuple-element((s32[], s32[], s32[]) %[[TUPLE]]), index=0
// CHECK-NEXT:  %[[GTE_2:.*]] = s32[] get-tuple-element((s32[], s32[], s32[]) %[[TUPLE]]), index=2
// CHECK-NEXT:  ROOT %[[CMP:.*]] = pred[] compare(s32[] %[[GTE_1]], s32[] %[[GTE_2]]), direction=LT

// CHECK-LABEL: ENTRY
// CHECK-NEXT:  %[[ARG_0:.*]] = (s32[], (s32[], (s32[]))) parameter(0)
// CHECK-NEXT:  %[[GTE_0:.*]] = s32[] get-tuple-element((s32[], (s32[], (s32[]))) %[[ARG_0]]), index=0
// CHECK-NEXT:  %[[GTE_1:.*]] = (s32[], (s32[])) get-tuple-element((s32[], (s32[], (s32[]))) %[[ARG_0]]), index=1
// CHECK-NEXT:  %[[GTE_2:.*]] = s32[] get-tuple-element((s32[], (s32[])) %[[GTE_1]]), index=0
// CHECK-NEXT:  %[[GTE_3:.*]] = (s32[]) get-tuple-element((s32[], (s32[])) %[[GTE_1]]), index=1
// CHECK-NEXT:  %[[GTE_4:.*]] = s32[] get-tuple-element((s32[]) %[[GTE_3]]), index=0
// CHECK-NEXT:  %[[TUPLE_0:.*]] = (s32[], s32[], s32[]) tuple(s32[] %[[GTE_0]], s32[] %[[GTE_2]], s32[] %[[GTE_4]])
// CHECK-NEXT:  %[[WHILE:.*]] = (s32[], s32[], s32[]) while((s32[], s32[], s32[]) %[[TUPLE_0]]), condition=%[[COND:.*]], body=%[[BODY:.*]]
// CHECK-NEXT:  %[[GTE_5:.*]] = s32[] get-tuple-element((s32[], s32[], s32[]) %[[WHILE]]), index=0
// CHECK-NEXT:  %[[GTE_6:.*]] = s32[] get-tuple-element((s32[], s32[], s32[]) %[[WHILE]]), index=1
// CHECK-NEXT:  %[[GTE_7:.*]] = s32[] get-tuple-element((s32[], s32[], s32[]) %[[WHILE]]), index=2
// CHECK-NEXT:  %[[TUPLE_1:.*]] = (s32[]) tuple(s32[] %[[GTE_7]])
// CHECK-NEXT:  %[[TUPLE_2:.*]] = (s32[], (s32[])) tuple(s32[] %[[GTE_6]], (s32[]) %[[TUPLE_1]])
// CHECK-NEXT:  ROOT %[[TUPLE_3:.*]] = (s32[], (s32[], (s32[]))) tuple(s32[] %[[GTE_5]], (s32[], (s32[])) %[[TUPLE_2]])

 func  @main(%arg0: tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>) -> tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>> {
    %0 = "mhlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>) -> tensor<i32>
    %1 = "mhlo.get_tuple_element"(%arg0) {index = 1 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>) -> tuple<tensor<i32>, tuple<tensor<i32>>>
    %2 = "mhlo.get_tuple_element"(%1) {index = 0 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>>>) -> tensor<i32>
    %3 = "mhlo.get_tuple_element"(%1) {index = 1 : i32} : (tuple<tensor<i32>, tuple<tensor<i32>>>) -> tuple<tensor<i32>>
    %4 = "mhlo.get_tuple_element"(%3) {index = 0 : i32} : (tuple<tensor<i32>>) -> tensor<i32>
    %5:3 = "mhlo.while"(%0, %2, %4) ( {
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>):  // no predecessors
      %9 = "mhlo.compare"(%arg1, %arg3) {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
      "mhlo.return"(%9) : (tensor<i1>) -> ()
    },  {
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>):  // no predecessors
      %9 = mhlo.add %arg1, %arg2 : tensor<i32>
      "mhlo.return"(%9, %arg2, %arg3) : (tensor<i32>, tensor<i32>, tensor<i32>) -> ()
    }) : (tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
    %6 = "mhlo.tuple"(%5#2) : (tensor<i32>) -> tuple<tensor<i32>>
    %7 = "mhlo.tuple"(%5#1, %6) : (tensor<i32>, tuple<tensor<i32>>) -> tuple<tensor<i32>, tuple<tensor<i32>>>
    %8 = "mhlo.tuple"(%5#0, %7) {xla_shape = "(s32[], (s32[], (s32[])))"} : (tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>) -> tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>
    return %8 : tuple<tensor<i32>, tuple<tensor<i32>, tuple<tensor<i32>>>>
}

// -----

// CHECK: HloModule

// CHECK-LABEL: %region_0.3
// CHECK-NEXT:  pred[] constant(false)
// CHECK-NEXT:  %[[ARG_0:.*]] = f32[3,3] parameter(0)
// CHECK-NEXT:  %[[CST_1:.*]] = f32[] constant(2)
// CHECK-NEXT:  %[[BDCAST:.*]] = f32[3,3] broadcast(f32[] %[[CST_1]]), dimensions={}
// CHECK-NEXT:  ROOT %[[ADD:.*]] = f32[3,3] add(f32[3,3] %[[ARG_0]], f32[3,3] %[[BDCAST]])

// CHECK-LABEL: %region_2.9
// CHECK-NEXT:   constant(false)
// CHECK-NEXT:   %[[ARG_0:.*]] = f32[] parameter(0)
// CHECK-NEXT:   %[[ARG_1:.*]] = f32[] parameter(1)
// CHECK-NEXT:   ROOT %[[ADD:.*]] = f32[] add(f32[] %[[ARG_0]], f32[] %[[ARG_1]])

// CHECK-LABEL: %region_1.14
// CHECK-NEXT:   pred[] constant(false)
// CHECK-NEXT:   %[[ARG_0:.*]] = f32[3,3] parameter(0)
// CHECK-NEXT:   %[[CST_0:.*]] = f32[] constant(0)
// CHECK-NEXT:   %[[REDUCE:.*]] = f32[] reduce(f32[3,3] %[[ARG_0]], f32[] %[[CST_0]]), dimensions={0,1}, to_apply=%region_2
// CHECK-NEXT:   %[[CST_1:.*]] = f32[] constant(100)
// CHECK-NEXT:   ROOT %[[CMP:.*]] = pred[] compare(f32[] %[[REDUCE]], f32[] %[[CST_1]]), direction=LT

// CHECK-LABEL: ENTRY
// CHECK-NEXT:  %[[CST_0:.*]] = pred[] constant(false)
// CHECK-NEXT:  %[[ARG_0:.*]] = f32[3,3] parameter(0)
// CHECK-NEXT:  ROOT %[[WHILE:.*]] = f32[3,3] while(f32[3,3] %[[ARG_0]]), condition=%[[COND:.*]], body=%[[BODY:.*]]

func @main(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %0 = mhlo.constant dense<false> : tensor<i1>
  %1 = "mhlo.while"(%arg0) ( {
  ^bb0(%arg1: tensor<3x3xf32>):  // no predecessors
    %2 = mhlo.constant dense<false> : tensor<i1>
    %3 = mhlo.constant dense<0.000000e+00> : tensor<f32>
    %4 = mhlo.reduce %arg1, %3 ( {
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):  // no predecessors
      %7 = mhlo.constant dense<false> : tensor<i1>
      %8 = mhlo.add %arg2, %arg3 : tensor<f32>
      "mhlo.return"(%8) : (tensor<f32>) -> ()
    }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<3x3xf32>, tensor<f32>) -> tensor<f32>
    %5 = mhlo.constant dense<1.000000e+02> : tensor<f32>
    %6 = "mhlo.compare"(%4, %5) {comparison_direction = "LT"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%6) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<3x3xf32>):  // no predecessors
    %2 = mhlo.constant dense<false> : tensor<i1>
    %3 = mhlo.constant dense<2.000000e+00> : tensor<f32>
    %4 = "mhlo.broadcast_in_dim"(%3) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<3x3xf32>
    %5 = mhlo.add %arg1, %4 : tensor<3x3xf32>
    "mhlo.return"(%5) : (tensor<3x3xf32>) -> ()
  }) : (tensor<3x3xf32>) -> tensor<3x3xf32>
  return %1 : tensor<3x3xf32>
}

// -----

// CHECK: HloModule

// CHECK-LABEL: %region_0.4
// CHECK-NEXT:  %[[ARG_TUPLE:.*]] = (s32[], s32[]) parameter(0)
// CHECK-NEXT:  %[[GTE_0:.*]] = s32[] get-tuple-element((s32[], s32[]) %[[ARG_TUPLE]]), index=0
// CHECK-NEXT:  %[[GTE_1:.*]] = s32[] get-tuple-element((s32[], s32[]) %[[ARG_TUPLE]]), index=1
// CHECK-NEXT:  %[[TUPLE_0:.*]] = (s32[], s32[]) tuple(s32[] %[[GTE_0]], s32[] %[[GTE_1]])
// CHECK-NEXT:  %[[CC:.*]] = (s32[], s32[]) custom-call(s32[] %[[GTE_0]], (s32[], s32[]) %[[TUPLE_0]])
// CHECK-NEXT:  %[[GTE_2:.*]] = s32[] get-tuple-element((s32[], s32[]) %[[CC]]), index=0
// CHECK-NEXT:  %[[GTE_3:.*]] = s32[] get-tuple-element((s32[], s32[]) %[[CC]]), index=1
// CHECK-NEXT:  ROOT %[[TUPLE_1:.*]] = (s32[], s32[]) tuple(s32[] %[[GTE_2]], s32[] %[[GTE_3]])

// CHECK-LABEL: %region_1.13
// CHECK-NEXT:  %[[ARG_TUPLE:.*]] = (s32[], s32[]) parameter(0)
// CHECK-NEXT:  %[[GTE_0:.*]] = s32[] get-tuple-element((s32[], s32[]) %[[ARG_TUPLE]]), index=0
// CHECK-NEXT:  %[[GTE_1:.*]] = s32[] get-tuple-element((s32[], s32[]) %[[ARG_TUPLE]]), index=1
// CHECK-NEXT:  ROOT %compare.17 = pred[] compare(s32[] %[[GTE_0]], s32[] %[[GTE_1]]), direction=LT

// CHECK-LABEL: ENTRY
// CHECK-NEXT:  %[[CST_0:.*]] = s32[] constant(0)
// CHECK-NEXT:  %[[ARG_0:.*]] = s32[] parameter(0)
// CHECK-NEXT:  %[[TUPLE:.*]] = (s32[], s32[]) tuple(s32[] %[[CST_0]], s32[] %[[ARG_0]])
// CHECK-NEXT:  %[[WHILE:.*]] = (s32[], s32[]) while((s32[], s32[]) %[[TUPLE]]), condition=%[[COND:.*]], body=%[[BODY:.*]]
// CHECK-NEXT:  ROOT %[[GTE_0:.*]] = s32[] get-tuple-element((s32[], s32[]) %[[WHILE]]), index=0
// CHECK-NEXT:  %[[GTE_1:.*]] = s32[] get-tuple-element((s32[], s32[]) %[[WHILE]]), index=1

func @main(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1:2 = "mhlo.while"(%0, %arg0) ( {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):  // no predecessors
    %2 = "mhlo.compare"(%arg1, %arg2) {comparison_direction = "LT"} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%2) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):  // no predecessors
    %2 = "mhlo.tuple"(%arg1, %arg2) : (tensor<i32>, tensor<i32>) -> tuple<tensor<i32>, tensor<i32>>
    %3 = "mhlo.custom_call"(%arg1, %2) {api_version = 1 : i32, backend_config = "bar", call_target_name = "foo", has_side_effect = false, xla_shape = "(s32[], s32[])"} : (tensor<i32>, tuple<tensor<i32>, tensor<i32>>) -> tuple<tensor<i32>, tensor<i32>>
    %4 = "mhlo.get_tuple_element"(%3) {index = 0 : i32} : (tuple<tensor<i32>, tensor<i32>>) -> tensor<i32>
    %5 = "mhlo.get_tuple_element"(%3) {index = 1 : i32} : (tuple<tensor<i32>, tensor<i32>>) -> tensor<i32>
    "mhlo.return"(%4, %5) : (tensor<i32>, tensor<i32>) -> ()
  }) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  return %1#0 : tensor<i32>
}

