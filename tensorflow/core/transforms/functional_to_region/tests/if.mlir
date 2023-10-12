// RUN: tfg-transforms-opt --tfg-functional-to-region %s | FileCheck %s

// CHECK: tfg.func @then_function
tfg.func @then_function(%first: tensor<*xi32> {tfg.name = "first"},
                        %second: tensor<*xf32> {tfg.name = "second"})
     -> (tensor<*xf32> {tfg.dtype = f32, tfg.name = "first/Identity"},
         tensor<*xi32> {tfg.dtype = i32, tfg.name = "second/Identity"})
 attributes {tf.input_shapes = [#tf_type.shape<>, #tf_type.shape<>]} {
  %Swap:2, %ctl = Swap(%first, %second) [%first.ctl, %second.ctl]
                    name("then_function/Swap")
                    {output_shapes = [#tf_type.shape<>, #tf_type.shape<>]}
                    : (tensor<*xi32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xi32>)
  %ctl_0 = NoOp [%ctl] name("NoOp") {output_shapes = []}
  %Identity, %ctl_1 = Identity(%Swap#0) [%ctl_0] name("first/Identity")
                        {output_shapes = [#tf_type.shape<>]}
                        : (tensor<*xf32>) -> (tensor<*xf32>)
  %Identity_0, %ctl_2 = Identity(%Swap#1) [%ctl_0] name("second/Identity")
                          {output_shapes = [#tf_type.shape<>]}
                          : (tensor<*xi32>) -> (tensor<*xi32>)
  return(%Identity, %Identity_0) [%ctl_1, %ctl_2]
    {tfg.control_ret_name_1 = "first/Identity",
     tfg.control_ret_name_2 = "second/Identity"}
    : tensor<*xf32>, tensor<*xi32>
}

// CHECK: tfg.func @else_function
tfg.func @else_function(%first: tensor<*xi32> {tfg.name = "first"},
                        %second: tensor<*xf32> {tfg.name = "second"})
     -> (tensor<*xf32> {tfg.dtype = f32, tfg.name = "first/Identity"},
         tensor<*xi32> {tfg.dtype = i32, tfg.name = "second/Identity"})
 attributes {tf.input_shapes = [#tf_type.shape<>, #tf_type.shape<>]} {
  %Identity, %ctl = Identity(%second) [%first.ctl] name("first/Identity")
                      {output_shapes = [#tf_type.shape<>]}
                      : (tensor<*xf32>) -> (tensor<*xf32>)
  %Identity_0, %ctl_0 = Identity(%first) [%second.ctl] name("second/Identity")
                          {output_shapes = [#tf_type.shape<>]}
                          : (tensor<*xi32>) -> (tensor<*xi32>)
  return(%Identity, %Identity_0) [%ctl, %ctl_0]
    {tfg.control_ret_name_1 = "first/Identity",
     tfg.control_ret_name_2 = "second/Identity"}
    : tensor<*xf32>, tensor<*xi32>
}

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  // CHECK: %[[ARGS:.*]]:2, %[[CTL:.*]] = Args
  %Args:2, %ctl = Args : () -> (tensor<*xi32>, tensor<*xf32>)
  // CHECK: %[[COND:.*]], %[[CTL_0:.*]] = Cond
  %Cond, %ctl_0 = Cond : () -> (tensor<*xi1>)
  // CHECK:      %[[IF:.*]]:2, %[[CTLS:.*]] = IfRegion %[[COND]] [%[[CTL_0]]] then {
  // CHECK-NEXT:   %[[SWAP:.*]]:2, %[[CTL_1:.*]] = Swap(%[[ARGS]]#0, %[[ARGS]]#1) [%[[CTL]], %[[CTL]]]
  // CHECK-NEXT:   %[[CTL_2:.*]] = NoOp [%[[CTL_1]]]
  // CHECK-NEXT:   %[[IDENTITY:.*]], %[[CTL_3:.*]] = Identity(%[[SWAP]]#0) [%[[CTL_2]]]
  // CHECK-NEXT:   %[[IDENTITY_4:.*]], %[[CTL_5:.*]] = Identity(%[[SWAP]]#1) [%[[CTL_2]]]
  // CHECK-NEXT:   yield(%[[IDENTITY]], %[[IDENTITY_4]]) [%[[CTL_3]], %[[CTL_5]]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   %[[IDENTITY:.*]], %[[CTL_1:.*]] = Identity(%[[ARGS]]#1) [%[[CTL]]]
  // CHECK-NEXT:   %[[IDENTITY_2:.*]], %[[CTL_3:.*]] = Identity(%[[ARGS]]#0) [%[[CTL]]]
  // CHECK-NEXT:   yield(%[[IDENTITY]], %[[IDENTITY_2]]) [%[[CTL_1]], %[[CTL_3]]]
  // CHECK-NEXT: } {_some_attr = 42 : index, _some_other_attr = "foo",
  // CHECK-SAME:   } : (tensor<*xi1>) -> (tensor<*xf32>, tensor<*xi32>)
  %If:2, %ctl_1 = If(%Cond, %Args#0, %Args#1) [%ctl_0]
                    {Tcond = i1, Tin = [i32, f32], Tout = [f32, i32],
                     _some_attr = 42 : index,
                     output_shapes = [#tf_type.shape<>, #tf_type.shape<>],
                     then_branch = #tf_type.func<@then_function, {}>,
                     else_branch = #tf_type.func<@else_function, {}>,
                     _some_other_attr = "foo"}
                    : (tensor<*xi1>, tensor<*xi32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xi32>)
}
