// RUN: tf-opt -tf-decompose-reduce-dataset %s | FileCheck %s

// CHECK-LABEL: func @skip_noncompiled_reduce_dataset
func.func @skip_noncompiled_reduce_dataset(
      %arg0 : tensor<!tf_type.variant>,
      %arg1: tensor<i64>
    ) {
    // CHECK: tf.ReduceDataset
    %1 = "tf.ReduceDataset"(%arg0, %arg1) {
      Targuments = [],
      Tstate = [i64], device = "",
      f = @__reduce_func_0, f._tf_data_function = true,
      output_shapes = [#tf_type.shape<>],
      output_types = [i64], use_inter_op_parallelism = true } : (tensor<!tf_type.variant>, tensor<i64>) -> (tensor<i64>)
    func.return
}

func.func private @__reduce_func_0(%arg0: tensor<i64> {tf._user_specified_name = "args_0"}) -> (tensor<i64>) attributes {tf._tf_data_function = true, tf.signature.is_stateful} {
    %0 = "tf.JustPretend"() : () -> (tensor<i64>)
    func.return %0 : tensor<i64>
}

// CHECK-LABEL: func @single_state_single_dataset_type_no_arguments
// CHECK-SAME: %[[ARG_0:.*]]: tensor<!tf_type.variant>, %[[ARG_1:.*]]: tensor<i64>
func.func @single_state_single_dataset_type_no_arguments(
      %arg0: tensor<!tf_type.variant>,
      %arg1: tensor<i64>
    ) {
    // CHECK:      %[[ANON_ITER:[0-9]*]] = "tf.AnonymousIteratorV3"
    // CHECK-SAME: output_shapes = [#tf_type.shape<32>]
    // CHECK-SAME: output_types = [f32]
    // CHECK-NEXT: "tf.MakeIterator"(%[[ARG_0]], %[[ANON_ITER]])
    // CHECK-NEXT: %[[COND:.*]] = "tf.Const"
    // CHECK-NEXT: "tf.WhileRegion"(%[[COND]], %[[ARG_1]])
    // CHECK-NEXT:   ^bb0(%[[ARG_2:.*]]: tensor<i1>, %[[ARG_3:.*]]: tensor<i64>)
    // CHECK-NEXT:     "tf.Yield"(%[[ARG_2]])
    // CHECK:        ^bb0(%[[ARG_4:.*]]: tensor<i1>, %[[ARG_5:.*]]: tensor<i64>)
    // CHECK:          %[[GET_NEXT:[0-9]*]] = "tf.IteratorGetNextAsOptional"(%[[ANON_ITER]])
    // CHECK-NEXT:     %[[HAS_VALUE:[0-9]*]] = "tf.OptionalHasValue"(%[[GET_NEXT]])
    // CHECK-NEXT:     %[[IF:.*]] = "tf.IfRegion"(%[[HAS_VALUE]])
    // CHECK-NEXT:       %[[GET_VALUE:[0-9]*]] = "tf.OptionalGetValue"(%[[GET_NEXT]])
    // CHECK-NEXT:       %[[FUNC_CALL:[0-9]*]] = func.call @__reduce_func_1(%[[ARG_5]], %[[GET_VALUE]])
    // CHECK-SAME:       _xla_compile_device_type = "TPU"
    // CHECK-SAME:       device = "/job:localhost/replica:0/task:0/device:TPU:1"
    // CHECK:            "tf.Yield"(%[[FUNC_CALL]])
    // CHECK:            "tf.Yield"(%[[ARG_5]])
    // CHECK:          "tf.Yield"(%[[HAS_VALUE]], %[[IF]])
    %1 = "tf.ReduceDataset"(%arg0, %arg1) {
      Targuments = [],
      Tstate = [i64], device = "/job:localhost/replica:0/task:0/device:TPU:1",
      f = @__reduce_func_1, f._tf_data_function = true,
      output_shapes = [#tf_type.shape<>],
      output_types = [i64], use_inter_op_parallelism = true, _xla_compile_device_type="TPU"} : (tensor<!tf_type.variant>, tensor<i64>) -> (tensor<i64>)
    func.return
}

func.func private @__reduce_func_1(%arg0: tensor<i64> {tf._user_specified_name = "args_0"}, %arg1: tensor<32xf32> {tf._user_specified_name = "args_1"}) -> (tensor<i64>) attributes {tf._tf_data_function = true, tf.signature.is_stateful} {
    %0 = "tf.JustPretend"(%arg1) : (tensor<32xf32>) -> (tensor<i64>)
    func.return %0 : tensor<i64>
}

// CHECK-LABEL: func @single_state_single_dataset_type_multiple_arguments
// CHECK-SAME: %[[ARG_0:.*]]: tensor<!tf_type.variant>, %[[ARG_1:.*]]: tensor<i64>, %[[ARG_2:.*]]: tensor<!tf_type.resource<tensor<64xf32>>>, %[[ARG_3:.*]]: tensor<!tf_type.resource<tensor<128xf32>>>
func.func @single_state_single_dataset_type_multiple_arguments(
      %arg0: tensor<!tf_type.variant>,
      %arg1: tensor<i64>,
      %arg2: tensor<!tf_type.resource<tensor<64xf32>>>,
      %arg3: tensor<!tf_type.resource<tensor<128xf32>>>
    ) {
    // CHECK:      %[[ANON_ITER:[0-9]*]] = "tf.AnonymousIteratorV3"
    // CHECK-SAME: output_shapes = [#tf_type.shape<32>]
    // CHECK-SAME: output_types = [f32]
    // CHECK-NEXT: "tf.MakeIterator"(%[[ARG_0]], %[[ANON_ITER]])
    // CHECK-NEXT: %[[COND:.*]] = "tf.Const"
    // CHECK-NEXT: "tf.WhileRegion"(%[[COND]], %[[ARG_1]], %[[ARG_2]], %[[ARG_3]])
    // CHECK-NEXT:   ^bb0(%[[ARG_4:.*]]: tensor<i1>, %[[ARG_5:.*]]: tensor<i64>, %[[ARG_6:.*]]: tensor<!tf_type.resource<tensor<64xf32>>>, %[[ARG_7:.*]]: tensor<!tf_type.resource<tensor<128xf32>>>)
    // CHECK-NEXT:     "tf.Yield"(%[[ARG_4]])
    // CHECK:        ^bb0(%[[ARG_8:.*]]: tensor<i1>, %[[ARG_9:.*]]: tensor<i64>, %[[ARG_10:.*]]: tensor<!tf_type.resource<tensor<64xf32>>>, %[[ARG_11:.*]]: tensor<!tf_type.resource<tensor<128xf32>>>)
    // CHECK:          %[[GET_NEXT:[0-9]*]] = "tf.IteratorGetNextAsOptional"(%[[ANON_ITER]])
    // CHECK-NEXT:     %[[HAS_VALUE:[0-9]*]] = "tf.OptionalHasValue"(%[[GET_NEXT]])
    // CHECK-NEXT:     %[[IF:.*]] = "tf.IfRegion"(%[[HAS_VALUE]])
    // CHECK-NEXT:       %[[GET_VALUE:[0-9]*]] = "tf.OptionalGetValue"(%[[GET_NEXT]])
    // CHECK-NEXT:       %[[FUNC_CALL:[0-9]*]] = func.call @__reduce_func_2(%[[ARG_9]], %[[GET_VALUE]], %[[ARG_10]], %[[ARG_11]])
    // CHECK-SAME:       _xla_compile_device_type = "TPU"
    // CHECK:            "tf.Yield"(%[[FUNC_CALL]])
    // CHECK:            "tf.Yield"(%[[ARG_9]])
    // CHECK:          "tf.Yield"(%[[HAS_VALUE]], %[[IF]], %[[ARG_10]], %[[ARG_11]])
    %1 = "tf.ReduceDataset"(%arg0, %arg1, %arg2, %arg3) {
      Targuments = [!tf_type.resource, !tf_type.resource],
      Tstate = [i64], device = "",
      f = @__reduce_func_2, f._tf_data_function = true,
      output_shapes = [#tf_type.shape<>],
      output_types = [i64], use_inter_op_parallelism = true, _xla_compile_device_type="TPU"} : (tensor<!tf_type.variant>, tensor<i64>, tensor<!tf_type.resource<tensor<64xf32>>>, tensor<!tf_type.resource<tensor<128xf32>>>) -> (tensor<i64>)
    func.return
}

func.func private @__reduce_func_2(%arg0: tensor<i64> {tf._user_specified_name = "args_0"}, %arg1: tensor<32xf32> {tf._user_specified_name = "args_1"}, %arg2: tensor<!tf_type.resource<tensor<64xf32>>>, %arg3: tensor<!tf_type.resource<tensor<128xf32>>>)
  -> (tensor<i64>) attributes {tf._tf_data_function = true, tf.signature.is_stateful} {
    %0 = "tf.JustPretend"(%arg1, %arg2, %arg3) : (tensor<32xf32>, tensor<!tf_type.resource<tensor<64xf32>>>, tensor<!tf_type.resource<tensor<128xf32>>>) -> (tensor<i64>)
    func.return %0 : tensor<i64>
}

// CHECK-LABEL: func @multiple_state_multiple_dataset_type_multiple_arguments
// CHECK-SAME: %[[ARG_0:.*]]: tensor<!tf_type.variant>, %[[ARG_1:.*]]: tensor<i64>, %[[ARG_2:.*]]: tensor<i32>, %[[ARG_3:.*]]: tensor<!tf_type.resource<tensor<64xf32>>>, %[[ARG_4:.*]]: tensor<!tf_type.resource<tensor<128xf32>>>
func.func @multiple_state_multiple_dataset_type_multiple_arguments(
      %arg0: tensor<!tf_type.variant>,
      %arg1: tensor<i64>,
      %arg2: tensor<i32>,
      %arg3: tensor<!tf_type.resource<tensor<64xf32>>>,
      %arg4: tensor<!tf_type.resource<tensor<128xf32>>>
    ) -> (tensor<i64>, tensor<i32>) {
    // CHECK:      %[[ANON_ITER:[0-9]*]] = "tf.AnonymousIteratorV3"
    // CHECK-SAME: output_shapes = [#tf_type.shape<32>, #tf_type.shape<64>]
    // CHECK-SAME: output_types = [f32, f64]
    // CHECK-NEXT: "tf.MakeIterator"(%[[ARG_0]], %[[ANON_ITER]])
    // CHECK-NEXT: %[[COND:.*]] = "tf.Const"
    // CHECK-NEXT: %[[WHILE:[0-9]*]]:5 = "tf.WhileRegion"(%[[COND]], %[[ARG_1]], %[[ARG_2]], %[[ARG_3]], %[[ARG_4]])
    // CHECK-NEXT:   ^bb0(%[[ARG_5:.*]]: tensor<i1>, %[[ARG_6:.*]]: tensor<i64>, %[[ARG_7:.*]]: tensor<i32>, %[[ARG_8:.*]]: tensor<!tf_type.resource<tensor<64xf32>>>, %[[ARG_9:.*]]: tensor<!tf_type.resource<tensor<128xf32>>>)
    // CHECK-NEXT:     "tf.Yield"(%[[ARG_5]])
    // CHECK:        ^bb0(%[[ARG_10:.*]]: tensor<i1>, %[[ARG_11:.*]]: tensor<i64>, %[[ARG_12:.*]]: tensor<i32>, %[[ARG_13:.*]]: tensor<!tf_type.resource<tensor<64xf32>>>, %[[ARG_14:.*]]: tensor<!tf_type.resource<tensor<128xf32>>>)
    // CHECK:          %[[GET_NEXT:[0-9]*]] = "tf.IteratorGetNextAsOptional"(%[[ANON_ITER]])
    // CHECK-NEXT:     %[[HAS_VALUE:[0-9]*]] = "tf.OptionalHasValue"(%[[GET_NEXT]])
    // CHECK-NEXT:     %[[IF:.*]]:2 = "tf.IfRegion"(%[[HAS_VALUE]])
    // CHECK-NEXT:       %[[GET_VALUE:[0-9]*]]:2 = "tf.OptionalGetValue"(%[[GET_NEXT]])
    // CHECK-NEXT:       %[[FUNC_CALL:[0-9]*]]:2 = func.call @__reduce_func_3(%[[ARG_11]], %[[ARG_12]], %[[GET_VALUE]]#0, %[[GET_VALUE]]#1, %[[ARG_13]], %[[ARG_14]])
    // CHECK-SAME:       _xla_compile_device_type = "TPU"
    // CHECK:            "tf.Yield"(%[[FUNC_CALL]]#0, %[[FUNC_CALL]]#1)
    // CHECK:            "tf.Yield"(%[[ARG_11]], %[[ARG_12]])
    // CHECK:          "tf.Yield"(%[[HAS_VALUE]], %[[IF]]#0, %[[IF]]#1, %[[ARG_13]], %[[ARG_14]])
    // CHECK:     return %[[WHILE]]#1, %[[WHILE]]#2
    %1:2 = "tf.ReduceDataset"(%arg0, %arg1, %arg2, %arg3, %arg4) {
      Targuments = [!tf_type.resource, !tf_type.resource],
      Tstate = [i64, i32], device = "",
      f = @__reduce_func_3, f._tf_data_function = true,
      output_shapes = [#tf_type.shape<>, #tf_type.shape<>],
      output_types = [i64, i32], use_inter_op_parallelism = true, _xla_compile_device_type="TPU"} : (tensor<!tf_type.variant>, tensor<i64>, tensor<i32>, tensor<!tf_type.resource<tensor<64xf32>>>, tensor<!tf_type.resource<tensor<128xf32>>>) -> (tensor<i64>, tensor<i32>)
    func.return %1#0, %1#1: tensor<i64>, tensor<i32>
}

func.func private @__reduce_func_3(%arg0: tensor<i64> {tf._user_specified_name = "args_0"}, %arg1: tensor<i32> {tf._user_specified_name = "args_1"},
  %arg2: tensor<32xf32> {tf._user_specified_name = "args_2"}, %arg3: tensor<64xf64> {tf._user_specified_name = "args_3"},
  %arg4: tensor<!tf_type.resource<tensor<64xf32>>>, %arg5: tensor<!tf_type.resource<tensor<128xf32>>>)
    -> (tensor<i64>, tensor<i32>) attributes {tf._tf_data_function = true, tf.signature.is_stateful} {
    %0:2 = "tf.JustPretend"(%arg2, %arg3, %arg4, %arg5) : (tensor<32xf32>, tensor<64xf64>, tensor<!tf_type.resource<tensor<64xf32>>>, tensor<!tf_type.resource<tensor<128xf32>>>) -> (tensor<i64>, tensor<i32>)
    func.return %0#0, %0#1: tensor<i64>, tensor<i32>
}
