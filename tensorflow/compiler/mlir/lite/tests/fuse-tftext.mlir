// RUN: tf-opt -tfl-prepare-composite-funcs-tf -tfl-fuse-tftext=true %s | FileCheck %s

func.func private @whitespace_tokenizer_rank1(%arg0: tensor<1x!tf_type.string> {tf._user_specified_name = "input"}) -> (tensor<?x!tf_type.string>, tensor<?xi64>) attributes {tf._input_shapes = [#tf_type.shape<1>], tf._implements = #tf_type.func<@"tftext:WhitespaceTokenizer", {}>, tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<[0, 1]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "tf.Const"() {value = dense<[]> : tensor<0xi64>} : () -> tensor<0xi64>
  %2 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
  %3 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %4 = "tf.Const"() {value = dense<[[0], [1]]> : tensor<2x1xi64>} : () -> tensor<2x1xi64>
  %5 = "tf.Const"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
  %6 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %7 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %8 = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
  %9 = "tf.Const"() {value = dense<[]> : tensor<0xi32>} : () -> tensor<0xi32>
  %10 = "tf.Const"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  %11 = "tf.Const"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %12 = "tf.Const"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
  %13 = "tf.Const"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %14 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %15 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %16 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %17 = "tf.If"(%2, %2, %13, %13) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_RaggedConcat_assert_equal_1_Assert_AssertGuard_false_3210, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_RaggedConcat_assert_equal_1_Assert_AssertGuard_true_3200} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %18 = "tf.Identity"(%17) {device = ""} : (tensor<i1>) -> tensor<i1>
  %19 = "tf.StringLength"(%arg0) {device = "", unit = "BYTE"} : (tensor<1x!tf_type.string>) -> tensor<1xi32>
  %20 = "tf.ExpandDims"(%19, %7) {device = ""} : (tensor<1xi32>, tensor<i32>) -> tensor<1x1xi32>
  %21 = "tf.Cast"(%20) {Truncate = false, device = ""} : (tensor<1x1xi32>) -> tensor<1x1xi64>
  %22 = "tf.Reshape"(%21, %12) {device = ""} : (tensor<1x1xi64>, tensor<1xi64>) -> tensor<1xi64>
  %23 = "tf.Reshape"(%arg0, %5) {device = ""} : (tensor<1x!tf_type.string>, tensor<1xi32>) -> tensor<1x!tf_type.string>
  %24:3 = "tf.UnicodeDecodeWithOffsets"(%23) {Tsplits = i64, device = "", errors = "replace", input_encoding = "UTF-8", replace_control_characters = false, replacement_char = 65533 : i64} : (tensor<1x!tf_type.string>) -> (tensor<2xi64>, tensor<?xi32>, tensor<?xi64>)
  %25 = "tf.StridedSlice"(%24#0, %15, %5, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi64>
  %26 = "tf.AddV2"(%25, %13) {device = ""} : (tensor<1xi64>, tensor<i64>) -> tensor<1xi64>
  %27 = "tf.StridedSlice"(%24#0, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi64>
  %28 = "tf.Minimum"(%26, %27) {device = ""} : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
  %29:2 = "tf.RaggedRange"(%28, %27, %13) {T = i64, Tsplits = i64, device = ""} : (tensor<1xi64>, tensor<1xi64>, tensor<i64>) -> (tensor<2xi64>, tensor<?xi64>)
  %30 = "tf.StridedSlice"(%29#0, %5, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %31 = "tf.AddV2"(%30, %12) {device = ""} : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
  %32 = "tf.ConcatV2"(%29#0, %31, %14) {device = ""} : (tensor<2xi64>, tensor<1xi64>, tensor<i32>) -> tensor<3xi64>
  %33 = "tf.GatherV2"(%24#2, %29#1, %14) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %34 = "tf.ConcatV2"(%33, %22, %14) {device = ""} : (tensor<?xi64>, tensor<1xi64>, tensor<i32>) -> tensor<?xi64>
  %35:2 = "tf.RaggedGather"(%32, %34, %0) {OUTPUT_RAGGED_RANK = 1 : i64, PARAMS_RAGGED_RANK = 1 : i64, Tindices = i64, Tsplits = i64, Tvalues = i64, device = ""} : (tensor<3xi64>, tensor<?xi64>, tensor<2xi64>) -> (tensor<?xi64>, tensor<?xi64>)
  %36:5 = "tf.WhitespaceTokenizeWithOffsets"(%24#1, %24#0) {Tsplits = i64, device = ""} : (tensor<?xi32>, tensor<2xi64>) -> (tensor<?xi32>, tensor<?xi64>, tensor<?xi64>, tensor<?xi64>, tensor<?xi64>)
  %37 = "tf.StridedSlice"(%36#1, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %38 = "tf.Equal"(%37, %10) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %39 = "tf.All"(%38, %9) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %40 = "tf.If"(%39, %39, %37, %10) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_3970, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_3960} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %41 = "tf.Identity"(%40) {device = ""} : (tensor<i1>) -> tensor<i1>
  %42 = "tf.StridedSlice"(%36#1, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %43 = "tf.StridedSlice"(%36#1, %15, %5, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %44 = "tf.Sub"(%42, %43) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %45 = "tf.LessEqual"(%10, %44) {device = ""} : (tensor<i64>, tensor<?xi64>) -> tensor<?xi1>
  %46 = "tf.All"(%45, %15) {device = "", keep_dims = false} : (tensor<?xi1>, tensor<1xi32>) -> tensor<i1>
  %47 = "tf.If"(%46, %46, %44) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_4330, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_4320} : (tensor<i1>, tensor<i1>, tensor<?xi64>) -> tensor<i1>
  %48 = "tf.Identity"(%47) {device = ""} : (tensor<i1>) -> tensor<i1>
  %49 = "tf.Identity"(%36#1) {_class = ["loc:@WhitespaceTokenize/WhitespaceTokenizeWithOffsets"], device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %50 = "tf.StridedSlice"(%49, %5, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %51 = "tf.Shape"(%36#0) {device = ""} : (tensor<?xi32>) -> tensor<1xi64>
  %52 = "tf.StridedSlice"(%51, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %53 = "tf.Equal"(%50, %52) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %54 = "tf.All"(%53, %9) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %55 = "tf.If"(%54, %54, %50, %52) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_false_4670, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_true_4660} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %56 = "tf.Identity"(%55) {device = ""} : (tensor<i1>) -> tensor<i1>
  %57 = "tf.Identity"(%49) {_class = ["loc:@WhitespaceTokenize/WhitespaceTokenizeWithOffsets"], device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %58 = "tf.Shape"(%57) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %59 = "tf.StridedSlice"(%58, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %60 = "tf.Sub"(%59, %13) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %61 = "tf.StridedSlice"(%36#4, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %62 = "tf.Equal"(%61, %10) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %63 = "tf.All"(%62, %9) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %64 = "tf.If"(%63, %63, %61, %10) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_5040, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_5030} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %65 = "tf.Identity"(%64) {device = ""} : (tensor<i1>) -> tensor<i1>
  %66 = "tf.StridedSlice"(%36#4, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %67 = "tf.StridedSlice"(%36#4, %15, %5, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %68 = "tf.Sub"(%66, %67) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %69 = "tf.LessEqual"(%10, %68) {device = ""} : (tensor<i64>, tensor<?xi64>) -> tensor<?xi1>
  %70 = "tf.All"(%69, %15) {device = "", keep_dims = false} : (tensor<?xi1>, tensor<1xi32>) -> tensor<i1>
  %71 = "tf.If"(%70, %70, %68) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_5400, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_5390} : (tensor<i1>, tensor<i1>, tensor<?xi64>) -> tensor<i1>
  %72 = "tf.Identity"(%71) {device = ""} : (tensor<i1>) -> tensor<i1>
  %73 = "tf.Identity"(%36#4) {_class = ["loc:@WhitespaceTokenize/WhitespaceTokenizeWithOffsets"], device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %74 = "tf.StridedSlice"(%73, %5, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %75 = "tf.Equal"(%74, %60) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %76 = "tf.All"(%75, %9) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %77 = "tf.If"(%76, %76, %74, %60) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_false_5760, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_true_5750} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %78 = "tf.Identity"(%77) {device = ""} : (tensor<i1>) -> tensor<i1>
  %79 = "tf.Identity"(%73) {_class = ["loc:@WhitespaceTokenize/WhitespaceTokenizeWithOffsets"], device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %80 = "tf.StridedSlice"(%36#4, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %81 = "tf.Equal"(%80, %10) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %82 = "tf.All"(%81, %9) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %83 = "tf.If"(%82, %82, %80, %10) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_6110, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_6100} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %84 = "tf.Identity"(%83) {device = ""} : (tensor<i1>) -> tensor<i1>
  %85 = "tf.StridedSlice"(%36#4, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %86 = "tf.StridedSlice"(%36#4, %15, %5, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %87 = "tf.Sub"(%85, %86) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %88 = "tf.LessEqual"(%10, %87) {device = ""} : (tensor<i64>, tensor<?xi64>) -> tensor<?xi1>
  %89 = "tf.All"(%88, %15) {device = "", keep_dims = false} : (tensor<?xi1>, tensor<1xi32>) -> tensor<i1>
  %90 = "tf.If"(%89, %89, %87) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_6470, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_6460} : (tensor<i1>, tensor<i1>, tensor<?xi64>) -> tensor<i1>
  %91 = "tf.Identity"(%90) {device = ""} : (tensor<i1>) -> tensor<i1>
  %92 = "tf.Identity"(%36#4) {_class = ["loc:@WhitespaceTokenize/WhitespaceTokenizeWithOffsets"], device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %93 = "tf.StridedSlice"(%92, %5, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %94 = "tf.Shape"(%36#2) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %95 = "tf.StridedSlice"(%94, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %96 = "tf.Equal"(%93, %95) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %97 = "tf.All"(%96, %9) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %98 = "tf.If"(%97, %97, %93, %95) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_false_6810, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_true_6800} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %99 = "tf.Identity"(%98) {device = ""} : (tensor<i1>) -> tensor<i1>
  %100 = "tf.Identity"(%92) {_class = ["loc:@WhitespaceTokenize/WhitespaceTokenizeWithOffsets"], device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %101 = "tf.Shape"(%100) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %102 = "tf.StridedSlice"(%101, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %103 = "tf.Sub"(%102, %13) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %104 = "tf.Equal"(%103, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %105 = "tf.LogicalOr"(%104, %2) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %106 = "tf.Equal"(%103, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %107 = "tf.LogicalOr"(%105, %106) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %108 = "tf.StridedSlice"(%100, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %109 = "tf.StridedSlice"(%100, %15, %5, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %110 = "tf.Sub"(%108, %109) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %111 = "tf.Shape"(%100) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %112 = "tf.StridedSlice"(%111, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %113 = "tf.Sub"(%112, %13) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %114 = "tf.Equal"(%113, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %115 = "tf.ExpandDims"(%100, %7) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %116 = "tf.Shape"(%100) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %117 = "tf.StridedSlice"(%116, %15, %15, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %118 = "tf.StridedSlice"(%116, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %119 = "tf.StridedSlice"(%116, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %120 = "tf.StridedSlice"(%36#4, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %121 = "tf.Equal"(%120, %10) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %122 = "tf.All"(%121, %9) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %123 = "tf.If"(%122, %122, %120, %10) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_7180, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_7170} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %124 = "tf.Identity"(%123) {device = ""} : (tensor<i1>) -> tensor<i1>
  %125 = "tf.StridedSlice"(%36#4, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %126 = "tf.StridedSlice"(%36#4, %15, %5, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %127 = "tf.Sub"(%125, %126) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %128 = "tf.LessEqual"(%10, %127) {device = ""} : (tensor<i64>, tensor<?xi64>) -> tensor<?xi1>
  %129 = "tf.All"(%128, %15) {device = "", keep_dims = false} : (tensor<?xi1>, tensor<1xi32>) -> tensor<i1>
  %130 = "tf.If"(%129, %129, %127) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_7540, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_7530} : (tensor<i1>, tensor<i1>, tensor<?xi64>) -> tensor<i1>
  %131 = "tf.Identity"(%130) {device = ""} : (tensor<i1>) -> tensor<i1>
  %132 = "tf.Identity"(%36#4) {_class = ["loc:@WhitespaceTokenize/WhitespaceTokenizeWithOffsets"], device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %133 = "tf.StridedSlice"(%132, %5, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %134 = "tf.Shape"(%36#3) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %135 = "tf.StridedSlice"(%134, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %136 = "tf.Equal"(%133, %135) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %137 = "tf.All"(%136, %9) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %138 = "tf.If"(%137, %137, %133, %135) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_false_7880, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_true_7870} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %139 = "tf.Identity"(%138) {device = ""} : (tensor<i1>) -> tensor<i1>
  %140 = "tf.Identity"(%132) {_class = ["loc:@WhitespaceTokenize/WhitespaceTokenizeWithOffsets"], device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %141 = "tf.Shape"(%140) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %142 = "tf.StridedSlice"(%141, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %143 = "tf.Sub"(%142, %13) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %144 = "tf.Equal"(%143, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %145 = "tf.LogicalOr"(%144, %2) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %146 = "tf.Equal"(%143, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %147 = "tf.LogicalOr"(%145, %146) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %148 = "tf.StridedSlice"(%140, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %149 = "tf.StridedSlice"(%140, %15, %5, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %150 = "tf.Sub"(%148, %149) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %151 = "tf.Shape"(%140) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %152 = "tf.StridedSlice"(%151, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %153 = "tf.Sub"(%152, %13) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %154 = "tf.Equal"(%153, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %155 = "tf.ExpandDims"(%140, %7) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %156 = "tf.Shape"(%140) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %157 = "tf.StridedSlice"(%156, %15, %15, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %158 = "tf.StridedSlice"(%156, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %159 = "tf.StridedSlice"(%156, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %160 = "tf.StridedSlice"(%140, %5, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %161 = "tf.Range"(%10, %160, %13) {device = ""} : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<?xi64>
  %162 = "tf.StridedSlice"(%140, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %163 = "tf.StridedSlice"(%140, %15, %5, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %164 = "tf.Sub"(%162, %163) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %165 = "tf.If"(%107, %107, %13, %103) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_RaggedGather_Assert_AssertGuard_false_8680, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_RaggedGather_Assert_AssertGuard_true_8670} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %166 = "tf.Identity"(%165) {device = ""} : (tensor<i1>) -> tensor<i1>
  %167 = "tf.Equal"(%103, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %168 = "tf.Select"(%167, %13, %103) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %169 = "tf.Equal"(%168, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %170 = "tf.LogicalOr"(%169, %2) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %171 = "tf.Equal"(%168, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %172 = "tf.LogicalOr"(%170, %171) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %173 = "tf.Select"(%114, %168, %13) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %174 = "tf.Pack"(%173, %13) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %175 = "tf.StridedSlice"(%174, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %176 = "tf.Cast"(%175) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %177 = "tf.Reshape"(%176, %9) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %178 = "tf.Pack"(%7, %177) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %179 = "tf.Tile"(%115, %178) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %180 = "tf.Mul"(%177, %118) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %181 = "tf.Pack"(%180) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %182 = "tf.ConcatV2"(%117, %181, %119, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %183 = "tf.Reshape"(%179, %182) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %184 = "tf.Shape"(%183) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %185 = "tf.StridedSlice"(%184, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %186 = "tf.Pack"(%175) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %187 = "tf.StridedSlice"(%183, %186, %11, %12) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %188 = "tf.Sub"(%185, %175) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %189 = "tf.Pack"(%188) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %190 = "tf.StridedSlice"(%183, %11, %189, %12) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %191:2 = "tf.RaggedRange"(%190, %187, %13) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %192 = "tf.Select"(%2, %168, %13) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %193 = "tf.Pack"(%192, %13) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %194 = "tf.StridedSlice"(%193, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %195 = "tf.Cast"(%194) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %196 = "tf.Reshape"(%195, %9) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %197 = "tf.Pack"(%7, %196) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %198 = "tf.Tile"(%4, %197) {device = ""} : (tensor<2x1xi64>, tensor<2xi32>) -> tensor<2x?xi64>
  %199 = "tf.Mul"(%196, %8) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %200 = "tf.Pack"(%199) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %201 = "tf.ConcatV2"(%9, %200, %9, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %202 = "tf.Reshape"(%198, %201) {device = ""} : (tensor<2x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %203 = "tf.Shape"(%202) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %204 = "tf.StridedSlice"(%203, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %205 = "tf.Pack"(%194) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %206 = "tf.StridedSlice"(%202, %205, %11, %12) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %207 = "tf.Sub"(%204, %194) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %208 = "tf.Pack"(%207) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %209 = "tf.StridedSlice"(%202, %11, %208, %12) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %210:2 = "tf.RaggedRange"(%209, %206, %13) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %211 = "tf.StridedSlice"(%193, %15, %16, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi64>
  %212 = "tf.StridedSlice"(%193, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %213 = "tf.Mul"(%212, %12) {device = ""} : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
  %214 = "tf.Tile"(%213, %211) {device = ""} : (tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %215 = "tf.Cumsum"(%214, %14) {device = "", exclusive = false, reverse = false} : (tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %216 = "tf.ConcatV2"(%11, %215, %3) {device = ""} : (tensor<1xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %217 = "tf.StridedSlice"(%216, %15, %5, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %218 = "tf.ExpandDims"(%217, %7) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %219 = "tf.Shape"(%217) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %220 = "tf.StridedSlice"(%219, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %221 = "tf.Pack"(%220) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %222 = "tf.StridedSlice"(%216, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %223 = "tf.ExpandDims"(%222, %7) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %224 = "tf.Shape"(%222) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %225 = "tf.StridedSlice"(%224, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %226 = "tf.Pack"(%225) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %227 = "tf.Equal"(%103, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %228 = "tf.Select"(%227, %168, %13) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %229 = "tf.Cast"(%228) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %230 = "tf.Reshape"(%229, %9) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %231 = "tf.Pack"(%7, %230) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %232 = "tf.Mul"(%230, %8) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %233 = "tf.Pack"(%232) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %234 = "tf.ConcatV2"(%9, %233, %9, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %235 = "tf.Pack"(%228) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %236 = "tf.Pack"(%10, %103) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %237 = "tf.ExpandDims"(%236, %7) {device = ""} : (tensor<2xi64>, tensor<i32>) -> tensor<2x1xi64>
  %238 = "tf.Tile"(%237, %231) {device = ""} : (tensor<2x1xi64>, tensor<2xi32>) -> tensor<2x?xi64>
  %239 = "tf.Reshape"(%238, %234) {device = ""} : (tensor<2x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %240 = "tf.Shape"(%239) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %241 = "tf.StridedSlice"(%240, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %242 = "tf.Sub"(%241, %228) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %243 = "tf.Pack"(%242) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %244 = "tf.StridedSlice"(%239, %11, %243, %12) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %245 = "tf.StridedSlice"(%239, %235, %11, %12) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %246:2 = "tf.RaggedRange"(%244, %245, %13) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %247 = "tf.GatherV2"(%110, %246#1, %14) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %248 = "tf.Cast"(%247) {Truncate = false, device = ""} : (tensor<?xi64>) -> tensor<?xi32>
  %249 = "tf.BroadcastTo"(%248, %221) {device = ""} : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %250 = "tf.Max"(%249, %15) {device = "", keep_dims = false} : (tensor<?xi32>, tensor<1xi32>) -> tensor<i32>
  %251 = "tf.Maximum"(%14, %250) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %252 = "tf.Range"(%14, %251, %7) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  %253 = "tf.Pack"(%7, %251) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %254 = "tf.Tile"(%218, %253) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %255 = "tf.Shape"(%254) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %256 = "tf.StridedSlice"(%255, %15, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %257 = "tf.Prod"(%256, %15) {device = "", keep_dims = false} : (tensor<2xi32>, tensor<1xi32>) -> tensor<i32>
  %258 = "tf.Pack"(%257) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %259 = "tf.Shape"(%254) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %260 = "tf.StridedSlice"(%259, %15, %15, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %261 = "tf.Shape"(%254) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %262 = "tf.StridedSlice"(%261, %6, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %263 = "tf.ConcatV2"(%260, %258, %262, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %264 = "tf.Reshape"(%254, %263) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %265 = "tf.ExpandDims"(%249, %3) {device = ""} : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
  %266 = "tf.Less"(%252, %265) {device = ""} : (tensor<?xi32>, tensor<?x1xi32>) -> tensor<?x?xi1>
  %267 = "tf.Reshape"(%266, %5) {device = ""} : (tensor<?x?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %268 = "tf.Where"(%267) {device = ""} : (tensor<?xi1>) -> tensor<?x1xi64>
  %269 = "tf.Squeeze"(%268) {device = "", squeeze_dims = [1]} : (tensor<?x1xi64>) -> tensor<?xi64>
  %270 = "tf.GatherV2"(%264, %269, %14) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %271 = "tf.Cast"(%247) {Truncate = false, device = ""} : (tensor<?xi64>) -> tensor<?xi32>
  %272 = "tf.BroadcastTo"(%271, %226) {device = ""} : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %273 = "tf.Max"(%272, %15) {device = "", keep_dims = false} : (tensor<?xi32>, tensor<1xi32>) -> tensor<i32>
  %274 = "tf.Maximum"(%14, %273) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %275 = "tf.Range"(%14, %274, %7) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  %276 = "tf.Pack"(%7, %274) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %277 = "tf.Tile"(%223, %276) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %278 = "tf.Shape"(%277) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %279 = "tf.StridedSlice"(%278, %15, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %280 = "tf.Prod"(%279, %15) {device = "", keep_dims = false} : (tensor<2xi32>, tensor<1xi32>) -> tensor<i32>
  %281 = "tf.Pack"(%280) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %282 = "tf.Shape"(%277) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %283 = "tf.StridedSlice"(%282, %15, %15, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %284 = "tf.Shape"(%277) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %285 = "tf.StridedSlice"(%284, %6, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %286 = "tf.ConcatV2"(%283, %281, %285, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %287 = "tf.Reshape"(%277, %286) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %288 = "tf.ExpandDims"(%272, %3) {device = ""} : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
  %289 = "tf.Less"(%275, %288) {device = ""} : (tensor<?xi32>, tensor<?x1xi32>) -> tensor<?x?xi1>
  %290 = "tf.Reshape"(%289, %5) {device = ""} : (tensor<?x?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %291 = "tf.Where"(%290) {device = ""} : (tensor<?xi1>) -> tensor<?x1xi64>
  %292 = "tf.Squeeze"(%291) {device = "", squeeze_dims = [1]} : (tensor<?x1xi64>) -> tensor<?xi64>
  %293 = "tf.GatherV2"(%287, %292, %14) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %294:2 = "tf.RaggedRange"(%270, %293, %13) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %295 = "tf.If"(%172, %172, %168, %13) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_RaggedGather_Assert_1_AssertGuard_false_9750, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_RaggedGather_Assert_1_AssertGuard_true_9740} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %296 = "tf.Identity"(%295) {device = ""} : (tensor<i1>) -> tensor<i1>
  %297 = "tf.Select"(%2, %168, %13) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %298 = "tf.Pack"(%297) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %299 = "tf.ConcatV2"(%1, %298, %12, %14) {device = ""} : (tensor<0xi64>, tensor<1xi64>, tensor<1xi64>, tensor<i32>) -> tensor<2xi64>
  %300 = "tf.StridedSlice"(%299, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %301 = "tf.Equal"(%300, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %302 = "tf.StridedSlice"(%299, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %303 = "tf.StridedSlice"(%299, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %304 = "tf.Equal"(%303, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %305 = "tf.If"(%304, %304, %303, %247) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_RaggedGather_Assert_2_AssertGuard_false_10240, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_RaggedGather_Assert_2_AssertGuard_true_10230} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<?xi64>) -> tensor<i1>
  %306 = "tf.Identity"(%305) {device = ""} : (tensor<i1>) -> tensor<i1>
  %307 = "tf.If"(%301, %301, %247, %302) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_RaggedGather_Assert_3_AssertGuard_false_10600, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_RaggedGather_Assert_3_AssertGuard_true_10590} : (tensor<i1>, tensor<i1>, tensor<?xi64>, tensor<i64>) -> tensor<i1>
  %308 = "tf.If"(%147, %147, %13, %143) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_Assert_AssertGuard_false_15300, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_Assert_AssertGuard_true_15290} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %309 = "tf.Identity"(%308) {device = ""} : (tensor<i1>) -> tensor<i1>
  %310 = "tf.Equal"(%143, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %311 = "tf.Select"(%310, %13, %143) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %312 = "tf.Equal"(%311, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %313 = "tf.LogicalOr"(%312, %2) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %314 = "tf.Equal"(%311, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %315 = "tf.LogicalOr"(%313, %314) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %316 = "tf.Select"(%154, %311, %13) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %317 = "tf.Pack"(%316, %13) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %318 = "tf.StridedSlice"(%317, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %319 = "tf.Cast"(%318) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %320 = "tf.Reshape"(%319, %9) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %321 = "tf.Pack"(%7, %320) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %322 = "tf.Tile"(%155, %321) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %323 = "tf.Mul"(%320, %158) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %324 = "tf.Pack"(%323) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %325 = "tf.ConcatV2"(%157, %324, %159, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %326 = "tf.Reshape"(%322, %325) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %327 = "tf.Shape"(%326) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %328 = "tf.StridedSlice"(%327, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %329 = "tf.Pack"(%318) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %330 = "tf.StridedSlice"(%326, %329, %11, %12) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %331 = "tf.Sub"(%328, %318) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %332 = "tf.Pack"(%331) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %333 = "tf.StridedSlice"(%326, %11, %332, %12) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %334:2 = "tf.RaggedRange"(%333, %330, %13) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %335 = "tf.GatherV2"(%161, %334#1, %14) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %336 = "tf.StridedSlice"(%317, %15, %16, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi64>
  %337 = "tf.StridedSlice"(%317, %15, %16, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi64>
  %338 = "tf.StridedSlice"(%317, %6, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi64>
  %339 = "tf.ConcatV2"(%337, %338, %14) {device = ""} : (tensor<1xi64>, tensor<0xi64>, tensor<i32>) -> tensor<1xi64>
  %340 = "tf.StridedSlice"(%317, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %341 = "tf.Mul"(%164, %340) {device = ""} : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
  %342 = "tf.Tile"(%341, %336) {device = ""} : (tensor<?xi64>, tensor<1xi64>) -> tensor<?xi64>
  %343 = "tf.Cumsum"(%342, %14) {device = "", exclusive = false, reverse = false} : (tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %344 = "tf.ConcatV2"(%11, %343, %3) {device = ""} : (tensor<1xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %345 = "tf.Shape"(%344) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %346 = "tf.StridedSlice"(%345, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %347 = "tf.Sub"(%346, %13) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %348 = "tf.Equal"(%347, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %349 = "tf.LogicalOr"(%348, %2) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %350 = "tf.Equal"(%347, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %351 = "tf.LogicalOr"(%349, %350) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %352 = "tf.StridedSlice"(%344, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %353 = "tf.StridedSlice"(%344, %15, %5, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %354 = "tf.Sub"(%352, %353) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %355 = "tf.Shape"(%344) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %356 = "tf.StridedSlice"(%355, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %357 = "tf.Sub"(%356, %13) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %358 = "tf.Equal"(%357, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %359 = "tf.ExpandDims"(%344, %7) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %360 = "tf.Shape"(%344) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %361 = "tf.StridedSlice"(%360, %15, %15, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %362 = "tf.StridedSlice"(%360, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %363 = "tf.StridedSlice"(%360, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %364 = "tf.Select"(%2, %311, %13) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %365 = "tf.Pack"(%364, %13) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %366 = "tf.StridedSlice"(%365, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %367 = "tf.Cast"(%366) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %368 = "tf.Reshape"(%367, %9) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %369 = "tf.Pack"(%7, %368) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %370 = "tf.Tile"(%4, %369) {device = ""} : (tensor<2x1xi64>, tensor<2xi32>) -> tensor<2x?xi64>
  %371 = "tf.Mul"(%368, %8) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %372 = "tf.Pack"(%371) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %373 = "tf.ConcatV2"(%9, %372, %9, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %374 = "tf.Reshape"(%370, %373) {device = ""} : (tensor<2x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %375 = "tf.Shape"(%374) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %376 = "tf.StridedSlice"(%375, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %377 = "tf.Pack"(%366) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %378 = "tf.StridedSlice"(%374, %377, %11, %12) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %379 = "tf.Sub"(%376, %366) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %380 = "tf.Pack"(%379) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %381 = "tf.StridedSlice"(%374, %11, %380, %12) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %382:2 = "tf.RaggedRange"(%381, %378, %13) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %383 = "tf.GatherV2"(%11, %382#1, %14) {batch_dims = 0 : i64, device = ""} : (tensor<1xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %384 = "tf.GatherV2"(%12, %383, %14) {batch_dims = 0 : i64, device = ""} : (tensor<1xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %385 = "tf.StridedSlice"(%365, %15, %16, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi64>
  %386 = "tf.StridedSlice"(%365, %15, %16, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi64>
  %387 = "tf.StridedSlice"(%365, %6, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi64>
  %388 = "tf.ConcatV2"(%386, %387, %14) {device = ""} : (tensor<1xi64>, tensor<0xi64>, tensor<i32>) -> tensor<1xi64>
  %389 = "tf.Tile"(%384, %388) {device = ""} : (tensor<?xi64>, tensor<1xi64>) -> tensor<?xi64>
  %390 = "tf.StridedSlice"(%365, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %391 = "tf.Mul"(%390, %12) {device = ""} : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
  %392 = "tf.Tile"(%391, %385) {device = ""} : (tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %393 = "tf.Cumsum"(%392, %14) {device = "", exclusive = false, reverse = false} : (tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %394 = "tf.ConcatV2"(%11, %393, %3) {device = ""} : (tensor<1xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %395 = "tf.StridedSlice"(%394, %15, %5, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %396 = "tf.ExpandDims"(%395, %7) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %397 = "tf.Shape"(%395) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %398 = "tf.StridedSlice"(%397, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %399 = "tf.Pack"(%398) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %400 = "tf.StridedSlice"(%394, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %401 = "tf.ExpandDims"(%400, %7) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %402 = "tf.Shape"(%400) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %403 = "tf.StridedSlice"(%402, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %404 = "tf.Pack"(%403) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %405 = "tf.Equal"(%143, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %406 = "tf.Select"(%405, %311, %13) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %407 = "tf.Cast"(%406) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %408 = "tf.Reshape"(%407, %9) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %409 = "tf.Pack"(%7, %408) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %410 = "tf.Mul"(%408, %8) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %411 = "tf.Pack"(%410) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %412 = "tf.ConcatV2"(%9, %411, %9, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %413 = "tf.Pack"(%406) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %414 = "tf.Pack"(%10, %143) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %415 = "tf.ExpandDims"(%414, %7) {device = ""} : (tensor<2xi64>, tensor<i32>) -> tensor<2x1xi64>
  %416 = "tf.Tile"(%415, %409) {device = ""} : (tensor<2x1xi64>, tensor<2xi32>) -> tensor<2x?xi64>
  %417 = "tf.Reshape"(%416, %412) {device = ""} : (tensor<2x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %418 = "tf.Shape"(%417) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %419 = "tf.StridedSlice"(%418, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %420 = "tf.Sub"(%419, %406) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %421 = "tf.Pack"(%420) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %422 = "tf.StridedSlice"(%417, %11, %421, %12) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %423 = "tf.StridedSlice"(%417, %413, %11, %12) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %424:2 = "tf.RaggedRange"(%422, %423, %13) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %425 = "tf.GatherV2"(%150, %424#1, %14) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %426 = "tf.Cast"(%425) {Truncate = false, device = ""} : (tensor<?xi64>) -> tensor<?xi32>
  %427 = "tf.BroadcastTo"(%426, %399) {device = ""} : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %428 = "tf.Max"(%427, %15) {device = "", keep_dims = false} : (tensor<?xi32>, tensor<1xi32>) -> tensor<i32>
  %429 = "tf.Maximum"(%14, %428) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %430 = "tf.Range"(%14, %429, %7) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  %431 = "tf.Pack"(%7, %429) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %432 = "tf.Tile"(%396, %431) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %433 = "tf.Shape"(%432) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %434 = "tf.StridedSlice"(%433, %15, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %435 = "tf.Prod"(%434, %15) {device = "", keep_dims = false} : (tensor<2xi32>, tensor<1xi32>) -> tensor<i32>
  %436 = "tf.Pack"(%435) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %437 = "tf.Shape"(%432) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %438 = "tf.StridedSlice"(%437, %15, %15, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %439 = "tf.Shape"(%432) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %440 = "tf.StridedSlice"(%439, %6, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %441 = "tf.ConcatV2"(%438, %436, %440, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %442 = "tf.Reshape"(%432, %441) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %443 = "tf.ExpandDims"(%427, %3) {device = ""} : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
  %444 = "tf.Less"(%430, %443) {device = ""} : (tensor<?xi32>, tensor<?x1xi32>) -> tensor<?x?xi1>
  %445 = "tf.Reshape"(%444, %5) {device = ""} : (tensor<?x?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %446 = "tf.Where"(%445) {device = ""} : (tensor<?xi1>) -> tensor<?x1xi64>
  %447 = "tf.Squeeze"(%446) {device = "", squeeze_dims = [1]} : (tensor<?x1xi64>) -> tensor<?xi64>
  %448 = "tf.GatherV2"(%442, %447, %14) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %449 = "tf.Cast"(%425) {Truncate = false, device = ""} : (tensor<?xi64>) -> tensor<?xi32>
  %450 = "tf.BroadcastTo"(%449, %404) {device = ""} : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %451 = "tf.Max"(%450, %15) {device = "", keep_dims = false} : (tensor<?xi32>, tensor<1xi32>) -> tensor<i32>
  %452 = "tf.Maximum"(%14, %451) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %453 = "tf.Range"(%14, %452, %7) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  %454 = "tf.Pack"(%7, %452) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %455 = "tf.Tile"(%401, %454) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %456 = "tf.Shape"(%455) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %457 = "tf.StridedSlice"(%456, %15, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %458 = "tf.Prod"(%457, %15) {device = "", keep_dims = false} : (tensor<2xi32>, tensor<1xi32>) -> tensor<i32>
  %459 = "tf.Pack"(%458) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %460 = "tf.Shape"(%455) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %461 = "tf.StridedSlice"(%460, %15, %15, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %462 = "tf.Shape"(%455) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %463 = "tf.StridedSlice"(%462, %6, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %464 = "tf.ConcatV2"(%461, %459, %463, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %465 = "tf.Reshape"(%455, %464) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %466 = "tf.ExpandDims"(%450, %3) {device = ""} : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
  %467 = "tf.Less"(%453, %466) {device = ""} : (tensor<?xi32>, tensor<?x1xi32>) -> tensor<?x?xi1>
  %468 = "tf.Reshape"(%467, %5) {device = ""} : (tensor<?x?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %469 = "tf.Where"(%468) {device = ""} : (tensor<?xi1>) -> tensor<?x1xi64>
  %470 = "tf.Squeeze"(%469) {device = "", squeeze_dims = [1]} : (tensor<?x1xi64>) -> tensor<?xi64>
  %471 = "tf.GatherV2"(%465, %470, %14) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %472:2 = "tf.RaggedRange"(%448, %471, %13) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %473 = "tf.GatherV2"(%389, %472#1, %14) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %474 = "tf.If"(%315, %315, %311, %13) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_Assert_1_AssertGuard_false_16370, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_Assert_1_AssertGuard_true_16360} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %475 = "tf.Identity"(%474) {device = ""} : (tensor<i1>) -> tensor<i1>
  %476 = "tf.Select"(%2, %311, %13) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %477 = "tf.Pack"(%476) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %478 = "tf.ConcatV2"(%1, %477, %12, %14) {device = ""} : (tensor<0xi64>, tensor<1xi64>, tensor<1xi64>, tensor<i32>) -> tensor<2xi64>
  %479 = "tf.StridedSlice"(%478, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %480 = "tf.Equal"(%479, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %481 = "tf.StridedSlice"(%478, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %482 = "tf.StridedSlice"(%478, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %483 = "tf.Equal"(%482, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %484 = "tf.If"(%483, %483, %482, %425) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_Assert_2_AssertGuard_false_16860, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_Assert_2_AssertGuard_true_16850} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<?xi64>) -> tensor<i1>
  %485 = "tf.Identity"(%484) {device = ""} : (tensor<i1>) -> tensor<i1>
  %486 = "tf.If"(%480, %480, %425, %481) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_Assert_3_AssertGuard_false_17220, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_Assert_3_AssertGuard_true_17210} : (tensor<i1>, tensor<i1>, tensor<?xi64>, tensor<i64>) -> tensor<i1>
  %487 = "tf.Identity"(%486) {device = ""} : (tensor<i1>) -> tensor<i1>
  %488 = "tf.If"(%351, %351, %13, %347) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_RaggedGather_1_Assert_AssertGuard_false_21900, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_RaggedGather_1_Assert_AssertGuard_true_21890} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %489 = "tf.Identity"(%488) {device = ""} : (tensor<i1>) -> tensor<i1>
  %490 = "tf.Equal"(%347, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %491 = "tf.Select"(%490, %13, %347) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %492 = "tf.Equal"(%491, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %493 = "tf.LogicalOr"(%492, %2) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %494 = "tf.Equal"(%491, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %495 = "tf.LogicalOr"(%493, %494) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %496 = "tf.Select"(%358, %491, %13) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %497 = "tf.Pack"(%496, %13) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %498 = "tf.StridedSlice"(%497, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %499 = "tf.Cast"(%498) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %500 = "tf.Reshape"(%499, %9) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %501 = "tf.Pack"(%7, %500) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %502 = "tf.Tile"(%359, %501) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %503 = "tf.Mul"(%500, %362) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %504 = "tf.Pack"(%503) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %505 = "tf.ConcatV2"(%361, %504, %363, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %506 = "tf.Reshape"(%502, %505) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %507 = "tf.Shape"(%506) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %508 = "tf.StridedSlice"(%507, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %509 = "tf.Pack"(%498) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %510 = "tf.StridedSlice"(%506, %509, %11, %12) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %511 = "tf.Sub"(%508, %498) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %512 = "tf.Pack"(%511) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %513 = "tf.StridedSlice"(%506, %11, %512, %12) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %514:2 = "tf.RaggedRange"(%513, %510, %13) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %515 = "tf.Select"(%2, %491, %13) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %516 = "tf.Pack"(%515, %13) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %517 = "tf.StridedSlice"(%516, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %518 = "tf.Cast"(%517) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %519 = "tf.Reshape"(%518, %9) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %520 = "tf.Pack"(%7, %519) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %521 = "tf.Tile"(%4, %520) {device = ""} : (tensor<2x1xi64>, tensor<2xi32>) -> tensor<2x?xi64>
  %522 = "tf.Mul"(%519, %8) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %523 = "tf.Pack"(%522) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %524 = "tf.ConcatV2"(%9, %523, %9, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %525 = "tf.Reshape"(%521, %524) {device = ""} : (tensor<2x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %526 = "tf.Shape"(%525) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %527 = "tf.StridedSlice"(%526, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %528 = "tf.Pack"(%517) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %529 = "tf.StridedSlice"(%525, %528, %11, %12) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %530 = "tf.Sub"(%527, %517) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %531 = "tf.Pack"(%530) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %532 = "tf.StridedSlice"(%525, %11, %531, %12) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %533:2 = "tf.RaggedRange"(%532, %529, %13) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %534 = "tf.StridedSlice"(%516, %15, %16, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi64>
  %535 = "tf.StridedSlice"(%516, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %536 = "tf.Mul"(%535, %12) {device = ""} : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
  %537 = "tf.Tile"(%536, %534) {device = ""} : (tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %538 = "tf.Cumsum"(%537, %14) {device = "", exclusive = false, reverse = false} : (tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %539 = "tf.ConcatV2"(%11, %538, %3) {device = ""} : (tensor<1xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %540 = "tf.StridedSlice"(%539, %15, %5, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %541 = "tf.ExpandDims"(%540, %7) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %542 = "tf.Shape"(%540) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %543 = "tf.StridedSlice"(%542, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %544 = "tf.Pack"(%543) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %545 = "tf.StridedSlice"(%539, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %546 = "tf.ExpandDims"(%545, %7) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %547 = "tf.Shape"(%545) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %548 = "tf.StridedSlice"(%547, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %549 = "tf.Pack"(%548) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %550 = "tf.Equal"(%347, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %551 = "tf.Select"(%550, %491, %13) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %552 = "tf.Cast"(%551) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %553 = "tf.Reshape"(%552, %9) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %554 = "tf.Pack"(%7, %553) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %555 = "tf.Mul"(%553, %8) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %556 = "tf.Pack"(%555) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %557 = "tf.ConcatV2"(%9, %556, %9, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %558 = "tf.Pack"(%551) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %559 = "tf.Pack"(%10, %347) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %560 = "tf.ExpandDims"(%559, %7) {device = ""} : (tensor<2xi64>, tensor<i32>) -> tensor<2x1xi64>
  %561 = "tf.Tile"(%560, %554) {device = ""} : (tensor<2x1xi64>, tensor<2xi32>) -> tensor<2x?xi64>
  %562 = "tf.Reshape"(%561, %557) {device = ""} : (tensor<2x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %563 = "tf.Shape"(%562) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %564 = "tf.StridedSlice"(%563, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %565 = "tf.Sub"(%564, %551) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %566 = "tf.Pack"(%565) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %567 = "tf.StridedSlice"(%562, %11, %566, %12) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %568 = "tf.StridedSlice"(%562, %558, %11, %12) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %569:2 = "tf.RaggedRange"(%567, %568, %13) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %570 = "tf.GatherV2"(%354, %569#1, %14) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %571 = "tf.Cast"(%570) {Truncate = false, device = ""} : (tensor<?xi64>) -> tensor<?xi32>
  %572 = "tf.BroadcastTo"(%571, %544) {device = ""} : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %573 = "tf.Max"(%572, %15) {device = "", keep_dims = false} : (tensor<?xi32>, tensor<1xi32>) -> tensor<i32>
  %574 = "tf.Maximum"(%14, %573) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %575 = "tf.Range"(%14, %574, %7) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  %576 = "tf.Pack"(%7, %574) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %577 = "tf.Tile"(%541, %576) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %578 = "tf.Shape"(%577) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %579 = "tf.StridedSlice"(%578, %15, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %580 = "tf.Prod"(%579, %15) {device = "", keep_dims = false} : (tensor<2xi32>, tensor<1xi32>) -> tensor<i32>
  %581 = "tf.Pack"(%580) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %582 = "tf.Shape"(%577) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %583 = "tf.StridedSlice"(%582, %15, %15, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %584 = "tf.Shape"(%577) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %585 = "tf.StridedSlice"(%584, %6, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %586 = "tf.ConcatV2"(%583, %581, %585, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %587 = "tf.Reshape"(%577, %586) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %588 = "tf.ExpandDims"(%572, %3) {device = ""} : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
  %589 = "tf.Less"(%575, %588) {device = ""} : (tensor<?xi32>, tensor<?x1xi32>) -> tensor<?x?xi1>
  %590 = "tf.Reshape"(%589, %5) {device = ""} : (tensor<?x?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %591 = "tf.Where"(%590) {device = ""} : (tensor<?xi1>) -> tensor<?x1xi64>
  %592 = "tf.Squeeze"(%591) {device = "", squeeze_dims = [1]} : (tensor<?x1xi64>) -> tensor<?xi64>
  %593 = "tf.GatherV2"(%587, %592, %14) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %594 = "tf.Cast"(%570) {Truncate = false, device = ""} : (tensor<?xi64>) -> tensor<?xi32>
  %595 = "tf.BroadcastTo"(%594, %549) {device = ""} : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %596 = "tf.Max"(%595, %15) {device = "", keep_dims = false} : (tensor<?xi32>, tensor<1xi32>) -> tensor<i32>
  %597 = "tf.Maximum"(%14, %596) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %598 = "tf.Range"(%14, %597, %7) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  %599 = "tf.Pack"(%7, %597) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %600 = "tf.Tile"(%546, %599) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %601 = "tf.Shape"(%600) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %602 = "tf.StridedSlice"(%601, %15, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %603 = "tf.Prod"(%602, %15) {device = "", keep_dims = false} : (tensor<2xi32>, tensor<1xi32>) -> tensor<i32>
  %604 = "tf.Pack"(%603) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %605 = "tf.Shape"(%600) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %606 = "tf.StridedSlice"(%605, %15, %15, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %607 = "tf.Shape"(%600) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %608 = "tf.StridedSlice"(%607, %6, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %609 = "tf.ConcatV2"(%606, %604, %608, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %610 = "tf.Reshape"(%600, %609) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %611 = "tf.ExpandDims"(%595, %3) {device = ""} : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
  %612 = "tf.Less"(%598, %611) {device = ""} : (tensor<?xi32>, tensor<?x1xi32>) -> tensor<?x?xi1>
  %613 = "tf.Reshape"(%612, %5) {device = ""} : (tensor<?x?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %614 = "tf.Where"(%613) {device = ""} : (tensor<?xi1>) -> tensor<?x1xi64>
  %615 = "tf.Squeeze"(%614) {device = "", squeeze_dims = [1]} : (tensor<?x1xi64>) -> tensor<?xi64>
  %616 = "tf.GatherV2"(%610, %615, %14) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %617:2 = "tf.RaggedRange"(%593, %616, %13) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %618 = "tf.If"(%495, %495, %491, %13) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_RaggedGather_1_Assert_1_AssertGuard_false_22970, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_RaggedGather_1_Assert_1_AssertGuard_true_22960} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %619 = "tf.Identity"(%618) {device = ""} : (tensor<i1>) -> tensor<i1>
  %620 = "tf.Select"(%2, %491, %13) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %621 = "tf.Pack"(%620) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %622 = "tf.ConcatV2"(%1, %621, %12, %14) {device = ""} : (tensor<0xi64>, tensor<1xi64>, tensor<1xi64>, tensor<i32>) -> tensor<2xi64>
  %623 = "tf.StridedSlice"(%622, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %624 = "tf.Equal"(%623, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %625 = "tf.StridedSlice"(%622, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %626 = "tf.StridedSlice"(%622, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %627 = "tf.Equal"(%626, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %628 = "tf.If"(%627, %627, %626, %570) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_RaggedGather_1_Assert_2_AssertGuard_false_23460, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_RaggedGather_1_Assert_2_AssertGuard_true_23450} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<?xi64>) -> tensor<i1>
  %629 = "tf.Identity"(%628) {device = ""} : (tensor<i1>) -> tensor<i1>
  %630 = "tf.If"(%624, %624, %570, %625) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_RaggedGather_1_Assert_3_AssertGuard_false_23820, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_RaggedGather_1_Assert_3_AssertGuard_true_23810} : (tensor<i1>, tensor<i1>, tensor<?xi64>, tensor<i64>) -> tensor<i1>
  %631 = "tf.Identity"(%79) {device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %632 = "tf.Identity"(%630) {device = ""} : (tensor<i1>) -> tensor<i1>
  %633 = "tf.Identity"(%307) {device = ""} : (tensor<i1>) -> tensor<i1>
  %634 = "tf.Shape"(%36#2) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %635 = "tf.StridedSlice"(%634, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %636 = "tf.Cast"(%635) {Truncate = false, device = ""} : (tensor<0xi32>) -> tensor<0xi64>
  %637 = "tf.Identity"(%636) {device = ""} : (tensor<0xi64>) -> tensor<0xi64>
  %638 = "tf.Shape"(%36#3) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %639 = "tf.StridedSlice"(%638, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %640 = "tf.Cast"(%639) {Truncate = false, device = ""} : (tensor<0xi32>) -> tensor<0xi64>
  %641 = "tf.Identity"(%640) {device = ""} : (tensor<0xi64>) -> tensor<0xi64>
  %642 = "tf.GatherV2"(%36#3, %335, %14) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %643 = "tf.Tile"(%642, %339) {device = ""} : (tensor<?xi64>, tensor<1xi64>) -> tensor<?xi64>
  %644 = "tf.Sub"(%643, %473) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %645 = "tf.Shape"(%644) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %646 = "tf.StridedSlice"(%645, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %647 = "tf.Cast"(%646) {Truncate = false, device = ""} : (tensor<0xi32>) -> tensor<0xi64>
  %648 = "tf.Identity"(%647) {device = ""} : (tensor<0xi64>) -> tensor<0xi64>
  %649 = "tf.UnicodeEncode"(%36#0, %57) {Tsplits = i64, device = "", errors = "replace", output_encoding = "UTF-8", replacement_char = 65533 : i64} : (tensor<?xi32>, tensor<?xi64>) -> tensor<?x!tf_type.string>
  %650 = "tf.Identity"(%649) {device = ""} : (tensor<?x!tf_type.string>) -> tensor<?x!tf_type.string>
  func.return %650, %631 : tensor<?x!tf_type.string>, tensor<?xi64>
}
func.func @WhitespaceTokenize_RaggedConcat_assert_equal_1_Assert_AssertGuard_false_3210(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Input tensors have incompatible shapes."> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/RaggedConcat/RaggedFromTensor/Const:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/RaggedConcat/RaggedNRows/Const:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedConcat_assert_equal_1_Assert_AssertGuard_true_3200(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_3970(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:zero"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits/RowPartitionFromRowSplits/Const:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_3960(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_4330(%arg0: tensor<i1>, %arg1: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:monotonic"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x >= 0 did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits/RowPartitionFromRowSplits/sub:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<?xi64>) -> ()
  %3 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %4 = "tf.Identity"(%3) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %4 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_4320(%arg0: tensor<i1>, %arg1: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_false_4670(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to _from_row_partition do not form a valid RaggedTensor"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits/strided_slice_1:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_true_4660(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_5040(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:zero"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits_1/RowPartitionFromRowSplits/Const:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_5030(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_5400(%arg0: tensor<i1>, %arg1: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:monotonic"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x >= 0 did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits_1/RowPartitionFromRowSplits/sub:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<?xi64>) -> ()
  %3 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %4 = "tf.Identity"(%3) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %4 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_5390(%arg0: tensor<i1>, %arg1: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_false_5760(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to _from_row_partition do not form a valid RaggedTensor"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits_1/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits_1/RaggedNRows/sub:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_true_5750(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_6110(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:zero"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/RaggedFromNestedRowSplits_1/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/RaggedFromNestedRowSplits_1/RaggedFromRowSplits/RowPartitionFromRowSplits/Const:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_6100(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_6470(%arg0: tensor<i1>, %arg1: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:monotonic"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x >= 0 did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/RaggedFromNestedRowSplits_1/RaggedFromRowSplits/RowPartitionFromRowSplits/sub:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<?xi64>) -> ()
  %3 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %4 = "tf.Identity"(%3) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %4 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_6460(%arg0: tensor<i1>, %arg1: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_false_6810(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to _from_row_partition do not form a valid RaggedTensor"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/RaggedFromNestedRowSplits_1/RaggedFromRowSplits/strided_slice_1:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/RaggedFromNestedRowSplits_1/RaggedFromRowSplits/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_true_6800(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_7180(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:zero"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/RaggedFromNestedRowSplits_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/RaggedFromNestedRowSplits_2/RaggedFromRowSplits/RowPartitionFromRowSplits/Const:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_7170(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_7540(%arg0: tensor<i1>, %arg1: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:monotonic"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x >= 0 did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/RaggedFromNestedRowSplits_2/RaggedFromRowSplits/RowPartitionFromRowSplits/sub:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<?xi64>) -> ()
  %3 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %4 = "tf.Identity"(%3) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %4 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_7530(%arg0: tensor<i1>, %arg1: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_false_7880(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to _from_row_partition do not form a valid RaggedTensor"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/RaggedFromNestedRowSplits_2/RaggedFromRowSplits/strided_slice_1:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/RaggedFromNestedRowSplits_2/RaggedFromRowSplits/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_true_7870(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedGather_Assert_AssertGuard_false_8680(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedGather_Assert_AssertGuard_true_8670(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedGather_Assert_1_AssertGuard_false_9750(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedGather_Assert_1_AssertGuard_true_9740(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedGather_Assert_2_AssertGuard_false_10240(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<?>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<?xi64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedGather_Assert_2_AssertGuard_true_10230(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<?>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedGather_Assert_3_AssertGuard_false_10600(%arg0: tensor<i1>, %arg1: tensor<?xi64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<?xi64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedGather_Assert_3_AssertGuard_true_10590(%arg0: tensor<i1>, %arg1: tensor<?xi64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_Assert_AssertGuard_false_15300(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_Assert_AssertGuard_true_15290(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_Assert_1_AssertGuard_false_16370(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_Assert_1_AssertGuard_true_16360(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_Assert_2_AssertGuard_false_16860(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<?>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<?xi64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_Assert_2_AssertGuard_true_16850(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<?>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_Assert_3_AssertGuard_false_17220(%arg0: tensor<i1>, %arg1: tensor<?xi64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<?xi64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_Assert_3_AssertGuard_true_17210(%arg0: tensor<i1>, %arg1: tensor<?xi64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedGather_1_Assert_AssertGuard_false_21900(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedGather_1_Assert_AssertGuard_true_21890(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedGather_1_Assert_1_AssertGuard_false_22970(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedGather_1_Assert_1_AssertGuard_true_22960(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedGather_1_Assert_2_AssertGuard_false_23460(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<?>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<?xi64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedGather_1_Assert_2_AssertGuard_true_23450(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<?>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedGather_1_Assert_3_AssertGuard_false_23820(%arg0: tensor<i1>, %arg1: tensor<?xi64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<?xi64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_RaggedGather_1_Assert_3_AssertGuard_true_23810(%arg0: tensor<i1>, %arg1: tensor<?xi64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}

// CHECK:  func private @whitespace_tokenizer_rank1(%arg0: tensor<1x!tf_type.string> {tf._user_specified_name = "input"}) -> (tensor<?x!tf_type.string>, tensor<?xi64>) attributes {tf._implements = #tf_type.func<@"tftext:WhitespaceTokenizer", {}>, tf._input_shapes = [#tf_type.shape<1>], tf.signature.is_stateful} {
// CHECK:  %0:2 = "tfl.custom"(%arg0) {custom_code = "tftext:WhitespaceTokenizer", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>} : (tensor<1x!tf_type.string>) -> (tensor<?x!tf_type.string>, tensor<?xi64>)
// CHECK:  return %0#0, %0#1 : tensor<?x!tf_type.string>, tensor<?xi64>

func.func private @whitespace_tokenizer_rank2(%arg0: tensor<?x1x!tf_type.string> {tf._user_specified_name = "input"}) -> (tensor<?x!tf_type.string>, tensor<?xi64>, tensor<?xi64>) attributes {tf._input_shapes = [#tf_type.shape<?x1>], tf._implements = #tf_type.func<@"tftext:WhitespaceTokenizer", {}>, tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<[]> : tensor<0xi64>} : () -> tensor<0xi64>
  %1 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
  %2 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %3 = "tf.Const"() {value = dense<[[0], [1]]> : tensor<2x1xi64>} : () -> tensor<2x1xi64>
  %4 = "tf.Const"() {value = dense<[2, -1]> : tensor<2xi32>} : () -> tensor<2xi32>
  %5 = "tf.Const"() {value = dense<2> : tensor<i64>} : () -> tensor<i64>
  %6 = "tf.Const"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
  %7 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %8 = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  %9 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %10 = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
  %11 = "tf.Const"() {value = dense<[]> : tensor<0xi32>} : () -> tensor<0xi32>
  %12 = "tf.Const"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  %13 = "tf.Const"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %14 = "tf.Const"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
  %15 = "tf.Const"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %16 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %17 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %18 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %19 = "tf.Shape"(%arg0) {device = ""} : (tensor<?x1x!tf_type.string>) -> tensor<2xi64>
  %20 = "tf.StridedSlice"(%19, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %21 = "tf.StridedSlice"(%19, %18, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %22 = "tf.Mul"(%20, %21) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %23 = "tf.Pack"(%22) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %24 = "tf.StridedSlice"(%19, %7, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi64>
  %25 = "tf.ConcatV2"(%23, %24, %16) {device = ""} : (tensor<1xi64>, tensor<0xi64>, tensor<i32>) -> tensor<1xi64>
  %26 = "tf.Reshape"(%arg0, %25) {device = ""} : (tensor<?x1x!tf_type.string>, tensor<1xi64>) -> tensor<?x!tf_type.string>
  %27 = "tf.StringLength"(%26) {device = "", unit = "BYTE"} : (tensor<?x!tf_type.string>) -> tensor<?xi32>
  %28 = "tf.ExpandDims"(%27, %9) {device = ""} : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
  %29 = "tf.Cast"(%28) {Truncate = false, device = ""} : (tensor<?x1xi32>) -> tensor<?x1xi64>
  %30 = "tf.Shape"(%29) {device = ""} : (tensor<?x1xi64>) -> tensor<2xi64>
  %31 = "tf.StridedSlice"(%30, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %32 = "tf.StridedSlice"(%30, %18, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %33 = "tf.Mul"(%31, %32) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %34 = "tf.Pack"(%33) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %35 = "tf.StridedSlice"(%30, %7, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi64>
  %36 = "tf.ConcatV2"(%34, %35, %16) {device = ""} : (tensor<1xi64>, tensor<0xi64>, tensor<i32>) -> tensor<1xi64>
  %37 = "tf.Reshape"(%29, %36) {device = ""} : (tensor<?x1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %38 = "tf.StridedSlice"(%30, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %39 = "tf.AddV2"(%38, %15) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %40 = "tf.Range"(%12, %39, %15) {device = ""} : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<?xi64>
  %41 = "tf.Mul"(%40, %15) {device = ""} : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
  %42 = "tf.Reshape"(%26, %6) {device = ""} : (tensor<?x!tf_type.string>, tensor<1xi32>) -> tensor<?x!tf_type.string>
  %43:3 = "tf.UnicodeDecodeWithOffsets"(%42) {Tsplits = i64, device = "", errors = "replace", input_encoding = "UTF-8", replace_control_characters = false, replacement_char = 65533 : i64} : (tensor<?x!tf_type.string>) -> (tensor<?xi64>, tensor<?xi32>, tensor<?xi64>)
  %44 = "tf.StridedSlice"(%43#0, %17, %6, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %45 = "tf.Shape"(%44) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %46 = "tf.ConcatV2"(%45, %18, %16) {device = ""} : (tensor<1xi32>, tensor<1xi32>, tensor<i32>) -> tensor<2xi32>
  %47 = "tf.Reshape"(%44, %46) {device = ""} : (tensor<?xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %48 = "tf.Shape"(%47) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi64>
  %49 = "tf.StridedSlice"(%48, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %50 = "tf.AddV2"(%49, %15) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %51 = "tf.Range"(%12, %50, %15) {device = ""} : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<?xi64>
  %52 = "tf.Mul"(%51, %15) {device = ""} : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
  %53 = "tf.ExpandDims"(%52, %9) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %54 = "tf.Shape"(%52) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %55 = "tf.StridedSlice"(%54, %17, %17, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %56 = "tf.StridedSlice"(%54, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %57 = "tf.StridedSlice"(%54, %18, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %58 = "tf.StridedSlice"(%52, %18, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %59 = "tf.StridedSlice"(%52, %17, %6, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %60 = "tf.Sub"(%58, %59) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %61 = "tf.Shape"(%47) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %62 = "tf.Cast"(%61) {Truncate = false, device = ""} : (tensor<2xi32>) -> tensor<2xi64>
  %63 = "tf.StridedSlice"(%62, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %64 = "tf.Equal"(%63, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %65 = "tf.StridedSlice"(%62, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %66 = "tf.Equal"(%65, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %67 = "tf.StridedSlice"(%62, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %68 = "tf.Shape"(%47) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %69 = "tf.Cast"(%68) {Truncate = false, device = ""} : (tensor<2xi32>) -> tensor<2xi64>
  %70 = "tf.StridedSlice"(%69, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %71 = "tf.Equal"(%70, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %72 = "tf.StridedSlice"(%43#0, %17, %6, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %73 = "tf.AddV2"(%72, %15) {device = ""} : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
  %74 = "tf.StridedSlice"(%43#0, %18, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %75 = "tf.Minimum"(%73, %74) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %76:2 = "tf.RaggedRange"(%75, %74, %15) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %77 = "tf.Shape"(%76#0) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %78 = "tf.StridedSlice"(%77, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %79 = "tf.Sub"(%78, %15) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %80 = "tf.Equal"(%38, %79) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %81 = "tf.All"(%80, %11) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %82 = "tf.If"(%81, %81, %38, %79) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedConcat_assert_equal_1_Assert_AssertGuard_false_99640, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedConcat_assert_equal_1_Assert_AssertGuard_true_99630} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %83 = "tf.Identity"(%82) {device = ""} : (tensor<i1>) -> tensor<i1>
  %84 = "tf.StridedSlice"(%41, %18, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %85 = "tf.Mul"(%79, %5) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %86 = "tf.Range"(%12, %85, %15) {device = ""} : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<?xi64>
  %87 = "tf.Reshape"(%86, %4) {device = ""} : (tensor<?xi64>, tensor<2xi32>) -> tensor<2x?xi64>
  %88 = "tf.Transpose"(%87, %8) {device = ""} : (tensor<2x?xi64>, tensor<2xi32>) -> tensor<?x2xi64>
  %89 = "tf.Reshape"(%88, %6) {device = ""} : (tensor<?x2xi64>, tensor<1xi32>) -> tensor<?xi64>
  %90 = "tf.StridedSlice"(%76#0, %6, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %91 = "tf.AddV2"(%84, %90) {device = ""} : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
  %92 = "tf.ConcatV2"(%76#0, %91, %16) {device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %93 = "tf.GatherV2"(%43#2, %76#1, %16) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %94 = "tf.ConcatV2"(%93, %37, %16) {device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %95:2 = "tf.RaggedGather"(%92, %94, %89) {OUTPUT_RAGGED_RANK = 1 : i64, PARAMS_RAGGED_RANK = 1 : i64, Tindices = i64, Tsplits = i64, Tvalues = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<?xi64>) -> (tensor<?xi64>, tensor<?xi64>)
  %96 = "tf.StridedSlice"(%95#0, %17, %17, %7) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %97 = "tf.StridedSlice"(%96, %17, %6, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %98 = "tf.Shape"(%97) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %99 = "tf.ConcatV2"(%98, %18, %16) {device = ""} : (tensor<1xi32>, tensor<1xi32>, tensor<i32>) -> tensor<2xi32>
  %100 = "tf.Reshape"(%97, %99) {device = ""} : (tensor<?xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %101 = "tf.Shape"(%100) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi64>
  %102 = "tf.StridedSlice"(%101, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %103 = "tf.AddV2"(%102, %15) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %104 = "tf.Range"(%12, %103, %15) {device = ""} : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<?xi64>
  %105 = "tf.Mul"(%104, %15) {device = ""} : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
  %106 = "tf.ExpandDims"(%105, %9) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %107 = "tf.Shape"(%105) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %108 = "tf.StridedSlice"(%107, %17, %17, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %109 = "tf.StridedSlice"(%107, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %110 = "tf.StridedSlice"(%107, %18, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %111 = "tf.StridedSlice"(%105, %18, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %112 = "tf.StridedSlice"(%105, %17, %6, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %113 = "tf.Sub"(%111, %112) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %114 = "tf.Shape"(%100) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %115 = "tf.Cast"(%114) {Truncate = false, device = ""} : (tensor<2xi32>) -> tensor<2xi64>
  %116 = "tf.StridedSlice"(%115, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %117 = "tf.Equal"(%116, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %118 = "tf.StridedSlice"(%115, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %119 = "tf.Equal"(%118, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %120 = "tf.StridedSlice"(%115, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %121 = "tf.Shape"(%100) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %122 = "tf.Cast"(%121) {Truncate = false, device = ""} : (tensor<2xi32>) -> tensor<2xi64>
  %123 = "tf.StridedSlice"(%122, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %124 = "tf.Equal"(%123, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %125:5 = "tf.WhitespaceTokenizeWithOffsets"(%43#1, %43#0) {Tsplits = i64, device = ""} : (tensor<?xi32>, tensor<?xi64>) -> (tensor<?xi32>, tensor<?xi64>, tensor<?xi64>, tensor<?xi64>, tensor<?xi64>)
  %126 = "tf.StridedSlice"(%125#1, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %127 = "tf.Equal"(%126, %12) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %128 = "tf.All"(%127, %11) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %129 = "tf.If"(%128, %128, %126, %12) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_100400, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_100390} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %130 = "tf.Identity"(%129) {device = ""} : (tensor<i1>) -> tensor<i1>
  %131 = "tf.StridedSlice"(%125#1, %18, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %132 = "tf.StridedSlice"(%125#1, %17, %6, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %133 = "tf.Sub"(%131, %132) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %134 = "tf.LessEqual"(%12, %133) {device = ""} : (tensor<i64>, tensor<?xi64>) -> tensor<?xi1>
  %135 = "tf.All"(%134, %17) {device = "", keep_dims = false} : (tensor<?xi1>, tensor<1xi32>) -> tensor<i1>
  %136 = "tf.If"(%135, %135, %133) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_100760, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_100750} : (tensor<i1>, tensor<i1>, tensor<?xi64>) -> tensor<i1>
  %137 = "tf.Identity"(%136) {device = ""} : (tensor<i1>) -> tensor<i1>
  %138 = "tf.Identity"(%125#1) {_class = ["loc:@WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenizeWithOffsets"], device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %139 = "tf.StridedSlice"(%138, %6, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %140 = "tf.Shape"(%125#0) {device = ""} : (tensor<?xi32>) -> tensor<1xi64>
  %141 = "tf.StridedSlice"(%140, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %142 = "tf.Equal"(%139, %141) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %143 = "tf.All"(%142, %11) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %144 = "tf.If"(%143, %143, %139, %141) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_false_101100, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_true_101090} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %145 = "tf.Identity"(%144) {device = ""} : (tensor<i1>) -> tensor<i1>
  %146 = "tf.Identity"(%138) {_class = ["loc:@WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenizeWithOffsets"], device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %147 = "tf.Shape"(%146) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %148 = "tf.StridedSlice"(%147, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %149 = "tf.Sub"(%148, %15) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %150 = "tf.StridedSlice"(%125#4, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %151 = "tf.Equal"(%150, %12) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %152 = "tf.All"(%151, %11) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %153 = "tf.If"(%152, %152, %150, %12) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_101470, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_101460} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %154 = "tf.Identity"(%153) {device = ""} : (tensor<i1>) -> tensor<i1>
  %155 = "tf.StridedSlice"(%125#4, %18, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %156 = "tf.StridedSlice"(%125#4, %17, %6, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %157 = "tf.Sub"(%155, %156) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %158 = "tf.LessEqual"(%12, %157) {device = ""} : (tensor<i64>, tensor<?xi64>) -> tensor<?xi1>
  %159 = "tf.All"(%158, %17) {device = "", keep_dims = false} : (tensor<?xi1>, tensor<1xi32>) -> tensor<i1>
  %160 = "tf.If"(%159, %159, %157) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_101830, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_101820} : (tensor<i1>, tensor<i1>, tensor<?xi64>) -> tensor<i1>
  %161 = "tf.Identity"(%160) {device = ""} : (tensor<i1>) -> tensor<i1>
  %162 = "tf.Identity"(%125#4) {_class = ["loc:@WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenizeWithOffsets"], device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %163 = "tf.StridedSlice"(%162, %6, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %164 = "tf.Equal"(%163, %149) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %165 = "tf.All"(%164, %11) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %166 = "tf.If"(%165, %165, %163, %149) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_false_102190, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_true_102180} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %167 = "tf.Identity"(%166) {device = ""} : (tensor<i1>) -> tensor<i1>
  %168 = "tf.Identity"(%162) {_class = ["loc:@WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenizeWithOffsets"], device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %169 = "tf.StridedSlice"(%125#4, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %170 = "tf.Equal"(%169, %12) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %171 = "tf.All"(%170, %11) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %172 = "tf.If"(%171, %171, %169, %12) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_102540, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_102530} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %173 = "tf.Identity"(%172) {device = ""} : (tensor<i1>) -> tensor<i1>
  %174 = "tf.StridedSlice"(%125#4, %18, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %175 = "tf.StridedSlice"(%125#4, %17, %6, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %176 = "tf.Sub"(%174, %175) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %177 = "tf.LessEqual"(%12, %176) {device = ""} : (tensor<i64>, tensor<?xi64>) -> tensor<?xi1>
  %178 = "tf.All"(%177, %17) {device = "", keep_dims = false} : (tensor<?xi1>, tensor<1xi32>) -> tensor<i1>
  %179 = "tf.If"(%178, %178, %176) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_102900, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_102890} : (tensor<i1>, tensor<i1>, tensor<?xi64>) -> tensor<i1>
  %180 = "tf.Identity"(%179) {device = ""} : (tensor<i1>) -> tensor<i1>
  %181 = "tf.Identity"(%125#4) {_class = ["loc:@WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenizeWithOffsets"], device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %182 = "tf.StridedSlice"(%181, %6, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %183 = "tf.Shape"(%125#2) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %184 = "tf.StridedSlice"(%183, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %185 = "tf.Equal"(%182, %184) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %186 = "tf.All"(%185, %11) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %187 = "tf.If"(%186, %186, %182, %184) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_false_103240, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_true_103230} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %188 = "tf.Identity"(%187) {device = ""} : (tensor<i1>) -> tensor<i1>
  %189 = "tf.Identity"(%181) {_class = ["loc:@WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenizeWithOffsets"], device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %190 = "tf.Shape"(%189) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %191 = "tf.StridedSlice"(%190, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %192 = "tf.Sub"(%191, %15) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %193 = "tf.Equal"(%192, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %194 = "tf.LogicalOr"(%64, %193) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %195 = "tf.Equal"(%192, %63) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %196 = "tf.LogicalOr"(%194, %195) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %197 = "tf.StridedSlice"(%189, %18, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %198 = "tf.StridedSlice"(%189, %17, %6, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %199 = "tf.Sub"(%197, %198) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %200 = "tf.Shape"(%189) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %201 = "tf.StridedSlice"(%200, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %202 = "tf.Sub"(%201, %15) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %203 = "tf.Equal"(%202, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %204 = "tf.ExpandDims"(%189, %9) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %205 = "tf.Shape"(%189) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %206 = "tf.StridedSlice"(%205, %17, %17, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %207 = "tf.StridedSlice"(%205, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %208 = "tf.StridedSlice"(%205, %18, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %209 = "tf.StridedSlice"(%125#4, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %210 = "tf.Equal"(%209, %12) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %211 = "tf.All"(%210, %11) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %212 = "tf.If"(%211, %211, %209, %12) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_103610, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_103600} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %213 = "tf.Identity"(%212) {device = ""} : (tensor<i1>) -> tensor<i1>
  %214 = "tf.StridedSlice"(%125#4, %18, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %215 = "tf.StridedSlice"(%125#4, %17, %6, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %216 = "tf.Sub"(%214, %215) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %217 = "tf.LessEqual"(%12, %216) {device = ""} : (tensor<i64>, tensor<?xi64>) -> tensor<?xi1>
  %218 = "tf.All"(%217, %17) {device = "", keep_dims = false} : (tensor<?xi1>, tensor<1xi32>) -> tensor<i1>
  %219 = "tf.If"(%218, %218, %216) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_103970, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_103960} : (tensor<i1>, tensor<i1>, tensor<?xi64>) -> tensor<i1>
  %220 = "tf.Identity"(%219) {device = ""} : (tensor<i1>) -> tensor<i1>
  %221 = "tf.Identity"(%125#4) {_class = ["loc:@WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenizeWithOffsets"], device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %222 = "tf.StridedSlice"(%221, %6, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %223 = "tf.Shape"(%125#3) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %224 = "tf.StridedSlice"(%223, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %225 = "tf.Equal"(%222, %224) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %226 = "tf.All"(%225, %11) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %227 = "tf.If"(%226, %226, %222, %224) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_false_104310, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_true_104300} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %228 = "tf.Identity"(%227) {device = ""} : (tensor<i1>) -> tensor<i1>
  %229 = "tf.Identity"(%221) {_class = ["loc:@WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenizeWithOffsets"], device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %230 = "tf.Shape"(%229) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %231 = "tf.StridedSlice"(%230, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %232 = "tf.Sub"(%231, %15) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %233 = "tf.Equal"(%232, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %234 = "tf.LogicalOr"(%233, %1) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %235 = "tf.Equal"(%232, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %236 = "tf.LogicalOr"(%234, %235) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %237 = "tf.StridedSlice"(%229, %18, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %238 = "tf.StridedSlice"(%229, %17, %6, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %239 = "tf.Sub"(%237, %238) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %240 = "tf.Shape"(%229) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %241 = "tf.StridedSlice"(%240, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %242 = "tf.Sub"(%241, %15) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %243 = "tf.Equal"(%242, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %244 = "tf.ExpandDims"(%229, %9) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %245 = "tf.Shape"(%229) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %246 = "tf.StridedSlice"(%245, %17, %17, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %247 = "tf.StridedSlice"(%245, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %248 = "tf.StridedSlice"(%245, %18, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %249 = "tf.StridedSlice"(%229, %6, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %250 = "tf.Range"(%12, %249, %15) {device = ""} : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<?xi64>
  %251 = "tf.StridedSlice"(%229, %18, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %252 = "tf.StridedSlice"(%229, %17, %6, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %253 = "tf.Sub"(%251, %252) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %254 = "tf.If"(%196, %196, %63, %192) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_AssertGuard_false_105110, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_AssertGuard_true_105100} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %255 = "tf.Identity"(%254) {device = ""} : (tensor<i1>) -> tensor<i1>
  %256 = "tf.Equal"(%192, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %257 = "tf.Select"(%256, %63, %192) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %258 = "tf.Equal"(%257, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %259 = "tf.LogicalOr"(%258, %66) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %260 = "tf.Equal"(%65, %257) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %261 = "tf.LogicalOr"(%259, %260) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %262 = "tf.Select"(%203, %257, %15) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %263 = "tf.Pack"(%262, %15) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %264 = "tf.StridedSlice"(%263, %18, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %265 = "tf.Cast"(%264) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %266 = "tf.Reshape"(%265, %11) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %267 = "tf.Pack"(%9, %266) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %268 = "tf.Tile"(%204, %267) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %269 = "tf.Mul"(%266, %207) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %270 = "tf.Pack"(%269) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %271 = "tf.ConcatV2"(%206, %270, %208, %16) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %272 = "tf.Reshape"(%268, %271) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %273 = "tf.Shape"(%272) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %274 = "tf.StridedSlice"(%273, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %275 = "tf.Pack"(%264) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %276 = "tf.StridedSlice"(%272, %275, %13, %14) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %277 = "tf.Sub"(%274, %264) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %278 = "tf.Pack"(%277) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %279 = "tf.StridedSlice"(%272, %13, %278, %14) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %280:2 = "tf.RaggedRange"(%279, %276, %15) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %281 = "tf.Select"(%71, %257, %15) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %282 = "tf.Pack"(%281, %15) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %283 = "tf.StridedSlice"(%282, %18, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %284 = "tf.Cast"(%283) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %285 = "tf.Reshape"(%284, %11) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %286 = "tf.Pack"(%9, %285) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %287 = "tf.Tile"(%53, %286) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %288 = "tf.Mul"(%285, %56) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %289 = "tf.Pack"(%288) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %290 = "tf.ConcatV2"(%55, %289, %57, %16) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %291 = "tf.Reshape"(%287, %290) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %292 = "tf.Shape"(%291) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %293 = "tf.StridedSlice"(%292, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %294 = "tf.Pack"(%283) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %295 = "tf.StridedSlice"(%291, %294, %13, %14) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %296 = "tf.Sub"(%293, %283) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %297 = "tf.Pack"(%296) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %298 = "tf.StridedSlice"(%291, %13, %297, %14) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %299:2 = "tf.RaggedRange"(%298, %295, %15) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %300 = "tf.StridedSlice"(%282, %17, %18, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi64>
  %301 = "tf.StridedSlice"(%282, %18, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %302 = "tf.Mul"(%60, %301) {device = ""} : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
  %303 = "tf.Tile"(%302, %300) {device = ""} : (tensor<?xi64>, tensor<1xi64>) -> tensor<?xi64>
  %304 = "tf.Cumsum"(%303, %16) {device = "", exclusive = false, reverse = false} : (tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %305 = "tf.ConcatV2"(%13, %304, %2) {device = ""} : (tensor<1xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %306 = "tf.StridedSlice"(%305, %17, %6, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %307 = "tf.ExpandDims"(%306, %9) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %308 = "tf.Shape"(%306) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %309 = "tf.StridedSlice"(%308, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %310 = "tf.Pack"(%309) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %311 = "tf.StridedSlice"(%305, %18, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %312 = "tf.ExpandDims"(%311, %9) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %313 = "tf.Shape"(%311) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %314 = "tf.StridedSlice"(%313, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %315 = "tf.Pack"(%314) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %316 = "tf.Equal"(%192, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %317 = "tf.Select"(%316, %257, %15) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %318 = "tf.Cast"(%317) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %319 = "tf.Reshape"(%318, %11) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %320 = "tf.Pack"(%9, %319) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %321 = "tf.Mul"(%319, %10) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %322 = "tf.Pack"(%321) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %323 = "tf.ConcatV2"(%11, %322, %11, %16) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %324 = "tf.Pack"(%317) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %325 = "tf.Pack"(%12, %192) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %326 = "tf.ExpandDims"(%325, %9) {device = ""} : (tensor<2xi64>, tensor<i32>) -> tensor<2x1xi64>
  %327 = "tf.Tile"(%326, %320) {device = ""} : (tensor<2x1xi64>, tensor<2xi32>) -> tensor<2x?xi64>
  %328 = "tf.Reshape"(%327, %323) {device = ""} : (tensor<2x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %329 = "tf.Shape"(%328) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %330 = "tf.StridedSlice"(%329, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %331 = "tf.Sub"(%330, %317) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %332 = "tf.Pack"(%331) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %333 = "tf.StridedSlice"(%328, %13, %332, %14) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %334 = "tf.StridedSlice"(%328, %324, %13, %14) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %335:2 = "tf.RaggedRange"(%333, %334, %15) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %336 = "tf.GatherV2"(%199, %335#1, %16) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %337 = "tf.Cast"(%336) {Truncate = false, device = ""} : (tensor<?xi64>) -> tensor<?xi32>
  %338 = "tf.BroadcastTo"(%337, %310) {device = ""} : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %339 = "tf.Max"(%338, %17) {device = "", keep_dims = false} : (tensor<?xi32>, tensor<1xi32>) -> tensor<i32>
  %340 = "tf.Maximum"(%16, %339) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %341 = "tf.Range"(%16, %340, %9) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  %342 = "tf.Pack"(%9, %340) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %343 = "tf.Tile"(%307, %342) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %344 = "tf.Shape"(%343) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %345 = "tf.StridedSlice"(%344, %17, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %346 = "tf.Prod"(%345, %17) {device = "", keep_dims = false} : (tensor<2xi32>, tensor<1xi32>) -> tensor<i32>
  %347 = "tf.Pack"(%346) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %348 = "tf.Shape"(%343) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %349 = "tf.StridedSlice"(%348, %17, %17, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %350 = "tf.Shape"(%343) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %351 = "tf.StridedSlice"(%350, %7, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %352 = "tf.ConcatV2"(%349, %347, %351, %16) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %353 = "tf.Reshape"(%343, %352) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %354 = "tf.ExpandDims"(%338, %2) {device = ""} : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
  %355 = "tf.Less"(%341, %354) {device = ""} : (tensor<?xi32>, tensor<?x1xi32>) -> tensor<?x?xi1>
  %356 = "tf.Reshape"(%355, %6) {device = ""} : (tensor<?x?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %357 = "tf.Where"(%356) {device = ""} : (tensor<?xi1>) -> tensor<?x1xi64>
  %358 = "tf.Squeeze"(%357) {device = "", squeeze_dims = [1]} : (tensor<?x1xi64>) -> tensor<?xi64>
  %359 = "tf.GatherV2"(%353, %358, %16) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %360 = "tf.Cast"(%336) {Truncate = false, device = ""} : (tensor<?xi64>) -> tensor<?xi32>
  %361 = "tf.BroadcastTo"(%360, %315) {device = ""} : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %362 = "tf.Max"(%361, %17) {device = "", keep_dims = false} : (tensor<?xi32>, tensor<1xi32>) -> tensor<i32>
  %363 = "tf.Maximum"(%16, %362) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %364 = "tf.Range"(%16, %363, %9) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  %365 = "tf.Pack"(%9, %363) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %366 = "tf.Tile"(%312, %365) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %367 = "tf.Shape"(%366) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %368 = "tf.StridedSlice"(%367, %17, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %369 = "tf.Prod"(%368, %17) {device = "", keep_dims = false} : (tensor<2xi32>, tensor<1xi32>) -> tensor<i32>
  %370 = "tf.Pack"(%369) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %371 = "tf.Shape"(%366) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %372 = "tf.StridedSlice"(%371, %17, %17, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %373 = "tf.Shape"(%366) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %374 = "tf.StridedSlice"(%373, %7, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %375 = "tf.ConcatV2"(%372, %370, %374, %16) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %376 = "tf.Reshape"(%366, %375) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %377 = "tf.ExpandDims"(%361, %2) {device = ""} : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
  %378 = "tf.Less"(%364, %377) {device = ""} : (tensor<?xi32>, tensor<?x1xi32>) -> tensor<?x?xi1>
  %379 = "tf.Reshape"(%378, %6) {device = ""} : (tensor<?x?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %380 = "tf.Where"(%379) {device = ""} : (tensor<?xi1>) -> tensor<?x1xi64>
  %381 = "tf.Squeeze"(%380) {device = "", squeeze_dims = [1]} : (tensor<?x1xi64>) -> tensor<?xi64>
  %382 = "tf.GatherV2"(%376, %381, %16) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %383:2 = "tf.RaggedRange"(%359, %382, %15) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %384 = "tf.If"(%261, %261, %257, %67) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_1_AssertGuard_false_106180, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_1_AssertGuard_true_106170} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %385 = "tf.Identity"(%384) {device = ""} : (tensor<i1>) -> tensor<i1>
  %386 = "tf.StridedSlice"(%62, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %387 = "tf.Equal"(%386, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %388 = "tf.Select"(%387, %257, %386) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %389 = "tf.Pack"(%388) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %390 = "tf.StridedSlice"(%62, %17, %17, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi64>
  %391 = "tf.StridedSlice"(%62, %18, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi64>
  %392 = "tf.ConcatV2"(%390, %389, %391, %16) {device = ""} : (tensor<0xi64>, tensor<1xi64>, tensor<1xi64>, tensor<i32>) -> tensor<2xi64>
  %393 = "tf.StridedSlice"(%392, %18, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %394 = "tf.Equal"(%393, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %395 = "tf.StridedSlice"(%392, %18, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %396 = "tf.StridedSlice"(%392, %18, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %397 = "tf.Equal"(%396, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %398 = "tf.If"(%397, %397, %396, %336) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_2_AssertGuard_false_106670, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_2_AssertGuard_true_106660} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<?xi64>) -> tensor<i1>
  %399 = "tf.Identity"(%398) {device = ""} : (tensor<i1>) -> tensor<i1>
  %400 = "tf.If"(%394, %394, %336, %395) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_3_AssertGuard_false_107030, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_3_AssertGuard_true_107020} : (tensor<i1>, tensor<i1>, tensor<?xi64>, tensor<i64>) -> tensor<i1>
  %401 = "tf.If"(%236, %236, %15, %232) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_Assert_AssertGuard_false_111870, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_Assert_AssertGuard_true_111860} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %402 = "tf.Identity"(%401) {device = ""} : (tensor<i1>) -> tensor<i1>
  %403 = "tf.Equal"(%232, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %404 = "tf.Select"(%403, %15, %232) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %405 = "tf.Equal"(%404, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %406 = "tf.LogicalOr"(%405, %1) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %407 = "tf.Equal"(%404, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %408 = "tf.LogicalOr"(%406, %407) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %409 = "tf.Select"(%243, %404, %15) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %410 = "tf.Pack"(%409, %15) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %411 = "tf.StridedSlice"(%410, %18, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %412 = "tf.Cast"(%411) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %413 = "tf.Reshape"(%412, %11) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %414 = "tf.Pack"(%9, %413) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %415 = "tf.Tile"(%244, %414) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %416 = "tf.Mul"(%413, %247) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %417 = "tf.Pack"(%416) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %418 = "tf.ConcatV2"(%246, %417, %248, %16) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %419 = "tf.Reshape"(%415, %418) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %420 = "tf.Shape"(%419) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %421 = "tf.StridedSlice"(%420, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %422 = "tf.Pack"(%411) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %423 = "tf.StridedSlice"(%419, %422, %13, %14) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %424 = "tf.Sub"(%421, %411) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %425 = "tf.Pack"(%424) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %426 = "tf.StridedSlice"(%419, %13, %425, %14) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %427:2 = "tf.RaggedRange"(%426, %423, %15) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %428 = "tf.GatherV2"(%250, %427#1, %16) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %429 = "tf.StridedSlice"(%410, %17, %18, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi64>
  %430 = "tf.StridedSlice"(%410, %17, %18, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi64>
  %431 = "tf.StridedSlice"(%410, %7, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi64>
  %432 = "tf.ConcatV2"(%430, %431, %16) {device = ""} : (tensor<1xi64>, tensor<0xi64>, tensor<i32>) -> tensor<1xi64>
  %433 = "tf.StridedSlice"(%410, %18, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %434 = "tf.Mul"(%253, %433) {device = ""} : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
  %435 = "tf.Tile"(%434, %429) {device = ""} : (tensor<?xi64>, tensor<1xi64>) -> tensor<?xi64>
  %436 = "tf.Cumsum"(%435, %16) {device = "", exclusive = false, reverse = false} : (tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %437 = "tf.ConcatV2"(%13, %436, %2) {device = ""} : (tensor<1xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %438 = "tf.Shape"(%437) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %439 = "tf.StridedSlice"(%438, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %440 = "tf.Sub"(%439, %15) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %441 = "tf.Equal"(%440, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %442 = "tf.LogicalOr"(%117, %441) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %443 = "tf.Equal"(%440, %116) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %444 = "tf.LogicalOr"(%442, %443) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %445 = "tf.StridedSlice"(%437, %18, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %446 = "tf.StridedSlice"(%437, %17, %6, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %447 = "tf.Sub"(%445, %446) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %448 = "tf.Shape"(%437) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %449 = "tf.StridedSlice"(%448, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %450 = "tf.Sub"(%449, %15) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %451 = "tf.Equal"(%450, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %452 = "tf.ExpandDims"(%437, %9) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %453 = "tf.Shape"(%437) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %454 = "tf.StridedSlice"(%453, %17, %17, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %455 = "tf.StridedSlice"(%453, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %456 = "tf.StridedSlice"(%453, %18, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %457 = "tf.Select"(%1, %404, %15) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %458 = "tf.Pack"(%457, %15) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %459 = "tf.StridedSlice"(%458, %18, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %460 = "tf.Cast"(%459) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %461 = "tf.Reshape"(%460, %11) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %462 = "tf.Pack"(%9, %461) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %463 = "tf.Tile"(%3, %462) {device = ""} : (tensor<2x1xi64>, tensor<2xi32>) -> tensor<2x?xi64>
  %464 = "tf.Mul"(%461, %10) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %465 = "tf.Pack"(%464) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %466 = "tf.ConcatV2"(%11, %465, %11, %16) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %467 = "tf.Reshape"(%463, %466) {device = ""} : (tensor<2x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %468 = "tf.Shape"(%467) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %469 = "tf.StridedSlice"(%468, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %470 = "tf.Pack"(%459) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %471 = "tf.StridedSlice"(%467, %470, %13, %14) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %472 = "tf.Sub"(%469, %459) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %473 = "tf.Pack"(%472) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %474 = "tf.StridedSlice"(%467, %13, %473, %14) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %475:2 = "tf.RaggedRange"(%474, %471, %15) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %476 = "tf.GatherV2"(%13, %475#1, %16) {batch_dims = 0 : i64, device = ""} : (tensor<1xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %477 = "tf.GatherV2"(%14, %476, %16) {batch_dims = 0 : i64, device = ""} : (tensor<1xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %478 = "tf.StridedSlice"(%458, %17, %18, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi64>
  %479 = "tf.StridedSlice"(%458, %17, %18, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi64>
  %480 = "tf.StridedSlice"(%458, %7, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi64>
  %481 = "tf.ConcatV2"(%479, %480, %16) {device = ""} : (tensor<1xi64>, tensor<0xi64>, tensor<i32>) -> tensor<1xi64>
  %482 = "tf.Tile"(%477, %481) {device = ""} : (tensor<?xi64>, tensor<1xi64>) -> tensor<?xi64>
  %483 = "tf.StridedSlice"(%458, %18, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %484 = "tf.Mul"(%483, %14) {device = ""} : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
  %485 = "tf.Tile"(%484, %478) {device = ""} : (tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %486 = "tf.Cumsum"(%485, %16) {device = "", exclusive = false, reverse = false} : (tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %487 = "tf.ConcatV2"(%13, %486, %2) {device = ""} : (tensor<1xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %488 = "tf.StridedSlice"(%487, %17, %6, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %489 = "tf.ExpandDims"(%488, %9) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %490 = "tf.Shape"(%488) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %491 = "tf.StridedSlice"(%490, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %492 = "tf.Pack"(%491) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %493 = "tf.StridedSlice"(%487, %18, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %494 = "tf.ExpandDims"(%493, %9) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %495 = "tf.Shape"(%493) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %496 = "tf.StridedSlice"(%495, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %497 = "tf.Pack"(%496) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %498 = "tf.Equal"(%232, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %499 = "tf.Select"(%498, %404, %15) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %500 = "tf.Cast"(%499) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %501 = "tf.Reshape"(%500, %11) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %502 = "tf.Pack"(%9, %501) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %503 = "tf.Mul"(%501, %10) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %504 = "tf.Pack"(%503) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %505 = "tf.ConcatV2"(%11, %504, %11, %16) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %506 = "tf.Pack"(%499) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %507 = "tf.Pack"(%12, %232) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %508 = "tf.ExpandDims"(%507, %9) {device = ""} : (tensor<2xi64>, tensor<i32>) -> tensor<2x1xi64>
  %509 = "tf.Tile"(%508, %502) {device = ""} : (tensor<2x1xi64>, tensor<2xi32>) -> tensor<2x?xi64>
  %510 = "tf.Reshape"(%509, %505) {device = ""} : (tensor<2x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %511 = "tf.Shape"(%510) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %512 = "tf.StridedSlice"(%511, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %513 = "tf.Sub"(%512, %499) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %514 = "tf.Pack"(%513) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %515 = "tf.StridedSlice"(%510, %13, %514, %14) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %516 = "tf.StridedSlice"(%510, %506, %13, %14) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %517:2 = "tf.RaggedRange"(%515, %516, %15) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %518 = "tf.GatherV2"(%239, %517#1, %16) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %519 = "tf.Cast"(%518) {Truncate = false, device = ""} : (tensor<?xi64>) -> tensor<?xi32>
  %520 = "tf.BroadcastTo"(%519, %492) {device = ""} : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %521 = "tf.Max"(%520, %17) {device = "", keep_dims = false} : (tensor<?xi32>, tensor<1xi32>) -> tensor<i32>
  %522 = "tf.Maximum"(%16, %521) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %523 = "tf.Range"(%16, %522, %9) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  %524 = "tf.Pack"(%9, %522) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %525 = "tf.Tile"(%489, %524) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %526 = "tf.Shape"(%525) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %527 = "tf.StridedSlice"(%526, %17, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %528 = "tf.Prod"(%527, %17) {device = "", keep_dims = false} : (tensor<2xi32>, tensor<1xi32>) -> tensor<i32>
  %529 = "tf.Pack"(%528) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %530 = "tf.Shape"(%525) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %531 = "tf.StridedSlice"(%530, %17, %17, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %532 = "tf.Shape"(%525) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %533 = "tf.StridedSlice"(%532, %7, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %534 = "tf.ConcatV2"(%531, %529, %533, %16) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %535 = "tf.Reshape"(%525, %534) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %536 = "tf.ExpandDims"(%520, %2) {device = ""} : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
  %537 = "tf.Less"(%523, %536) {device = ""} : (tensor<?xi32>, tensor<?x1xi32>) -> tensor<?x?xi1>
  %538 = "tf.Reshape"(%537, %6) {device = ""} : (tensor<?x?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %539 = "tf.Where"(%538) {device = ""} : (tensor<?xi1>) -> tensor<?x1xi64>
  %540 = "tf.Squeeze"(%539) {device = "", squeeze_dims = [1]} : (tensor<?x1xi64>) -> tensor<?xi64>
  %541 = "tf.GatherV2"(%535, %540, %16) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %542 = "tf.Cast"(%518) {Truncate = false, device = ""} : (tensor<?xi64>) -> tensor<?xi32>
  %543 = "tf.BroadcastTo"(%542, %497) {device = ""} : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %544 = "tf.Max"(%543, %17) {device = "", keep_dims = false} : (tensor<?xi32>, tensor<1xi32>) -> tensor<i32>
  %545 = "tf.Maximum"(%16, %544) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %546 = "tf.Range"(%16, %545, %9) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  %547 = "tf.Pack"(%9, %545) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %548 = "tf.Tile"(%494, %547) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %549 = "tf.Shape"(%548) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %550 = "tf.StridedSlice"(%549, %17, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %551 = "tf.Prod"(%550, %17) {device = "", keep_dims = false} : (tensor<2xi32>, tensor<1xi32>) -> tensor<i32>
  %552 = "tf.Pack"(%551) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %553 = "tf.Shape"(%548) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %554 = "tf.StridedSlice"(%553, %17, %17, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %555 = "tf.Shape"(%548) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %556 = "tf.StridedSlice"(%555, %7, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %557 = "tf.ConcatV2"(%554, %552, %556, %16) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %558 = "tf.Reshape"(%548, %557) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %559 = "tf.ExpandDims"(%543, %2) {device = ""} : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
  %560 = "tf.Less"(%546, %559) {device = ""} : (tensor<?xi32>, tensor<?x1xi32>) -> tensor<?x?xi1>
  %561 = "tf.Reshape"(%560, %6) {device = ""} : (tensor<?x?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %562 = "tf.Where"(%561) {device = ""} : (tensor<?xi1>) -> tensor<?x1xi64>
  %563 = "tf.Squeeze"(%562) {device = "", squeeze_dims = [1]} : (tensor<?x1xi64>) -> tensor<?xi64>
  %564 = "tf.GatherV2"(%558, %563, %16) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %565:2 = "tf.RaggedRange"(%541, %564, %15) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %566 = "tf.GatherV2"(%482, %565#1, %16) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %567 = "tf.If"(%408, %408, %404, %15) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_Assert_1_AssertGuard_false_112940, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_Assert_1_AssertGuard_true_112930} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %568 = "tf.Identity"(%567) {device = ""} : (tensor<i1>) -> tensor<i1>
  %569 = "tf.Select"(%1, %404, %15) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %570 = "tf.Pack"(%569) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %571 = "tf.ConcatV2"(%0, %570, %14, %16) {device = ""} : (tensor<0xi64>, tensor<1xi64>, tensor<1xi64>, tensor<i32>) -> tensor<2xi64>
  %572 = "tf.StridedSlice"(%571, %18, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %573 = "tf.Equal"(%572, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %574 = "tf.StridedSlice"(%571, %18, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %575 = "tf.StridedSlice"(%571, %18, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %576 = "tf.Equal"(%575, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %577 = "tf.If"(%576, %576, %575, %518) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_Assert_2_AssertGuard_false_113430, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_Assert_2_AssertGuard_true_113420} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<?xi64>) -> tensor<i1>
  %578 = "tf.Identity"(%577) {device = ""} : (tensor<i1>) -> tensor<i1>
  %579 = "tf.If"(%573, %573, %518, %574) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_Assert_3_AssertGuard_false_113790, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_Assert_3_AssertGuard_true_113780} : (tensor<i1>, tensor<i1>, tensor<?xi64>, tensor<i64>) -> tensor<i1>
  %580 = "tf.Identity"(%579) {device = ""} : (tensor<i1>) -> tensor<i1>
  %581 = "tf.If"(%444, %444, %116, %440) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_AssertGuard_false_118470, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_AssertGuard_true_118460} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %582 = "tf.Identity"(%581) {device = ""} : (tensor<i1>) -> tensor<i1>
  %583 = "tf.Equal"(%440, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %584 = "tf.Select"(%583, %116, %440) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %585 = "tf.Equal"(%584, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %586 = "tf.LogicalOr"(%585, %119) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %587 = "tf.Equal"(%118, %584) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %588 = "tf.LogicalOr"(%586, %587) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %589 = "tf.Select"(%451, %584, %15) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %590 = "tf.Pack"(%589, %15) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %591 = "tf.StridedSlice"(%590, %18, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %592 = "tf.Cast"(%591) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %593 = "tf.Reshape"(%592, %11) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %594 = "tf.Pack"(%9, %593) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %595 = "tf.Tile"(%452, %594) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %596 = "tf.Mul"(%593, %455) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %597 = "tf.Pack"(%596) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %598 = "tf.ConcatV2"(%454, %597, %456, %16) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %599 = "tf.Reshape"(%595, %598) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %600 = "tf.Shape"(%599) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %601 = "tf.StridedSlice"(%600, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %602 = "tf.Pack"(%591) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %603 = "tf.StridedSlice"(%599, %602, %13, %14) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %604 = "tf.Sub"(%601, %591) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %605 = "tf.Pack"(%604) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %606 = "tf.StridedSlice"(%599, %13, %605, %14) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %607:2 = "tf.RaggedRange"(%606, %603, %15) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %608 = "tf.Select"(%124, %584, %15) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %609 = "tf.Pack"(%608, %15) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %610 = "tf.StridedSlice"(%609, %18, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %611 = "tf.Cast"(%610) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %612 = "tf.Reshape"(%611, %11) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %613 = "tf.Pack"(%9, %612) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %614 = "tf.Tile"(%106, %613) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %615 = "tf.Mul"(%612, %109) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %616 = "tf.Pack"(%615) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %617 = "tf.ConcatV2"(%108, %616, %110, %16) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %618 = "tf.Reshape"(%614, %617) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %619 = "tf.Shape"(%618) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %620 = "tf.StridedSlice"(%619, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %621 = "tf.Pack"(%610) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %622 = "tf.StridedSlice"(%618, %621, %13, %14) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %623 = "tf.Sub"(%620, %610) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %624 = "tf.Pack"(%623) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %625 = "tf.StridedSlice"(%618, %13, %624, %14) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %626:2 = "tf.RaggedRange"(%625, %622, %15) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %627 = "tf.StridedSlice"(%609, %17, %18, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi64>
  %628 = "tf.StridedSlice"(%609, %18, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %629 = "tf.Mul"(%113, %628) {device = ""} : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
  %630 = "tf.Tile"(%629, %627) {device = ""} : (tensor<?xi64>, tensor<1xi64>) -> tensor<?xi64>
  %631 = "tf.Cumsum"(%630, %16) {device = "", exclusive = false, reverse = false} : (tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %632 = "tf.ConcatV2"(%13, %631, %2) {device = ""} : (tensor<1xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %633 = "tf.StridedSlice"(%632, %17, %6, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %634 = "tf.ExpandDims"(%633, %9) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %635 = "tf.Shape"(%633) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %636 = "tf.StridedSlice"(%635, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %637 = "tf.Pack"(%636) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %638 = "tf.StridedSlice"(%632, %18, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %639 = "tf.ExpandDims"(%638, %9) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %640 = "tf.Shape"(%638) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %641 = "tf.StridedSlice"(%640, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %642 = "tf.Pack"(%641) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %643 = "tf.Equal"(%440, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %644 = "tf.Select"(%643, %584, %15) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %645 = "tf.Cast"(%644) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %646 = "tf.Reshape"(%645, %11) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %647 = "tf.Pack"(%9, %646) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %648 = "tf.Mul"(%646, %10) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %649 = "tf.Pack"(%648) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %650 = "tf.ConcatV2"(%11, %649, %11, %16) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %651 = "tf.Pack"(%644) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %652 = "tf.Pack"(%12, %440) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %653 = "tf.ExpandDims"(%652, %9) {device = ""} : (tensor<2xi64>, tensor<i32>) -> tensor<2x1xi64>
  %654 = "tf.Tile"(%653, %647) {device = ""} : (tensor<2x1xi64>, tensor<2xi32>) -> tensor<2x?xi64>
  %655 = "tf.Reshape"(%654, %650) {device = ""} : (tensor<2x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %656 = "tf.Shape"(%655) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %657 = "tf.StridedSlice"(%656, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %658 = "tf.Sub"(%657, %644) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %659 = "tf.Pack"(%658) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %660 = "tf.StridedSlice"(%655, %13, %659, %14) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %661 = "tf.StridedSlice"(%655, %651, %13, %14) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %662:2 = "tf.RaggedRange"(%660, %661, %15) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %663 = "tf.GatherV2"(%447, %662#1, %16) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %664 = "tf.Cast"(%663) {Truncate = false, device = ""} : (tensor<?xi64>) -> tensor<?xi32>
  %665 = "tf.BroadcastTo"(%664, %637) {device = ""} : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %666 = "tf.Max"(%665, %17) {device = "", keep_dims = false} : (tensor<?xi32>, tensor<1xi32>) -> tensor<i32>
  %667 = "tf.Maximum"(%16, %666) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %668 = "tf.Range"(%16, %667, %9) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  %669 = "tf.Pack"(%9, %667) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %670 = "tf.Tile"(%634, %669) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %671 = "tf.Shape"(%670) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %672 = "tf.StridedSlice"(%671, %17, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %673 = "tf.Prod"(%672, %17) {device = "", keep_dims = false} : (tensor<2xi32>, tensor<1xi32>) -> tensor<i32>
  %674 = "tf.Pack"(%673) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %675 = "tf.Shape"(%670) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %676 = "tf.StridedSlice"(%675, %17, %17, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %677 = "tf.Shape"(%670) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %678 = "tf.StridedSlice"(%677, %7, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %679 = "tf.ConcatV2"(%676, %674, %678, %16) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %680 = "tf.Reshape"(%670, %679) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %681 = "tf.ExpandDims"(%665, %2) {device = ""} : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
  %682 = "tf.Less"(%668, %681) {device = ""} : (tensor<?xi32>, tensor<?x1xi32>) -> tensor<?x?xi1>
  %683 = "tf.Reshape"(%682, %6) {device = ""} : (tensor<?x?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %684 = "tf.Where"(%683) {device = ""} : (tensor<?xi1>) -> tensor<?x1xi64>
  %685 = "tf.Squeeze"(%684) {device = "", squeeze_dims = [1]} : (tensor<?x1xi64>) -> tensor<?xi64>
  %686 = "tf.GatherV2"(%680, %685, %16) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %687 = "tf.Cast"(%663) {Truncate = false, device = ""} : (tensor<?xi64>) -> tensor<?xi32>
  %688 = "tf.BroadcastTo"(%687, %642) {device = ""} : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %689 = "tf.Max"(%688, %17) {device = "", keep_dims = false} : (tensor<?xi32>, tensor<1xi32>) -> tensor<i32>
  %690 = "tf.Maximum"(%16, %689) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %691 = "tf.Range"(%16, %690, %9) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  %692 = "tf.Pack"(%9, %690) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %693 = "tf.Tile"(%639, %692) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %694 = "tf.Shape"(%693) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %695 = "tf.StridedSlice"(%694, %17, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %696 = "tf.Prod"(%695, %17) {device = "", keep_dims = false} : (tensor<2xi32>, tensor<1xi32>) -> tensor<i32>
  %697 = "tf.Pack"(%696) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %698 = "tf.Shape"(%693) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %699 = "tf.StridedSlice"(%698, %17, %17, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %700 = "tf.Shape"(%693) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %701 = "tf.StridedSlice"(%700, %7, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %702 = "tf.ConcatV2"(%699, %697, %701, %16) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %703 = "tf.Reshape"(%693, %702) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %704 = "tf.ExpandDims"(%688, %2) {device = ""} : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
  %705 = "tf.Less"(%691, %704) {device = ""} : (tensor<?xi32>, tensor<?x1xi32>) -> tensor<?x?xi1>
  %706 = "tf.Reshape"(%705, %6) {device = ""} : (tensor<?x?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %707 = "tf.Where"(%706) {device = ""} : (tensor<?xi1>) -> tensor<?x1xi64>
  %708 = "tf.Squeeze"(%707) {device = "", squeeze_dims = [1]} : (tensor<?x1xi64>) -> tensor<?xi64>
  %709 = "tf.GatherV2"(%703, %708, %16) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %710:2 = "tf.RaggedRange"(%686, %709, %15) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %711 = "tf.If"(%588, %588, %584, %120) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_1_AssertGuard_false_119540, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_1_AssertGuard_true_119530} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %712 = "tf.Identity"(%711) {device = ""} : (tensor<i1>) -> tensor<i1>
  %713 = "tf.StridedSlice"(%115, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %714 = "tf.Equal"(%713, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %715 = "tf.Select"(%714, %584, %713) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %716 = "tf.Pack"(%715) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %717 = "tf.StridedSlice"(%115, %17, %17, %18) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi64>
  %718 = "tf.StridedSlice"(%115, %18, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi64>
  %719 = "tf.ConcatV2"(%717, %716, %718, %16) {device = ""} : (tensor<0xi64>, tensor<1xi64>, tensor<1xi64>, tensor<i32>) -> tensor<2xi64>
  %720 = "tf.StridedSlice"(%719, %18, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %721 = "tf.Equal"(%720, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %722 = "tf.StridedSlice"(%719, %18, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %723 = "tf.StridedSlice"(%719, %18, %7, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %724 = "tf.Equal"(%723, %15) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %725 = "tf.If"(%724, %724, %723, %663) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_2_AssertGuard_false_120030, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_2_AssertGuard_true_120020} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<?xi64>) -> tensor<i1>
  %726 = "tf.Identity"(%725) {device = ""} : (tensor<i1>) -> tensor<i1>
  %727 = "tf.If"(%721, %721, %663, %722) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_3_AssertGuard_false_120390, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_3_AssertGuard_true_120380} : (tensor<i1>, tensor<i1>, tensor<?xi64>, tensor<i64>) -> tensor<i1>
  %728 = "tf.Identity"(%168) {device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %729 = "tf.Identity"(%727) {device = ""} : (tensor<i1>) -> tensor<i1>
  %730 = "tf.Identity"(%400) {device = ""} : (tensor<i1>) -> tensor<i1>
  %731 = "tf.Shape"(%125#2) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %732 = "tf.StridedSlice"(%731, %18, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %733 = "tf.Cast"(%732) {Truncate = false, device = ""} : (tensor<0xi32>) -> tensor<0xi64>
  %734 = "tf.Identity"(%733) {device = ""} : (tensor<0xi64>) -> tensor<0xi64>
  %735 = "tf.Shape"(%125#3) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %736 = "tf.StridedSlice"(%735, %18, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %737 = "tf.Cast"(%736) {Truncate = false, device = ""} : (tensor<0xi32>) -> tensor<0xi64>
  %738 = "tf.Identity"(%737) {device = ""} : (tensor<0xi64>) -> tensor<0xi64>
  %739 = "tf.GatherV2"(%125#3, %428, %16) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %740 = "tf.Tile"(%739, %432) {device = ""} : (tensor<?xi64>, tensor<1xi64>) -> tensor<?xi64>
  %741 = "tf.Sub"(%740, %566) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %742 = "tf.Shape"(%741) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %743 = "tf.StridedSlice"(%742, %18, %17, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %744 = "tf.Cast"(%743) {Truncate = false, device = ""} : (tensor<0xi32>) -> tensor<0xi64>
  %745 = "tf.Identity"(%744) {device = ""} : (tensor<0xi64>) -> tensor<0xi64>
  %746 = "tf.UnicodeEncode"(%125#0, %146) {Tsplits = i64, device = "", errors = "replace", output_encoding = "UTF-8", replacement_char = 65533 : i64} : (tensor<?xi32>, tensor<?xi64>) -> tensor<?x!tf_type.string>
  %747 = "tf.Identity"(%746) {device = ""} : (tensor<?x!tf_type.string>) -> tensor<?x!tf_type.string>
  %748 = "tf.StridedSlice"(%19, %17, %18, %18) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %749 = "tf.AddV2"(%748, %15) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %750 = "tf.Range"(%12, %749, %15) {device = ""} : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<?xi64>
  %751 = "tf.Mul"(%750, %15) {device = ""} : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
  %752 = "tf.Identity"(%751) {device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  func.return %747, %752, %728 : tensor<?x!tf_type.string>, tensor<?xi64>, tensor<?xi64>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedConcat_assert_equal_1_Assert_AssertGuard_false_99640(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Input tensors have incompatible shapes."> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/RaggedConcat/RaggedFromTensor/strided_slice_4:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/RaggedConcat/RaggedNRows/sub:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedConcat_assert_equal_1_Assert_AssertGuard_true_99630(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_100400(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:zero"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits/RowPartitionFromRowSplits/Const:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_100390(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_100760(%arg0: tensor<i1>, %arg1: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:monotonic"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x >= 0 did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits/RowPartitionFromRowSplits/sub:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<?xi64>) -> ()
  %3 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %4 = "tf.Identity"(%3) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %4 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_100750(%arg0: tensor<i1>, %arg1: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_false_101100(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to _from_row_partition do not form a valid RaggedTensor"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits/strided_slice_1:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_true_101090(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_101470(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:zero"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits_1/RowPartitionFromRowSplits/Const:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_101460(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_101830(%arg0: tensor<i1>, %arg1: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:monotonic"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x >= 0 did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits_1/RowPartitionFromRowSplits/sub:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<?xi64>) -> ()
  %3 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %4 = "tf.Identity"(%3) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %4 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_101820(%arg0: tensor<i1>, %arg1: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_false_102190(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to _from_row_partition do not form a valid RaggedTensor"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits_1/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits_1/RaggedNRows/sub:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_true_102180(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_102540(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:zero"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits_1/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits_1/RaggedFromRowSplits/RowPartitionFromRowSplits/Const:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_102530(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_102900(%arg0: tensor<i1>, %arg1: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:monotonic"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x >= 0 did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits_1/RaggedFromRowSplits/RowPartitionFromRowSplits/sub:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<?xi64>) -> ()
  %3 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %4 = "tf.Identity"(%3) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %4 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_102890(%arg0: tensor<i1>, %arg1: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_false_103240(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to _from_row_partition do not form a valid RaggedTensor"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits_1/RaggedFromRowSplits/strided_slice_1:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits_1/RaggedFromRowSplits/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_true_103230(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_103610(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:zero"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits_2/RaggedFromRowSplits/RowPartitionFromRowSplits/Const:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_103600(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_103970(%arg0: tensor<i1>, %arg1: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:monotonic"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x >= 0 did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits_2/RaggedFromRowSplits/RowPartitionFromRowSplits/sub:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<?xi64>) -> ()
  %3 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %4 = "tf.Identity"(%3) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %4 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_103960(%arg0: tensor<i1>, %arg1: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_false_104310(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to _from_row_partition do not form a valid RaggedTensor"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits_2/RaggedFromRowSplits/strided_slice_1:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits_2/RaggedFromRowSplits/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_true_104300(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_AssertGuard_false_105110(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_AssertGuard_true_105100(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_1_AssertGuard_false_106180(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_1_AssertGuard_true_106170(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_2_AssertGuard_false_106670(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<?>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<?xi64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_2_AssertGuard_true_106660(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<?>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_3_AssertGuard_false_107030(%arg0: tensor<i1>, %arg1: tensor<?xi64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<?xi64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_3_AssertGuard_true_107020(%arg0: tensor<i1>, %arg1: tensor<?xi64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_Assert_AssertGuard_false_111870(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_Assert_AssertGuard_true_111860(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_Assert_1_AssertGuard_false_112940(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_Assert_1_AssertGuard_true_112930(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_Assert_2_AssertGuard_false_113430(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<?>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<?xi64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_Assert_2_AssertGuard_true_113420(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<?>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_Assert_3_AssertGuard_false_113790(%arg0: tensor<i1>, %arg1: tensor<?xi64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<?xi64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_Assert_3_AssertGuard_true_113780(%arg0: tensor<i1>, %arg1: tensor<?xi64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_AssertGuard_false_118470(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_AssertGuard_true_118460(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_1_AssertGuard_false_119540(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_1_AssertGuard_true_119530(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_2_AssertGuard_false_120030(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<?>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<?xi64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_2_AssertGuard_true_120020(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<?>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_3_AssertGuard_false_120390(%arg0: tensor<i1>, %arg1: tensor<?xi64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<?xi64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_3_AssertGuard_true_120380(%arg0: tensor<i1>, %arg1: tensor<?xi64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}



// CHECK:  func private @whitespace_tokenizer_rank2(%arg0: tensor<?x1x!tf_type.string> {tf._user_specified_name = "input"}) -> (tensor<?x!tf_type.string>, tensor<?xi64>, tensor<?xi64>) attributes {tf._implements = #tf_type.func<@"tftext:WhitespaceTokenizer", {}>, tf._input_shapes = [#tf_type.shape<?x1>], tf.signature.is_stateful} {
// CHECK:  %0:3 = "tfl.custom"(%arg0) {custom_code = "tftext:WhitespaceTokenizer", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>} : (tensor<?x1x!tf_type.string>) -> (tensor<?x!tf_type.string>, tensor<?xi64>, tensor<?xi64>)
// CHECK:  return %0#0, %0#1, %0#2 : tensor<?x!tf_type.string>, tensor<?xi64>, tensor<?xi64>

func.func private @whitespace_tokenizer_rank0(%arg0: tensor<!tf_type.string> {tf._user_specified_name = "input"}) -> tensor<?x!tf_type.string> attributes {tf._input_shapes = [#tf_type.shape<>], tf._implements = #tf_type.func<@"tftext:WhitespaceTokenizer", {}>, tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<[0, 1]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "tf.Const"() {value = dense<[]> : tensor<0xi64>} : () -> tensor<0xi64>
  %2 = "tf.Const"() {value = dense<true> : tensor<i1>} : () -> tensor<i1>
  %3 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %4 = "tf.Const"() {value = dense<[[0], [1]]> : tensor<2x1xi64>} : () -> tensor<2x1xi64>
  %5 = "tf.Const"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
  %6 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %7 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %8 = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
  %9 = "tf.Const"() {value = dense<[]> : tensor<0xi32>} : () -> tensor<0xi32>
  %10 = "tf.Const"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  %11 = "tf.Const"() {value = dense<0> : tensor<1xi64>} : () -> tensor<1xi64>
  %12 = "tf.Const"() {value = dense<1> : tensor<1xi64>} : () -> tensor<1xi64>
  %13 = "tf.Const"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %14 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %15 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %16 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %17 = "tf.If"(%2, %2, %13, %13) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedConcat_assert_equal_1_Assert_AssertGuard_false_3220, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedConcat_assert_equal_1_Assert_AssertGuard_true_3210} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %18 = "tf.Identity"(%17) {device = ""} : (tensor<i1>) -> tensor<i1>
  %19 = "tf.Pack"(%arg0) {axis = 0 : i64, device = ""} : (tensor<!tf_type.string>) -> tensor<1x!tf_type.string>
  %20 = "tf.StringLength"(%19) {device = "", unit = "BYTE"} : (tensor<1x!tf_type.string>) -> tensor<1xi32>
  %21 = "tf.ExpandDims"(%20, %7) {device = ""} : (tensor<1xi32>, tensor<i32>) -> tensor<1x1xi32>
  %22 = "tf.Cast"(%21) {Truncate = false, device = ""} : (tensor<1x1xi32>) -> tensor<1x1xi64>
  %23 = "tf.Reshape"(%22, %12) {device = ""} : (tensor<1x1xi64>, tensor<1xi64>) -> tensor<1xi64>
  %24 = "tf.Reshape"(%19, %5) {device = ""} : (tensor<1x!tf_type.string>, tensor<1xi32>) -> tensor<1x!tf_type.string>
  %25:3 = "tf.UnicodeDecodeWithOffsets"(%24) {Tsplits = i64, device = "", errors = "replace", input_encoding = "UTF-8", replace_control_characters = false, replacement_char = 65533 : i64} : (tensor<1x!tf_type.string>) -> (tensor<2xi64>, tensor<?xi32>, tensor<?xi64>)
  %26 = "tf.StridedSlice"(%25#0, %15, %5, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi64>
  %27 = "tf.AddV2"(%26, %13) {device = ""} : (tensor<1xi64>, tensor<i64>) -> tensor<1xi64>
  %28 = "tf.StridedSlice"(%25#0, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi64>
  %29 = "tf.Minimum"(%27, %28) {device = ""} : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
  %30:2 = "tf.RaggedRange"(%29, %28, %13) {T = i64, Tsplits = i64, device = ""} : (tensor<1xi64>, tensor<1xi64>, tensor<i64>) -> (tensor<2xi64>, tensor<?xi64>)
  %31 = "tf.StridedSlice"(%30#0, %5, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %32 = "tf.AddV2"(%31, %12) {device = ""} : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
  %33 = "tf.ConcatV2"(%30#0, %32, %14) {device = ""} : (tensor<2xi64>, tensor<1xi64>, tensor<i32>) -> tensor<3xi64>
  %34 = "tf.GatherV2"(%25#2, %30#1, %14) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %35 = "tf.ConcatV2"(%34, %23, %14) {device = ""} : (tensor<?xi64>, tensor<1xi64>, tensor<i32>) -> tensor<?xi64>
  %36:2 = "tf.RaggedGather"(%33, %35, %0) {OUTPUT_RAGGED_RANK = 1 : i64, PARAMS_RAGGED_RANK = 1 : i64, Tindices = i64, Tsplits = i64, Tvalues = i64, device = ""} : (tensor<3xi64>, tensor<?xi64>, tensor<2xi64>) -> (tensor<?xi64>, tensor<?xi64>)
  %37:5 = "tf.WhitespaceTokenizeWithOffsets"(%25#1, %25#0) {Tsplits = i64, device = ""} : (tensor<?xi32>, tensor<2xi64>) -> (tensor<?xi32>, tensor<?xi64>, tensor<?xi64>, tensor<?xi64>, tensor<?xi64>)
  %38 = "tf.StridedSlice"(%37#1, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %39 = "tf.Equal"(%38, %10) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %40 = "tf.All"(%39, %9) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %41 = "tf.If"(%40, %40, %38, %10) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_3980, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_3970} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %42 = "tf.Identity"(%41) {device = ""} : (tensor<i1>) -> tensor<i1>
  %43 = "tf.StridedSlice"(%37#1, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %44 = "tf.StridedSlice"(%37#1, %15, %5, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %45 = "tf.Sub"(%43, %44) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %46 = "tf.LessEqual"(%10, %45) {device = ""} : (tensor<i64>, tensor<?xi64>) -> tensor<?xi1>
  %47 = "tf.All"(%46, %15) {device = "", keep_dims = false} : (tensor<?xi1>, tensor<1xi32>) -> tensor<i1>
  %48 = "tf.If"(%47, %47, %45) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_4340, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_4330} : (tensor<i1>, tensor<i1>, tensor<?xi64>) -> tensor<i1>
  %49 = "tf.Identity"(%48) {device = ""} : (tensor<i1>) -> tensor<i1>
  %50 = "tf.Identity"(%37#1) {_class = ["loc:@WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenizeWithOffsets"], device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %51 = "tf.StridedSlice"(%50, %5, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %52 = "tf.Shape"(%37#0) {device = ""} : (tensor<?xi32>) -> tensor<1xi64>
  %53 = "tf.StridedSlice"(%52, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %54 = "tf.Equal"(%51, %53) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %55 = "tf.All"(%54, %9) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %56 = "tf.If"(%55, %55, %51, %53) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_false_4680, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_true_4670} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %57 = "tf.Identity"(%56) {device = ""} : (tensor<i1>) -> tensor<i1>
  %58 = "tf.Identity"(%50) {_class = ["loc:@WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenizeWithOffsets"], device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %59 = "tf.Shape"(%58) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %60 = "tf.StridedSlice"(%59, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %61 = "tf.Sub"(%60, %13) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %62 = "tf.StridedSlice"(%37#4, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %63 = "tf.Equal"(%62, %10) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %64 = "tf.All"(%63, %9) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %65 = "tf.If"(%64, %64, %62, %10) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_5050, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_5040} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %66 = "tf.Identity"(%65) {device = ""} : (tensor<i1>) -> tensor<i1>
  %67 = "tf.StridedSlice"(%37#4, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %68 = "tf.StridedSlice"(%37#4, %15, %5, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %69 = "tf.Sub"(%67, %68) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %70 = "tf.LessEqual"(%10, %69) {device = ""} : (tensor<i64>, tensor<?xi64>) -> tensor<?xi1>
  %71 = "tf.All"(%70, %15) {device = "", keep_dims = false} : (tensor<?xi1>, tensor<1xi32>) -> tensor<i1>
  %72 = "tf.If"(%71, %71, %69) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_5410, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_5400} : (tensor<i1>, tensor<i1>, tensor<?xi64>) -> tensor<i1>
  %73 = "tf.Identity"(%72) {device = ""} : (tensor<i1>) -> tensor<i1>
  %74 = "tf.Identity"(%37#4) {_class = ["loc:@WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenizeWithOffsets"], device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %75 = "tf.StridedSlice"(%74, %5, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %76 = "tf.Equal"(%75, %61) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %77 = "tf.All"(%76, %9) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %78 = "tf.If"(%77, %77, %75, %61) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_false_5770, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_true_5760} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %79 = "tf.Identity"(%78) {device = ""} : (tensor<i1>) -> tensor<i1>
  %80 = "tf.Identity"(%74) {_class = ["loc:@WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenizeWithOffsets"], device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %81 = "tf.StridedSlice"(%37#4, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %82 = "tf.Equal"(%81, %10) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %83 = "tf.All"(%82, %9) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %84 = "tf.If"(%83, %83, %81, %10) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_6120, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_6110} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %85 = "tf.Identity"(%84) {device = ""} : (tensor<i1>) -> tensor<i1>
  %86 = "tf.StridedSlice"(%37#4, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %87 = "tf.StridedSlice"(%37#4, %15, %5, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %88 = "tf.Sub"(%86, %87) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %89 = "tf.LessEqual"(%10, %88) {device = ""} : (tensor<i64>, tensor<?xi64>) -> tensor<?xi1>
  %90 = "tf.All"(%89, %15) {device = "", keep_dims = false} : (tensor<?xi1>, tensor<1xi32>) -> tensor<i1>
  %91 = "tf.If"(%90, %90, %88) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_6480, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_6470} : (tensor<i1>, tensor<i1>, tensor<?xi64>) -> tensor<i1>
  %92 = "tf.Identity"(%91) {device = ""} : (tensor<i1>) -> tensor<i1>
  %93 = "tf.Identity"(%37#4) {_class = ["loc:@WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenizeWithOffsets"], device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %94 = "tf.StridedSlice"(%93, %5, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %95 = "tf.Shape"(%37#2) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %96 = "tf.StridedSlice"(%95, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %97 = "tf.Equal"(%94, %96) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %98 = "tf.All"(%97, %9) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %99 = "tf.If"(%98, %98, %94, %96) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_false_6820, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_true_6810} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %100 = "tf.Identity"(%99) {device = ""} : (tensor<i1>) -> tensor<i1>
  %101 = "tf.Identity"(%93) {_class = ["loc:@WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenizeWithOffsets"], device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %102 = "tf.Shape"(%101) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %103 = "tf.StridedSlice"(%102, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %104 = "tf.Sub"(%103, %13) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %105 = "tf.Equal"(%104, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %106 = "tf.LogicalOr"(%105, %2) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %107 = "tf.Equal"(%104, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %108 = "tf.LogicalOr"(%106, %107) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %109 = "tf.StridedSlice"(%101, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %110 = "tf.StridedSlice"(%101, %15, %5, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %111 = "tf.Sub"(%109, %110) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %112 = "tf.Shape"(%101) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %113 = "tf.StridedSlice"(%112, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %114 = "tf.Sub"(%113, %13) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %115 = "tf.Equal"(%114, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %116 = "tf.ExpandDims"(%101, %7) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %117 = "tf.Shape"(%101) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %118 = "tf.StridedSlice"(%117, %15, %15, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %119 = "tf.StridedSlice"(%117, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %120 = "tf.StridedSlice"(%117, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %121 = "tf.StridedSlice"(%37#4, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %122 = "tf.Equal"(%121, %10) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %123 = "tf.All"(%122, %9) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %124 = "tf.If"(%123, %123, %121, %10) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_7190, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_7180} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %125 = "tf.Identity"(%124) {device = ""} : (tensor<i1>) -> tensor<i1>
  %126 = "tf.StridedSlice"(%37#4, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %127 = "tf.StridedSlice"(%37#4, %15, %5, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %128 = "tf.Sub"(%126, %127) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %129 = "tf.LessEqual"(%10, %128) {device = ""} : (tensor<i64>, tensor<?xi64>) -> tensor<?xi1>
  %130 = "tf.All"(%129, %15) {device = "", keep_dims = false} : (tensor<?xi1>, tensor<1xi32>) -> tensor<i1>
  %131 = "tf.If"(%130, %130, %128) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_7550, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_7540} : (tensor<i1>, tensor<i1>, tensor<?xi64>) -> tensor<i1>
  %132 = "tf.Identity"(%131) {device = ""} : (tensor<i1>) -> tensor<i1>
  %133 = "tf.Identity"(%37#4) {_class = ["loc:@WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenizeWithOffsets"], device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %134 = "tf.StridedSlice"(%133, %5, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %135 = "tf.Shape"(%37#3) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %136 = "tf.StridedSlice"(%135, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %137 = "tf.Equal"(%134, %136) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %138 = "tf.All"(%137, %9) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %139 = "tf.If"(%138, %138, %134, %136) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_false_7890, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_true_7880} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %140 = "tf.Identity"(%139) {device = ""} : (tensor<i1>) -> tensor<i1>
  %141 = "tf.Identity"(%133) {_class = ["loc:@WhitespaceTokenize/WhitespaceTokenize/WhitespaceTokenizeWithOffsets"], device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %142 = "tf.Shape"(%141) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %143 = "tf.StridedSlice"(%142, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %144 = "tf.Sub"(%143, %13) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %145 = "tf.Equal"(%144, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %146 = "tf.LogicalOr"(%145, %2) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %147 = "tf.Equal"(%144, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %148 = "tf.LogicalOr"(%146, %147) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %149 = "tf.StridedSlice"(%141, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %150 = "tf.StridedSlice"(%141, %15, %5, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %151 = "tf.Sub"(%149, %150) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %152 = "tf.Shape"(%141) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %153 = "tf.StridedSlice"(%152, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %154 = "tf.Sub"(%153, %13) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %155 = "tf.Equal"(%154, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %156 = "tf.ExpandDims"(%141, %7) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %157 = "tf.Shape"(%141) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %158 = "tf.StridedSlice"(%157, %15, %15, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %159 = "tf.StridedSlice"(%157, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %160 = "tf.StridedSlice"(%157, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %161 = "tf.StridedSlice"(%141, %5, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %162 = "tf.Range"(%10, %161, %13) {device = ""} : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<?xi64>
  %163 = "tf.StridedSlice"(%141, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %164 = "tf.StridedSlice"(%141, %15, %5, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %165 = "tf.Sub"(%163, %164) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %166 = "tf.If"(%108, %108, %13, %104) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_AssertGuard_false_8690, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_AssertGuard_true_8680} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %167 = "tf.Identity"(%166) {device = ""} : (tensor<i1>) -> tensor<i1>
  %168 = "tf.Equal"(%104, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %169 = "tf.Select"(%168, %13, %104) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %170 = "tf.Equal"(%169, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %171 = "tf.LogicalOr"(%170, %2) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %172 = "tf.Equal"(%169, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %173 = "tf.LogicalOr"(%171, %172) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %174 = "tf.Select"(%115, %169, %13) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %175 = "tf.Pack"(%174, %13) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %176 = "tf.StridedSlice"(%175, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %177 = "tf.Cast"(%176) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %178 = "tf.Reshape"(%177, %9) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %179 = "tf.Pack"(%7, %178) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %180 = "tf.Tile"(%116, %179) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %181 = "tf.Mul"(%178, %119) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %182 = "tf.Pack"(%181) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %183 = "tf.ConcatV2"(%118, %182, %120, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %184 = "tf.Reshape"(%180, %183) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %185 = "tf.Shape"(%184) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %186 = "tf.StridedSlice"(%185, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %187 = "tf.Pack"(%176) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %188 = "tf.StridedSlice"(%184, %187, %11, %12) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %189 = "tf.Sub"(%186, %176) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %190 = "tf.Pack"(%189) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %191 = "tf.StridedSlice"(%184, %11, %190, %12) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %192:2 = "tf.RaggedRange"(%191, %188, %13) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %193 = "tf.Select"(%2, %169, %13) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %194 = "tf.Pack"(%193, %13) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %195 = "tf.StridedSlice"(%194, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %196 = "tf.Cast"(%195) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %197 = "tf.Reshape"(%196, %9) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %198 = "tf.Pack"(%7, %197) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %199 = "tf.Tile"(%4, %198) {device = ""} : (tensor<2x1xi64>, tensor<2xi32>) -> tensor<2x?xi64>
  %200 = "tf.Mul"(%197, %8) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %201 = "tf.Pack"(%200) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %202 = "tf.ConcatV2"(%9, %201, %9, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %203 = "tf.Reshape"(%199, %202) {device = ""} : (tensor<2x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %204 = "tf.Shape"(%203) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %205 = "tf.StridedSlice"(%204, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %206 = "tf.Pack"(%195) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %207 = "tf.StridedSlice"(%203, %206, %11, %12) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %208 = "tf.Sub"(%205, %195) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %209 = "tf.Pack"(%208) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %210 = "tf.StridedSlice"(%203, %11, %209, %12) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %211:2 = "tf.RaggedRange"(%210, %207, %13) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %212 = "tf.StridedSlice"(%194, %15, %16, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi64>
  %213 = "tf.StridedSlice"(%194, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %214 = "tf.Mul"(%213, %12) {device = ""} : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
  %215 = "tf.Tile"(%214, %212) {device = ""} : (tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %216 = "tf.Cumsum"(%215, %14) {device = "", exclusive = false, reverse = false} : (tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %217 = "tf.ConcatV2"(%11, %216, %3) {device = ""} : (tensor<1xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %218 = "tf.StridedSlice"(%217, %15, %5, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %219 = "tf.ExpandDims"(%218, %7) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %220 = "tf.Shape"(%218) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %221 = "tf.StridedSlice"(%220, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %222 = "tf.Pack"(%221) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %223 = "tf.StridedSlice"(%217, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %224 = "tf.ExpandDims"(%223, %7) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %225 = "tf.Shape"(%223) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %226 = "tf.StridedSlice"(%225, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %227 = "tf.Pack"(%226) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %228 = "tf.Equal"(%104, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %229 = "tf.Select"(%228, %169, %13) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %230 = "tf.Cast"(%229) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %231 = "tf.Reshape"(%230, %9) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %232 = "tf.Pack"(%7, %231) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %233 = "tf.Mul"(%231, %8) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %234 = "tf.Pack"(%233) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %235 = "tf.ConcatV2"(%9, %234, %9, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %236 = "tf.Pack"(%229) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %237 = "tf.Pack"(%10, %104) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %238 = "tf.ExpandDims"(%237, %7) {device = ""} : (tensor<2xi64>, tensor<i32>) -> tensor<2x1xi64>
  %239 = "tf.Tile"(%238, %232) {device = ""} : (tensor<2x1xi64>, tensor<2xi32>) -> tensor<2x?xi64>
  %240 = "tf.Reshape"(%239, %235) {device = ""} : (tensor<2x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %241 = "tf.Shape"(%240) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %242 = "tf.StridedSlice"(%241, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %243 = "tf.Sub"(%242, %229) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %244 = "tf.Pack"(%243) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %245 = "tf.StridedSlice"(%240, %11, %244, %12) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %246 = "tf.StridedSlice"(%240, %236, %11, %12) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %247:2 = "tf.RaggedRange"(%245, %246, %13) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %248 = "tf.GatherV2"(%111, %247#1, %14) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %249 = "tf.Cast"(%248) {Truncate = false, device = ""} : (tensor<?xi64>) -> tensor<?xi32>
  %250 = "tf.BroadcastTo"(%249, %222) {device = ""} : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %251 = "tf.Max"(%250, %15) {device = "", keep_dims = false} : (tensor<?xi32>, tensor<1xi32>) -> tensor<i32>
  %252 = "tf.Maximum"(%14, %251) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %253 = "tf.Range"(%14, %252, %7) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  %254 = "tf.Pack"(%7, %252) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %255 = "tf.Tile"(%219, %254) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %256 = "tf.Shape"(%255) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %257 = "tf.StridedSlice"(%256, %15, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %258 = "tf.Prod"(%257, %15) {device = "", keep_dims = false} : (tensor<2xi32>, tensor<1xi32>) -> tensor<i32>
  %259 = "tf.Pack"(%258) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %260 = "tf.Shape"(%255) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %261 = "tf.StridedSlice"(%260, %15, %15, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %262 = "tf.Shape"(%255) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %263 = "tf.StridedSlice"(%262, %6, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %264 = "tf.ConcatV2"(%261, %259, %263, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %265 = "tf.Reshape"(%255, %264) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %266 = "tf.ExpandDims"(%250, %3) {device = ""} : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
  %267 = "tf.Less"(%253, %266) {device = ""} : (tensor<?xi32>, tensor<?x1xi32>) -> tensor<?x?xi1>
  %268 = "tf.Reshape"(%267, %5) {device = ""} : (tensor<?x?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %269 = "tf.Where"(%268) {device = ""} : (tensor<?xi1>) -> tensor<?x1xi64>
  %270 = "tf.Squeeze"(%269) {device = "", squeeze_dims = [1]} : (tensor<?x1xi64>) -> tensor<?xi64>
  %271 = "tf.GatherV2"(%265, %270, %14) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %272 = "tf.Cast"(%248) {Truncate = false, device = ""} : (tensor<?xi64>) -> tensor<?xi32>
  %273 = "tf.BroadcastTo"(%272, %227) {device = ""} : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %274 = "tf.Max"(%273, %15) {device = "", keep_dims = false} : (tensor<?xi32>, tensor<1xi32>) -> tensor<i32>
  %275 = "tf.Maximum"(%14, %274) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %276 = "tf.Range"(%14, %275, %7) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  %277 = "tf.Pack"(%7, %275) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %278 = "tf.Tile"(%224, %277) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %279 = "tf.Shape"(%278) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %280 = "tf.StridedSlice"(%279, %15, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %281 = "tf.Prod"(%280, %15) {device = "", keep_dims = false} : (tensor<2xi32>, tensor<1xi32>) -> tensor<i32>
  %282 = "tf.Pack"(%281) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %283 = "tf.Shape"(%278) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %284 = "tf.StridedSlice"(%283, %15, %15, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %285 = "tf.Shape"(%278) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %286 = "tf.StridedSlice"(%285, %6, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %287 = "tf.ConcatV2"(%284, %282, %286, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %288 = "tf.Reshape"(%278, %287) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %289 = "tf.ExpandDims"(%273, %3) {device = ""} : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
  %290 = "tf.Less"(%276, %289) {device = ""} : (tensor<?xi32>, tensor<?x1xi32>) -> tensor<?x?xi1>
  %291 = "tf.Reshape"(%290, %5) {device = ""} : (tensor<?x?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %292 = "tf.Where"(%291) {device = ""} : (tensor<?xi1>) -> tensor<?x1xi64>
  %293 = "tf.Squeeze"(%292) {device = "", squeeze_dims = [1]} : (tensor<?x1xi64>) -> tensor<?xi64>
  %294 = "tf.GatherV2"(%288, %293, %14) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %295:2 = "tf.RaggedRange"(%271, %294, %13) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %296 = "tf.If"(%173, %173, %169, %13) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_1_AssertGuard_false_9760, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_1_AssertGuard_true_9750} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %297 = "tf.Identity"(%296) {device = ""} : (tensor<i1>) -> tensor<i1>
  %298 = "tf.Select"(%2, %169, %13) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %299 = "tf.Pack"(%298) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %300 = "tf.ConcatV2"(%1, %299, %12, %14) {device = ""} : (tensor<0xi64>, tensor<1xi64>, tensor<1xi64>, tensor<i32>) -> tensor<2xi64>
  %301 = "tf.StridedSlice"(%300, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %302 = "tf.Equal"(%301, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %303 = "tf.StridedSlice"(%300, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %304 = "tf.StridedSlice"(%300, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %305 = "tf.Equal"(%304, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %306 = "tf.If"(%305, %305, %304, %248) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_2_AssertGuard_false_10250, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_2_AssertGuard_true_10240} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<?xi64>) -> tensor<i1>
  %307 = "tf.Identity"(%306) {device = ""} : (tensor<i1>) -> tensor<i1>
  %308 = "tf.If"(%302, %302, %248, %303) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_3_AssertGuard_false_10610, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_3_AssertGuard_true_10600} : (tensor<i1>, tensor<i1>, tensor<?xi64>, tensor<i64>) -> tensor<i1>
  %309 = "tf.If"(%148, %148, %13, %144) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_Assert_AssertGuard_false_15310, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_Assert_AssertGuard_true_15300} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %310 = "tf.Identity"(%309) {device = ""} : (tensor<i1>) -> tensor<i1>
  %311 = "tf.Equal"(%144, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %312 = "tf.Select"(%311, %13, %144) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %313 = "tf.Equal"(%312, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %314 = "tf.LogicalOr"(%313, %2) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %315 = "tf.Equal"(%312, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %316 = "tf.LogicalOr"(%314, %315) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %317 = "tf.Select"(%155, %312, %13) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %318 = "tf.Pack"(%317, %13) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %319 = "tf.StridedSlice"(%318, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %320 = "tf.Cast"(%319) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %321 = "tf.Reshape"(%320, %9) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %322 = "tf.Pack"(%7, %321) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %323 = "tf.Tile"(%156, %322) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %324 = "tf.Mul"(%321, %159) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %325 = "tf.Pack"(%324) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %326 = "tf.ConcatV2"(%158, %325, %160, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %327 = "tf.Reshape"(%323, %326) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %328 = "tf.Shape"(%327) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %329 = "tf.StridedSlice"(%328, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %330 = "tf.Pack"(%319) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %331 = "tf.StridedSlice"(%327, %330, %11, %12) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %332 = "tf.Sub"(%329, %319) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %333 = "tf.Pack"(%332) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %334 = "tf.StridedSlice"(%327, %11, %333, %12) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %335:2 = "tf.RaggedRange"(%334, %331, %13) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %336 = "tf.GatherV2"(%162, %335#1, %14) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %337 = "tf.StridedSlice"(%318, %15, %16, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi64>
  %338 = "tf.StridedSlice"(%318, %15, %16, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi64>
  %339 = "tf.StridedSlice"(%318, %6, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi64>
  %340 = "tf.ConcatV2"(%338, %339, %14) {device = ""} : (tensor<1xi64>, tensor<0xi64>, tensor<i32>) -> tensor<1xi64>
  %341 = "tf.StridedSlice"(%318, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %342 = "tf.Mul"(%165, %341) {device = ""} : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
  %343 = "tf.Tile"(%342, %337) {device = ""} : (tensor<?xi64>, tensor<1xi64>) -> tensor<?xi64>
  %344 = "tf.Cumsum"(%343, %14) {device = "", exclusive = false, reverse = false} : (tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %345 = "tf.ConcatV2"(%11, %344, %3) {device = ""} : (tensor<1xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %346 = "tf.Shape"(%345) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %347 = "tf.StridedSlice"(%346, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %348 = "tf.Sub"(%347, %13) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %349 = "tf.Equal"(%348, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %350 = "tf.LogicalOr"(%349, %2) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %351 = "tf.Equal"(%348, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %352 = "tf.LogicalOr"(%350, %351) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %353 = "tf.StridedSlice"(%345, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %354 = "tf.StridedSlice"(%345, %15, %5, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %355 = "tf.Sub"(%353, %354) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %356 = "tf.Shape"(%345) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %357 = "tf.StridedSlice"(%356, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %358 = "tf.Sub"(%357, %13) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %359 = "tf.Equal"(%358, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %360 = "tf.ExpandDims"(%345, %7) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %361 = "tf.Shape"(%345) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %362 = "tf.StridedSlice"(%361, %15, %15, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %363 = "tf.StridedSlice"(%361, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %364 = "tf.StridedSlice"(%361, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %365 = "tf.Select"(%2, %312, %13) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %366 = "tf.Pack"(%365, %13) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %367 = "tf.StridedSlice"(%366, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %368 = "tf.Cast"(%367) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %369 = "tf.Reshape"(%368, %9) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %370 = "tf.Pack"(%7, %369) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %371 = "tf.Tile"(%4, %370) {device = ""} : (tensor<2x1xi64>, tensor<2xi32>) -> tensor<2x?xi64>
  %372 = "tf.Mul"(%369, %8) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %373 = "tf.Pack"(%372) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %374 = "tf.ConcatV2"(%9, %373, %9, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %375 = "tf.Reshape"(%371, %374) {device = ""} : (tensor<2x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %376 = "tf.Shape"(%375) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %377 = "tf.StridedSlice"(%376, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %378 = "tf.Pack"(%367) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %379 = "tf.StridedSlice"(%375, %378, %11, %12) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %380 = "tf.Sub"(%377, %367) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %381 = "tf.Pack"(%380) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %382 = "tf.StridedSlice"(%375, %11, %381, %12) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %383:2 = "tf.RaggedRange"(%382, %379, %13) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %384 = "tf.GatherV2"(%11, %383#1, %14) {batch_dims = 0 : i64, device = ""} : (tensor<1xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %385 = "tf.GatherV2"(%12, %384, %14) {batch_dims = 0 : i64, device = ""} : (tensor<1xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %386 = "tf.StridedSlice"(%366, %15, %16, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi64>
  %387 = "tf.StridedSlice"(%366, %15, %16, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi64>
  %388 = "tf.StridedSlice"(%366, %6, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi64>
  %389 = "tf.ConcatV2"(%387, %388, %14) {device = ""} : (tensor<1xi64>, tensor<0xi64>, tensor<i32>) -> tensor<1xi64>
  %390 = "tf.Tile"(%385, %389) {device = ""} : (tensor<?xi64>, tensor<1xi64>) -> tensor<?xi64>
  %391 = "tf.StridedSlice"(%366, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %392 = "tf.Mul"(%391, %12) {device = ""} : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
  %393 = "tf.Tile"(%392, %386) {device = ""} : (tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %394 = "tf.Cumsum"(%393, %14) {device = "", exclusive = false, reverse = false} : (tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %395 = "tf.ConcatV2"(%11, %394, %3) {device = ""} : (tensor<1xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %396 = "tf.StridedSlice"(%395, %15, %5, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %397 = "tf.ExpandDims"(%396, %7) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %398 = "tf.Shape"(%396) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %399 = "tf.StridedSlice"(%398, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %400 = "tf.Pack"(%399) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %401 = "tf.StridedSlice"(%395, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %402 = "tf.ExpandDims"(%401, %7) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %403 = "tf.Shape"(%401) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %404 = "tf.StridedSlice"(%403, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %405 = "tf.Pack"(%404) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %406 = "tf.Equal"(%144, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %407 = "tf.Select"(%406, %312, %13) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %408 = "tf.Cast"(%407) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %409 = "tf.Reshape"(%408, %9) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %410 = "tf.Pack"(%7, %409) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %411 = "tf.Mul"(%409, %8) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %412 = "tf.Pack"(%411) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %413 = "tf.ConcatV2"(%9, %412, %9, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %414 = "tf.Pack"(%407) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %415 = "tf.Pack"(%10, %144) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %416 = "tf.ExpandDims"(%415, %7) {device = ""} : (tensor<2xi64>, tensor<i32>) -> tensor<2x1xi64>
  %417 = "tf.Tile"(%416, %410) {device = ""} : (tensor<2x1xi64>, tensor<2xi32>) -> tensor<2x?xi64>
  %418 = "tf.Reshape"(%417, %413) {device = ""} : (tensor<2x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %419 = "tf.Shape"(%418) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %420 = "tf.StridedSlice"(%419, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %421 = "tf.Sub"(%420, %407) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %422 = "tf.Pack"(%421) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %423 = "tf.StridedSlice"(%418, %11, %422, %12) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %424 = "tf.StridedSlice"(%418, %414, %11, %12) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %425:2 = "tf.RaggedRange"(%423, %424, %13) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %426 = "tf.GatherV2"(%151, %425#1, %14) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %427 = "tf.Cast"(%426) {Truncate = false, device = ""} : (tensor<?xi64>) -> tensor<?xi32>
  %428 = "tf.BroadcastTo"(%427, %400) {device = ""} : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %429 = "tf.Max"(%428, %15) {device = "", keep_dims = false} : (tensor<?xi32>, tensor<1xi32>) -> tensor<i32>
  %430 = "tf.Maximum"(%14, %429) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %431 = "tf.Range"(%14, %430, %7) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  %432 = "tf.Pack"(%7, %430) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %433 = "tf.Tile"(%397, %432) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %434 = "tf.Shape"(%433) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %435 = "tf.StridedSlice"(%434, %15, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %436 = "tf.Prod"(%435, %15) {device = "", keep_dims = false} : (tensor<2xi32>, tensor<1xi32>) -> tensor<i32>
  %437 = "tf.Pack"(%436) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %438 = "tf.Shape"(%433) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %439 = "tf.StridedSlice"(%438, %15, %15, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %440 = "tf.Shape"(%433) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %441 = "tf.StridedSlice"(%440, %6, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %442 = "tf.ConcatV2"(%439, %437, %441, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %443 = "tf.Reshape"(%433, %442) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %444 = "tf.ExpandDims"(%428, %3) {device = ""} : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
  %445 = "tf.Less"(%431, %444) {device = ""} : (tensor<?xi32>, tensor<?x1xi32>) -> tensor<?x?xi1>
  %446 = "tf.Reshape"(%445, %5) {device = ""} : (tensor<?x?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %447 = "tf.Where"(%446) {device = ""} : (tensor<?xi1>) -> tensor<?x1xi64>
  %448 = "tf.Squeeze"(%447) {device = "", squeeze_dims = [1]} : (tensor<?x1xi64>) -> tensor<?xi64>
  %449 = "tf.GatherV2"(%443, %448, %14) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %450 = "tf.Cast"(%426) {Truncate = false, device = ""} : (tensor<?xi64>) -> tensor<?xi32>
  %451 = "tf.BroadcastTo"(%450, %405) {device = ""} : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %452 = "tf.Max"(%451, %15) {device = "", keep_dims = false} : (tensor<?xi32>, tensor<1xi32>) -> tensor<i32>
  %453 = "tf.Maximum"(%14, %452) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %454 = "tf.Range"(%14, %453, %7) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  %455 = "tf.Pack"(%7, %453) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %456 = "tf.Tile"(%402, %455) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %457 = "tf.Shape"(%456) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %458 = "tf.StridedSlice"(%457, %15, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %459 = "tf.Prod"(%458, %15) {device = "", keep_dims = false} : (tensor<2xi32>, tensor<1xi32>) -> tensor<i32>
  %460 = "tf.Pack"(%459) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %461 = "tf.Shape"(%456) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %462 = "tf.StridedSlice"(%461, %15, %15, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %463 = "tf.Shape"(%456) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %464 = "tf.StridedSlice"(%463, %6, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %465 = "tf.ConcatV2"(%462, %460, %464, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %466 = "tf.Reshape"(%456, %465) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %467 = "tf.ExpandDims"(%451, %3) {device = ""} : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
  %468 = "tf.Less"(%454, %467) {device = ""} : (tensor<?xi32>, tensor<?x1xi32>) -> tensor<?x?xi1>
  %469 = "tf.Reshape"(%468, %5) {device = ""} : (tensor<?x?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %470 = "tf.Where"(%469) {device = ""} : (tensor<?xi1>) -> tensor<?x1xi64>
  %471 = "tf.Squeeze"(%470) {device = "", squeeze_dims = [1]} : (tensor<?x1xi64>) -> tensor<?xi64>
  %472 = "tf.GatherV2"(%466, %471, %14) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %473:2 = "tf.RaggedRange"(%449, %472, %13) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %474 = "tf.GatherV2"(%390, %473#1, %14) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %475 = "tf.If"(%316, %316, %312, %13) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_Assert_1_AssertGuard_false_16380, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_Assert_1_AssertGuard_true_16370} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %476 = "tf.Identity"(%475) {device = ""} : (tensor<i1>) -> tensor<i1>
  %477 = "tf.Select"(%2, %312, %13) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %478 = "tf.Pack"(%477) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %479 = "tf.ConcatV2"(%1, %478, %12, %14) {device = ""} : (tensor<0xi64>, tensor<1xi64>, tensor<1xi64>, tensor<i32>) -> tensor<2xi64>
  %480 = "tf.StridedSlice"(%479, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %481 = "tf.Equal"(%480, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %482 = "tf.StridedSlice"(%479, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %483 = "tf.StridedSlice"(%479, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %484 = "tf.Equal"(%483, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %485 = "tf.If"(%484, %484, %483, %426) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_Assert_2_AssertGuard_false_16870, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_Assert_2_AssertGuard_true_16860} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<?xi64>) -> tensor<i1>
  %486 = "tf.Identity"(%485) {device = ""} : (tensor<i1>) -> tensor<i1>
  %487 = "tf.If"(%481, %481, %426, %482) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_Assert_3_AssertGuard_false_17230, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_Assert_3_AssertGuard_true_17220} : (tensor<i1>, tensor<i1>, tensor<?xi64>, tensor<i64>) -> tensor<i1>
  %488 = "tf.Identity"(%487) {device = ""} : (tensor<i1>) -> tensor<i1>
  %489 = "tf.If"(%352, %352, %13, %348) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_AssertGuard_false_21910, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_AssertGuard_true_21900} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %490 = "tf.Identity"(%489) {device = ""} : (tensor<i1>) -> tensor<i1>
  %491 = "tf.Equal"(%348, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %492 = "tf.Select"(%491, %13, %348) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %493 = "tf.Equal"(%492, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %494 = "tf.LogicalOr"(%493, %2) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %495 = "tf.Equal"(%492, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %496 = "tf.LogicalOr"(%494, %495) {device = ""} : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %497 = "tf.Select"(%359, %492, %13) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %498 = "tf.Pack"(%497, %13) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %499 = "tf.StridedSlice"(%498, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %500 = "tf.Cast"(%499) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %501 = "tf.Reshape"(%500, %9) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %502 = "tf.Pack"(%7, %501) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %503 = "tf.Tile"(%360, %502) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %504 = "tf.Mul"(%501, %363) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %505 = "tf.Pack"(%504) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %506 = "tf.ConcatV2"(%362, %505, %364, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %507 = "tf.Reshape"(%503, %506) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %508 = "tf.Shape"(%507) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %509 = "tf.StridedSlice"(%508, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %510 = "tf.Pack"(%499) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %511 = "tf.StridedSlice"(%507, %510, %11, %12) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %512 = "tf.Sub"(%509, %499) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %513 = "tf.Pack"(%512) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %514 = "tf.StridedSlice"(%507, %11, %513, %12) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %515:2 = "tf.RaggedRange"(%514, %511, %13) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %516 = "tf.Select"(%2, %492, %13) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %517 = "tf.Pack"(%516, %13) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %518 = "tf.StridedSlice"(%517, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %519 = "tf.Cast"(%518) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %520 = "tf.Reshape"(%519, %9) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %521 = "tf.Pack"(%7, %520) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %522 = "tf.Tile"(%4, %521) {device = ""} : (tensor<2x1xi64>, tensor<2xi32>) -> tensor<2x?xi64>
  %523 = "tf.Mul"(%520, %8) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %524 = "tf.Pack"(%523) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %525 = "tf.ConcatV2"(%9, %524, %9, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %526 = "tf.Reshape"(%522, %525) {device = ""} : (tensor<2x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %527 = "tf.Shape"(%526) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %528 = "tf.StridedSlice"(%527, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %529 = "tf.Pack"(%518) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %530 = "tf.StridedSlice"(%526, %529, %11, %12) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %531 = "tf.Sub"(%528, %518) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %532 = "tf.Pack"(%531) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %533 = "tf.StridedSlice"(%526, %11, %532, %12) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %534:2 = "tf.RaggedRange"(%533, %530, %13) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %535 = "tf.StridedSlice"(%517, %15, %16, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi64>
  %536 = "tf.StridedSlice"(%517, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %537 = "tf.Mul"(%536, %12) {device = ""} : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
  %538 = "tf.Tile"(%537, %535) {device = ""} : (tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %539 = "tf.Cumsum"(%538, %14) {device = "", exclusive = false, reverse = false} : (tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %540 = "tf.ConcatV2"(%11, %539, %3) {device = ""} : (tensor<1xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %541 = "tf.StridedSlice"(%540, %15, %5, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %542 = "tf.ExpandDims"(%541, %7) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %543 = "tf.Shape"(%541) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %544 = "tf.StridedSlice"(%543, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %545 = "tf.Pack"(%544) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %546 = "tf.StridedSlice"(%540, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %547 = "tf.ExpandDims"(%546, %7) {device = ""} : (tensor<?xi64>, tensor<i32>) -> tensor<?x1xi64>
  %548 = "tf.Shape"(%546) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %549 = "tf.StridedSlice"(%548, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %550 = "tf.Pack"(%549) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %551 = "tf.Equal"(%348, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %552 = "tf.Select"(%551, %492, %13) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %553 = "tf.Cast"(%552) {Truncate = false, device = ""} : (tensor<i64>) -> tensor<i32>
  %554 = "tf.Reshape"(%553, %9) {device = ""} : (tensor<i32>, tensor<0xi32>) -> tensor<i32>
  %555 = "tf.Pack"(%7, %554) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %556 = "tf.Mul"(%554, %8) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %557 = "tf.Pack"(%556) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %558 = "tf.ConcatV2"(%9, %557, %9, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %559 = "tf.Pack"(%552) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %560 = "tf.Pack"(%10, %348) {axis = 0 : i64, device = ""} : (tensor<i64>, tensor<i64>) -> tensor<2xi64>
  %561 = "tf.ExpandDims"(%560, %7) {device = ""} : (tensor<2xi64>, tensor<i32>) -> tensor<2x1xi64>
  %562 = "tf.Tile"(%561, %555) {device = ""} : (tensor<2x1xi64>, tensor<2xi32>) -> tensor<2x?xi64>
  %563 = "tf.Reshape"(%562, %558) {device = ""} : (tensor<2x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %564 = "tf.Shape"(%563) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %565 = "tf.StridedSlice"(%564, %15, %16, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %566 = "tf.Sub"(%565, %552) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %567 = "tf.Pack"(%566) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %568 = "tf.StridedSlice"(%563, %11, %567, %12) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %569 = "tf.StridedSlice"(%563, %559, %11, %12) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<?xi64>
  %570:2 = "tf.RaggedRange"(%568, %569, %13) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %571 = "tf.GatherV2"(%355, %570#1, %14) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %572 = "tf.Cast"(%571) {Truncate = false, device = ""} : (tensor<?xi64>) -> tensor<?xi32>
  %573 = "tf.BroadcastTo"(%572, %545) {device = ""} : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %574 = "tf.Max"(%573, %15) {device = "", keep_dims = false} : (tensor<?xi32>, tensor<1xi32>) -> tensor<i32>
  %575 = "tf.Maximum"(%14, %574) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %576 = "tf.Range"(%14, %575, %7) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  %577 = "tf.Pack"(%7, %575) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %578 = "tf.Tile"(%542, %577) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %579 = "tf.Shape"(%578) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %580 = "tf.StridedSlice"(%579, %15, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %581 = "tf.Prod"(%580, %15) {device = "", keep_dims = false} : (tensor<2xi32>, tensor<1xi32>) -> tensor<i32>
  %582 = "tf.Pack"(%581) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %583 = "tf.Shape"(%578) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %584 = "tf.StridedSlice"(%583, %15, %15, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %585 = "tf.Shape"(%578) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %586 = "tf.StridedSlice"(%585, %6, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %587 = "tf.ConcatV2"(%584, %582, %586, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %588 = "tf.Reshape"(%578, %587) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %589 = "tf.ExpandDims"(%573, %3) {device = ""} : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
  %590 = "tf.Less"(%576, %589) {device = ""} : (tensor<?xi32>, tensor<?x1xi32>) -> tensor<?x?xi1>
  %591 = "tf.Reshape"(%590, %5) {device = ""} : (tensor<?x?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %592 = "tf.Where"(%591) {device = ""} : (tensor<?xi1>) -> tensor<?x1xi64>
  %593 = "tf.Squeeze"(%592) {device = "", squeeze_dims = [1]} : (tensor<?x1xi64>) -> tensor<?xi64>
  %594 = "tf.GatherV2"(%588, %593, %14) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %595 = "tf.Cast"(%571) {Truncate = false, device = ""} : (tensor<?xi64>) -> tensor<?xi32>
  %596 = "tf.BroadcastTo"(%595, %550) {device = ""} : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  %597 = "tf.Max"(%596, %15) {device = "", keep_dims = false} : (tensor<?xi32>, tensor<1xi32>) -> tensor<i32>
  %598 = "tf.Maximum"(%14, %597) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %599 = "tf.Range"(%14, %598, %7) {device = ""} : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  %600 = "tf.Pack"(%7, %598) {axis = 0 : i64, device = ""} : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  %601 = "tf.Tile"(%547, %600) {device = ""} : (tensor<?x1xi64>, tensor<2xi32>) -> tensor<?x?xi64>
  %602 = "tf.Shape"(%601) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %603 = "tf.StridedSlice"(%602, %15, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %604 = "tf.Prod"(%603, %15) {device = "", keep_dims = false} : (tensor<2xi32>, tensor<1xi32>) -> tensor<i32>
  %605 = "tf.Pack"(%604) {axis = 0 : i64, device = ""} : (tensor<i32>) -> tensor<1xi32>
  %606 = "tf.Shape"(%601) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %607 = "tf.StridedSlice"(%606, %15, %15, %16) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %608 = "tf.Shape"(%601) {device = ""} : (tensor<?x?xi64>) -> tensor<2xi32>
  %609 = "tf.StridedSlice"(%608, %6, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %610 = "tf.ConcatV2"(%607, %605, %609, %14) {device = ""} : (tensor<0xi32>, tensor<1xi32>, tensor<0xi32>, tensor<i32>) -> tensor<1xi32>
  %611 = "tf.Reshape"(%601, %610) {device = ""} : (tensor<?x?xi64>, tensor<1xi32>) -> tensor<?xi64>
  %612 = "tf.ExpandDims"(%596, %3) {device = ""} : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
  %613 = "tf.Less"(%599, %612) {device = ""} : (tensor<?xi32>, tensor<?x1xi32>) -> tensor<?x?xi1>
  %614 = "tf.Reshape"(%613, %5) {device = ""} : (tensor<?x?xi1>, tensor<1xi32>) -> tensor<?xi1>
  %615 = "tf.Where"(%614) {device = ""} : (tensor<?xi1>) -> tensor<?x1xi64>
  %616 = "tf.Squeeze"(%615) {device = "", squeeze_dims = [1]} : (tensor<?x1xi64>) -> tensor<?xi64>
  %617 = "tf.GatherV2"(%611, %616, %14) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %618:2 = "tf.RaggedRange"(%594, %617, %13) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %619 = "tf.If"(%496, %496, %492, %13) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_1_AssertGuard_false_22980, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_1_AssertGuard_true_22970} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
  %620 = "tf.Identity"(%619) {device = ""} : (tensor<i1>) -> tensor<i1>
  %621 = "tf.Select"(%2, %492, %13) {device = ""} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  %622 = "tf.Pack"(%621) {axis = 0 : i64, device = ""} : (tensor<i64>) -> tensor<1xi64>
  %623 = "tf.ConcatV2"(%1, %622, %12, %14) {device = ""} : (tensor<0xi64>, tensor<1xi64>, tensor<1xi64>, tensor<i32>) -> tensor<2xi64>
  %624 = "tf.StridedSlice"(%623, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %625 = "tf.Equal"(%624, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %626 = "tf.StridedSlice"(%623, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %627 = "tf.StridedSlice"(%623, %16, %6, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %628 = "tf.Equal"(%627, %13) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %629 = "tf.If"(%628, %628, %627, %571) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_2_AssertGuard_false_23470, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_2_AssertGuard_true_23460} : (tensor<i1>, tensor<i1>, tensor<i64>, tensor<?xi64>) -> tensor<i1>
  %630 = "tf.Identity"(%629) {device = ""} : (tensor<i1>) -> tensor<i1>
  %631 = "tf.If"(%625, %625, %571, %626) {_lower_using_switch_merge = true, _read_only_resource_inputs = [], device = "", else_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_3_AssertGuard_false_23830, is_stateless = false, output_shapes = [#tf_type.shape<>], then_branch = @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_3_AssertGuard_true_23820} : (tensor<i1>, tensor<i1>, tensor<?xi64>, tensor<i64>) -> tensor<i1>
  %632 = "tf.Identity"(%631) {device = ""} : (tensor<i1>) -> tensor<i1>
  %633 = "tf.Identity"(%308) {device = ""} : (tensor<i1>) -> tensor<i1>
  %634 = "tf.Shape"(%37#2) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %635 = "tf.StridedSlice"(%634, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %636 = "tf.Cast"(%635) {Truncate = false, device = ""} : (tensor<0xi32>) -> tensor<0xi64>
  %637 = "tf.Identity"(%636) {device = ""} : (tensor<0xi64>) -> tensor<0xi64>
  %638 = "tf.Shape"(%37#3) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %639 = "tf.StridedSlice"(%638, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %640 = "tf.Cast"(%639) {Truncate = false, device = ""} : (tensor<0xi32>) -> tensor<0xi64>
  %641 = "tf.Identity"(%640) {device = ""} : (tensor<0xi64>) -> tensor<0xi64>
  %642 = "tf.GatherV2"(%37#3, %336, %14) {batch_dims = 0 : i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i32>) -> tensor<?xi64>
  %643 = "tf.Tile"(%642, %340) {device = ""} : (tensor<?xi64>, tensor<1xi64>) -> tensor<?xi64>
  %644 = "tf.Sub"(%643, %474) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %645 = "tf.Shape"(%644) {device = ""} : (tensor<?xi64>) -> tensor<1xi32>
  %646 = "tf.StridedSlice"(%645, %16, %15, %16) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<0xi32>
  %647 = "tf.Cast"(%646) {Truncate = false, device = ""} : (tensor<0xi32>) -> tensor<0xi64>
  %648 = "tf.Identity"(%647) {device = ""} : (tensor<0xi64>) -> tensor<0xi64>
  %649 = "tf.UnicodeEncode"(%37#0, %58) {Tsplits = i64, device = "", errors = "replace", output_encoding = "UTF-8", replacement_char = 65533 : i64} : (tensor<?xi32>, tensor<?xi64>) -> tensor<?x!tf_type.string>
  %650 = "tf.Identity"(%649) {device = ""} : (tensor<?x!tf_type.string>) -> tensor<?x!tf_type.string>
  func.return %650 : tensor<?x!tf_type.string>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedConcat_assert_equal_1_Assert_AssertGuard_false_3220(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Input tensors have incompatible shapes."> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/WhitespaceTokenize/RaggedConcat/RaggedFromTensor/Const:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/WhitespaceTokenize/RaggedConcat/RaggedNRows/Const:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedConcat_assert_equal_1_Assert_AssertGuard_true_3210(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_3980(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:zero"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits/RowPartitionFromRowSplits/Const:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_3970(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_4340(%arg0: tensor<i1>, %arg1: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:monotonic"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x >= 0 did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits/RowPartitionFromRowSplits/sub:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<?xi64>) -> ()
  %3 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %4 = "tf.Identity"(%3) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %4 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_4330(%arg0: tensor<i1>, %arg1: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_false_4680(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to _from_row_partition do not form a valid RaggedTensor"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits/strided_slice_1:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_true_4670(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_5050(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:zero"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits_1/RowPartitionFromRowSplits/Const:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_5040(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_5410(%arg0: tensor<i1>, %arg1: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:monotonic"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x >= 0 did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits_1/RowPartitionFromRowSplits/sub:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<?xi64>) -> ()
  %3 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %4 = "tf.Identity"(%3) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %4 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_5400(%arg0: tensor<i1>, %arg1: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_false_5770(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to _from_row_partition do not form a valid RaggedTensor"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits_1/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits/RaggedFromRowSplits_1/RaggedNRows/sub:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_RaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_true_5760(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_6120(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:zero"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits_1/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits_1/RaggedFromRowSplits/RowPartitionFromRowSplits/Const:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_6110(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_6480(%arg0: tensor<i1>, %arg1: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:monotonic"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x >= 0 did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits_1/RaggedFromRowSplits/RowPartitionFromRowSplits/sub:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<?xi64>) -> ()
  %3 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %4 = "tf.Identity"(%3) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %4 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_6470(%arg0: tensor<i1>, %arg1: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_false_6820(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to _from_row_partition do not form a valid RaggedTensor"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits_1/RaggedFromRowSplits/strided_slice_1:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits_1/RaggedFromRowSplits/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_1_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_true_6810(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_7190(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:zero"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits_2/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits_2/RaggedFromRowSplits/RowPartitionFromRowSplits/Const:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_7180(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_7550(%arg0: tensor<i1>, %arg1: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:monotonic"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x >= 0 did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits_2/RaggedFromRowSplits/RowPartitionFromRowSplits/sub:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<?xi64>) -> ()
  %3 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %4 = "tf.Identity"(%3) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %4 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_7540(%arg0: tensor<i1>, %arg1: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_false_7890(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to _from_row_partition do not form a valid RaggedTensor"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits_2/RaggedFromRowSplits/strided_slice_1:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (WhitespaceTokenize/WhitespaceTokenize/RaggedFromNestedRowSplits_2/RaggedFromRowSplits/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedFromNestedRowSplits_2_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_true_7880(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_AssertGuard_false_8690(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_AssertGuard_true_8680(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_1_AssertGuard_false_9760(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_1_AssertGuard_true_9750(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_2_AssertGuard_false_10250(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<?>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<?xi64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_2_AssertGuard_true_10240(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<?>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_3_AssertGuard_false_10610(%arg0: tensor<i1>, %arg1: tensor<?xi64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<?xi64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_Assert_3_AssertGuard_true_10600(%arg0: tensor<i1>, %arg1: tensor<?xi64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_Assert_AssertGuard_false_15310(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_Assert_AssertGuard_true_15300(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_Assert_1_AssertGuard_false_16380(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_Assert_1_AssertGuard_true_16370(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_Assert_2_AssertGuard_false_16870(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<?>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<?xi64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_Assert_2_AssertGuard_true_16860(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<?>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_Assert_3_AssertGuard_false_17230(%arg0: tensor<i1>, %arg1: tensor<?xi64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<?xi64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_Assert_3_AssertGuard_true_17220(%arg0: tensor<i1>, %arg1: tensor<?xi64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_AssertGuard_false_21910(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_AssertGuard_true_21900(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_1_AssertGuard_false_22980(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_1_AssertGuard_true_22970(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_2_AssertGuard_false_23470(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<?>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<?xi64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_2_AssertGuard_true_23460(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<?>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_3_AssertGuard_false_23830(%arg0: tensor<i1>, %arg1: tensor<?xi64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Unable to broadcast: dimension size mismatch in dimension"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.Const"() {value = dense<"lengths="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"dim_size="> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 10 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<i32>, tensor<!tf_type.string>, tensor<?xi64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func @WhitespaceTokenize_WhitespaceTokenize_RaggedGather_1_Assert_3_AssertGuard_true_23820(%arg0: tensor<i1>, %arg1: tensor<?xi64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}

// CHECK: func private @whitespace_tokenizer_rank0(%arg0: tensor<!tf_type.string> {tf._user_specified_name = "input"}) -> tensor<?x!tf_type.string> attributes {tf._implements = #tf_type.func<@"tftext:WhitespaceTokenizer", {}>, tf._input_shapes = [#tf_type.shape<>], tf.signature.is_stateful} {
// CHECK: %0 = "tfl.custom"(%arg0) {custom_code = "tftext:WhitespaceTokenizer", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>} : (tensor<!tf_type.string>) -> tensor<?x!tf_type.string>
// CHECK: return %0 : tensor<?x!tf_type.string>

func.func @ngrams(%arg0: tensor<?x!tf_type.string> {tf._user_specified_name = "input"}) -> tensor<?x!tf_type.string> attributes {tf._input_shapes = [#tf_type.shape<?>], tf._implements = #tf_type.func<@"tftext:Ngrams", {axis = -1 : i64, reduction_type = "STRING_JOIN", string_separator = " ", width = 2 : i64}>} {
  %0 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Const"() {value = dense<[0, -1]> : tensor<2xi32>} : () -> tensor<2xi32>
  %2 = "tf.Const"() {value = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  %3 = "tf.Const"() {value = dense<0> : tensor<2xi32>} : () -> tensor<2xi32>
  %4 = "tf.Const"() {value = dense<1> : tensor<2xi32>} : () -> tensor<2xi32>
  %5 = "tf.StridedSlice"(%arg0, %3, %1, %4) {begin_mask = 0 : i64, device = "", ellipsis_mask = 1 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?x!tf_type.string>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x!tf_type.string>
  %6 = "tf.StridedSlice"(%arg0, %2, %3, %4) {begin_mask = 0 : i64, device = "", ellipsis_mask = 1 : i64, end_mask = 2 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?x!tf_type.string>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<?x!tf_type.string>
  %7 = "tf.Pack"(%5, %6) {axis = -1 : i64, device = ""} : (tensor<?x!tf_type.string>, tensor<?x!tf_type.string>) -> tensor<?x2x!tf_type.string>
  %8 = "tf.ReduceJoin"(%7, %0) {device = "", keep_dims = false, separator = " "} : (tensor<?x2x!tf_type.string>, tensor<i32>) -> tensor<?x!tf_type.string>
  %9 = "tf.Identity"(%8) {device = ""} : (tensor<?x!tf_type.string>) -> tensor<?x!tf_type.string>
  func.return %9 : tensor<?x!tf_type.string>
}

// CHECK: func @ngrams(%arg0: tensor<?x!tf_type.string> {tf._user_specified_name = "input"}) -> tensor<?x!tf_type.string> attributes {tf._implements = #tf_type.func<@"tftext:Ngrams", {axis = -1 : i64, reduction_type = "STRING_JOIN", string_separator = " ", width = 2 : i64}>, tf._input_shapes = [#tf_type.shape<?>]} {
// CHECK:   %0 = "tfl.custom"(%arg0) {custom_code = "tftext:Ngrams", custom_option = opaque<"tfl", "0x776964746800737472696E675F736570617261746F72000120006178697300726564756374696F6E5F74797065000B535452494E475F4A4F494E0004221E383F040104FF152D0204141404082401"> : tensor<78xi8>} : (tensor<?x!tf_type.string>) -> tensor<?x!tf_type.string>
// CHECK:   return %0 : tensor<?x!tf_type.string>
// CHECK: }

func.func private @ngrams_ragged_rank_2(%arg0: tensor<?x!tf_type.string> {tf._user_specified_name = "values"}, %arg1: tensor<3xi64> {tf._user_specified_name = "args_0"}, %arg2: tensor<?xi64> {tf._user_specified_name = "args_1"}) -> (tensor<?x!tf_type.string>, tensor<3xi64>, tensor<?xi64>) attributes {tf._implements = #tf_type.func<@"tftext:Ngrams", {axis = -1 : i64, reduction_type = "STRING_JOIN", string_separator = "", width = 2 : i64}>, tf._input_shapes = [#tf_type.shape<?>, #tf_type.shape<3>, #tf_type.shape<?>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Const"() {value = dense<-1> : tensor<i64>} : () -> tensor<i64>
  %2 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %3 = "tf.Const"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
  %4 = "tf.Const"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  %5 = "tf.Const"() {value = dense<> : tensor<0xi32>} : () -> tensor<0xi32>
  %6 = "tf.Const"() {value = dense<-1> : tensor<1xi32>} : () -> tensor<1xi32>
  %7 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %8 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  %9 = "tf.StridedSlice"(%arg1, %7, %8, %8) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<3xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %10 = "tf.Equal"(%9, %4) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %11 = "tf.All"(%10, %5) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %12 = "tf.StridedSlice"(%arg1, %8, %7, %8) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<3xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi64>
  %13 = "tf.StridedSlice"(%arg1, %7, %6, %8) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<3xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<2xi64>
  %14 = "tf.Sub"(%12, %13) {device = ""} : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi64>
  %15 = "tf.LessEqual"(%4, %14) {device = ""} : (tensor<i64>, tensor<2xi64>) -> tensor<2xi1>
  %16 = "tf.All"(%15, %7) {device = "", keep_dims = false} : (tensor<2xi1>, tensor<1xi32>) -> tensor<i1>
  %17 = "tf.StridedSlice"(%arg2, %7, %8, %8) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %18 = "tf.Equal"(%17, %4) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %19 = "tf.All"(%18, %5) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %20 = "tf.IfRegion"(%19) ({
    %72 = "func.call"(%19, %17, %4) {callee = @RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_27770} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
    "tf.Yield"(%72) : (tensor<i1>) -> ()
  },  {
    %72 = "func.call"(%19, %17, %4) {callee = @RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_27780} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
    "tf.Yield"(%72) : (tensor<i1>) -> ()
  }) {is_stateless = false} : (tensor<i1>) -> tensor<i1>
  %21 = "tf.Identity"(%20) {device = ""} : (tensor<i1>) -> tensor<i1>
  %22 = "tf.StridedSlice"(%arg2, %8, %7, %8) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %23 = "tf.StridedSlice"(%arg2, %7, %6, %8) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %24 = "tf.Sub"(%22, %23) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %25 = "tf.LessEqual"(%4, %24) {device = ""} : (tensor<i64>, tensor<?xi64>) -> tensor<?xi1>
  %26 = "tf.All"(%25, %7) {device = "", keep_dims = false} : (tensor<?xi1>, tensor<1xi32>) -> tensor<i1>
  %27 = "tf.IfRegion"(%26) ({
    %72 = "func.call"(%26, %24) {callee = @RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_28130} : (tensor<i1>, tensor<?xi64>) -> tensor<i1>
    "tf.Yield"(%72) : (tensor<i1>) -> ()
  },  {
    %72 = "func.call"(%26, %24) {callee = @RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_28140} : (tensor<i1>, tensor<?xi64>) -> tensor<i1>
    "tf.Yield"(%72) : (tensor<i1>) -> ()
  }) {is_stateless = false} : (tensor<i1>) -> tensor<i1>
  %28 = "tf.Identity"(%27) {device = ""} : (tensor<i1>) -> tensor<i1>
  %29 = "tf.Identity"(%arg2) {_class = ["loc:@args_1"], device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %30 = "tf.StridedSlice"(%29, %6, %7, %8) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %31 = "tf.Shape"(%arg0) {device = ""} : (tensor<?x!tf_type.string>) -> tensor<1xi64>
  %32 = "tf.StridedSlice"(%31, %7, %8, %8) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %33 = "tf.Equal"(%30, %32) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %34 = "tf.All"(%33, %5) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %35 = "tf.IfRegion"(%34) ({
    %72 = "func.call"(%34, %30, %32) {callee = @RaggedFromNestedRowSplits_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_true_28500} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
    "tf.Yield"(%72) : (tensor<i1>) -> ()
  },  {
    %72 = "func.call"(%34, %30, %32) {callee = @RaggedFromNestedRowSplits_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_false_28510} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
    "tf.Yield"(%72) : (tensor<i1>) -> ()
  }) {is_stateless = false} : (tensor<i1>) -> tensor<i1>
  %36 = "tf.Identity"(%35) {device = ""} : (tensor<i1>) -> tensor<i1>
  %37 = "tf.Identity"(%29) {_class = ["loc:@args_1"], device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %38 = "tf.StridedSlice"(%37, %7, %6, %8) {begin_mask = 1 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %39 = "tf.StridedSlice"(%37, %8, %7, %8) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 1 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<?xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi64>
  %40 = "tf.Minimum"(%38, %39) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %41 = "tf.AddV2"(%39, %1) {device = ""} : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
  %42 = "tf.Maximum"(%41, %38) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %43:2 = "tf.RaggedRange"(%40, %42, %3) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %44 = "tf.GatherV2"(%arg0, %43#1, %2) {batch_dims = 0 : i64, device = ""} : (tensor<?x!tf_type.string>, tensor<?xi64>, tensor<i32>) -> tensor<?x!tf_type.string>
  %45 = "tf.AddV2"(%38, %3) {device = ""} : (tensor<?xi64>, tensor<i64>) -> tensor<?xi64>
  %46 = "tf.Minimum"(%45, %39) {device = ""} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi64>
  %47:2 = "tf.RaggedRange"(%46, %39, %3) {T = i64, Tsplits = i64, device = ""} : (tensor<?xi64>, tensor<?xi64>, tensor<i64>) -> (tensor<?xi64>, tensor<?xi64>)
  %48 = "tf.Equal"(%43#0, %47#0) {device = "", incompatible_shape_error = true} : (tensor<?xi64>, tensor<?xi64>) -> tensor<?xi1>
  %49 = "tf.All"(%48, %7) {device = "", keep_dims = false} : (tensor<?xi1>, tensor<1xi32>) -> tensor<i1>
  %50 = "tf.GatherV2"(%arg0, %47#1, %2) {batch_dims = 0 : i64, device = ""} : (tensor<?x!tf_type.string>, tensor<?xi64>, tensor<i32>) -> tensor<?x!tf_type.string>
  %51 = "tf.Shape"(%37) {device = ""} : (tensor<?xi64>) -> tensor<1xi64>
  %52 = "tf.StridedSlice"(%51, %7, %8, %8) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<1xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %53 = "tf.Sub"(%52, %3) {device = ""} : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %54 = "tf.IfRegion"(%11) ({
    %72 = "func.call"(%11, %9, %4) {callee = @RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_28900} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
    "tf.Yield"(%72) : (tensor<i1>) -> ()
  },  {
    %72 = "func.call"(%11, %9, %4) {callee = @RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_28910} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
    "tf.Yield"(%72) : (tensor<i1>) -> ()
  }) {is_stateless = false} : (tensor<i1>) -> tensor<i1>
  %55 = "tf.Identity"(%54) {device = ""} : (tensor<i1>) -> tensor<i1>
  %56 = "tf.IfRegion"(%16) ({
    %72 = "func.call"(%16, %14) {callee = @RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_29260} : (tensor<i1>, tensor<2xi64>) -> tensor<i1>
    "tf.Yield"(%72) : (tensor<i1>) -> ()
  },  {
    %72 = "func.call"(%16, %14) {callee = @RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_29270} : (tensor<i1>, tensor<2xi64>) -> tensor<i1>
    "tf.Yield"(%72) : (tensor<i1>) -> ()
  }) {is_stateless = false} : (tensor<i1>) -> tensor<i1>
  %57 = "tf.Identity"(%56) {device = ""} : (tensor<i1>) -> tensor<i1>
  %58 = "tf.Identity"(%arg1) {_class = ["loc:@args_0"], device = ""} : (tensor<3xi64>) -> tensor<3xi64>
  %59 = "tf.StridedSlice"(%58, %6, %7, %8) {begin_mask = 0 : i64, device = "", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<3xi64>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i64>
  %60 = "tf.Equal"(%59, %53) {device = "", incompatible_shape_error = true} : (tensor<i64>, tensor<i64>) -> tensor<i1>
  %61 = "tf.All"(%60, %5) {device = "", keep_dims = false} : (tensor<i1>, tensor<0xi32>) -> tensor<i1>
  %62 = "tf.IfRegion"(%61) ({
    %72 = "func.call"(%61, %59, %53) {callee = @RaggedFromNestedRowSplits_RaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_true_29650} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
    "tf.Yield"(%72) : (tensor<i1>) -> ()
  },  {
    %72 = "func.call"(%61, %59, %53) {callee = @RaggedFromNestedRowSplits_RaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_false_29660} : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i1>
    "tf.Yield"(%72) : (tensor<i1>) -> ()
  }) {is_stateless = false} : (tensor<i1>) -> tensor<i1>
  %63 = "tf.IfRegion"(%49) ({
    %72 = "func.call"(%49, %43#0, %47#0) {callee = @NGrams_SlidingWindow_RaggedConcat_assert_equal_2_Assert_AssertGuard_true_30330} : (tensor<i1>, tensor<?xi64>, tensor<?xi64>) -> tensor<i1>
    "tf.Yield"(%72) : (tensor<i1>) -> ()
  },  {
    %72 = "func.call"(%49, %43#0, %47#0) {callee = @NGrams_SlidingWindow_RaggedConcat_assert_equal_2_Assert_AssertGuard_false_30340} : (tensor<i1>, tensor<?xi64>, tensor<?xi64>) -> tensor<i1>
    "tf.Yield"(%72) : (tensor<i1>) -> ()
  }) {is_stateless = false} : (tensor<i1>) -> tensor<i1>
  %64 = "tf.Identity"(%43#0) {device = ""} : (tensor<?xi64>) -> tensor<?xi64>
  %65 = "tf.Identity"(%63) {device = ""} : (tensor<i1>) -> tensor<i1>
  %66 = "tf.Pack"(%44, %50) {axis = 1 : i64, device = ""} : (tensor<?x!tf_type.string>, tensor<?x!tf_type.string>) -> tensor<?x2x!tf_type.string>
  %67 = "tf.ReduceJoin"(%66, %0) {device = "", keep_dims = false, separator = ""} : (tensor<?x2x!tf_type.string>, tensor<i32>) -> tensor<?x!tf_type.string>
  %68 = "tf.Identity"(%67) {device = ""} : (tensor<?x!tf_type.string>) -> tensor<?x!tf_type.string>
  %69 = "tf.Identity"(%62) {device = ""} : (tensor<i1>) -> tensor<i1>
  %70 = "tf.Identity"(%58) {_class = ["loc:@args_0"], device = ""} : (tensor<3xi64>) -> tensor<3xi64>
  %71 = "tf.Identity"(%70) {device = ""} : (tensor<3xi64>) -> tensor<3xi64>
  func.return %68, %71, %64 : tensor<?x!tf_type.string>, tensor<3xi64>, tensor<?xi64>
}
func.func private @RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_27770(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func private @RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_27780(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:zero"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (RaggedFromNestedRowSplits/RaggedFromRowSplits/RowPartitionFromRowSplits/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (RaggedFromNestedRowSplits/RaggedFromRowSplits/RowPartitionFromRowSplits/Const:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func private @RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_28130(%arg0: tensor<i1>, %arg1: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func private @RaggedFromNestedRowSplits_RaggedFromRowSplits_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_28140(%arg0: tensor<i1>, %arg1: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:monotonic"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x >= 0 did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (RaggedFromNestedRowSplits/RaggedFromRowSplits/RowPartitionFromRowSplits/sub:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<?xi64>) -> ()
  %3 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %4 = "tf.Identity"(%3) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %4 : tensor<i1>
}
func.func private @RaggedFromNestedRowSplits_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_true_28500(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func private @RaggedFromNestedRowSplits_RaggedFromRowSplits_assert_equal_1_Assert_AssertGuard_false_28510(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to _from_row_partition do not form a valid RaggedTensor"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (RaggedFromNestedRowSplits/RaggedFromRowSplits/strided_slice_1:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (RaggedFromNestedRowSplits/RaggedFromRowSplits/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
"tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func private @RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_true_28900(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func private @RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_equal_1_Assert_AssertGuard_false_28910(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:zero"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (RaggedFromNestedRowSplits/RaggedFromRowSplits_1/RowPartitionFromRowSplits/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (RaggedFromNestedRowSplits/RaggedFromRowSplits_1/RowPartitionFromRowSplits/Const:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func private @RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_true_29260(%arg0: tensor<i1>, %arg1: tensor<2xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<2>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func private @RaggedFromNestedRowSplits_RaggedFromRowSplits_1_RowPartitionFromRowSplits_assert_non_negative_assert_less_equal_Assert_AssertGuard_false_29270(%arg0: tensor<i1>, %arg1: tensor<2xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<2>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to from_row_splits do not form a valid RaggedTensor:monotonic"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x >= 0 did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (RaggedFromNestedRowSplits/RaggedFromRowSplits_1/RowPartitionFromRowSplits/sub:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<2xi64>) -> ()
  %3 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %4 = "tf.Identity"(%3) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %4 : tensor<i1>
}
func.func private @RaggedFromNestedRowSplits_RaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_true_29650(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func private @RaggedFromNestedRowSplits_RaggedFromRowSplits_1_assert_equal_1_Assert_AssertGuard_false_29660(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Arguments to _from_row_partition do not form a valid RaggedTensor"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (RaggedFromNestedRowSplits/RaggedFromRowSplits_1/strided_slice:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (RaggedFromNestedRowSplits/RaggedFromRowSplits_1/RaggedNRows/sub:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>, tensor<!tf_type.string>, tensor<i64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
func.func private @NGrams_SlidingWindow_RaggedConcat_assert_equal_2_Assert_AssertGuard_true_30330(%arg0: tensor<i1>, %arg1: tensor<?xi64>, %arg2: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>, #tf_type.shape<?>]} {
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %1 = "tf.Identity"(%0) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %1 : tensor<i1>
}
func.func private @NGrams_SlidingWindow_RaggedConcat_assert_equal_2_Assert_AssertGuard_false_30340(%arg0: tensor<i1>, %arg1: tensor<?xi64>, %arg2: tensor<?xi64>) -> tensor<i1> attributes {tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<?>, #tf_type.shape<?>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<"Inputs must have identical ragged splits"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %1 = "tf.Const"() {value = dense<"Condition x == y did not hold element-wise:"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %2 = "tf.Const"() {value = dense<"x (NGrams/SlidingWindow/RaggedGetItem/RaggedRange:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  %3 = "tf.Const"() {value = dense<"y (NGrams/SlidingWindow/RaggedGetItem_1/RaggedRange:0) = "> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
  "tf.Assert"(%arg0, %0, %1, %2, %arg1, %3, %arg2) {device = "", summarize = 3 : i64} : (tensor<i1>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<?xi64>, tensor<!tf_type.string>, tensor<?xi64>) -> ()
  %4 = "tf.Identity"(%arg0) {device = ""} : (tensor<i1>) -> tensor<i1>
  %5 = "tf.Identity"(%4) {device = ""} : (tensor<i1>) -> tensor<i1>
  func.return %5 : tensor<i1>
}
// CHECK:  func private @ngrams_ragged_rank_2(%arg0: tensor<?x!tf_type.string> {tf._user_specified_name = "values"}, %arg1: tensor<3xi64> {tf._user_specified_name = "args_0"}, %arg2: tensor<?xi64> {tf._user_specified_name = "args_1"}) -> (tensor<?x!tf_type.string>, tensor<3xi64>, tensor<?xi64>) attributes {tf._implements = #tf_type.func<@"tftext:Ngrams", {axis = -1 : i64, reduction_type = "STRING_JOIN", string_separator = "", width = 2 : i64}>, tf._input_shapes = [#tf_type.shape<?>, #tf_type.shape<3>, #tf_type.shape<?>], tf.signature.is_stateful} {
// CHECK:    %0:3 = "tfl.custom"(%arg0, %arg1, %arg2) {custom_code = "tftext:Ngrams", custom_option = opaque<"tfl", "0x776964746800737472696E675F736570617261746F720000006178697300726564756374696F6E5F74797065000B535452494E475F4A4F494E0004221E373E040104FF152C0204141404082401"> : tensor<77xi8>} : (tensor<?x!tf_type.string>, tensor<3xi64>, tensor<?xi64>) -> (tensor<?x!tf_type.string>, tensor<3xi64>, tensor<?xi64>)
// CHECK:    return %0#0, %0#1, %0#2 : tensor<?x!tf_type.string>, tensor<3xi64>, tensor<?xi64>


func.func private @sgnn_projection(%arg0: tensor<?x!tf_type.string> {tf._user_specified_name = "values"}, %arg1: tensor<?xi64> {tf._user_specified_name = "row_splits"}) -> tensor<?x10xf64> attributes {tf._implements = #tf_type.func<@"tftext:custom:SgnnProjection", {buckets = 2147483647 : i64, hash_seed = [1902835825, -1475704015, 473120514, 1254202069, 1558833093, 1756181982, 1906603252, -1034142694, 542842690, 535515822]}>, tf._input_shapes = [#tf_type.shape<?>, #tf_type.shape<?>], tf.signature.is_stateful} {
  %0 = "tf.Const"() {value = dense<[[1902835825], [-1475704015], [473120514], [1254202069], [1558833093], [1756181982], [1906603252], [-1034142694], [542842690], [535515822]]> : tensor<10x1xi64>} : () -> tensor<10x1xi64>
  %1 = "tf.StringToHashBucketFast"(%arg0) {device = "", num_buckets = 2147483647 : i64} : (tensor<?x!tf_type.string>) -> tensor<?xi64>
  %2 = "tf.Sgnn"(%1, %0) {device = ""} : (tensor<?xi64>, tensor<10x1xi64>) -> tensor<10x?xf64>
  %3 = "tf.Const"() {value = dense<[-1, 10]> : tensor<2xi64>} : () -> tensor<2xi64>
  %4 = "tf.Reshape"(%2, %3) : (tensor<10x?xf64>, tensor<2xi64>) -> tensor<?x10xf64>
  func.return %4 : tensor<?x10xf64>
}


// CHECK: func private @sgnn_projection(%arg0: tensor<?x!tf_type.string> {tf._user_specified_name = "values"}, %arg1: tensor<?xi64> {tf._user_specified_name = "row_splits"}) -> tensor<?x10xf64> attributes {tf._implements = #tf_type.func<@"tftext:custom:SgnnProjection", {buckets = 2147483647 : i64, hash_seed = [1902835825, -1475704015, 473120514, 1254202069, 1558833093, 1756181982, 1906603252, -1034142694, 542842690, 535515822]}>, tf._input_shapes = [#tf_type.shape<?>, #tf_type.shape<?>], tf.signature.is_stateful} {
// CHECK:   %0 = "tfl.custom"(%arg0, %arg1) {custom_code = "tftext:custom:SgnnProjection", custom_option = opaque<"tfl", "0x686173685F736565640000000A00000071F86A71318B0AA8023F331CD59AC14AC5E7E95CDE35AD68F474A4711A3C5CC2421F5B20AE52EB1F6275636B6574730002094200030000000100000002000000FFFFFF7F44000000062E0A2601"> : tensor<93xi8>} : (tensor<?x!tf_type.string>, tensor<?xi64>) -> tensor<?x10xf64>
// CHECK:   return %0 : tensor<?x10xf64>
