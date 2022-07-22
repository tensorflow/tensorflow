// RUN: tf-tfrt-opt -tfrt-test-cost-analysis -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: test_cheap_ops_0
func.func @test_cheap_ops_0(%arg: tensor<?x!tf_type.string>) -> (tensor<?x8xf32>) {
    // expected-remark@+1 {{Cost: 1}}
    %0 = "tf.Const"() {value = dense<> : tensor<0xi64>} : () -> tensor<0xi64>
    // expected-remark@+1 {{Cost: 1}}
    %1 = "tf.Const"() {value = dense<"has_login_page_feature"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
    // expected-remark@+1 {{Cost: 1}}
    %2 = "tf.Const"() {value = dense<"num_terms_inside_postform"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
    // expected-remark@+1 {{Cost: 1}}
    %3 = "tf.Const"() {value = dense<"num_terms_outside_postform"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
    // expected-remark@+1 {{Cost: 1}}
    %4 = "tf.Const"() {value = dense<"num_terms_outside_postform_without_bp"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
    // expected-remark@+1 {{Cost: 1}}
    %5 = "tf.Const"() {value = dense<"password_not_in_bp_area"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
    // expected-remark@+1 {{Cost: 1}}
    %6 = "tf.Const"() {value = dense<"query_params_contains_url"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
    // expected-remark@+1 {{Cost: 1}}
    %7 = "tf.Const"() {value = dense<"title_with_login_phase"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
    // expected-remark@+1 {{Cost: 1}}
    %8 = "tf.Const"() {value = dense<"url_contains_login_terms"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
    // expected-remark@+1 {{Cost: 1}}
    %9 = "tf.Const"() {value = dense<> : tensor<0x!tf_type.string>} : () -> tensor<0x!tf_type.string>
    // expected-remark@+1 {{Cost: 1}}
    %10 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    // expected-remark@+1 {{Cost: 1}}
    %11 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
    // expected-remark@+1 {{Cost: 19}}
    %dense_values:8 = "tf.ParseExample"(%arg, %9, %1, %2, %3, %4, %5, %6, %7, %8, %0, %0, %0, %0, %0, %0, %0, %0) {dense_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>], device = "/job:localhost/replica:0/task:0/device:CPU:0", operand_segment_sizes = dense<[1, 1, 0, 8, 8]> : vector<5xi32>, result_segment_sizes = dense<[0, 0, 0, 8]> : vector<4xi32>} : (tensor<?x!tf_type.string>, tensor<0x!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<0xi64>, tensor<0xi64>, tensor<0xi64>, tensor<0xi64>, tensor<0xi64>, tensor<0xi64>, tensor<0xi64>, tensor<0xi64>) -> (tensor<?xi64>, tensor<?xi64>, tensor<?xi64>, tensor<?xi64>, tensor<?xi64>, tensor<?xi64>, tensor<?xi64>, tensor<?xi64>)
    // expected-remark@+1 {{Cost: 2}}
    %28 = "tf.Cast"(%dense_values#0) {Truncate = false, device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?xi64>) -> tensor<?xf32>
    // expected-remark@+1 {{Cost: 2}}
    %29 = "tf.Cast"(%dense_values#1) {Truncate = false, device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?xi64>) -> tensor<?xf32>
    // expected-remark@+1 {{Cost: 2}}
    %30 = "tf.Cast"(%dense_values#2) {Truncate = false, device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?xi64>) -> tensor<?xf32>
    // expected-remark@+1 {{Cost: 2}}
    %31 = "tf.Cast"(%dense_values#3) {Truncate = false, device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?xi64>) -> tensor<?xf32>
    // expected-remark@+1 {{Cost: 2}}
    %32 = "tf.Cast"(%dense_values#4) {Truncate = false, device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?xi64>) -> tensor<?xf32>
    // expected-remark@+1 {{Cost: 2}}
    %33 = "tf.Cast"(%dense_values#5) {Truncate = false, device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?xi64>) -> tensor<?xf32>
    // expected-remark@+1 {{Cost: 2}}
    %34 = "tf.Cast"(%dense_values#6) {Truncate = false, device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?xi64>) -> tensor<?xf32>
    // expected-remark@+1 {{Cost: 2}}
    %35 = "tf.Cast"(%dense_values#7) {Truncate = false, device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?xi64>) -> tensor<?xf32>
    // expected-remark@+1 {{Cost: 1}}
    %36 = "tf.ExpandDims"(%28, %11) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?xf32>, tensor<i32>) -> tensor<?x1xf32>
    // expected-remark@+1 {{Cost: 1}}
    %37 = "tf.ExpandDims"(%29, %11) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?xf32>, tensor<i32>) -> tensor<?x1xf32>
    // expected-remark@+1 {{Cost: 1}}
    %38 = "tf.ExpandDims"(%30, %11) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?xf32>, tensor<i32>) -> tensor<?x1xf32>
    // expected-remark@+1 {{Cost: 1}}
    %39 = "tf.ExpandDims"(%31, %11) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?xf32>, tensor<i32>) -> tensor<?x1xf32>
    // expected-remark@+1 {{Cost: 1}}
    %40 = "tf.ExpandDims"(%32, %11) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?xf32>, tensor<i32>) -> tensor<?x1xf32>
    // expected-remark@+1 {{Cost: 1}}
    %41 = "tf.ExpandDims"(%33, %11) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?xf32>, tensor<i32>) -> tensor<?x1xf32>
    // expected-remark@+1 {{Cost: 1}}
    %42 = "tf.ExpandDims"(%34, %11) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?xf32>, tensor<i32>) -> tensor<?x1xf32>
    // expected-remark@+1 {{Cost: 1}}
    %43 = "tf.ExpandDims"(%35, %11) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?xf32>, tensor<i32>) -> tensor<?x1xf32>
    // expected-remark@+1 {{Cost: 10}}
    %44 = "tf.ConcatV2"(%36, %37, %38, %39, %40, %41, %42, %43, %10) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?x1xf32>, tensor<?x1xf32>, tensor<?x1xf32>, tensor<?x1xf32>, tensor<?x1xf32>, tensor<?x1xf32>, tensor<?x1xf32>, tensor<?x1xf32>, tensor<i32>) -> tensor<?x8xf32>
    // expected-remark@+1 {{Cost: 1}}
    func.return %44 : tensor<?x8xf32>
}

// CHECK-LABEL: test_cheap_ops_1
func.func @test_cheap_ops_1(%arg: tensor<?x8x?x?xf32>) -> (tensor<4xi32>, tensor<?x8x?x?xf32>) {
    // expected-remark@+1 {{Cost: 1}}
    %0 = "tf.Const"() {value = dense<8> : tensor<i32>} : () -> tensor<i32>
    // expected-remark@+1 {{Cost: 1}}
    %1 = "tf.Const"() {value = dense<4> : tensor<1xi32>} : () -> tensor<1xi32>
    // expected-remark@+1 {{Cost: 1}}
    %2 = "tf.Const"() {value = dense<64> : tensor<i32>} : () -> tensor<i32>
    // expected-remark@+1 {{Cost: 1}}
    %3 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
    // expected-remark@+1 {{Cost: 1}}
    %4 = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
    // expected-remark@+1 {{Cost: 1}}
    %5 = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    // expected-remark@+1 {{Cost: 1}}
    %6 = "tf.Const"() {value = dense<3> : tensor<1xi32>} : () -> tensor<1xi32>
    // expected-remark@+1 {{Cost: 9}}
    %7 = "tf.Softmax"(%arg) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?x8x?x?xf32>) -> tensor<?x8x?x?xf32>
    // expected-remark@+1 {{Cost: 1}}
    %8 = "tf.Shape"(%7) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?x8x?x?xf32>) -> tensor<4xi32>
    // expected-remark@+1 {{Cost: 1}}
    %9 = "tf.StridedSlice"(%8, %5, %4, %4) {begin_mask = 0 : i64, device = "/job:localhost/replica:0/task:0/device:CPU:0", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
    // expected-remark@+1 {{Cost: 1}}
    %10 = "tf.StridedSlice"(%8, %3, %6, %4) {begin_mask = 0 : i64, device = "/job:localhost/replica:0/task:0/device:CPU:0", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
    // expected-remark@+1 {{Cost: 5}}
    %11 = "tf.Pack"(%9, %0, %10, %2) {axis = 0 : i64, device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4xi32>
    // expected-remark@+1 {{Cost: 1}}
    %12 = "tf.StridedSlice"(%8, %6, %1, %4) {begin_mask = 0 : i64, device = "/job:localhost/replica:0/task:0/device:CPU:0", ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<4xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
    // expected-remark@+1 {{Cost: 5}}
    %13 = "tf.Pack"(%9, %0, %10, %12) {axis = 0 : i64, device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4xi32>
    // expected-remark@+1 {{Cost: 1}}
    %14 = "tf.Reshape"(%7, %13) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<?x8x?x?xf32>, tensor<4xi32>) -> tensor<?x8x?x?xf32>
    // expected-remark@+1 {{Cost: 8}}
    func.return %11, %14 : tensor<4xi32>, tensor<?x8x?x?xf32>
}

// CHECK-LABEL: test_expensive_ops
func.func @test_expensive_ops(%arg: tensor<?x512xf32>) -> tensor<?x512xf32> {
    // expected-remark@+1 {{Cost: 1}}
    %0 = "tf.VarHandleOp"() {allowed_devices = [], container = "", device = "/job:localhost/replica:0/task:0/device:CPU:0", shared_name = "var"} : () -> tensor<!tf_type.resource<tensor<512x512xf32>>>
    // expected-remark@+1 {{Cost: 2}}
    %1 = "tf.ReadVariableOp"(%0) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<!tf_type.resource<tensor<512x512xf32>>>) -> tensor<512x512xf32>
    // 262657 = 1 + 512 + 512 * 512
    // expected-remark@+1 {{Cost: 262657}}
    %2 = "tf.MatMul"(%arg, %1) {device = "/job:localhost/replica:0/task:0/device:CPU:0", transpose_a = false, transpose_b = false} : (tensor<?x512xf32>, tensor<512x512xf32>) -> tensor<?x512xf32>
    // expected-remark@+1 {{Cost: 512}}
    func.return %2 : tensor<?x512xf32>
}

// CHECK-LABEL: test_dynamic_shape
func.func @test_dynamic_shape(%key: tensor<?x!tf_type.string>, %value: tensor<8xi64>) -> tensor<*xi1> {
    // expected-remark@+1 {{Cost: 1}}
    %default = "tf.Const"() {device = "/job:localhost/replica:0/task:0/device:CPU:0", value = dense<-1> : tensor<i64>} : () -> tensor<i64>
    // expected-remark@+1 {{Cost: 1}}
    %0 = "tf.HashTableV2"() {container = "", device = "/job:localhost/replica:0/task:0/device:CPU:0", key_dtype = !tf_type.string, shared_name = "hash_table", use_node_name_sharing = false, value_dtype = i64} : () -> tensor<!tf_type.resource>
    // expected-remark@+1 {{Cost: 1024}}
    %1 = "tf.LookupTableFindV2"(%0, %key, %default) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<!tf_type.resource>, tensor<?x!tf_type.string>, tensor<i64>) -> tensor<*xi64>
    // 17 = 1 + 8 + 8
    // expected-remark@+1 {{Cost: 17}}
    %2 = "tf.NotEqual"(%1, %value) {device = "/job:localhost/replica:0/task:0/device:CPU:0", incompatible_shape_error = true} : (tensor<*xi64>, tensor<8xi64>) -> tensor<*xi1>
    // expected-remark@+1 {{Cost: 8}}
    func.return %2 : tensor<*xi1>
}

// CHECK-LABEL: test_gather
func.func @test_gather(%arg0 : tensor<1x2x20xf32>, %arg1 : tensor<3x5xi32>) -> (tensor<1x3x5x20xf32>){
    // expected-remark@+1 {{Cost: 1}}
    %0 = "tf.Const"() { value = dense<[1]> : tensor<1xi32> } : () -> tensor<1xi32>
    // expected-remark@+1 {{Cost: 300}}
    %1 = "tf.GatherV2"(%arg0, %arg1, %0) : (tensor<1x2x20xf32>, tensor<3x5xi32>, tensor<1xi32>) -> tensor<1x3x5x20xf32>
    // expected-remark@+1 {{Cost: 40}}
    func.return %1 : tensor<1x3x5x20xf32>
}

// CHECK-LABEL: test_sparse_segment_sum
func.func @test_sparse_segment_sum(%indices: tensor<3xi64>, %segment_ids: tensor<3xi64>) -> (tensor<?x28xf32>){
    // expected-remark@+1 {{Cost: 1}}
    %data = "tf.Const"() { value = dense<0.1> : tensor<476x28xf32> } : () -> tensor<476x28xf32>
    // expected-remark@+1 {{Cost: 28}}
    %1 = "tf.SparseSegmentSum"(%data, %indices, %segment_ids) {device = "/job:localhost/replica:0/task:0/device:CPU:0"} : (tensor<476x28xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<?x28xf32>
    // expected-remark@+1 {{Cost: 3}}
    func.return %1 : tensor<?x28xf32>
}
