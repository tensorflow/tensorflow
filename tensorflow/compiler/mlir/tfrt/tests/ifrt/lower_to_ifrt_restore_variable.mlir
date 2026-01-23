// RUN: tf-tfrt-opt -split-input-file -verify-diagnostics -lower-to-ifrt-restore-variable %s | FileCheck %s


// -----
// single variable

// CHECK-LABEL:   func.func @restore_single() {
// CHECK-NEXT:     [[PREFIX:%.*]] = "tf.Const"() <{value = dense<"restore_ariables"> : tensor<!tf_type.string>}> : () -> tensor<!tf_type.string>
// CHECK-NEXT:     [[SLICE:%.*]] = "tf.Const"() <{value = dense<""> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
// CHECK-NEXT:     [[NAME:%.*]] = "tf.Const"() <{value = dense<"y"> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
// CHECK-NEXT:     [[HANDLEY:%.*]] = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
// CHECK-NEXT:     "tf.IfrtRestoreVariableOp"([[PREFIX]], [[NAME]], [[SLICE]], [[HANDLEY]])
// CHECK-SAME:        {restored_dtypes = [f32], truncate_in_cast = array<i1: false>}
// CHECK-NOT:       "tf.RestoreV2"
// CHECK-NEXT:     return

module {
  func.func @restore_single() {
    %cst = "tf.Const"() <{value = dense<"restore_ariables"> : tensor<!tf_type.string>}> : () -> tensor<!tf_type.string>
    %cst_0 = "tf.Const"() <{value = dense<""> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
    %cst_1 = "tf.Const"() <{value = dense<"y"> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
    %0 = "tf.RestoreV2"(%cst, %cst_1, %cst_0): (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>) -> tensor<3x1xf32>
    %1 = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
    "tf.AssignVariableOp"(%1, %0) : (tensor<!tf_type.resource<tensor<3x1xf32>>>, tensor<3x1xf32>) -> ()
    return
  }
}

// -----
// single variable: VarHandleOp is before RestoreV2

// CHECK-LABEL:   func.func @varhandle_before_restore() {
// CHECK-NEXT:     [[PREFIX:%.*]] = "tf.Const"() <{value = dense<"restore_ariables"> : tensor<!tf_type.string>}> : () -> tensor<!tf_type.string>
// CHECK-NEXT:     [[SLICE:%.*]] = "tf.Const"() <{value = dense<""> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
// CHECK-NEXT:     [[NAME:%.*]] = "tf.Const"() <{value = dense<"y"> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
// CHECK-NEXT:     [[HANDLEY:%.*]] = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
// CHECK-NEXT:     "tf.IfrtRestoreVariableOp"([[PREFIX]], [[NAME]], [[SLICE]], [[HANDLEY]])
// CHECK-SAME:        {restored_dtypes = [f32], truncate_in_cast = array<i1: false>}
// CHECK-NOT:       "tf.RestoreV2"
// CHECK-NEXT:     return

module {
  func.func @varhandle_before_restore() {
    %cst = "tf.Const"() <{value = dense<"restore_ariables"> : tensor<!tf_type.string>}> : () -> tensor<!tf_type.string>
    %cst_0 = "tf.Const"() <{value = dense<""> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
    %cst_1 = "tf.Const"() <{value = dense<"y"> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
    %1 = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
    %0 = "tf.RestoreV2"(%cst, %cst_1, %cst_0): (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>) -> tensor<3x1xf32>
    "tf.AssignVariableOp"(%1, %0) : (tensor<!tf_type.resource<tensor<3x1xf32>>>, tensor<3x1xf32>) -> ()
    return
  }
}


// -----
// multiple variables

// CHECK-LABEL:   func.func @restore_multiple() {
// CHECK-NEXT:     [[PREFIX:%.*]] = "tf.Const"()
// CHECK-NEXT:     [[SLICE:%.*]] = "tf.Const"()
// CHECK-NEXT:     [[NAME:%.*]] = "tf.Const"()
// CHECK-NEXT:     [[HANDLEY:%.*]] = "tf.VarHandleOp"() <{container = "x", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
// CHECK-NEXT:     [[HANDLEZ:%.*]] = "tf.VarHandleOp"() <{container = "x", shared_name = "z"}> : () -> tensor<!tf_type.resource<tensor<1x3xf32>>>
// CHECK-NEXT:     "tf.IfrtRestoreVariableOp"([[PREFIX]], [[NAME]], [[SLICE]], [[HANDLEY]], [[HANDLEZ]])
// CHECK-SAME:        {restored_dtypes = [f32, f32], truncate_in_cast = array<i1: false, false>}
// CHECK-NOT:       "tf.RestoreV2"
// CHECK-NEXT:     return

module {
  func.func @restore_multiple() {
    %cst = "tf.Const"() <{value = dense<"restore_ariables"> : tensor<!tf_type.string>}> : () -> tensor<!tf_type.string>
    %cst_0 = "tf.Const"() <{value = dense<["", ""]> : tensor<2x!tf_type.string>}> : () -> tensor<2x!tf_type.string>
    %cst_1 = "tf.Const"() <{value = dense<["y", "z"]> : tensor<2x!tf_type.string>}> : () -> tensor<2x!tf_type.string>
    %0:2 = "tf.RestoreV2"(%cst, %cst_1, %cst_0): (tensor<!tf_type.string>, tensor<2x!tf_type.string>, tensor<2x!tf_type.string>) -> (tensor<3x1xf32>, tensor<1x3xf32>)
    %1 = "tf.VarHandleOp"() <{container = "x", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
    "tf.AssignVariableOp"(%1, %0#0) : (tensor<!tf_type.resource<tensor<3x1xf32>>>, tensor<3x1xf32>) -> ()
    %2 = "tf.VarHandleOp"() <{container = "x", shared_name = "z"}> : () -> tensor<!tf_type.resource<tensor<1x3xf32>>>
    "tf.AssignVariableOp"(%2, %0#1) : (tensor<!tf_type.resource<tensor<1x3xf32>>>, tensor<1x3xf32>) -> ()
    return
  }
}

// -----
// Restored variable is not assigned with a name is an error.

module {
  func.func @unassigned_restore_return_error() {
    %cst = "tf.Const"() <{value = dense<"restore_ariables"> : tensor<!tf_type.string>}> : () -> tensor<!tf_type.string>
    %cst_0 = "tf.Const"() <{value = dense<["", ""]> : tensor<2x!tf_type.string>}> : () -> tensor<2x!tf_type.string>
    %cst_1 = "tf.Const"() <{value = dense<["y", "z"]> : tensor<2x!tf_type.string>}> : () -> tensor<2x!tf_type.string>
    //expected-error@below {{'tf.RestoreV2' op expects 2 valid users, but got 1}}
    %0:2 = "tf.RestoreV2"(%cst, %cst_1, %cst_0): (tensor<!tf_type.string>, tensor<2x!tf_type.string>, tensor<2x!tf_type.string>) -> (tensor<3x1xf32>, tensor<1x3xf32>)
    %1 = "tf.VarHandleOp"() <{container = "x", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
    "tf.AssignVariableOp"(%1, %0#0) : (tensor<!tf_type.resource<tensor<3x1xf32>>>, tensor<3x1xf32>) -> ()
    return
  }
}

// -----
// Unsupported OP from RestoreV2 to AssignVariableOp is an error.

module {
  func.func @unassigned_restore_return_error() {
    %cst = "tf.Const"() <{value = dense<"restore_ariables"> : tensor<!tf_type.string>}> : () -> tensor<!tf_type.string>
    %cst_0 = "tf.Const"() <{value = dense<["", ""]> : tensor<2x!tf_type.string>}> : () -> tensor<2x!tf_type.string>
    %cst_1 = "tf.Const"() <{value = dense<["y", "z"]> : tensor<2x!tf_type.string>}> : () -> tensor<2x!tf_type.string>
    %0 = "tf.RestoreV2"(%cst, %cst_1, %cst_0): (tensor<!tf_type.string>, tensor<2x!tf_type.string>, tensor<2x!tf_type.string>) -> (tensor<3x1xf32>)
    //expected-error@below {{'tf.ReluOp' op is not a supported user of RestoreV2Op}}
    %2 = "tf.ReluOp"(%0) : (tensor<3x1xf32>) -> tensor<3x1xf32>
    %1 = "tf.VarHandleOp"() <{container = "x", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xf32>>>
    "tf.AssignVariableOp"(%1, %2) : (tensor<!tf_type.resource<tensor<3x1xf32>>>, tensor<3x1xf32>) -> ()
    return
  }
}



// -----
// variable with cast
// CHECK-LABEL:   func.func @restore_with_cast() {
// CHECK-NEXT:     [[PREFIX:%.*]] = "tf.Const"() <{value = dense<"restore_ariables"> : tensor<!tf_type.string>}> : () -> tensor<!tf_type.string>
// CHECK-NEXT:     [[SLICE:%.*]] = "tf.Const"() <{value = dense<""> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
// CHECK-NEXT:     [[NAME:%.*]] = "tf.Const"() <{value = dense<"y"> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
// CHECK-NEXT:     [[HANDLEY:%.*]] = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xbf16>>>
// CHECK-NEXT:     "tf.IfrtRestoreVariableOp"([[PREFIX]], [[NAME]], [[SLICE]], [[HANDLEY]])
// CHECK-SAME:        {restored_dtypes = [f32], truncate_in_cast = array<i1: false>}
// CHECK-NOT:       "tf.RestoreV2"
// CHECK-NEXT:     return

module {
  func.func @restore_with_cast() {
    %cst = "tf.Const"() <{value = dense<"restore_ariables"> : tensor<!tf_type.string>}> : () -> tensor<!tf_type.string>
    %cst_0 = "tf.Const"() <{value = dense<""> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
    %cst_1 = "tf.Const"() <{value = dense<"y"> : tensor<1x!tf_type.string>}> : () -> tensor<1x!tf_type.string>
    %0 = "tf.RestoreV2"(%cst, %cst_1, %cst_0): (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>) -> tensor<3x1xf32>
    %1 = "tf.Cast"(%0) <{truncate = false}> : (tensor<3x1xf32>) -> tensor<3x1xbf16>
    %2 = "tf.VarHandleOp"() <{container = "", shared_name = "y"}> : () -> tensor<!tf_type.resource<tensor<3x1xbf16>>>
    "tf.AssignVariableOp"(%2, %1) : (tensor<!tf_type.resource<tensor<3x1xbf16>>>, tensor<3x1xbf16>) -> ()
    return
  }
}


// -----
// variable and table lookup
// CHECK-LABEL: func.func @restore_var_and_table()
// CHECK-NEXT: [[PREFIX:%.*]] = "tf.Const"() <{value = dense<"model/foo"> : tensor<!tf_type.string>}> : () -> tensor<!tf_type.string>
// CHECK-NEXT: [[SLICE:%.*]] = "tf.Const"() <{value = dense<["var1", "table1_keys", "table1_vals"]> : tensor<3x!tf_type.string>}> : () -> tensor<3x!tf_type.string>
// CHECK-NEXT: [[NAME:%.*]] = "tf.Const"() <{value = dense<""> : tensor<3x!tf_type.string>}> : () -> tensor<3x!tf_type.string>
// CHECK-NEXT: [[VAR:%[0-9]+]] = "tf.VarHandleOp"() <{container = "", shared_name = "var1"}> : () -> tensor<!tf_type.resource<tensor<*xi32>>>
// CHECK-NEXT: [[TABLE:%[0-9]+]] = "tf.HashTableV2"() <{container = "", key_dtype = i64, shared_name = "table1", value_dtype = f32}> : () -> tensor<!tf_type.resource>
// CHECK-NEXT: "tf.IfrtRestoreVariableOp"([[PREFIX]], [[SLICE]], [[NAME]], [[VAR]], [[TABLE]], [[TABLE]])
// CHECK-SAME:  <{restored_dtypes = [i32, i64, f32], truncate_in_cast = array<i1: false, false, false>}> : (tensor<!tf_type.string>, tensor<3x!tf_type.string>, tensor<3x!tf_type.string>, tensor<!tf_type.resource<tensor<*xi32>>>, tensor<!tf_type.resource>, tensor<!tf_type.resource>) -> ()
// CHECK-NEXT: return
module {
  func.func @restore_var_and_table() {
    %cst = "tf.Const"() {value = dense<"model/foo"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
    %cst_0 = "tf.Const"() {value = dense<["var1", "table1_keys", "table1_vals"]> : tensor<3x!tf_type.string>} : () -> tensor<3x!tf_type.string>
    %cst_1 = "tf.Const"() {value = dense<["", "", ""]> : tensor<3x!tf_type.string>} : () -> tensor<3x!tf_type.string>
    %0:3 = "tf.RestoreV2"(%cst, %cst_0, %cst_1) {dtypes = [i32, i64, f32]} : (tensor<!tf_type.string>, tensor<3x!tf_type.string>, tensor<3x!tf_type.string>) -> (tensor<*xi32>, tensor<*xi64>, tensor<*xf32>)
    %1 = "tf.VarHandleOp"() {container = "", shared_name = "var1"} : () -> tensor<!tf_type.resource<tensor<*xi32>>>
    "tf.AssignVariableOp"(%1, %0#0) : (tensor<!tf_type.resource<tensor<*xi32>>>, tensor<*xi32>) -> ()
    %2 = "tf.HashTableV2"() {container = "", key_dtype = i64, shared_name = "table1", value_dtype = f32} : () -> tensor<!tf_type.resource>
    "tf.LookupTableImportV2"(%2, %0#1, %0#2) : (tensor<!tf_type.resource>, tensor<*xi64>, tensor<*xf32>) -> ()
    return
  }
}

// -----
// variable and dense table lookup
// CHECK-LABEL: func.func @restore_var_and_dense_table()
// CHECK-NEXT: [[PREFIX:%.*]] = "tf.Const"() <{value = dense<"model/foo"> : tensor<!tf_type.string>}> : () -> tensor<!tf_type.string>
// CHECK-NEXT: [[SLICE:%.*]] = "tf.Const"() <{value = dense<["var1", "table1_keys", "table1_vals"]> : tensor<3x!tf_type.string>}> : () -> tensor<3x!tf_type.string>
// CHECK-NEXT: [[NAME:%.*]] = "tf.Const"() <{value = dense<""> : tensor<3x!tf_type.string>}> : () -> tensor<3x!tf_type.string>
// CHECK-NEXT: [[VAR:%[0-9]+]] = "tf.VarHandleOp"() <{container = "", shared_name = "var1"}> : () -> tensor<!tf_type.resource<tensor<*xi32>>>
// CHECK-NEXT: [[EMPTYKEY:%.*]] = "tf.Const"() <{value = dense<-1> : tensor<i64>}> : () -> tensor<i64>
// CHECK-NEXT: [[DEFAULTVAL:%.*]] = "tf.Const"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK-NEXT: [[TABLE:%[0-9]+]] = "tf.MutableDenseHashTableV2"([[EMPTYKEY]], [[DEFAULTVAL]]) <{container = "", shared_name = "table1", value_dtype = f32}> {key_dtype = i64} : (tensor<i64>, tensor<f32>) -> tensor<!tf_type.resource>
// CHECK-NEXT: "tf.IfrtRestoreVariableOp"([[PREFIX]], [[SLICE]], [[NAME]], [[VAR]], [[TABLE]], [[TABLE]])
// CHECK-SAME:  <{restored_dtypes = [i32, i64, f32], truncate_in_cast = array<i1: false, false, false>}> : (tensor<!tf_type.string>, tensor<3x!tf_type.string>, tensor<3x!tf_type.string>, tensor<!tf_type.resource<tensor<*xi32>>>, tensor<!tf_type.resource>, tensor<!tf_type.resource>) -> ()
// CHECK-NEXT: return
module {
  func.func @restore_var_and_dense_table() {
    %cst = "tf.Const"() {value = dense<"model/foo"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
    %cst_0 = "tf.Const"() {value = dense<["var1", "table1_keys", "table1_vals"]> : tensor<3x!tf_type.string>} : () -> tensor<3x!tf_type.string>
    %cst_1 = "tf.Const"() {value = dense<["", "", ""]> : tensor<3x!tf_type.string>} : () -> tensor<3x!tf_type.string>
    %0:3 = "tf.RestoreV2"(%cst, %cst_0, %cst_1) {dtypes = [i32, i64, f32]} : (tensor<!tf_type.string>, tensor<3x!tf_type.string>, tensor<3x!tf_type.string>) -> (tensor<*xi32>, tensor<*xi64>, tensor<*xf32>)
    %1 = "tf.VarHandleOp"() {container = "", shared_name = "var1"} : () -> tensor<!tf_type.resource<tensor<*xi32>>>
    "tf.AssignVariableOp"(%1, %0#0) : (tensor<!tf_type.resource<tensor<*xi32>>>, tensor<*xi32>) -> ()
    %empty_key = "tf.Const"() {value = dense<-1> : tensor<i64>} : () -> tensor<i64>
    %default_value = "tf.Const"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
    %2 = "tf.MutableDenseHashTableV2"(%empty_key, %default_value) {container = "", key_dtype = i64, shared_name = "table1", value_dtype = f32} : (tensor<i64>, tensor<f32>) -> tensor<!tf_type.resource>
    "tf.LookupTableImportV2"(%2, %0#1, %0#2) : (tensor<!tf_type.resource>, tensor<*xi64>, tensor<*xf32>) -> ()
    return
  }
}
