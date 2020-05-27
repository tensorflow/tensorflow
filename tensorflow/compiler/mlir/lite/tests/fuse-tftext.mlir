// RUN: tf-opt -tfl-prepare-composite-funcs-tf -tfl-fuse-tftext=true %s -split-input-file | FileCheck %s --dump-input-on-failure
module {

  func @_whitespace_func(%arg0: tensor<1x!tf.string>) -> (tensor<?x!tf.string>, tensor<?xi64>) attributes {tf._GrapplerSpecializedFunc = true, tf._input_shapes = [#tf.shape<1>], tf.api_implements = "tftext:WhitespaceTokenizer", tf.signature.is_stateful} {
    %0 = "tf.op1"(%arg0)  : (tensor<1x!tf.string>) -> (tensor<?x!tf.string>)
    %1 = "tf.Const"() {value = dense<-1> : tensor<i64>} : () -> tensor<?xi64>
    %2:2 = "tf.op2"(%arg0, %1) : (tensor<1x!tf.string>, tensor<?xi64>) -> (tensor<?x!tf.string>, tensor<?xi64>)
    return %2#0, %2#1 : tensor<?x!tf.string>, tensor<?xi64>
  }

  // CHECK: func @_whitespace_func(%arg0: tensor<1x!tf.string>) -> (tensor<?x!tf.string>, tensor<?xi64>) attributes {tf._GrapplerSpecializedFunc = true, tf._input_shapes = [#tf.shape<1>], tf.api_implements = "tftext:WhitespaceTokenizer", tf.signature.is_stateful} {
  // CHECK:  "tfl.custom"(%arg0) {custom_code = "tftext:WhitespaceTokenizer", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>} : (tensor<1x!tf.string>) -> (tensor<?x!tf.string>, tensor<?xi64>)
  // CHECK:  return %0#0, %0#1 : tensor<?x!tf.string>, tensor<?xi64>
}
