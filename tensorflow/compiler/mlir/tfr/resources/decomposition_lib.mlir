// A test resource file which contains some pre-defined internal tfr.functions
// for decomposition and external tfr.functions for raising the decomposition
// result to the ops in the TF dialect.
//
// All the tfr.func functions are supposed to be translated from the Python
// function with tf.composite annotation.
// All the external tfr.func functions modeles the op signature defined by
// OpDefs.

tfr.func @tf__my_add_n(%values: !tfr.tensor_list,
                       %n: i64 {tfr.name="N"}) -> !tfr.tensor {
  %index = constant 0 : index
  %cst = constant 1 : i64
  %eq = cmpi eq, %n, %cst : i64
  %v1 = tfr.get_element %values[%index] : (!tfr.tensor_list, index) -> !tfr.tensor
  %res = scf.if %eq -> !tfr.tensor {
    scf.yield %v1 : !tfr.tensor
  } else {
    %step = index_cast %cst : i64 to index
    %end = index_cast %n : i64 to index
    %reduce = scf.for %i = %step to %end step %step iter_args(%reduce_iter=%v1) -> !tfr.tensor {
      %v = tfr.get_element %values[%i] : (!tfr.tensor_list, index) -> !tfr.tensor
      %reduce_next =  tfr.call @tf__add(%reduce_iter, %v) : (!tfr.tensor, !tfr.tensor) -> !tfr.tensor
      scf.yield %reduce_next : !tfr.tensor
    }
    scf.yield %reduce : !tfr.tensor
  }
  tfr.return %res : !tfr.tensor
}

tfr.func @tf__my_add_n_(!tfr.tensor_list<N,T>, i64 {tfr.name="N"}) -> !tfr.tensor attributes {N,T}

// Translated from tf.compose Python function.
tfr.func @tf__my_biased_dense(%input: !tfr.tensor, %weight: !tfr.tensor,
                              %bias: !tfr.tensor,
                              %act: !tfr.attr{tfr.name="act", tfr.default=""}) -> !tfr.tensor {
  %dot = tfr.call @tf__mat_mul(%input, %weight) : (!tfr.tensor, !tfr.tensor) -> !tfr.tensor
  %add = tfr.call @tf__add(%dot, %bias) : (!tfr.tensor, !tfr.tensor) -> !tfr.tensor

  %relu = tfr.constant "relu" -> !tfr.attr
  %relu6 = tfr.constant "relu6" -> !tfr.attr

  %is_relu = tfr.equal %act, %relu -> i1
  %res = scf.if %is_relu -> !tfr.tensor {
    %applied_relu = tfr.call @tf__relu(%add) : (!tfr.tensor) -> !tfr.tensor
    scf.yield %applied_relu : !tfr.tensor
  } else {
    %is_relu6 = tfr.equal %act, %relu6 -> i1
    %res1 = scf.if %is_relu6 -> !tfr.tensor {
      %applied_relu6 = tfr.call @tf__relu6(%add) : (!tfr.tensor) -> !tfr.tensor
      scf.yield %applied_relu6 : !tfr.tensor
    } else {
      scf.yield %add : !tfr.tensor
    }
    scf.yield %res1 : !tfr.tensor
  }
  tfr.return %res : !tfr.tensor
}

tfr.func @tf__my_biased_dense_(!tfr.tensor<T>, !tfr.tensor<T>, !tfr.tensor<T>,
    !tfr.attr{tfr.name="act", tfr.default=""}) -> !tfr.tensor attributes {T}

// This is a wong decomposition and used to verify that tf.Elu isn't decomposed
// since its kernel has been registered.
tfr.func @tf__elu_(%input: !tfr.tensor) -> !tfr.tensor {
  tfr.return %input : !tfr.tensor
}

// Translated from:
//
// REGISTER_OP("Add")
//     .Input("x: T")
//     .Input("y: T")
//     .Output("z: T")
//     .Attr(
//         "T: {bfloat16, half, float, double, uint8, int8, int16, int32, int64, "
//         "complex64, complex128, string}")
tfr.func @tf__add_(!tfr.tensor<T>, !tfr.tensor<T>)
    -> !tfr.tensor<T> attributes{T}

// Translated from:
//
// REGISTER_OP("MatMul")
//     .Input("a: T")
//     .Input("b: T")
//     .Output("product: T")
//     .Attr("transpose_a: bool = false")
//     .Attr("transpose_b: bool = false")
//     .Attr("T: {bfloat16, half, float, double, int32, int64, complex64, complex128}")
// T is a derived attribute.
// transpose_a and transpose_b is materialized attributes.
tfr.func @tf__mat_mul_(!tfr.tensor<T>, !tfr.tensor<T>,
                      i1 {tfr.name="transpose_a", tfr.default=false},
                      i1 {tfr.name="transpose_b", tfr.default=false})
    -> !tfr.tensor<T> attributes{T}

// Translated from:
//
// REGISTER_OP("Relu")
//     .Input("features: T")
//     .Output("activations: T")
//     .Attr("T: {realnumbertype, qint8}")
// T is a derived attribute.
tfr.func @tf__relu_(!tfr.tensor<T>) -> !tfr.tensor<T> attributes{T}


// Translated from:
//
// REGISTER_OP("Relu6")
//     .Input("features: T")
//     .Output("activations: T")
//     .Attr("T: {realnumbertype}")
// T is a derived attribute.
tfr.func @tf__relu6_(!tfr.tensor<T>) -> !tfr.tensor<T> attributes{T}
