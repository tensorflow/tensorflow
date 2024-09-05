// RUN: tfrt_translate -mlir-to-bef %s | tf_bef_executor | FileCheck %s

// Initialize all the variables used in the model by creating variable handles
// and assigning the variables with the correct values. This function takes an
// input chain and returns an output chain and a list of variable handles that
// have been value-initialized. The returned tfd.tf_tensors are TF tensors that
// contain tensorflow::ResourceHandle.
func.func @mnist_init_variables(%start_c : !tfrt.chain) -> (
  !tfrt.chain, !tfd.tf_tensor, !tfd.tf_tensor, !tfd.tf_tensor, !tfd.tf_tensor) {

  // Init w1.
  %init_w1_c, %w1_h = "tfd.delegate_kernel"(%start_c) {
    _name = "VarHandleOp", attr0_name = "container", attr0_value = "string$",
    attr1_name = "shared_name", attr1_value = "string$w1",
    attr2_name = "dtype", attr2_value = "tfdtype$DT_INT32",
    attr3_name = "shape", attr3_value = "tfshape$[2,2]"
  } : (!tfrt.chain) -> (!tfrt.chain, !tfd.tf_tensor)

  %const_w1_c, %w1_const_tensor = "tfd.delegate_kernel"(%init_w1_c) {
    _name = "Const", attr0_name = "dtype", attr0_value = "tfdtype$DT_INT32",
    attr1_name = "value",
    attr1_value = "tftensor$dtype:DT_INT32 tensor_shape { dim { size: 2 } dim { size: 2 }} int_val: 1 int_val: 1 int_val: 1 int_val: 1"
  } : (!tfrt.chain) -> (!tfrt.chain, !tfd.tf_tensor)

  %assignv_w1_c = "tfd.delegate_kernel"(%const_w1_c, %w1_h, %w1_const_tensor) {
    _name = "AssignVariableOp",
    attr0_name = "dtype",attr0_value = "tfdtype$DT_INT32"
  } : (!tfrt.chain, !tfd.tf_tensor, !tfd.tf_tensor) -> (!tfrt.chain)

  // Init b1.
  %init_b1_c, %b1_h = "tfd.delegate_kernel"(%assignv_w1_c) {
    _name = "VarHandleOp", attr0_name = "container", attr0_value = "string$",
    attr1_name = "shared_name", attr1_value = "string$b1",
    attr2_name = "dtype", attr2_value = "tfdtype$DT_INT32",
    attr3_name = "shape", attr3_value = "tfshape$[2,2]"
  } : (!tfrt.chain) -> (!tfrt.chain, !tfd.tf_tensor)

  %const_b1_c, %b1_const_tensor = "tfd.delegate_kernel"(%init_b1_c) {
    _name = "Const", attr0_name = "dtype", attr0_value = "tfdtype$DT_INT32",
    attr1_name = "value",
    attr1_value = "tftensor$dtype:DT_INT32 tensor_shape { dim { size: 2 } dim { size: 2 }} int_val: 1 int_val: 1 int_val: 1 int_val: 1"
  } : (!tfrt.chain) -> (!tfrt.chain, !tfd.tf_tensor)

  %assignv_b1_c = "tfd.delegate_kernel"(%const_b1_c, %b1_h, %b1_const_tensor) {
    _name = "AssignVariableOp",
    attr0_name = "dtype", attr0_value = "tfdtype$DT_INT32"
  } : (!tfrt.chain, !tfd.tf_tensor, !tfd.tf_tensor) -> (!tfrt.chain)

  // Init w2.
  %init_w2_c, %w2_h = "tfd.delegate_kernel"(%assignv_b1_c) {
    _name = "VarHandleOp", attr0_name = "container", attr0_value = "string$",
    attr1_name = "shared_name", attr1_value = "string$w2",
    attr2_name = "dtype", attr2_value = "tfdtype$DT_INT32",
    attr3_name = "shape", attr3_value = "tfshape$[2,2]"
  } : (!tfrt.chain) -> (!tfrt.chain, !tfd.tf_tensor)

  %const_w2_c, %w2_const_tensor = "tfd.delegate_kernel"(%init_w2_c) {
    _name = "Const", attr0_name = "dtype", attr0_value = "tfdtype$DT_INT32",
    attr1_name = "value",
    attr1_value = "tftensor$dtype:DT_INT32 tensor_shape { dim { size: 2 } dim { size: 2 }} int_val: 1 int_val: 1 int_val: 1 int_val: 1"
  } : (!tfrt.chain) -> (!tfrt.chain, !tfd.tf_tensor)

  %assignv_w2_c = "tfd.delegate_kernel"(%const_w2_c, %w2_h, %w2_const_tensor) {
    _name = "AssignVariableOp",
    attr0_name = "dtype",attr0_value = "tfdtype$DT_INT32"
  } : (!tfrt.chain, !tfd.tf_tensor, !tfd.tf_tensor) -> (!tfrt.chain)

  // Init b2.
  %init_b2_c, %b2_h = "tfd.delegate_kernel"(%assignv_w2_c) {
    _name = "VarHandleOp", attr0_name = "container", attr0_value = "string$",
    attr1_name = "shared_name", attr1_value = "string$b2",
    attr2_name = "dtype", attr2_value = "tfdtype$DT_INT32",
    attr3_name = "shape", attr3_value = "tfshape$[2,2]"
  } : (!tfrt.chain) -> (!tfrt.chain, !tfd.tf_tensor)

  %const_b2_c, %b2_const_tensor = "tfd.delegate_kernel"(%init_b2_c) {
    _name = "Const", attr0_name = "dtype", attr0_value = "tfdtype$DT_INT32",
    attr1_name = "value",
    attr1_value = "tftensor$dtype:DT_INT32 tensor_shape { dim { size: 2 } dim { size: 2 }} int_val: 1 int_val: 1 int_val: 1 int_val: 1"
  } : (!tfrt.chain) -> (!tfrt.chain, !tfd.tf_tensor)

  %assignv_b2_c = "tfd.delegate_kernel"(%const_b2_c, %b2_h, %b2_const_tensor) {
    _name = "AssignVariableOp",
    attr0_name = "dtype", attr0_value = "tfdtype$DT_INT32"
  } : (!tfrt.chain, !tfd.tf_tensor, !tfd.tf_tensor) -> (!tfrt.chain)

  tfrt.return %assignv_b2_c, %w1_h, %b1_h, %w2_h, %b2_h : !tfrt.chain, !tfd.tf_tensor, !tfd.tf_tensor, !tfd.tf_tensor, !tfd.tf_tensor
}

// This function represents one forward pass on the model. It takes an input
// chain and the variable handles for all model variables, and returns an output
// chain and the result tensor value.
func.func @inference_call(
  %start_c : !tfrt.chain, %inputx_tensor : !tfd.tf_tensor,
  %w1_h : !tfd.tf_tensor, %b1_h : !tfd.tf_tensor, %w2_h : !tfd.tf_tensor,
  %b2_h : !tfd.tf_tensor) -> (!tfrt.chain, !tfd.tf_tensor) {

  // Read all variables into tensors.
  %readv_c1, %w1_tensor = "tfd.delegate_kernel"(%start_c, %w1_h) {
    _name = "ReadVariableOp",
    attr0_name = "dtype", attr0_value = "tfdtype$DT_INT32"
  } : (!tfrt.chain, !tfd.tf_tensor) -> (!tfrt.chain, !tfd.tf_tensor)

  %readv_c2, %b1_tensor = "tfd.delegate_kernel"(%readv_c1, %b1_h) {
    _name = "ReadVariableOp",
    attr0_name = "dtype", attr0_value = "tfdtype$DT_INT32"
  } : (!tfrt.chain, !tfd.tf_tensor) -> (!tfrt.chain, !tfd.tf_tensor)

  %readv_c3, %w2_tensor = "tfd.delegate_kernel"(%readv_c2, %w2_h) {
    _name = "ReadVariableOp",
    attr0_name = "dtype", attr0_value = "tfdtype$DT_INT32"
  } : (!tfrt.chain, !tfd.tf_tensor) -> (!tfrt.chain, !tfd.tf_tensor)

  %readv_c4, %b2_tensor = "tfd.delegate_kernel"(%readv_c3, %b2_h) {
    _name = "ReadVariableOp",
    attr0_name = "dtype", attr0_value = "tfdtype$DT_INT32"
  } : (!tfrt.chain, !tfd.tf_tensor) -> (!tfrt.chain, !tfd.tf_tensor)

  // Compute the model forward pass.
  %matmul0_out_c, %matmul0_tensor = "tfd.delegate_kernel"(
    %readv_c4, %inputx_tensor, %w1_tensor) {
    _name = "MatMul",
    attr1_name = "transpose_a", attr1_value = "bool$false",
    attr2_name = "transpose_b", attr2_value = "bool$false"
  } : (!tfrt.chain, !tfd.tf_tensor, !tfd.tf_tensor) -> (
    !tfrt.chain, !tfd.tf_tensor)

  %add0_out_c, %add0_tensor = "tfd.delegate_kernel"(
    %matmul0_out_c, %matmul0_tensor, %b1_tensor) {
    _name = "AddV2"
  } : (!tfrt.chain, !tfd.tf_tensor, !tfd.tf_tensor) -> (
    !tfrt.chain, !tfd.tf_tensor)

  %relu0_out_c, %relu0_tensor = "tfd.delegate_kernel"(
    %add0_out_c, %add0_tensor) {
    _name = "Relu"
  } : (!tfrt.chain, !tfd.tf_tensor) -> (!tfrt.chain, !tfd.tf_tensor)

  %matmul1_out_c, %matmul1_tensor = "tfd.delegate_kernel"(
    %relu0_out_c, %relu0_tensor, %w2_tensor) {
    _name = "MatMul",
    attr1_name = "transpose_a", attr1_value = "bool$false",
    attr2_name = "transpose_b", attr2_value = "bool$false"
  } : (!tfrt.chain, !tfd.tf_tensor, !tfd.tf_tensor) -> (
    !tfrt.chain, !tfd.tf_tensor)

  %add1_out_c, %add1_tensor = "tfd.delegate_kernel"(
    %matmul1_out_c, %matmul1_tensor, %b2_tensor) {
    _name = "AddV2"
  } : (!tfrt.chain, !tfd.tf_tensor, !tfd.tf_tensor) -> (
    !tfrt.chain, !tfd.tf_tensor)

  %identity_out_c, %identity_tensor = "tfd.delegate_kernel"(
    %add1_out_c, %add1_tensor) {
    _name = "Identity"
  } : (!tfrt.chain, !tfd.tf_tensor) -> (!tfrt.chain, !tfd.tf_tensor)

  tfrt.return %identity_out_c, %identity_tensor : !tfrt.chain, !tfd.tf_tensor
}

// This is the main driver function. It creates the EagerContext, initializes
// all variables and runs one forward pass on the model.
// CHECK: --- Running 'mnist_delegate_test'
func.func @mnist_delegate_test() {
  %start_c = tfrt.new.chain

  // Init eager context.
  %context_init_c = "tfd.init_eager_context"(%start_c)
    : (!tfrt.chain) -> !tfrt.chain

  // Initialize all variables.
  %init_var_c, %w1_h, %b1_h, %w2_h, %b2_h
    = tfrt.call @mnist_init_variables(%context_init_c) : (!tfrt.chain) -> (
      !tfrt.chain, !tfd.tf_tensor, !tfd.tf_tensor, !tfd.tf_tensor,
      !tfd.tf_tensor)

  // Get input data.
  %get_input_c, %inputx_tensor = "tfd.delegate_kernel"(%init_var_c) {
    _name = "Const", attr0_name = "dtype", attr0_value = "tfdtype$DT_INT32",
    attr1_name = "value",
    attr1_value = "tftensor$dtype:DT_INT32 tensor_shape { dim { size: 2 } dim { size: 2 }} int_val: 1 int_val: 1 int_val: 1 int_val: 1"
  } : (!tfrt.chain) -> (!tfrt.chain, !tfd.tf_tensor)

  // Make one inference call.
  %inference_call_out_c, %inference_call_out_tensor
    = tfrt.call @inference_call(
      %get_input_c, %inputx_tensor, %w1_h, %b1_h, %w2_h, %b2_h) : (
      !tfrt.chain, !tfd.tf_tensor, !tfd.tf_tensor, !tfd.tf_tensor,
      !tfd.tf_tensor, !tfd.tf_tensor) -> (!tfrt.chain, !tfd.tf_tensor)

  // CHECK: shape = [2, 2], values = [7, 7, 7, 7]
  %print_c = "tfd.print_tft"(%inference_call_out_tensor, %inference_call_out_c)
    : (!tfd.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return
}
