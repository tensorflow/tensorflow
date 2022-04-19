// RUN: tf-opt %s -split-input-file -verify-diagnostics

// Check that a replicate with 'n' attribute that is less than 2 is invalid.
func.func @parser_replicate_n(%arg0: tensor<*xf32>) {
  tf_device.replicate([%arg0] as %input0: tensor<*xf32>) {n = 1 : i32} {
// expected-error@-1 {{'tf_device.replicate' expects 'n' to be at least 2, got 1}}
  }
  func.return
}

// -----

// Check that a replicate replicated inputs where operand sizes do not match
// 'n' is invalid.
func.func @parser_replicate_operand_count(%arg0: tensor<*xf32>) {
  tf_device.replicate([%arg0, %arg0, %arg0] as %input0: tensor<*xf32>) {n = 2 : i32} {
// expected-error@-1 {{'tf_device.replicate' expects number of operands for replicated input 0 to be 'n' (2), got 3}}
  }
  func.return
}

// -----

// Check that a replicate with incompatible operands and block argument type is
// invalid.
func.func @parser_replicate_operand_type(%arg0: tensor<*xi32>) {
// expected-note@-1 {{prior use here}}
  tf_device.replicate([%arg0, %arg0] as %input0: tensor<*xf32>) {n = 2 : i32} {
// expected-error@-1 {{use of value '%arg0' expects different type than prior uses: 'tensor<*xf32>' vs 'tensor<*xi32>'}}
  }
  func.return
}

// -----

// Check that a replicate with multiple blocks in its region is invalid.
func.func @parser_replicate_region() {
  tf_device.replicate() {n = 2 : i32} {
// expected-error@-1 {{custom op 'tf_device.replicate' expects a single block region}}
    cf.br ^bb
  ^bb:
    tf_device.return
  }
  func.return
}

// -----

// Check that a replicate with a bad terminator is invalid.
func.func @parser_replicate_terminator() {
  tf_device.replicate() {n = 2 : i32} {
// expected-error@-1 {{custom op 'tf_device.replicate' expects a tf_device.return terminator}}
    func.return
  }
  func.return
}

// -----

// Check that an empty replicate is invalid (replicate needs a region).
func.func @verifier_replicate_no_block() {
  "tf_device.replicate" () ({
// expected-error@-1 {{'tf_device.replicate' op region #0 ('body') failed to verify constraint: region with 1 blocks}}
  }) {n = 2 : i32, operand_segment_sizes = dense<[0, 0]> : vector<2xi32>} : () -> ()
  func.return
}

// -----

// Check that an empty replicate block is invalid.
func.func @verifier_replicate_empty_block() {
  "tf_device.replicate" () ({
// expected-error@-1 {{'tf_device.replicate' op expects a non-empty block}}
  ^entry:
  }) {n = 2 : i32, operand_segment_sizes = dense<[0, 0]> : vector<2xi32>} : () -> ()
  func.return
}

// -----

// Check that a replicate with a bad terminator is invalid.
func.func @verifier_replicate_terminator() {
  "tf_device.replicate" () ({
// expected-error@+2 {{'func.return' op expects parent op 'func.func'}}
  ^entry:
    func.return
  }) {n = 2 : i32, operand_segment_sizes = dense<[0, 0]> : vector<2xi32>} : () -> ()
  func.return
}

// -----

// Check that a replicate with 'n' attribute that is less than 2 is invalid.
func.func @verifier_replicate_n() {
  "tf_device.replicate" () ({
// expected-error@-1 {{'tf_device.replicate' op attribute 'n' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 2}}
  ^entry:
    tf_device.return
  }) {n = 1 : i32, operand_segment_sizes = dense<[0, 0]> : vector<2xi32>} : () -> ()
}

// -----

// Check that a replicate with mismatched 'n' attribute and device count is
// invalid.
func.func @verifier_replicate_n_device() {
  "tf_device.replicate" () ({
// expected-error@-1 {{'tf_device.replicate' op expects number of devices (2) to be equal to 'n' (3)}}
  ^entry:
    tf_device.return
  }) {devices = {TPU_REPLICATED_CORE_0 = ["/DEVICE:0", "/DEVICE:1"]}, n = 3 : i32, operand_segment_sizes = dense<[0, 0]> : vector<2xi32>} : () -> ()
}

// -----

// Check that replicate op's 'devices' attribute must consist of dictionary
// with values as list with size equal to 'n' attribute.
func.func @verifier_replicate_n_device_multiple_alias() {
  "tf_device.replicate" () ({
// expected-error@-1 {{'tf_device.replicate' op expects number of devices (2) to be equal to 'n' (3)}}
  ^entry:
    tf_device.return
  }) {devices = {TPU_REPLICATED_CORE_0 = ["/DEVICE:0", "/DEVICE:1"], TPU_REPLICATED_CORE_1 = ["/DEVICE:2"]}, n = 3 : i32, operand_segment_sizes = dense<[0, 0]> : vector<2xi32>} : () -> ()
}

// -----

// Check number of replicated inputs is evenly divisible by 'n'.
func.func @verifier_replicate_bad_operand_segment_sizes(%arg0: tensor<*xi32>) {
  "tf_device.replicate" (%arg0, %arg0, %arg0, %arg0) ({
// expected-error@-1 {{'tf_device.replicate' op expects number of replicated inputs (4) to be evenly divisible by 'n' (3)}}
  ^entry(%input0: tensor<*xi32>, %input1: tensor<*xi32>):
    tf_device.return
  }) {n = 3 : i32, operand_segment_sizes = dense<[4, 0]> : vector<2xi32>} : (tensor<*xi32>, tensor<*xi32>, tensor<*xi32>, tensor<*xi32>) -> ()
}

// -----

// Check number of replicated inputs / 'n' + number of packed inputs matches the
// number of block arguments.
func.func @verifier_replicate_num_block_args(%arg0: tensor<*xi32>) {
  "tf_device.replicate" (%arg0, %arg0, %arg0, %arg0, %arg0) ({
// expected-error@-1 {{'tf_device.replicate' op expects number of block arguments (2) to be equal to number of replicated inputs (3) / 'n' (3) + number of packed inputs (2)}}
  ^entry(%input0: tensor<*xi32>, %input1: tensor<*xi32>):
    tf_device.return
  }) {n = 3 : i32, operand_segment_sizes = dense<[3, 2]> : vector<2xi32>} : (tensor<*xi32>, tensor<*xi32>, tensor<*xi32>, tensor<*xi32>, tensor<*xi32>) -> ()
}

// -----

// Check that a replicate with incompatible replicated operand and block
// argument type is invalid.
func.func @verifier_replicate_replicated_operand_block_arg_type(%arg0: tensor<*xi32>, %arg1: tensor<*xi1>) {
  "tf_device.replicate" (%arg0, %arg1) ({
// expected-error@-1 {{'tf_device.replicate' op expects operand 1 ('tensor<*xi1>') and block argument 0 ('tensor<*xi32>') to have compatible types}}
  ^entry(%input0: tensor<*xi32>):
    tf_device.return
  }) {n = 2 : i32, operand_segment_sizes = dense<[2, 0]> : vector<2xi32>} : (tensor<*xi32>, tensor<*xi1>) -> ()
}

// -----

// Check that a replicate with incompatible packed operand and block argument
// type is invalid.
func.func @verifier_replicate_packed_operand_block_arg_type(%arg0: tensor<*xi1>) {
  "tf_device.replicate" (%arg0) ({
// expected-error@-1 {{'tf_device.replicate' op expects operand 0 ('tensor<*xi1>') and block argument 0 ('tensor<*xi32>') to have compatible types}}
  ^entry(%input0: tensor<*xi32>):
    tf_device.return
  }) {n = 2 : i32, operand_segment_sizes = dense<[0, 1]> : vector<2xi32>} : (tensor<*xi1>) -> ()
}

// -----

// Check that a replicate with mismatched result and terminator operand counts
// is invalid.
func.func @verifier_replicate_result_return_operand_count(%arg0: tensor<*xi32>) {
  %result:3 = "tf_device.replicate" () ({
// expected-error@-1 {{'tf_device.replicate' op expects number of results (3) to be equal to 'n' * number of terminator operands (2 * 1)}}
  ^entry:
    tf_device.return %arg0 : tensor<*xi32>
  }) {n = 2 : i32, operand_segment_sizes = dense<[0, 0]> : vector<2xi32>} : () -> (tensor<*xi32>, tensor<*xi32>, tensor<*xi32>)
}

// -----

// Check that a replicate with incompatible result and terminator operand type
// is invalid.
func.func @verifier_replicate_result_return_operand_type(%arg0: tensor<*xi32>) {
  %result:2 = "tf_device.replicate" () ({
// expected-error@-1 {{'tf_device.replicate' op incompatible types for result 1 and terminator operand 0}}
  ^entry:
    tf_device.return %arg0 : tensor<*xi32>
  }) {n = 2 : i32, operand_segment_sizes = dense<[0, 0]> : vector<2xi32>} : () -> (tensor<*xi32>, tensor<*xi1>)
}

// -----

// Check that a parallel_execute op with a single region is not allowed.
func.func @parallel_execute_single_region() {
  "tf_device.parallel_execute"() ({
// expected-error@-1 {{'tf_device.parallel_execute' op must have at least two regions.}}
    tf_device.return
  }) {} : () -> ()
  func.return
}

// -----

// Check that a parallel_execute op with empty regions are not allowed.
func.func @parallel_execute_empty_region() {
  "tf_device.parallel_execute"() ({
// expected-error@-1 {{'tf_device.parallel_execute' op region #0 ('regions') failed to verify constraint: region with 1 blocks}}
  },
  {
  tf_device.return
  }) {} : () -> ()
  func.return
}

// -----

// Check that a parallel_execute ops with invalid number of output types are
// not allowed.
func.func @parallel_execute_invalid_output_type_numbers() {
  "tf_device.parallel_execute"() ({
// expected-error@-1 {{'tf_device.parallel_execute' op number of output types (3) must match the total number of outputs from all regions (2).}}
    %0 = "tf.opA"() : () -> (tensor<*xi1>)
    %1 = "tf.opB"() : () -> (tensor<*xi32>)
    tf_device.return %0, %1 : tensor<*xi1>, tensor<*xi32>
  },
  {
    %2 = "tf.opC"() : () -> (tensor<*xi1>)
    tf_device.return
  }) {} : () -> (tensor<*xi1>, tensor<*xi32>, tensor<*xi32>)
  func.return
}

// -----

// Check that a parallel_execute ops with mismatching output types are not
// allowed.
func.func @parallel_execute_mismatched_output_types() {
  "tf_device.parallel_execute"() ({
// expected-error@-1 {{'tf_device.parallel_execute' op output types must be a concatenated list of output types for each regions.}}
    %0 = "tf.opA"() : () -> (tensor<*xi1>)
    %1 = "tf.opB"() : () -> (tensor<*xi32>)
    tf_device.return %0, %1 : tensor<*xi1>, tensor<*xi32>
  },
  {
    %2 = "tf.opC"() : () -> (tensor<*xi1>)
    tf_device.return
  }) {} : () -> (tensor<*xi1>, tensor<*xi1>)
  func.return
}

// -----

// Check that a parallel_execute ops with unmatching output types for
// multiple regions with data outputs are not allowed.
func.func @parallel_execute_regions_with_invalid_data_results() {
  "tf_device.parallel_execute"() ({
// expected-error@-1 {{'tf_device.parallel_execute' op output types must be a concatenated list of output types for each regions.}}
    %0 = "tf.opA"() : () -> (tensor<*xi1>)
    %1 = "tf.opB"() : () -> (tensor<*xi32>)
    tf_device.return %0, %1 : tensor<*xi1>, tensor<*xi32>
  },
  {
    %2 = "tf.opC"() : () -> (tensor<*xf32>)
    tf_device.return %2 : tensor<*xf32>
  }) {} : () -> (tensor<*xi1>, tensor<*xi32>, tensor<*xi1>)
  func.return
}
