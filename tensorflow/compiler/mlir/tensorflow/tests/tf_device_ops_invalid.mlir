// RUN: tf-opt %s -split-input-file -verify-diagnostics

// Check that a replicate with 'n' attribute that is less than 2 is invalid.
func @parser_replicate_n(%arg0: tensor<*xf32>) {
  tf_device.replicate([%arg0] as %input0: tensor<*xf32>) {n = 1 : i32} {
// expected-error@-1 {{'tf_device.replicate' expects 'n' to be at least 2, got 1}}
  }
  return
}

// -----

// Check that a replicate replicated inputs where operand sizes do not match
// 'n' is invalid.
func @parser_replicate_operand_count(%arg0: tensor<*xf32>) {
  tf_device.replicate([%arg0, %arg0, %arg0] as %input0: tensor<*xf32>) {n = 2 : i32} {
// expected-error@-1 {{'tf_device.replicate' expects number of operands for replicated input 0 to be 'n' (2), got 3}}
  }
  return
}

// -----

// Check that a replicate with incompatible operands and block argument type is
// invalid.
func @parser_replicate_operand_type(%arg0: tensor<*xi32>) {
// expected-note@-1 {{prior use here}}
  tf_device.replicate([%arg0, %arg0] as %input0: tensor<*xf32>) {n = 2 : i32} {
// expected-error@-1 {{use of value '%arg0' expects different type than prior uses: 'tensor<*xf32>' vs 'tensor<*xi32>'}}
  }
  return
}

// -----

// Check that a replicate with multiple blocks in its region is invalid.
func @parser_replicate_region() {
  tf_device.replicate() {n = 2 : i32} {
// expected-error@-1 {{custom op 'tf_device.replicate' expects a single block region}}
    br ^bb
  ^bb:
    tf_device.return
  }
  return
}

// -----

// Check that a replicate with a bad terminator is invalid.
func @parser_replicate_terminator() {
  tf_device.replicate() {n = 2 : i32} {
// expected-error@-1 {{custom op 'tf_device.replicate' expects a tf_device.return terminator}}
    return
  }
  return
}

// -----

// Check that an empty replicate is invalid (replicate needs a region).
func @verifier_replicate_no_block() {
  "tf_device.replicate" () ({
// expected-error@-1 {{'tf_device.replicate' op region #0 ('body') failed to verify constraint: region with 1 blocks}}
  }) {n = 2 : i32} : () -> ()
  return
}

// -----

// Check that an empty replicate block is invalid.
func @verifier_replicate_empty_block() {
  "tf_device.replicate" () ({
// expected-error@-1 {{'tf_device.replicate' op expects a non-empty block}}
  ^entry:
  }) {n = 2 : i32} : () -> ()
  return
}

// -----

// Check that a replicate with a bad terminator is invalid.
func @verifier_replicate_terminator() {
// expected-note@+1 {{in custom textual format, the absence of terminator implies 'tf_device.return'}}
  "tf_device.replicate" () ({
// expected-error@-1 {{'tf_device.replicate' op expects regions to end with 'tf_device.return', found 'std.return'}}
  ^entry:
    return
  }) {n = 2 : i32} : () -> ()
  return
}

// -----

// Check that a replicate with 'n' attribute that is less than 2 is invalid.
func @verifier_replicate_n() {
  "tf_device.replicate" () ({
// expected-error@-1 {{'tf_device.replicate' op attribute 'n' failed to satisfy constraint: 32-bit integer attribute whose minimal value is 2}}
  ^entry:
    tf_device.return
  }) {n = 1 : i32} : () -> ()
}

// -----

// Check that a replicate with mismatched 'n' attribute and device count is
// invalid.
func @verifier_replicate_n_device() {
  "tf_device.replicate" () ({
// expected-error@-1 {{'tf_device.replicate' op expects number of devices (2) to be equal to 'n' (3)}}
  ^entry:
    tf_device.return
  }) {n = 3 : i32, devices = ["/DEVICE:0", "/DEVICE:1"]} : () -> ()
}

// -----

// Check that a replicate with mismatched operand and block arg counts is
// invalid.
func @verifier_replicate_operand_block_arg_count(%arg0: tensor<*xi32>) {
  "tf_device.replicate" (%arg0, %arg0, %arg0) ({
// expected-error@-1 {{'tf_device.replicate' op expects number of operands (3) to be equal to 'n' * number of block arguments (2 * 1)}}
  ^entry(%input0: tensor<*xi32>):
    tf_device.return
  }) {n = 2 : i32} : (tensor<*xi32>, tensor<*xi32>, tensor<*xi32>) -> ()
}

// -----

// Check that a replicate with incompatible operand and block argument type is
// invalid.
func @verifier_replicate_operand_block_arg_type(%arg0: tensor<*xi32>, %arg1: tensor<*xi1>) {
  "tf_device.replicate" (%arg0, %arg1) ({
// expected-error@-1 {{'tf_device.replicate' op incompatible types for operand 1 and block argument 0}}
  ^entry(%input0: tensor<*xi32>):
    tf_device.return
  }) {n = 2 : i32} : (tensor<*xi32>, tensor<*xi1>) -> ()
}

// -----

// Check that a replicate with mismatched result and terminator operand counts
// is invalid.
func @verifier_replicate_result_return_operand_count(%arg0: tensor<*xi32>) {
  %result:3 = "tf_device.replicate" () ({
// expected-error@-1 {{'tf_device.replicate' op expects number of results (3) to be equal to 'n' * number of terminator operands (2 * 1)}}
  ^entry:
    tf_device.return %arg0 : tensor<*xi32>
  }) {n = 2 : i32} : () -> (tensor<*xi32>, tensor<*xi32>, tensor<*xi32>)
}

// -----

// Check that a replicate with incompatible result and terminator operand type
// is invalid.
func @verifier_replicate_result_return_operand_type(%arg0: tensor<*xi32>) {
  %result:2 = "tf_device.replicate" () ({
// expected-error@-1 {{'tf_device.replicate' op incompatible types for result 1 and terminator operand 0}}
  ^entry:
    tf_device.return %arg0 : tensor<*xi32>
  }) {n = 2 : i32} : () -> (tensor<*xi32>, tensor<*xi1>)
}
