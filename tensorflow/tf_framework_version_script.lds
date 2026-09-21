tensorflow {
  global:
    *;

  # Statically linked third-party code whose symbols must not reach the
  # process-wide resolution scope. LLVM in particular collides with other
  # copies loaded into the same process, for example the LLVM that llvmlite
  # and numba bring in, which resolve against TensorFlow's build and crash on
  # the mismatched layout.
  #
  # Matched by namespace rather than by substring so that TensorFlow's own
  # symbols are not caught: a plain *mlir* would also hide names such as
  # tensorflow::tf_xla_test_use_mlir. TensorFlow's symbols must stay exported,
  # because custom op libraries loaded through tf.load_op_library resolve
  # REGISTER_OP and REGISTER_KERNEL_BUILDER against this library.
  #
  # LLVM* covers the LLVM C API, whose symbols are not in the llvm namespace.
  local:
    extern "C++" {
      llvm::*;
      mlir::*;
    };
    LLVM*;
};
