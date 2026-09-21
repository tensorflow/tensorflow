tensorflow {
  global:
    *;

  # Statically linked third-party code whose symbols must not reach the
  # process-wide resolution scope. LLVM in particular collides with other
  # copies loaded into the same process, for example the LLVM that llvmlite
  # and numba bring in, which resolve against TensorFlow's build and crash on
  # the mismatched layout.
  #
  # These mirror the entries already hidden on macOS in tf_private_symbols.lds
  # and are deliberately narrow: TensorFlow's own symbols stay exported,
  # because custom op libraries loaded through tf.load_op_library resolve
  # REGISTER_OP and REGISTER_KERNEL_BUILDER against this library.
  local:
    *llvm*;
    *mlir*;
};
