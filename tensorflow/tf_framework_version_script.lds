tensorflow {
  global:
    *;

  # The C API of the statically linked LLVM. Exported, it lets another LLVM
  # loaded into the same process, such as the one llvmlite and numba bring in,
  # resolve its C API calls against TensorFlow's copy instead of its own. No
  # other TensorFlow library imports these symbols.
  #
  # The llvm:: and mlir:: C++ symbols have to stay exported: libtensorflow_cc
  # links against them from this library.
  local:
    LLVM*;
};
