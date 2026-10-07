tensorflow {
  global:
    *;

    # The target initializers are the one part of the LLVM C API that
    # TensorFlow calls: tfcompile's InitializeTargets(), which is linked into
    # libtensorflow_cc, imports them from this library. A global wildcard
    # takes precedence over the local LLVM* below.
    LLVMInitialize*;

  # The rest of the C API of the statically linked LLVM. Exported, it lets
  # another LLVM loaded into the same process, such as the one llvmlite and
  # numba bring in, resolve its C API calls against TensorFlow's copy instead
  # of its own.
  #
  # The llvm:: and mlir:: C++ symbols have to stay exported: libtensorflow_cc
  # links against them from this library.
  local:
    LLVM*;
};
