VERS_1.0 {
  # Hide libjpeg symbols to avoid symbol conflict with OpenCV
  local:
    *mlir*;
    # Do not re-export bundled hwloc. Its objects also land in other DSOs
    # (e.g. _xla_ops.so); hiding at compile time is the complete fix, this
    # is extra protection for libtensorflow_framework.so (tensorflow#125854).
    hwloc*;
};
