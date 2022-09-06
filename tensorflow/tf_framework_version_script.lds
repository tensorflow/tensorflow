VERS_1.0 {
  # Hide libjpeg symbols to avoid symbol conflict with OpenCV
  local:
    jpeg_*;
    jinit_*;
    jdiv_round_up;
    jround_up;
    jzero_far;
    jcopy_*;
    jsimd_*;
    hwloc_*;
    *mlir*;
};
