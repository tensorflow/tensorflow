VERS_1.0 {
  # Hide libjpeg symbols to avoid symbol conflict with OpenCV
  local:
    jpeg_*;
    jinit_*;
    jcopy_*;
    jsimd_*;
    hwloc_*;
    *mlir*;
};
