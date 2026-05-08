VERS_1.0 {
  # Export JNI and native C symbols.
  global:
    Java_*;
    TfLiteGpu*;

  # Hide everything else.
  local:
    *;
};
