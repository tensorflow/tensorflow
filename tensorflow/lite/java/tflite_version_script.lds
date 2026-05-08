VERS_1.0 {
  # Export JNI and native C symbols.
  global:
    Java_*;
    TfLite*;

  # Hide everything else.
  local:
    *;
};
