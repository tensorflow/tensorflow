VERS_1.0 {
  # Export JNI symbols.
  global:
    Java_*;
    JNI_OnLoad;
    JNI_OnUnload;

  # Hide everything else.
  local:
    *;
};
