VERS_1.0 {
  # Export JNI symbols.
  global:
    Java_*;
    JNI_OnLoad;
    JNI_OnUnload;
    # TODO(b/138605512): Remove this and build separate .so for c++ api ?
    TfLiteHexagon*;

  # Hide everything else.
  local:
    *;
};