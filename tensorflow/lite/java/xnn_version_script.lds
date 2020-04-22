VERS_1.0 {
  # Export JNI and native C symbols.
  global:
    Java_*;
    JNI_OnLoad;
    JNI_OnUnload;
    TfLiteXNNPack*;

  # Hide everything else.
  local:
    *;
};
