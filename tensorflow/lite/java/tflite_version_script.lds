VERS_1.0 {
  # Export JNI and native C symbols.
  global:
    Java_*;
    JNI_OnLoad;
    JNI_OnUnload;
    TfLite*;

  # Required for libunwind. This is needed if built and then run internally.
    google_find_phdr;

  # Hide everything else.
  local:
    *;
};
