VERS_1.0 {
  # Export symbols in c_api.h.
  global:
    TfLite*;

  # Hide everything else.
  local:
    *;
};
