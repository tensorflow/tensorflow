VERS_1.0 {
  # Export symbols in c_api.h.
  global:
    *TF_*;
    *TFE_*;

  # Hide everything else.
  local:
    *;
};
