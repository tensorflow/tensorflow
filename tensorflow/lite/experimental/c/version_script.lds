VERS_1.0 {
  # Export symbols in c_api.h.
  global:
    *TFL_*;

  # Hide everything else.
  local:
    *;
};
