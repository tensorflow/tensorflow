tensorflow {
  global:
    *;
  local:
    # Hide leftover unprefixed hwloc_* names. Bundled APIs are renamed
    # tf_hwloc_* (HWLOC_SYM_TRANSFORM) so they do not interpose system hwloc
    # and can still be shared across TF DSOs (tensorflow#125854).
    hwloc*;
};
