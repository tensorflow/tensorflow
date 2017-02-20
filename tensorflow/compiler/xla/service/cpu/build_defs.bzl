"""build_defs for service/cpu."""

def runtime_copts():
  """Returns copts used for CPU runtime libraries."""
  return (["-DEIGEN_AVOID_STL_ARRAY"] +
          select({
              "//tensorflow:android_arm": ["-mfpu=neon"],
              "//conditions:default": []}) +
          select({
              "//tensorflow:android": ["-O2"],
              "//conditions:default": []}))
