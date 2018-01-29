"""build_defs for service/cpu."""


def runtime_copts():
  """Returns copts used for CPU runtime libraries."""
  return (["-DEIGEN_AVOID_STL_ARRAY"] + select({
      "//tensorflow:android_arm": ["-mfpu=neon"],
      "//conditions:default": []
  }) + select({
      "//tensorflow:android": ["-O2"],
      "//conditions:default": []
  }))

def append_polly_dep_if_enabled(basic_deps):
    """Shorthand for select()'ing on whether we're building with Polly.
    Returns a select statement which evaluates to basic_deps if we're building
    with Polly enabled.  Otherwise, the select statement evaluates to basic_deps.
    """
    return select({
        "//third_party/llvm:using_polly": basic_deps.append("@llvm//:polly"),
        "//conditions:default": basic_deps
    })

