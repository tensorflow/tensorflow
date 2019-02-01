# Copyright 2017 Graphcore Ltd

def _poplar_autoconf_impl(repository_ctx):

  # Tensorflow build tag
  tf_poplar_build_tag = "UNKNOWN"
  if "TF_POPLAR_BUILD_TAG" in repository_ctx.os.environ:
    tf_poplar_build_tag = repository_ctx.os.environ["TF_POPLAR_BUILD_TAG"].strip()

  # Poplar release
  if "TF_POPLAR_BASE" in repository_ctx.os.environ:

    poplar_base = repository_ctx.os.environ["TF_POPLAR_BASE"].strip()

    if poplar_base == "":
      fail("TF_POPLAR_BASE not specified")

    if not repository_ctx.path(poplar_base + "/include").exists:
      fail("Cannot find poplar include path.")

    if not repository_ctx.path(poplar_base + "/lib").exists:
      fail("Cannot find poplar libary path.")

    if not repository_ctx.path(poplar_base + "/bin").exists:
      fail("Cannot find poplar bin path.")

    repository_ctx.symlink(poplar_base + "/include", "poplar/include")
    repository_ctx.symlink(poplar_base + "/lib", "poplar/lib")
    repository_ctx.symlink(poplar_base + "/bin", "poplar/bin")

    repository_ctx.template("poplar/BUILD",
        Label("//third_party/ipus/poplar_lib:BUILD_poplar.tpl"), {})
    repository_ctx.template("poplar/build_defs.bzl",
        Label("//third_party/ipus/poplar_lib:build_defs_poplar.tpl"),
        { "POPLAR_LIB_DIRECTORY" : poplar_base + "/lib",
	        "POPLIBS_LIB_DIRECTORY" : poplar_base + "/lib",
	        "TF_POPLAR_BUILD_TAG" : tf_poplar_build_tag })

    return

  # Poplar sandbox
  if "TF_POPLAR_SANDBOX" in repository_ctx.os.environ:

    poplar_base = repository_ctx.os.environ["TF_POPLAR_SANDBOX"].strip()

    if poplar_base == "":
      fail("TF_POPLAR_SANDBOX not specified")

    if not repository_ctx.path(poplar_base + "/poplar/include").exists:
      fail("Cannot find poplar/include path.")

    if not repository_ctx.path(poplar_base + "/poplibs/include").exists:
      fail("Cannot find poplibs/include path.")

    repository_ctx.symlink(poplar_base + "/poplar/include", "poplar/poplar/include")
    repository_ctx.symlink(poplar_base + "/poplar/lib", "poplar/poplar/lib")
    repository_ctx.symlink(poplar_base + "/poplar/bin", "poplar/poplar/bin")
    repository_ctx.symlink(poplar_base + "/poplibs/include", "poplar/poplibs/include")
    repository_ctx.symlink(poplar_base + "/poplibs/lib", "poplar/poplibs/lib")

    repository_ctx.template("poplar/BUILD",
        Label("//third_party/ipus/poplar_lib:BUILD_poplar_sandbox.tpl"), {})
    repository_ctx.template("poplar/build_defs.bzl",
        Label("//third_party/ipus/poplar_lib:build_defs_poplar.tpl"),
        { "POPLAR_LIB_DIRECTORY" : poplar_base + "/poplar/lib",
	        "POPLIBS_LIB_DIRECTORY" : poplar_base + "/poplibs/lib",
	        "TF_POPLAR_BUILD_TAG" : tf_poplar_build_tag })

    return


  # No Poplar
  repository_ctx.template("poplar/BUILD",
      Label("//third_party/ipus/poplar_lib:BUILD_nopoplar.tpl"), {})
  repository_ctx.template("poplar/build_defs.bzl",
      Label("//third_party/ipus/poplar_lib:build_defs_nopoplar.tpl"), {})


poplar_configure = repository_rule(
  implementation = _poplar_autoconf_impl,
  local = True,
)

