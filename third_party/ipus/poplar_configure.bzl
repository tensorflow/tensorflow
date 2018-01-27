# Copyright 2017 Graphcore Ltd

def _poplar_autoconf_impl(repository_ctx):
  if "TF_POPLAR_BASE" in repository_ctx.os.environ:

    poplar_base = repository_ctx.os.environ["TF_POPLAR_BASE"].strip()

    if poplar_base == "":
      fail("TF_POPLAR_BASE not specified")

    if not repository_ctx.path(poplar_base + "/include").exists:
      fail("Cannot find poplar include path.")

    if not repository_ctx.path(poplar_base + "/lib").exists:
      fail("Cannot find poplar libary path.")

    repository_ctx.symlink(poplar_base + "/colossus", "poplar/colossus")
    repository_ctx.symlink(poplar_base + "/include", "poplar/include")
    repository_ctx.symlink(poplar_base + "/lib", "poplar/lib")
    repository_ctx.symlink(poplar_base + "/bin", "poplar/bin")

    repository_ctx.template("poplar/BUILD",
        Label("//third_party/ipus/poplar_lib:BUILD_poplar.tpl"), {})
    repository_ctx.template("poplar/build_defs.bzl",
        Label("//third_party/ipus/poplar_lib:build_defs_poplar.tpl"),
        { "POPLAR_LIB_DIRECTORY" : poplar_base + "/lib" })

    return

  repository_ctx.template("poplar/BUILD",
      Label("//third_party/ipus/poplar_lib:BUILD_nopoplar.tpl"), {})
  repository_ctx.template("poplar/build_defs.bzl",
      Label("//third_party/ipus/poplar_lib:build_defs_nopoplar.tpl"), {})


poplar_configure = repository_rule(
  implementation = _poplar_autoconf_impl,
  local = True,
)

