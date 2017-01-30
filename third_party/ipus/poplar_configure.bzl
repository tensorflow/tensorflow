def _poplar_autoconf_impl(repository_ctx):
  if "TF_POPLAR_BASE" in repository_ctx.os.environ:

    if not "TF_POPLAR_BASE" in repository_ctx.os.environ:
      fail("TF_POPNN_BASE is not set")

    poplar_base = repository_ctx.os.environ["TF_POPLAR_BASE"].strip()
    popnn_base = repository_ctx.os.environ["TF_POPNN_BASE"].strip()

    if poplar_base == "":
      fail("TF_POPLAR_BASE not specified")

    if popnn_base == "":
      fail("TF_POPNN_BASE not specified")

    if not repository_ctx.path(poplar_base + "/include").exists:
      fail("Cannot find poplar include path.")

    if not repository_ctx.path(poplar_base + "/lib").exists:
      fail("Cannot find poplar libary path.")

    if not repository_ctx.path(popnn_base + "/include").exists:
      fail("Cannot find popnn include path.")

    if not repository_ctx.path(popnn_base + "/lib").exists:
      fail("Cannot find popnn libary path.")

    repository_ctx.symlink(poplar_base + "/include", "poplar/include")
    repository_ctx.symlink(popnn_base + "/include", "popnn/include")

    repository_ctx.symlink(poplar_base + "/lib", "poplar/lib")
    repository_ctx.symlink(popnn_base + "/lib", "popnn/lib")

    repository_ctx.template("poplar/BUILD", Label("//third_party/ipus/poplar_lib:BUILD_poplar.tpl"), {})
    repository_ctx.template("popnn/BUILD", Label("//third_party/ipus/poplar_lib:BUILD_popnn.tpl"), {})

    return

  repository_ctx.template("poplar/BUILD", Label("//third_party/ipus/poplar_lib:BUILD_nopoplar.tpl"), {})
  repository_ctx.template("popnn/BUILD", Label("//third_party/ipus/poplar_lib:BUILD_nopopnn.tpl"), {})


poplar_configure = repository_rule(
  implementation = _poplar_autoconf_impl,
  local = True,
)

