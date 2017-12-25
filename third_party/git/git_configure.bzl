"""Repository rule for Git autoconfiguration."""

def _git_conf_impl(repository_ctx):
  repository_ctx.template(
      "BUILD",
      Label("//third_party/git:BUILD.tpl"))

  tensorflow_root_path = str(repository_ctx.path(
      Label("@org_tensorflow//:BUILD")))[:-len("BUILD")]
  python_script_path = repository_ctx.path(
      Label("@org_tensorflow//tensorflow/tools/git:gen_git_source.py"))
  generated_files_path = repository_ctx.path("gen")

  repository_ctx.execute([
      python_script_path, "--configure", tensorflow_root_path,
      "--gen_root_path", generated_files_path], quiet=False)

git_configure = repository_rule(
    implementation = _git_conf_impl,
)
