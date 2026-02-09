"""
A rule to generate a requirements.in file for a given set of dependencies.
"""

def _py_deps_profile_impl(ctx):
    is_switch = False
    for var_name, var_val in ctx.attr.switch.items():
        is_switch = is_switch or ctx.os.environ.get(var_name, "") == var_val

    prefix = ctx.attr.pip_repo_name
    requirements_name = ctx.attr.requirements_in.name
    requirements_in_substitutions = {}
    build_content = ['exports_files(["{}"])'.format(requirements_name)]
    for k, v in ctx.attr.deps_map.items():
        repo_name = v[0] if is_switch else k
        requirements_in_substitutions[k + "\n"] = repo_name + "\n"
        requirements_in_substitutions[k + "\r\n"] = repo_name + "\r\n"
        aliased_targets = ["pkg"] + v[1:]
        norm_repo_name = repo_name.replace("-", "_")
        norm_alas_name = k.replace("-", "_")
        for target in aliased_targets:
            alias_name = "{}_{}".format(norm_alas_name, target)
            alias_value = "@{}_{}//:{}".format(prefix, norm_repo_name, target)
            build_content.append("""
alias(
    name = "{}",
    actual = "{}",
    visibility = ["//visibility:public"]
)
""".format(alias_name, alias_value))

    ctx.file("BUILD", "".join(build_content))
    ctx.template(
        requirements_name,
        ctx.attr.requirements_in,
        executable = False,
        substitutions = requirements_in_substitutions,
    )

py_deps_profile = repository_rule(
    implementation = _py_deps_profile_impl,
    attrs = {
        "requirements_in": attr.label(mandatory = True),
        "deps_map": attr.string_list_dict(mandatory = True),
        "pip_repo_name": attr.string(mandatory = True),
        "switch": attr.string_dict(mandatory = True),
    },
    local = True,
)
