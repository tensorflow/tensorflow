load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("@python//:defs.bzl", "compile_pip_requirements")
load("@python_version_repo//:py_version.bzl", "REQUIREMENTS")

compile_pip_requirements(
    name = "requirements",
    extra_args = [
        "--allow-unsafe",
        "--build-isolation",
    ],
    generate_hashes = True,
    requirements_in = "requirements.in",
    requirements_txt = REQUIREMENTS,
)

compile_pip_requirements(
    name = "requirements_nightly",
    data = ["test-requirements.txt"],
    extra_args = [
        "--allow-unsafe",
        "--build-isolation",
        "--extra-index-url=https://pypi.anaconda.org/scientific-python-nightly-wheels/simple",
        "--pre",
        "--upgrade",
    ],
    generate_hashes = False,
    requirements_in = "requirements.in",
    requirements_txt = REQUIREMENTS,
)

compile_pip_requirements(
    name = "requirements_dev",
    extra_args = [
        "--allow-unsafe",
        "--build-isolation",
        "--upgrade",
    ],
    generate_hashes = False,
    requirements_in = "requirements.in",
    requirements_txt = REQUIREMENTS,
)

# Flag indicating if the target requires pre-built wheel.
bool_flag(
    name = "wheel_dependency",
    build_setting_default = False,
)

config_setting(
    name = "enable_wheel_dependency",
    flag_values = {
        ":wheel_dependency": "True",
    },
)

filegroup(
    name = "manylinux_compliance_test",
    srcs = ["manylinux_compliance_test.py"],
    visibility = ["//visibility:public"],
)
