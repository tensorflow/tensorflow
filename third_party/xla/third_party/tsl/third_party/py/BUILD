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
