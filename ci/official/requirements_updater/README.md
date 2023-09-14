### Hermetic Python

Hermetic Python allows us not to rely on system-installed python and
system-installed python packages, instead we register our own python toolchain.
See https://github.com/bazelbuild/rules_python/ for more details.

#### Hermetic Python toolchain details

By default, Python 3.9 is used.

To set your own version for hermetic Python toolchain, use `TF_PYTHON_VERSION`
environment variable, e.g.

```
export TF_PYTHON_VERSION=3.10
```

To set a version from argument line, add to your command

```
--repo_env=TF_PYTHON_VERSION=3.10
```

### Requirements updater

Requirements updater is a standalone tool intended to simplify process of
updating requirements for multiple versions of Python.

#### How to update/add requirements

By default, the name of the input requirements file is `requirements.in`,
but it can be set using the `REQUIREMENTS_FILE_NAME` variable, for example:
```
export REQUIREMENTS_FILE_NAME=`my_requirements.in`
```

To set a version from the argument line, add to your command
```
--repo_env=REQUIREMENTS_FILE_NAME=`my_requirements.in`
```

#### How to run the updater

```
bash updater.sh
```

### How to add a new Python version

1) In the `WORKSPACE` file add a new version to `python_versions` argument of
the `python_register_multi_toolchains` function.

2) In `BUILD.bazel` file add a load statement for the new version, e.g.

```
load("@python//3.11:defs.bzl",
     compile_pip_requirements_3_11 = "compile_pip_requirements")
```

Add a new entry for the loaded `compile_pip_requirements`, e.g.

```
compile_pip_requirements_3_11(
    name = "requirements_3_11",
    extra_args = ["--allow-unsafe"],
    requirements_in = "requirements.in",
    requirements_txt = "requirements_lock_3_11.txt",
)
```

3) Add the version to `SUPPORTED_VERSIONS` in `updater.sh`, after that run the
 requirements updater tool.

4) As a result, a new `requirements_lock_3_11.txt` file should appear under the
root of tensorflow directory.
