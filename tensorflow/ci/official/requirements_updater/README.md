## Managing hermetic Python

To make sure that TensorFlow's build is reproducible, behaves uniformly across
supported platforms (Linux, Windows, MacOS) and is properly isolated from
specifics of a local system, we rely on hermetic Python (see
[rules_python](https://github.com/bazelbuild/rules_python)) for all build
and test commands executed via Bazel. This means that your system Python
installation will be ignored during the build and Python interpreter itself
as well as all the Python dependencies will be managed by bazel directly.

### Specifying Python version

The hermetic Python version is controlled by `HERMETIC_PYTHON_VERSION`
environment variable, which could be setin one of the following ways:

```
# Either add an entry to your `.bazelrc` file
build --repo_env=HERMETIC_PYTHON_VERSION=3.12

# OR pass it directly to your specific build command
bazel build <target> --repo_env=HERMETIC_PYTHON_VERSION=3.12

# OR set the environment variable globally in your shell:
export HERMETIC_PYTHON_VERSION=3.12
```

You may run builds and tests against different versions of Python sequentially
on the same machine by simply switching the value of `HERMETIC_PYTHON_VERSION`
between the runs. All the python-agnostic parts of the build cache from the
previous build will be preserved and reused for the subsequent builds.

### Specifying Python dependencies

During bazel build all TensorFlow's Python dependencies are pinned to their
specific versions. This is necessary to ensure reproducibility of the build.
The pinned versions of the full transitive closure of TensorFlow's dependencies
together with their corresponding hashes are specified in
`requirements_lock_<python version>.txt` files (e.g.
`requirements_lock_3_12.txt` for `Python 3.12`).

To update the lock files, make sure
`ci/official/requirements_updater/requirements.in` contains the desired direct
dependencies list and then execute the following command (which will call
[pip-compile](https://pypi.org/project/pip-tools/) under the hood):

```
bazel run //ci/official/requirements_updater:requirements.update --repo_env=HERMETIC_PYTHON_VERSION=3.12
```

where `3.12` is the `Python` version you wish to update.

Note, since it is still `pip` and `pip-compile` tools used under the hood, so
most of the command line arguments and features supported by those tools will be
acknowledged by the Bazel requirements updater command as well. For example, if
you wish the updater to consider pre-release versions simply pass `--pre`
argument to the bazel command:

```
bazel run //ci/official/requirements_updater:requirements.update --repo_env=HERMETIC_PYTHON_VERSION=3.12 -- --pre
```

If you need to upgrade all of the packages in requirements lock file, just pass
the `--upgrade` parameter:

```
bazel run //ci/official/requirements_updater:requirements.update --repo_env=HERMETIC_PYTHON_VERSION=3.12 -- --upgrade
```

For the full set of supported parameters please check
[pip-compile](https://pip-tools.readthedocs.io/en/latest/cli/pip-compile/)
documentation