# Hermetic Python

Hermetic Python allows not to rely on system-installed Python, and
system-installed Python packages. \
Instead, an independent Python toolchain is registered, ensuring the right
dependencies are always used. \
See https://github.com/bazelbuild/rules_python/ for more details.

### Specifying the Python version

Note: Only a number of minor Python versions are supported at any given time.

By default, the lowest supported version is used.

To set a different version, use the `TF_PYTHON_VERSION` environment variable,
e.g.

```
export TF_PYTHON_VERSION=3.11
```

To specify the version via a Bazel command argument, use the following:

```
--repo_env=TF_PYTHON_VERSION=3.11
```

## Requirements updater

Requirements updater is a standalone tool, intended to simplify process of
updating requirements for multiple minor versions of Python.

It takes in a file with a set of dependencies, and produces a more detailed
requirements file for each version, with hashes specified for each
dependency required, as well as their sub-dependencies.

### How to update/add requirements

By default, the name of the base requirements file is `requirements.in`, but it
can be set using the `REQUIREMENTS_FILE_NAME` variable. \
For example:

```
export REQUIREMENTS_FILE_NAME=my_requirements.in
```

To specify the file via a Bazel command argument, use the following:

```
--repo_env=REQUIREMENTS_FILE_NAME=my_requirements.in
```

### How to run the updater

```
bash updater.sh
```

## How to add a new Python version

Note: Updating the
[rules-python](https://github.com/bazelbuild/rules_python/releases) version may
be required before going through the steps below. This is due to the new Python
versions becoming available through `rules-python`. \
See
[here](https://github.com/tensorflow/tensorflow/commit/f91457f258fdd78f693044a57efa63a38335d1de),
and
[here](https://github.com/tensorflow/tensorflow/commit/052445e04ce20fd747657e0198a1bcec2b6dff5b),
for an example.

See
[this commit](https://github.com/tensorflow/tensorflow/commit/5f7f05a80aac9b01325a78ec3fcff0dbedb1cc23)
as a rough example of the steps below.

All the files referenced below are located in the same directory as this README,
unless indicated otherwise.

1) Add the new version to the `VERSIONS` variable inside
   `tensorflow/tools/toolchains/python/python_repo.bzl`. \
   While this isn't necessary for running the updater, it is required for
   actually using the new version with Tensorflow.

2) In the `WORKSPACE` file, add the new version to the `python_versions`
   parameter of the `python_register_multi_toolchains` function.

3) In the `BUILD.bazel` file, add a load statement for the new version, e.g.

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

   ```
      compile_pip_requirements_3_11(
          name = "requirements_3_11_release",
          extra_args = [
              "--allow-unsafe",
              "-P keras-nightly",
              "-P tb-nightly",
          ],
          requirements_in = "requirements.in",
          requirements_txt = "requirements_lock_3_11.txt",
      )
   ```

4) Add the version to `SUPPORTED_VERSIONS` in `updater.sh`, and
   `release_updater.sh`

5) Run the `updater.sh` shell script. \
   If the base requirements file hasn't yet been updated to account for the new
   Python version, which will require different versions for at least some
   dependencies, it will need to be updated now, for the script to run
   successfully.

6) A new `requirements_lock_3_11.txt` file should appear under the root of the
   `tensorflow` directory.
