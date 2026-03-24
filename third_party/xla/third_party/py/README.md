# ML wheels

Provides a standardized and efficient system for packaging and verifying ML
software.

## ML wheels features

- Standardized creation, validation (`auditwheel`) and testing of wheel
  artifacts.

- Availability of the final wheel artifacts in the Bazel Build phase, which
  enables testing of generated wheels by regular `py_test` targets
  together with the rest of the existing tests.

- Ability to use Bazel RBE for wheel creation and testing.

- Reproducible and unified steps for generating testing of the wheels on
  different platforms.

## Getting started

1. Integrate hermetic Python, C++ and CUDA (if needed) toolchains in the
   project.

   Examples:

   [JAX hermetic Python and C++ integration](https://github.com/jax-ml/jax/blob/006b2904720bf029cb4298ab963f8f50438e79df/WORKSPACE#L16-L84)

   [JAX hermetic CUDA integration](https://github.com/jax-ml/jax/blob/006b2904720bf029cb4298ab963f8f50438e79df/WORKSPACE#L131-L196)

   [TensorFlow hermetic Python integration](https://github.com/tensorflow/tensorflow/blob/5feca557408c3552494c4db03a02b36f9817bd37/WORKSPACE#L32-L67)

   [TensorFlow hermetic C++ and CUDA integration](https://github.com/tensorflow/tensorflow/blob/5feca557408c3552494c4db03a02b36f9817bd37/WORKSPACE#L94-L170)

2. Create python script that produces a wheel, and declare it as `py_binary`
   build rule.

   A common case scenario: a python script should take wheel sources provided in
   the arguments list, then do the required transformations and run command like
   `python -m build` in the folder with the collected resources.

   [JAX py_binary declaration](https://github.com/jax-ml/jax/blob/006b2904720bf029cb4298ab963f8f50438e79df/jaxlib/tools/BUILD.bazel#L230-L241)

   [TensorFlow py_binary declaration](https://github.com/tensorflow/tensorflow/blob/5feca557408c3552494c4db03a02b36f9817bd37/tensorflow/tools/pip_package/BUILD#L242-L253)

3. Create Bazel build rule that returns python wheel in the output.

   In a common case scenario, this Bazel rule runs `py_binary` (created in
   step 1) passed in the rule attributes.

   [JAX rule definition](https://github.com/tensorflow/tensorflow/blob/5feca557408c3552494c4db03a02b36f9817bd37/tensorflow/tools/pip_package/BUILD#L242-L253)

   [TensorFlow rule definition](https://github.com/tensorflow/tensorflow/blob/5feca557408c3552494c4db03a02b36f9817bd37/tensorflow/tools/pip_package/utils/tf_wheel.bzl#L137-L154)

   - The wheel sources should be provided in the wheel build rule attributes.

     To collect the wheel sources that are suitable for all types of Bazel
     builds, including cross-compile builds, the following build rules should be
     used: `collect_data_files`, `transitive_py_deps` from
     `@xla//third_party/py:python_wheel.bzl`, and `transitive_hdrs` from
     `@xla//xla/tsl:tsl.bzl`.

     [jaxlib wheel sources](https://github.com/jax-ml/jax/blob/006b2904720bf029cb4298ab963f8f50438e79df/jaxlib/tools/BUILD.bazel#L243-L265)

     [TensorFlow wheel sources](https://github.com/tensorflow/tensorflow/blob/5feca557408c3552494c4db03a02b36f9817bd37/tensorflow/tools/pip_package/BUILD#L312-L367)

   - the wheel name should conform to
     [PEP-491 naming convention](https://peps.python.org/pep-0491/#file-name-convention).

     [JAX example](https://github.com/jax-ml/jax/blob/006b2904720bf029cb4298ab963f8f50438e79df/jaxlib/jax.bzl#L326-L348)

     [TensorFlow example](https://github.com/tensorflow/tensorflow/blob/5feca557408c3552494c4db03a02b36f9817bd37/tensorflow/tools/pip_package/utils/tf_wheel.bzl#L56-L69)

   - Storing of the wheel version is custom, and should be implemented per
     project. It can be additional repository rule, or a constant in .bzl file.

     [JAX example](https://github.com/jax-ml/jax/blob/006b2904720bf029cb4298ab963f8f50438e79df/WORKSPACE#L108-L114)

     [Tensorflow example](https://github.com/tensorflow/tensorflow/blob/5feca557408c3552494c4db03a02b36f9817bd37/tensorflow/tf_version.bzl#L11)

   - The wheel suffix is controlled by a common repository rule
     `python_wheel_version_suffix_repository`, that should be called in
     `WORKSPACE` file.

     [JAX rule call](https://github.com/jax-ml/jax/blob/006b2904720bf029cb4298ab963f8f50438e79df/WORKSPACE#L127-L129)

     [Tensorflow rule call](https://github.com/tensorflow/tensorflow/blob/5feca557408c3552494c4db03a02b36f9817bd37/WORKSPACE#L92)

4. To verify manylinux tag compliance, use common py_binary
  `verify_manylinux_compliance_test`.

  [JAX tests](https://github.com/jax-ml/jax/blob/006b2904720bf029cb4298ab963f8f50438e79df/jaxlib/tools/BUILD.bazel#L626-L668)

  [Tensorflow test](https://github.com/tensorflow/tensorflow/blob/5feca557408c3552494c4db03a02b36f9817bd37/tensorflow/tools/pip_package/BUILD#L441-L450)

5. With the wheel build rule defined, one can run Bazel test targets dependent
  on the wheel instead of individual Bazel targets. To implement it, define
  `py_import` call. `py_import` target can be used in other python targets in
  the same way as `py_library`.

  [JAX example](https://github.com/jax-ml/jax/blob/006b2904720bf029cb4298ab963f8f50438e79df/jaxlib/tools/BUILD.bazel#L542-L570)

  [Tensorflow example](https://github.com/tensorflow/tensorflow/blob/5feca557408c3552494c4db03a02b36f9817bd37/tensorflow/tools/pip_package/BUILD#L452-L485)

  [TensorFlow tests dependent on `py_import`](https://github.com/tensorflow/tensorflow/blob/5feca557408c3552494c4db03a02b36f9817bd37/tensorflow/tools/pip_package/BUILD#L411-L439)
