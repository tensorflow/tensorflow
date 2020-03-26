"""Definitions for using tools like saved_model_cli."""

load("//tensorflow:tensorflow.bzl", "if_xla_available")
load("//tensorflow/compiler/aot:tfcompile.bzl", "target_llvm_triple")

def _maybe_force_compile(args, force_compile):
    if force_compile:
        return args
    else:
        return if_xla_available(args)

def saved_model_compile_aot(
        name,
        directory,
        filegroups,
        cpp_class,
        checkpoint_path = None,
        tag_set = "serve",
        signature_def = "serving_default",
        variables_to_feed = "",
        target_triple = None,
        force_without_xla_support_flag = True,
        tags = None):
    """Compile a SavedModel directory accessible from a filegroup.

    This target rule takes a path to a filegroup directory containing a
    SavedModel and generates a cc_library with an AOT compiled model.
    For extra details, see the help for saved_model_cli's aot_compile_cpu help.

    **NOTE** Any variables passed to `variables_to_feed` *must be set by the
    user*.  These variables will NOT be frozen and their values will be
    uninitialized in the compiled object (this applies to all input
    arguments from the signature as well).

    Example usage:

    ```
    saved_model_compile_aot(
      name = "aot_compiled_x_plus_y",
      cpp_class = "tensorflow::CompiledModel",
      directory = "//tensorflow/cc/saved_model:testdata/x_plus_y_v2_debuginfo",
      filegroups = [
          "//tensorflow/cc/saved_model:saved_model_half_plus_two",
      ]
    )

    cc_test(
      name = "test",
      srcs = ["test.cc"],
      deps = [
        "//tensorflow/core:test_main",
        ":aot_compiled_x_plus_y",
        "//tensorflow/core:test",
        "//tensorflow/core/platform:logging",
      ]),
    )

    In "test.cc":

    #include "third_party/tensorflow/python/tools/aot_compiled_x_plus_y.h"

    TEST(Test, Run) {
      tensorflow::CompiledModel model;
      CHECK(model.Run());
    }
    ```

    Args:
      name: The rule name, and the name prefix of the headers and object file
        emitted by this rule.
      directory: The bazel directory containing saved_model.pb and variables/
        subdirectories.
      filegroups: List of `filegroup` targets; these filegroups contain the
        files pointed to by `directory` and `checkpoint_path`.
      cpp_class: The name of the C++ class that will be generated, including
        namespace; e.g. "my_model::InferenceRunner".
      checkpoint_path: The bazel directory containing `variables.index`.  If
        not provided, then `$directory/variables/` is used
        (default for SavedModels).
      tag_set: The tag set to use in the SavedModel.
      signature_def: The name of the signature to use from the SavedModel.
      variables_to_feed: (optional) The names of the variables to feed, a comma
        separated string, or 'all'.  If empty, all variables will be frozen and none
        may be fed at runtime.

        **NOTE** Any variables passed to `variables_to_feed` *must be set by
        the user*.  These variables will NOT be frozen and their values will be
        uninitialized in the compiled object (this applies to all input
        arguments from the signature as well).
      target_triple: The LLVM target triple to use (defaults to current build's
        target architecture's triple).
      force_without_xla_support_flag: Whether to compile even when
        `--define=with_xla_support=true` is not set.  If `False`, and the
        define is not passed when building, then the created `cc_library`
        will be empty.  In this case, downstream targets should
        conditionally build using macro `tfcompile.bzl:if_xla_available`.
        This flag is used by the TensorFlow build to avoid building on
        architectures that do not support XLA.
      tags: List of target tags.
    """
    saved_model = "{}/saved_model.pb".format(directory)
    target_triple = target_triple or target_llvm_triple()
    variables_to_feed = variables_to_feed or "''"
    if checkpoint_path:
        checkpoint_cmd_args = (
            "--checkpoint_path \"$$(dirname $(location {}/variables.index))\" "
                .format(checkpoint_path)
        )
        checkpoint_srcs = ["{}/variables.index".format(checkpoint_path)]
    else:
        checkpoint_cmd_args = ""
        checkpoint_srcs = []

    native.genrule(
        name = "{}_gen".format(name),
        srcs = filegroups + [saved_model] + checkpoint_srcs,
        outs = [
            "{}.h".format(name),
            "{}.o".format(name),
            "{}_metadata.o".format(name),
            "{}_makefile.inc".format(name),
        ],
        cmd = (
            "$(location :saved_model_cli) aot_compile_cpu " +
            "--dir \"$$(dirname $(location {}))\" ".format(saved_model) +
            checkpoint_cmd_args +
            "--output_prefix $(@D)/{} ".format(name) +
            "--cpp_class {} ".format(cpp_class) +
            "--variables_to_feed {} ".format(variables_to_feed) +
            "--target_triple " + target_triple + " " +
            "--tag_set {} ".format(tag_set)
        ),
        tags = tags,
        tools = [
            "//tensorflow/python/tools:saved_model_cli",
        ],
    )

    native.cc_library(
        name = name,
        srcs = _maybe_force_compile(
            [
                ":{}.o".format(name),
                ":{}_metadata.o".format(name),
            ],
            force_compile = force_without_xla_support_flag,
        ),
        hdrs = _maybe_force_compile(
            [
                ":{}.h".format(name),
            ],
            force_compile = force_without_xla_support_flag,
        ),
        tags = tags,
        deps = _maybe_force_compile(
            [
                "//tensorflow/compiler/tf2xla:xla_compiled_cpu_runtime_standalone",
            ],
            force_compile = force_without_xla_support_flag,
        ),
    )
