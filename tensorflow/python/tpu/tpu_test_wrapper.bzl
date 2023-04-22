# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Provides Starlark helpers for Cloud TPU."""

def get_kwargs_for_wrapping(
        name,
        tags = None,
        args = [],
        **kwargs):
    """Generates the kwargs for constructing a wrapped TPU test.

    Args:
        name: Name of test. Will be prefixed by accelerator versions.
        tags: BUILD tags to apply to tests.
        args: Arguments to apply to tests.
        **kwargs: Additional named arguments to apply to tests.

    Returns:
        A dict to be splatted into a py_binary or py_test.
    """
    tags = tags or []

    tags = [
        "tpu",
        "no_pip",
        "no_gpu",
        "nomac",
        "local",
    ] + tags

    test_main = kwargs.get("srcs")
    if not test_main or len(test_main) > 1:
        fail('"srcs" should be a list of exactly one python file.')
    test_main = test_main[0]

    wrapper_src = _copy_test_source(
        "//tensorflow/python/tpu:tpu_test_wrapper.py",
    )

    # deps might be either a list or a depset, so we standardize here.
    deps = kwargs["deps"]
    if type(deps) == type(list()):
        deps = depset(deps)

    kwargs["python_version"] = kwargs.get("python_version", "PY3")
    kwargs["srcs"] = [wrapper_src] + kwargs["srcs"]
    kwargs["deps"] = depset(
        ["//tensorflow/python/tpu:tpu_test_deps"],
        transitive = [deps],
    )
    kwargs["main"] = wrapper_src

    args = [
        "--wrapped_tpu_test_module_relative=.%s" % test_main.rsplit(".", 1)[0],
    ] + args

    kwargs["name"] = name
    kwargs["tags"] = tags
    kwargs["args"] = args

    return kwargs

def _copy_test_source(src):
    """Creates a genrule copying src into the current directory.

    This silences a Bazel warning, and is necessary for relative import of the
    user test to work.

    This genrule checks existing rules to avoid duplicating the source if
    another call has already produced the file. Note that this will fail
    weirdly if two source files have the same filename, as whichever one is
    copied in first will win and other tests will unexpectedly run the wrong
    file. We don't expect to see this case, since we're only copying the one
    test wrapper around.

    Args:
        src: The source file we would like to use.

    Returns:
        The path of a copy of this source file, inside the current package.
    """
    name = src.rpartition(":")[-1].rpartition("/")[-1]  # Get basename.

    new_main = "%s/%s" % (native.package_name(), name)
    new_name = "_gen_" + name

    if not native.existing_rule(new_name):
        native.genrule(
            name = new_name,
            srcs = [src],
            outs = [new_main],
            cmd = "cp $< $@",
        )

    return new_main
