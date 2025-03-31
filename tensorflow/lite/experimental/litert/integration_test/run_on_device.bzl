# Copyright 2025 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module defines the `run_on_device` macro, which helps to execute a binary target on a device.
"""

load("//tensorflow:tensorflow.bzl", "if_oss")
load("//tensorflow/lite/experimental/litert/build_common:litert_build_defs.bzl", "absolute_label")

# DEVICE PATHS #####################################################################################

DEVICE_RLOCATION_ROOT = "/data/local/tmp/runfiles"

def device_rlocation(label = None, get_parent = False):
    """Get the path on device for a given label.

    Args:
        label: The label to get the path for. If None, returns the root path.
        get_parent: If true, get the parent directory of the resolved path.

    Returns:
        The path on device for the given label.
    """
    if not label:
        return DEVICE_RLOCATION_ROOT
    abs_label = absolute_label(label)
    res = DEVICE_RLOCATION_ROOT + "/" + abs_label.replace("//", "").replace(":", "/")
    if get_parent:
        return res[:res.rfind("/")]
    return res

def make_path_args(spec):
    """Formats shell path-like variable assignment exprs from common directories in given labels

    Useful for making things like LD_LIBRARY_PATH=... for paths on device.

    An entry of the spec contains a key, and a list of labels. Unique leaf directories paths are
    extracted from the labels and joined into a colon-separated string.

    Example:
    ```
    make_path_args({
        "LD_LIBRARY_PATH": [
            "// foo : bar",
        ],
        "ADSP_LIBRARY_PATH": [
            "// foo : baz",
            "// foo : bat"
        ],
    })
    ```
    will return:
    ```
    LD_LIBRARY_PATH=/data/local/tmp/runfiles/foo/bar
    ADSP_LIBRARY_PATH=/data/local/tmp/runfiles/foo/baz:/data/local/tmp/runfiles/foo/bat
    ```

    Args:
        spec: A dict of path variable names to lists of labels.

    Returns:
        A list of shell variable assignment expressions.
    """

    res = []
    for path_var, values in spec.items():
        # TODO: Figure out why OSS doesn't have `set` core datatype.
        dirs = []
        for v in values:
            parent = device_rlocation(v, True)
            if parent not in dirs:
                dirs.append(parent)
        res.append("{path_var}={paths}".format(
            path_var = path_var,
            paths = ":".join(dirs),
        ))
    return res

# DYNAMIC LIBRARY DEPENDENCIES #####################################################################

LITERT_CORE_LIBS = [
    "//tensorflow/lite/experimental/litert/c:libLiteRtRuntimeCApi.so",
]

def make_lib_spec(**kwargs):
    return struct(
        litert_base_libs = LITERT_CORE_LIBS,
        core_libs = kwargs["core_libs"],
        kernel_libs = kwargs["kernel_libs"],
        dispatch_lib = kwargs["dispatch_lib"],
        compiler_lib = kwargs["compiler_lib"],
    )

BASE_LIB_SPEC = make_lib_spec(
    core_libs = [],
    kernel_libs = [],
    dispatch_lib = None,
    compiler_lib = None,
)

def all_libs(spec):
    """
    Returns all the dynamic libraries needed for the given spec.

    Args:
        spec: The lib spec to get the libs for.

    Returns:
        A list of all the dynamic libraries needed for the given spec.
    """
    libs = spec.litert_base_libs + spec.core_libs + spec.kernel_libs
    for lib in [spec.dispatch_lib, spec.compiler_lib]:
        if lib:
            libs.append(lib)
    return libs

# QNN

QUALCOMM_LIB_SPEC = make_lib_spec(
    core_libs = [
        "//third_party/qairt/latest:lib/aarch64-android/libQnnHtp.so",
        "//third_party/qairt/latest:lib/aarch64-android/libQnnHtpV75Stub.so",
        "//third_party/qairt/latest:lib/aarch64-android/libQnnSystem.so",
        "//third_party/qairt/latest:lib/aarch64-android/libQnnHtpPrepare.so",
    ],
    kernel_libs = ["//third_party/qairt/latest:lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so"],
    dispatch_lib = "//tensorflow/lite/experimental/litert/vendors/qualcomm/dispatch:dispatch_api_so",
    compiler_lib = "//tensorflow/lite/experimental/litert/vendors/qualcomm/compiler:qnn_compiler_plugin_so",
)

# MTK
# TODO

# GOOGLE TENSOR
# TODO

def get_lib_spec(backend_id):
    """
    Returns the dynamic library spec for the given backend id.

    Args:
        backend_id: The backend id to get the lib spec for.

    Returns:
        The dynamic library spec for the given backend id.
    """
    if backend_id == "qualcomm":
        return QUALCOMM_LIB_SPEC
    if backend_id == "cpu":
        return BASE_LIB_SPEC
    else:
        fail("Unsupported backend id: {}".format(backend_id))

# RUN ON DEVICE MACRO ##############################################################################

def get_driver():
    return if_oss(
        "//tensorflow/lite/experimental/litert/integration_test:run_on_device_driver_OSS",
        "//tensorflow/lite/experimental/litert/integration_test/google:run_on_device_driver",
    )

def run_on_device(
        name,
        target,
        driver,
        data = [],
        exec_args = [],
        exec_env_vars = []):
    """
    Macro to execute a binary target on a device (locally through ADB).

    The output of this macro is an executable shell script that pushes all the necessary files to
    the device and executes the target with the given arguments and environment variables.

    Args:
        name: Name of the target.
        target: The binary target to execute on device.
        driver: The driver script to use for execution.
        data: List of data files to push to the device.
        exec_args: List of arguments to pass to the executable.
        exec_env_vars: List of environment variables to set before executing the target.
    """
    call_mobile_install = """
    echo '$(location {driver}) \
        --bin=$(rlocationpath {target}) \
        --data={data} \
        --do_exec=true \
        --exec_args={exec_args} \
        --exec_env_vars={exec_env_vars} \
        '\
        > $@
    """

    concat_targ_data = "$$(echo \"$(rlocationpaths {})\" | sed \"s/ /,/g\")"
    data_str = ",".join([concat_targ_data.format(d) for d in data])

    # NOTE: Tilde delimiter here (also see driver script) to allow passing list args to underlying
    # binary.
    exec_args_str = "~".join(["{}".format(a) for a in exec_args])
    exec_env_vars_str = ",".join(["{}".format(a) for a in exec_env_vars])

    driver_targ = driver.removesuffix(".sh")
    driver_sh = driver_targ + ".sh"

    cmd = call_mobile_install.format(
        driver = driver_sh,
        target = target,
        data = data_str,
        exec_args = exec_args_str,
        exec_env_vars = exec_env_vars_str,
    )

    exec_script = name + "_exec.sh"

    native.genrule(
        name = name + "_gen_script",
        srcs = [driver_sh] + [target] + data,
        outs = [exec_script],
        tags = ["manual", "notap"],
        cmd = cmd,
        testonly = True,
    )

    native.sh_binary(
        testonly = True,
        tags = ["manual", "notap"],
        name = name,
        deps = [driver_targ],
        srcs = [exec_script],
        data = [target] + data,
    )

def litert_integration_test(
        name,
        models,
        hw = "cpu",
        skips = []):
    """
    Higher level macro that configures run_on_device or a mobile test to run with gen_device_test.

    Args:
        name: Name of the target.
        models: A single target that may contain model or many models in the same directory.
        hw: The backend to test against (see gen_device_test).
        skips: List of substrings of models to skip.
    """

    # Get libs for the given backend.
    lib_spec = get_lib_spec(hw)

    # Accelerator option to pass to the compiled model api on device.
    hw_cfg = hw if hw == "cpu" else "npu"

    # Create env args for paths to dynamic libraries.
    env_args = make_path_args({
        "LD_LIBRARY_PATH": lib_spec.litert_base_libs + lib_spec.core_libs + [lib_spec.dispatch_lib, lib_spec.compiler_lib],
        "ADSP_LIBRARY_PATH": lib_spec.kernel_libs,
    })

    skips_str = ",".join(skips)

    # Create CLI args for the gen_device_test binary on device.
    cli_args = [
        "--model_path={}".format(device_rlocation(models)),
        "--dispatch_library_dir={}".format(device_rlocation(lib_spec.dispatch_lib, True)),
        "--compiler_library_dir={}".format(device_rlocation(lib_spec.compiler_lib, True)),
        "--hw={}".format(hw_cfg),
        "--skips={}".format(skips_str),
    ]

    data = [models] + all_libs(lib_spec)
    driver = get_driver()
    target = "//tensorflow/lite/experimental/litert/integration_test:gen_device_test"

    # TODO: Also kick off a xeno mobile test here.

    run_on_device(
        name = name,
        target = target,
        driver = driver,
        data = data,
        exec_args = cli_args,
        exec_env_vars = env_args,
    )
