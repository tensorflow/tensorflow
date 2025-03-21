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

DEVICE_RLOCATION_ROOT = "/data/local/tmp/runfiles"

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
    exec_args_str = ",".join(["{}".format(a) for a in exec_args])
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

def device_rlocation(label = None):
    if not label:
        return DEVICE_RLOCATION_ROOT
    abs_label = absolute_label(label)
    return DEVICE_RLOCATION_ROOT + "/" + abs_label.replace("//", "").replace(":", "/")

def get_driver():
    return if_oss(
        "//tensorflow/lite/experimental/litert/integration_test:run_on_device_driver_OSS",
        "//tensorflow/lite/experimental/litert/integration_test/google:run_on_device_driver",
    )
