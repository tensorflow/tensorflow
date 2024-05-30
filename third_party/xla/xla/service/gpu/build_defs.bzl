""" GPU-specific build macros.
"""

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured", "rocm_copts")
load("@local_tsl//tsl/platform/default:cuda_build_defs.bzl", "if_cuda_is_configured")
load("@bazel_skylib//lib:paths.bzl", "paths")
load(
    "//xla/tests:build_defs.bzl",
    "prepare_gpu_backend_data",
)

# buildifier: disable=out-of-order-load
# Internally this loads a macro, but in OSS this is a function
def register_extension_info(**_kwargs):
    pass

def get_cub_sort_kernel_types(name = ""):
    """ List of supported types for CUB sort kernels.
    """
    return [
        "f16",
        "f32",
        "f64",
        "s8",
        "s16",
        "s32",
        "s64",
        "u8",
        "u16",
        "u32",
        "u64",
        "u16_b16",
        "u16_b32",
        "u16_b64",
        "u32_b16",
        "u32_b32",
        "u32_b64",
        "u64_b16",
        "u64_b32",
        "u64_b64",
    ]

def build_cub_sort_kernels(name, types, local_defines = [], **kwargs):
    """ Create build rules for all CUB sort kernels.
    """
    for suffix in types:
        gpu_kernel_library(
            name = name + "_" + suffix,
            local_defines = local_defines + ["CUB_TYPE_" + suffix.upper()],
            **kwargs
        )

register_extension_info(extension = build_cub_sort_kernels, label_regex_for_dep = "{extension_name}_.*")

def gpu_kernel_library(name, copts = [], local_defines = [], **kwargs):
    cuda_library(
        name = name,
        local_defines = local_defines + if_cuda_is_configured(["GOOGLE_CUDA=1"]) +
                        if_rocm_is_configured(["TENSORFLOW_USE_ROCM=1"]),
        copts = copts + rocm_copts(),
        **kwargs
    )

register_extension_info(extension = gpu_kernel_library, label_regex_for_dep = "{extension_name}")

def gen_gpu_hlo_compile_tests(
        name,
        hlo_files,
        multihost_hlo_runner_binary_path,
        backends = [],
        disabled_backends = [],
        backend_tags = {},
        backend_args = {},
        xla_flags = []):
    """Macro to generate Bazel tests for compiling HLO files on a GPU.

    This macro creates individual Bazel test targets for each specified HLO file.
    These tests use the `cuda_hlo_runner_main` binary to attempt to compile the HLO
    files.

    Parses num_hosts, num_devices_per_host and num_replicas for each filename in `hlo_files`.
    E.g.
    - name_16x16_256replicas.hlo
      -> num_hosts=16, num_devices_per_host=16, num_replicas=256
    - name_4x8.hlo
      -> num_hosts=4,  num_devices_per_host=8,  num_replicas=1
    - name_x1.hlo
      -> num_hosts=1,  num_devices_per_host=1,  num_replicas=1

    Args:
      name: A string used to create unique names for the generated tests.
      hlo_files: A list of strings representing HLO filenames of to HLO files.
      multihost_hlo_runner_binary_path: Path to the multihost_hlo_runner binary.
      backends: A list of backends to generate tests for. Supported values: "cpu",
        "gpu". If this list is empty, the test will be generated for all supported
        backends.
      disabled_backends: A list of backends to NOT generate tests for.
      backend_tags: A dict mapping backend name to list of additional tags to
        use for that target.
      backend_args: A dict mapping backend name to list of additional args to
        use for that target.
      xla_flags: A list of XLA flags passed to multihost_hlo_runner.


    Example Usage:

    ```bazel
    gen_gpu_hlo_compile_tests(
      name = "my_gpu_compile_tests",
      hlo_filenames = glob([
        "dir/*.hlo",
      ]),
    )
    ```
    """

    content = "exec %s \"$$@\"" % multihost_hlo_runner_binary_path
    native.genrule(
        name = name + "_gensh",
        outs = [name + ".sh"],
        cmd = "echo '%s' > $@" % content,
        executable = True,
    )

    for filename in hlo_files:
        hlo_path = "%s/%s" % (native.package_name(), filename)

        relative_dirname = paths.dirname(filename)
        filename = paths.basename(filename)

        if relative_dirname:
            data_label = "//%s:%s/%s" % (native.package_name(), relative_dirname, filename)
        else:
            data_label = "//%s:%s" % (native.package_name(), filename)

        def parse_number_of_hosts_and_devices(device_topology):
            """
            Parse number of hosts and devices per host from the device topology.

            E.g.
              - for `device_topology=16x16`, num_hosts=16, num_devices_per_host=16.
              - for `device_topology=x6`, num_hosts=1, num_devices_per_host=6.
            """
            num_hosts, num_devices_per_host = device_topology.split("x")
            return (int(num_hosts) if num_hosts else 1, int(num_devices_per_host))

        filename = filename.removesuffix(".hlo")
        if filename.endswith("replicas"):
            _, device_topology, replica_string = filename.rsplit("_", 2)
            num_replicas = int(replica_string.removesuffix("replicas"))
            num_hosts, num_devices_per_host = parse_number_of_hosts_and_devices(device_topology)
        else:
            num_replicas = 1
            _, device_topology = filename.rsplit("_", 1)
            num_hosts, num_devices_per_host = parse_number_of_hosts_and_devices(device_topology)

        num_partitions = num_hosts * num_devices_per_host // num_replicas

        # Expand "gpu" backend name to specific GPU backends and update tags.
        backends, disabled_backends, backend_tags, backend_args = \
            prepare_gpu_backend_data(backends, disabled_backends, backend_tags, backend_args)

        backends = [
            backend
            for backend in backends
            if backend not in disabled_backends
        ]

        for backend in backends:
            native.sh_test(
                name = "gpu_compile_%s_%s_hlo_test" % (filename, backend),
                srcs = [name + "_gensh"],
                args = [
                    "--device_type=gpu",
                    "--run=false",
                    "--num_replicas=%d" % num_replicas,
                    "--num_partitions=%d" % num_partitions,
                    "--use_spmd_partitioning=true",
                    hlo_path,
                ] + xla_flags,
                data = ["//xla/tools/multihost_hlo_runner:cuda_hlo_runner_main", data_label],
                tags = backend_tags[backend] + ["requires-mem:16g"],
            )
