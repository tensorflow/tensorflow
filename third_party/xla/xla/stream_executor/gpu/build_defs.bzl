"""Helper macros for the GPU components in StreamExecutor."""

visibility([
    "//xla/stream_executor/...",
])

def _cc_embed_gpu_library(
        name,
        gpu_library_target,
        section_name,
        cpp_namespace,
        cpp_identifier = None,
        compatible_with = None,
        tags = [],
        testonly = None,
        **kwargs):
    if not cpp_identifier:
        cpp_identifier = "kFatbin_{}".format(name)

    cc_file = "{}.cc".format(name)
    header_file = "{}.h".format(name)
    fatbin_extractor_main = "//xla/stream_executor/gpu:fatbin_extractor_main"
    native.genrule(
        name = "{}_generate_cc_and_header".format(name),
        srcs = [gpu_library_target],
        outs = [
            cc_file,
            header_file,
        ],
        cmd = """
      $(execpath {}) \
        --output_cc_file=$(location {}) --output_header_file=$(location {}) --section_name={} \
        --cpp_namespace={} --cpp_identifier={} $(execpaths {})
    """.format(
            fatbin_extractor_main,
            cc_file,
            header_file,
            section_name,
            cpp_namespace,
            cpp_identifier,
            gpu_library_target,
        ),
        tools = [fatbin_extractor_main],
        compatible_with = compatible_with,
        tags = tags,
        testonly = testonly,
    )
    native.cc_library(
        name = name,
        hdrs = [header_file],
        srcs = [cc_file],
        deps = [
            "@com_google_absl//absl/types:span",
        ],
        compatible_with = compatible_with,
        tags = tags,
        testonly = testonly,
        **kwargs
    )

def cc_embed_cuda_library(
        name,
        cuda_library_target,
        cpp_namespace = "stream_executor::cuda",
        section_name = ".nv_fatbin",
        tags = [],
        **kwargs):
    """Embeds the fatbin segment of a CUDA library as a string in a C++ library.

    Limitations:
        - The cuda_library_target may only have a single .cu.cc file as each fatbin segment
            corresponds to a single .cu.cc file.

    Args:
        name: The name of the build rule.
        cuda_library_target: The target of the CUDA library to embed.
        cpp_namespace: The C++ namespace to place the resulting C++ code in.
        section_name: The name of the section in the ELF file to extract.
        tags: Any tags to pass to the resulting cc_library target.
        **kwargs: Any additional arguments to pass to the resulting cc_library target.
    """
    _cc_embed_gpu_library(
        name = name,
        gpu_library_target = cuda_library_target,
        cpp_namespace = cpp_namespace,
        section_name = section_name,
        tags = tags + ["cuda-only", "gpu"],
        **kwargs
    )

def cc_embed_rocm_library(
        name,
        rocm_library_target,
        cpp_namespace = "stream_executor::rocm",
        section_name = ".hip_fatbin",
        tags = [],
        **kwargs):
    """Embeds the fatbin segment of a ROCM library as a string in a C++ library.

    Limitations:
        - The rocm_library_target may only have a single .cu.cc file as each fatbin segment
            corresponds to a single .cu.cc file.

    Args:
        name: The name of the build rule.
        rocm_library_target: The target of the ROCM library to embed.
        cpp_namespace: The C++ namespace to place the resulting C++ code in.
        section_name: The name of the section in the ELF file to extract.
        tags: Any tags to pass to the resulting cc_library target.
        **kwargs: Any additional arguments to pass to the resulting cc_library target.
    """
    _cc_embed_gpu_library(
        name = name,
        gpu_library_target = rocm_library_target,
        cpp_namespace = cpp_namespace,
        section_name = section_name,
        tags = tags + ["rocm-only", "gpu"],
        **kwargs
    )
