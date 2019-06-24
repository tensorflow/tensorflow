# Copied from https://github.com/nelhage/rules_boost
load(
    "@org_tensorflow//third_party/boost:boost.bzl",
    "boost_library",
    "hdr_list",
    "includes_list",
)

_w_no_deprecated = select({
    ":linux": [
        "-Wno-deprecated-declarations",
    ],
    "//conditions:default": [],
})

config_setting(
    name = "linux",
    constraint_values = [
        "@bazel_tools//platforms:linux",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "osx",
    constraint_values = [
        "@bazel_tools//platforms:osx",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "windows",
    constraint_values = [
        "@bazel_tools//platforms:windows",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_arm",
    constraint_values = [
        "@bazel_tools//platforms:linux",
        "@bazel_tools//platforms:arm",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_x86_64",
    constraint_values = [
        "@bazel_tools//platforms:linux",
        "@bazel_tools//platforms:x86_64",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "osx_x86_64",
    constraint_values = [
        "@bazel_tools//platforms:osx",
        "@bazel_tools//platforms:x86_64",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "windows_x86_64",
    constraint_values = [
        "@bazel_tools//platforms:windows",
        "@bazel_tools//platforms:x86_64",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "x86_64",
    constraint_values = [
        "@bazel_tools//platforms:x86_64",
    ],
    visibility = ["//visibility:public"],
)

BOOST_CTX_ASM_SOURCES = select({
    ":linux_arm": [
        "libs/context/src/asm/jump_arm_aapcs_elf_gas.S",
        "libs/context/src/asm/make_arm_aapcs_elf_gas.S",
        "libs/context/src/asm/ontop_arm_aapcs_elf_gas.S",
    ],
    ":linux_x86_64": [
        "libs/context/src/asm/jump_x86_64_sysv_elf_gas.S",
        "libs/context/src/asm/make_x86_64_sysv_elf_gas.S",
        "libs/context/src/asm/ontop_x86_64_sysv_elf_gas.S",
    ],
    ":osx_x86_64": [
        "libs/context/src/asm/jump_x86_64_sysv_macho_gas.S",
        "libs/context/src/asm/make_x86_64_sysv_macho_gas.S",
        "libs/context/src/asm/ontop_x86_64_sysv_macho_gas.S",
    ],
    ":windows_x86_64": [
        "libs/context/src/asm/make_x86_64_ms_pe_masm.asm",
        "libs/context/src/asm/jump_x86_64_ms_pe_masm.asm",
        "libs/context/src/asm/ontop_x86_64_ms_pe_masm.asm",
    ],
})

boost_library(
    name = "context",
    srcs = BOOST_CTX_ASM_SOURCES + select({
        ":linux": [
            "libs/context/src/posix/stack_traits.cpp",
        ],
        ":osx_x86_64": [
            "libs/context/src/posix/stack_traits.cpp",
        ],
        ":windows_x86_64": [
            "libs/context/src/windows/stack_traits.cpp",
        ],
    }),
    exclude_src = [
        "libs/context/src/fiber.cpp",
        "libs/context/src/untested.cpp",
        "libs/context/src/continuation.cpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":assert",
        ":config",
        ":cstdint",
        ":detail",
        ":intrusive_ptr",
    ],
)

BOOST_FIBER_NUMA_SRCS = select({
    ":linux_arm": [
    ],
    ":linux_x86_64": [
        "libs/fiber/src/numa/linux/pin_thread.cpp",
        "libs/fiber/src/numa/linux/topology.cpp",
    ],
    ":windows_x86_64": [
        "libs/fiber/src/numa/windows/pin_thread.cpp",
        "libs/fiber/src/numa/windows/topology.cpp",
    ],
    "//conditions:default": [],
})

boost_library(
    name = "fiber",
    srcs = BOOST_FIBER_NUMA_SRCS + glob(["libs/fiber/src/algo/**/*.cpp"]) + [
        "libs/fiber/examples/asio/detail/yield.hpp",
    ],
    hdrs = [
        "libs/fiber/examples/asio/round_robin.hpp",
        "libs/fiber/examples/asio/yield.hpp",
    ],
    copts = select({
        ":windows_x86_64": [
            "/D_WIN32_WINNT=0x0601",
        ],
        "//conditions:default": [],
    }),
    exclude_src = ["libs/fiber/src/numa/**/*.cpp"],
    linkopts = select({
        ":linux": [
            "-lpthread",
        ],
        ":osx_x86_64": [
            "-lpthread",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        ":algorithm",
        ":assert",
        ":atomic",
        ":context",
        ":filesystem",
        ":format",
        ":intrusive",
        ":intrusive_ptr",
        ":pool",
        ":system",
        ":throw_exception",
    ],
)

boost_library(
    name = "pool",
    deps = [
        ":integer",
    ],
)

boost_library(
    name = "accumulators",
    deps = [
        ":concept_check",
        ":config",
        ":fusion",
        ":mpl",
        ":parameter",
        ":type_traits",
        ":utility",
        ":version",
    ],
)

boost_library(
    name = "algorithm",
    deps = [
        ":function",
        ":iterator",
        ":range",
    ],
)

boost_library(
    name = "align",
)

boost_library(
    name = "any",
    deps = [
        ":config",
        ":mpl",
        ":static_assert",
        ":throw_exception",
        ":type_index",
        ":type_traits",
        ":utility",
    ],
)

boost_library(
    name = "assign",
    deps = [
        ":config",
        ":detail",
        ":mpl",
        ":preprocessor",
        ":ptr_container",
        ":range",
        ":static_assert",
        ":type_traits",
    ],
)

boost_library(
    name = "atomic",
    hdrs = [
        "boost/memory_order.hpp",
    ],
    deps = [
        ":assert",
        ":config",
        ":cstdint",
        ":type_traits",
    ],
)

boost_library(
    name = "archive",
    deps = [
        ":assert",
        ":cstdint",
        ":integer",
        ":io",
        ":iterator",
        ":mpl",
        ":noncopyable",
        ":smart_ptr",
        ":spirit",
    ],
)

boost_library(
    name = "array",
    deps = [
        ":assert",
        ":config",
        ":core",
        ":functional",
        ":swap",
        ":throw_exception",
    ],
)

boost_library(
    name = "asio",
    defines = select({
        ":osx_x86_64": [
            "BOOST_ASIO_DISABLE_STD_EXPERIMENTAL_STRING_VIEW",
        ],
        "//conditions:default": [],
    }),
    linkopts = ["-lpthread"],
    deps = [
        ":bind",
        ":date_time",
        ":regex",
    ],
)

boost_library(
    name = "assert",
)

boost_library(
    name = "beast",
    deps = [
        ":asio",
        ":config",
        ":core",
        ":detail",
        ":endian",
        ":smart_ptr",
        ":static_assert",
        ":system",
        ":throw_exception",
        ":utility",
    ],
)

boost_library(
    name = "bimap",
    deps = [
        ":concept_check",
        ":multi_index",
        ":serialization",
    ],
)

boost_library(
    name = "bind",
    deps = [
        ":get_pointer",
        ":is_placeholder",
        ":mem_fn",
        ":ref",
        ":type",
        ":visit_each",
    ],
)

boost_library(
    name = "callable_traits",
    deps = [
    ],
)

boost_library(
    name = "call_traits",
)

boost_library(
    name = "cerrno",
)

boost_library(
    name = "checked_delete",
)

boost_library(
    name = "chrono",
    deps = [
        ":config",
        ":mpl",
        ":operators",
        ":predef",
        ":ratio",
        ":system",
        ":throw_exception",
        ":type_traits",
    ],
)

boost_library(
    name = "circular_buffer",
    deps = [
        ":call_traits",
        ":concept_check",
        ":config",
        ":container",
        ":detail",
        ":iterator",
        ":limits",
        ":move",
        ":static_assert",
        ":throw_exception",
        ":type_traits",
        ":utility",
    ],
)

boost_library(
    name = "compute",
    linkopts = select({
        ":osx_x86_64": [
            "-framework OpenCL",
        ],
        "//conditions:default": [
            "-lOpenCL",
        ],
    }),
    deps = [
        ":algorithm",
        ":chrono",
        ":config",
        ":throw_exception",
    ],
)

boost_library(
    name = "concept_archetype",
    deps = [
        ":config",
        ":iterator",
        ":mpl",
    ],
)

boost_library(
    name = "concept_check",
    deps = [
        ":concept",
        ":concept_archetype",
    ],
)

boost_library(
    name = "config",
    deps = [
        ":version",
    ],
)

boost_library(
    name = "concept",
)

boost_library(
    name = "container",
    srcs = [
        "libs/container/src/dlmalloc_ext_2_8_6.c",
    ],
    hdrs = [
        "libs/container/src/dlmalloc_2_8_6.c",
    ],
    deps = [
        ":config",
        ":core",
        ":intrusive",
        ":move",
    ],
)

boost_library(
    name = "container_hash",
    deps = [
        ":assert",
        ":config",
        ":core",
        ":integer",
        ":limits",
        ":type_traits",
    ],
)

boost_library(
    name = "conversion",
    hdrs = [
        "boost/cast.hpp",
        "boost/implicit_cast.hpp",
        "boost/lexical_cast.hpp",
        "boost/polymorphic_cast.hpp",
        "boost/polymorphic_pointer_cast.hpp",
    ],
    deps = [
        ":assert",
        ":config",
        ":fusion",
        ":throw_exception",
    ],
)

boost_library(
    name = "core",
    srcs = [
        "boost/checked_delete.hpp",
    ],
    deps = [
        ":config",
    ],
)

BOOST_CORO_SRCS = select({
    ":linux": [
        "libs/coroutine/src/posix/stack_traits.cpp",
    ],
    ":osx_x86_64": [
        "libs/coroutine/src/posix/stack_traits.cpp",
    ],
    ":windows_x86_64": [
        "libs/coroutine/src/windows/stack_traits.cpp",
    ],
})

boost_library(
    name = "coroutine",
    srcs = BOOST_CORO_SRCS + [
        "libs/coroutine/src/detail/coroutine_context.cpp",
    ],
    deps = [
        ":config",
        ":context",
        ":move",
        ":range",
        ":system",
        ":thread",
    ],
)

boost_library(
    name = "crc",
    deps = [
        ":config",
        ":integer",
        ":limits",
    ],
)

boost_library(
    name = "cstdint",
)

boost_library(
    name = "current_function",
)

boost_library(
    name = "date_time",
    srcs = glob(["libs/date_time/src/**/*.cpp"]),
    hdrs = glob(["libs/date_time/src/**/*.hpp"]),
    deps = [
        ":algorithm",
        ":io",
        ":lexical_cast",
        ":mpl",
        ":operators",
        ":smart_ptr",
        ":static_assert",
        ":tokenizer",
        ":type_traits",
    ],
)

boost_library(
    name = "detail",
    hdrs = [
        "boost/blank.hpp",
        "boost/blank_fwd.hpp",
        "boost/cstdlib.hpp",
    ],
    deps = [
        ":limits",
    ],
)

boost_library(
    name = "dynamic_bitset",
    deps = [
        ":config",
        ":core",
        ":detail",
        ":integer",
        ":move",
        ":throw_exception",
        ":utility",
    ],
)

boost_library(
    name = "enable_shared_from_this",
)

boost_library(
    name = "endian",
    deps = [
        ":config",
        ":core",
        ":cstdint",
        ":detail",
        ":predef",
        ":type_traits",
    ],
)

boost_library(
    name = "exception",
    hdrs = [
        "boost/exception_ptr.hpp",
    ],
    deps = [
        ":config",
        ":detail",
    ],
)

boost_library(
    name = "exception_ptr",
    deps = [
        ":config",
    ],
)

boost_library(
    name = "filesystem",
    copts = _w_no_deprecated,
    deps = [
        ":config",
        ":functional",
        ":io",
        ":iterator",
        ":range",
        ":smart_ptr",
        ":system",
        ":type_traits",
    ],
)

boost_library(
    name = "foreach",
    deps = [
        ":config",
        ":detail",
        ":iterator",
        ":mpl",
        ":noncopyable",
        ":range",
        ":type_traits",
        ":utility",
    ],
)

boost_library(
    name = "format",
    deps = [
        ":assert",
        ":config",
        ":detail",
        ":limits",
        ":optional",
        ":smart_ptr",
        ":throw_exception",
        ":timer",
        ":utility",
    ],
)

boost_library(
    name = "function",
    hdrs = [
        "boost/function_equal.hpp",
    ],
    deps = [
        ":bind",
        ":integer",
        ":type_index",
    ],
)

boost_library(
    name = "function_types",
)

boost_library(
    name = "functional",
    deps = [
        ":container_hash",
        ":detail",
        ":integer",
    ],
)

boost_library(
    name = "fusion",
    deps = [
        ":call_traits",
        ":config",
        ":core",
        ":detail",
        ":function_types",
        ":functional",
        ":get_pointer",
        ":mpl",
        ":preprocessor",
        ":ref",
        ":static_assert",
        ":tuple",
        ":type_traits",
        ":typeof",
        ":utility",
    ],
)

boost_library(
    name = "geometry",
    deps = [
        ":algorithm",
        ":call_traits",
        ":config",
        ":function_types",
        ":lexical_cast",
        ":math",
        ":mpl",
        ":numeric",
        ":qvm",
        ":range",
        ":rational",
        ":tokenizer",
        ":variant",
    ],
)

boost_library(
    name = "property_tree",
    deps = [
        ":any",
        ":bind",
        ":format",
        ":multi_index",
        ":optional",
        ":range",
        ":ref",
        ":throw_exception",
        ":utility",
    ],
)

boost_library(
    name = "get_pointer",
)

boost_library(
    name = "heap",
    deps = [
        ":parameter",
    ],
)

boost_library(
    name = "icl",
    deps = [
        ":concept_check",
    ],
)

boost_library(
    name = "implicit_cast",
)

boost_library(
    name = "is_placeholder",
)

boost_library(
    name = "integer",
    hdrs = [
        "boost/cstdint.hpp",
        "boost/integer_traits.hpp",
        "boost/pending/integer_log2.hpp",
        "boost/pending/lowest_bit.hpp",
    ],
    deps = [
        ":static_assert",
    ],
)

boost_library(
    name = "interprocess",
    deps = [
        ":assert",
        ":checked_delete",
        ":config",
        ":container",
        ":core",
        ":date_time",
        ":detail",
        ":integer",
        ":intrusive",
        ":limits",
        ":move",
        ":static_assert",
        ":type_traits",
        ":unordered",
        ":utility",
    ],
)

boost_library(
    name = "iterator",
    hdrs = [
        "boost/function_output_iterator.hpp",
        "boost/generator_iterator.hpp",
        "boost/indirect_reference.hpp",
        "boost/iterator_adaptors.hpp",
        "boost/next_prior.hpp",
        "boost/pointee.hpp",
        "boost/shared_container_iterator.hpp",
    ],
    deps = [
        ":detail",
        ":static_assert",
        ":type_traits",
        ":utility",
    ],
)

boost_library(
    name = "intrusive",
    deps = [
        ":assert",
        ":cstdint",
        ":noncopyable",
        ":static_assert",
    ],
)

boost_library(
    name = "intrusive_ptr",
    deps = [
        ":assert",
        ":detail",
        ":smart_ptr",
    ],
)

boost_library(
    name = "io",
)

boost_library(
    name = "iostreams",
    deps = [
        ":assert",
        ":bind",
        ":call_traits",
        ":checked_delete",
        ":config",
        ":detail",
        ":function",
        ":integer",
        ":mpl",
        ":noncopyable",
        ":preprocessor",
        ":random",
        ":range",
        ":ref",
        ":regex",
        ":shared_ptr",
        ":static_assert",
        ":throw_exception",
        ":type",
        ":type_traits",
        ":utility",
        "@net_zlib_zlib//:zlib",
        "@org_bzip_bzip2//:bz2lib",
        "@org_lzma_lzma//:lzma",
    ],
)

boost_library(
    name = "lexical_cast",
    deps = [
        ":array",
        ":chrono",
        ":config",
        ":container",
        ":detail",
        ":integer",
        ":limits",
        ":math",
        ":mpl",
        ":noncopyable",
        ":numeric_conversion",
        ":range",
        ":static_assert",
        ":throw_exception",
        ":type_traits",
    ],
)

boost_library(
    name = "limits",
)

boost_library(
    name = "locale",
    srcs =
        glob(
            [
                "libs/locale/src/**/*.cpp",
                "libs/locale/src/**/*.hpp",
                "libs/locale/src/**/*.ipp",
            ],
            exclude = [
                "libs/locale/src/win32/*.cpp",
                "libs/locale/src/icu/*.cpp",
            ],
        ),
    copts = [
        "-DBOOST_LOCALE_WITH_ICONV",
        "-DBOOST_LOCALE_NO_WINAPI_BACKEND",
    ] + _w_no_deprecated,
    deps = [
        ":assert",
        ":config",
        ":cstdint",
        ":smart_ptr",
        ":thread",
        ":unordered",
    ],
)

boost_library(
    name = "lockfree",
    deps = [
        ":align",
        ":array",
        ":assert",
        ":config",
        ":detail",
        ":noncopyable",
        ":parameter",
        ":predef",
        ":static_assert",
        ":type_traits",
        ":utility",
    ],
)

boost_library(
    name = "math",
    hdrs = [
        "boost/cstdint.hpp",
    ],
)

boost_library(
    name = "mem_fn",
)

boost_library(
    name = "move",
    deps = [
        ":assert",
        ":detail",
        ":static_assert",
    ],
)

boost_library(
    name = "mp11",
    deps = [
        ":config",
        ":detail",
    ],
)

boost_library(
    name = "mpl",
    deps = [
        ":move",
        ":preprocessor",
    ],
)

boost_library(
    name = "hana",
    deps = [
    ],
)

boost_library(
    name = "msm",
    deps = [
        ":any",
        ":bind",
        ":detail",
        ":function",
        ":fusion",
        ":mpl",
        ":noncopyable",
        ":parameter",
        ":proto",
        ":serialization",
        ":type_traits",
    ],
)

boost_library(
    name = "multi_index",
    hdrs = [
        "boost/multi_index_container.hpp",
        "boost/multi_index_container_fwd.hpp",
    ],
    deps = [
        ":foreach",
        ":serialization",
        ":static_assert",
        ":tuple",
    ],
)

boost_library(
    name = "multiprecision",
    deps = [
        ":config",
        ":cstdint",
        ":lexical_cast",
        ":math",
        ":mpl",
        ":predef",
        ":rational",
        ":throw_exception",
        ":type_traits",
        ":utility",
    ],
)

boost_library(
    name = "noncopyable",
    deps = [
        ":core",
    ],
)

boost_library(
    name = "none",
    hdrs = [
        "boost/none_t.hpp",
    ],
)

boost_library(
    name = "numeric_conversion",
    hdrs = glob([
        "boost/numeric/conversion/**/*.hpp",
    ]),
    deps = [
        ":config",
        ":detail",
        ":integer",
        ":limits",
        ":mpl",
        ":throw_exception",
        ":type",
        ":type_traits",
    ],
)

boost_library(
    name = "numeric_ublas",
    hdrs = glob([
        "boost/numeric/ublas/**/*.hpp",
    ]),
    deps = [
        ":concept_check",
        ":config",
        ":core",
        ":iterator",
        ":mpl",
        ":noncopyable",
        ":numeric",
        ":range",
        ":serialization",
        ":shared_array",
        ":static_assert",
        ":timer",
        ":type_traits",
        ":typeof",
        ":utility",
    ],
)

boost_library(
    name = "numeric_odeint",
    hdrs = glob([
        "boost/numeric/odeint/**/*.hpp",
    ]),
    deps = [
        ":fusion",
        ":lexical_cast",
        ":math",
        ":multi_array",
        ":numeric",
        ":serialization",
        ":units",
    ],
)

boost_library(
    name = "multi_array",
)

boost_library(
    name = "units",
)

boost_library(
    name = "operators",
)

boost_library(
    name = "optional",
    deps = [
        ":assert",
        ":config",
        ":detail",
        ":none",
        ":static_assert",
        ":throw_exception",
        ":type",
        ":type_traits",
        ":utility",
    ],
)

boost_library(
    name = "parameter",
)

boost_library(
    name = "predef",
)

boost_library(
    name = "preprocessor",
)

boost_library(
    name = "process",
    deps = [
        ":algorithm",
        ":asio",
        ":config",
        ":filesystem",
        ":fusion",
        ":system",
        ":winapi",
    ],
)

boost_library(
    name = "program_options",
    deps = [
        ":any",
        ":bind",
        ":config",
        ":detail",
        ":function",
        ":iterator",
        ":lexical_cast",
        ":limits",
        ":noncopyable",
        ":shared_ptr",
        ":static_assert",
        ":throw_exception",
        ":tokenizer",
        ":type_traits",
        ":version",
    ],
)

boost_library(
    name = "ptr_container",
    deps = [
        ":assert",
        ":checked_delete",
        ":circular_buffer",
        ":config",
        ":iterator",
        ":mpl",
        ":range",
        ":serialization",
        ":static_assert",
        ":type_traits",
        ":unordered",
        ":utility",
    ],
)

boost_library(
    name = "qvm",
    deps = [
        ":assert",
        ":core",
        ":exception",
        ":static_assert",
        ":throw_exception",
        ":utility",
    ],
)

boost_library(
    name = "random",
    deps = [
        ":assert",
        ":config",
        ":detail",
        ":foreach",
        ":integer",
        ":lexical_cast",
        ":limits",
        ":math",
        ":mpl",
        ":multi_index",
        ":noncopyable",
        ":operators",
        ":range",
        ":regex",
        ":shared_ptr",
        ":static_assert",
        ":system",
        ":throw_exception",
        ":timer",
        ":tuple",
        ":type_traits",
        ":utility",
    ],
)

boost_library(
    name = "range",
    deps = [
        ":array",
        ":assert",
        ":concept_check",
        ":config",
        ":detail",
        ":functional",
        ":integer",
        ":iterator",
        ":mpl",
        ":noncopyable",
        ":optional",
        ":preprocessor",
        ":ref",
        ":regex",
        ":static_assert",
        ":tuple",
        ":type_traits",
        ":utility",
    ],
)

boost_library(
    name = "ratio",
    deps = [
        ":integer",
    ],
)

boost_library(
    name = "rational",
    deps = [
        ":assert",
        ":call_traits",
        ":config",
        ":detail",
        ":integer",
        ":operators",
        ":static_assert",
        ":throw_exception",
        ":type_traits",
        ":utility",
    ],
)

boost_library(
    name = "ref",
    deps = [
        ":config",
        ":core",
        ":detail",
        ":utility",
    ],
)

boost_library(
    name = "regex",
    hdrs = [
        "boost/cregex.hpp",
    ],
    defines = [
        "BOOST_FALLTHROUGH",
    ],
    deps = [
        ":assert",
        ":config",
        ":cstdint",
        ":detail",
        ":exception",
        ":functional",
        ":integer",
        ":limits",
        ":mpl",
        ":predef",
        ":ref",
        ":smart_ptr",
        ":throw_exception",
        ":type_traits",
    ],
)

boost_library(
    name = "scope_exit",
    deps = [
        ":config",
        ":detail",
        ":function",
        ":mpl",
        ":preprocessor",
        ":type_traits",
        ":typeof",
        ":utility",
    ],
)

boost_library(
    name = "scoped_array",
    deps = [
        ":checked_delete",
    ],
)

boost_library(
    name = "scoped_ptr",
    deps = [
        ":checked_delete",
    ],
)

boost_library(
    name = "shared_ptr",
    deps = [
        ":checked_delete",
    ],
)

boost_library(
    name = "shared_array",
    deps = [
        ":checked_delete",
    ],
)

boost_library(
    name = "signals2",
    deps = [
        ":assert",
        ":bind",
        ":checked_delete",
        ":config",
        ":core",
        ":detail",
        ":function",
        ":iterator",
        ":mpl",
        ":multi_index",
        ":noncopyable",
        ":optional",
        ":parameter",
        ":predef",
        ":preprocessor",
        ":ref",
        ":scoped_ptr",
        ":shared_ptr",
        ":smart_ptr",
        ":swap",
        ":throw_exception",
        ":tuple",
        ":type_traits",
        ":utility",
        ":variant",
        ":visit_each",
    ],
)

boost_library(
    name = "serialization",
    deps = [
        ":archive",
        ":array",
        ":call_traits",
        ":config",
        ":detail",
        ":function",
        ":operators",
        ":type_traits",
    ],
)

boost_library(
    name = "smart_ptr",
    hdrs = [
        "boost/enable_shared_from_this.hpp",
        "boost/make_shared.hpp",
        "boost/make_unique.hpp",
        "boost/pointer_to_other.hpp",
        "boost/weak_ptr.hpp",
    ],
    deps = [
        ":align",
        ":core",
        ":predef",
        ":scoped_array",
        ":scoped_ptr",
        ":shared_array",
        ":shared_ptr",
        ":throw_exception",
        ":utility",
    ],
)

boost_library(
    name = "spirit",
    deps = [
        ":optional",
        ":phoenix",
        ":range",
        ":ref",
        ":utility",
    ],
)

BOOST_STACKTRACE_SOURCES = select({
    ":linux_arm": [
        "libs/stacktrace/src/basic.cpp",
        "libs/stacktrace/src/noop.cpp",
    ],
    ":linux_x86_64": [
        "libs/stacktrace/src/backtrace.cpp",
    ],
    ":osx_x86_64": [
        "libs/stacktrace/src/addr2line.cpp",
    ],
    ":windows_x86_64": [
        "libs/stacktrace/src/windbg.cpp",
        "libs/stacktrace/src/windbg_cached.cpp",
    ],
})

boost_library(
    name = "stacktrace",
    srcs = BOOST_STACKTRACE_SOURCES,
    defines = select({
        ":osx_x86_64": [
            "BOOST_STACKTRACE_GNU_SOURCE_NOT_REQUIRED",
        ],
        "//conditions:default": [],
    }),
    exclude_src = ["libs/stacktrace/src/*.cpp"],
    linkopts = select({
        ":linux_x86_64": [
            "-lbacktrace -ldl",
        ],
        "//conditions:default": [],
    }),
    deps = [
        ":array",
        ":config",
        ":core",
        ":detail",
        ":lexical_cast",
        ":static_assert",
        ":type_traits",
    ],
)

boost_library(
    name = "static_assert",
)

boost_library(
    name = "system",
    deps = [
        ":assert",
        ":cerrno",
        ":config",
        ":core",
        ":cstdint",
        ":noncopyable",
        ":predef",
        ":utility",
    ],
)

boost_library(
    name = "swap",
    deps = [
        ":core",
    ],
)

boost_library(
    name = "throw_exception",
    deps = [
        ":current_function",
        ":detail",
        ":exception",
    ],
)

boost_library(
    name = "thread",
    srcs = select({
        ":linux": [
            "libs/thread/src/pthread/once.cpp",
            "libs/thread/src/pthread/thread.cpp",
        ],
        ":osx_x86_64": [
            "libs/thread/src/pthread/once.cpp",
            "libs/thread/src/pthread/thread.cpp",
        ],
        ":windows_x86_64": [
            "libs/thread/src/win32/thread.cpp",
            "libs/thread/src/win32/tss_dll.cpp",
            "libs/thread/src/win32/tss_pe.cpp",
        ],
    }),
    hdrs = select({
        ":linux": [
            "libs/thread/src/pthread/once_atomic.cpp",
        ],
        ":osx_x86_64": [
            "libs/thread/src/pthread/once_atomic.cpp",
        ],
        ":windows_x86_64": [],
    }),
    defines = select({
        ":linux_arm": [],
        ":linux_x86_64": [],
        ":osx_x86_64": [],
        ":windows_x86_64": [
            "BOOST_ALL_NO_LIB",
            "BOOST_THREAD_USE_LIB",
            "BOOST_WIN32_THREAD",
        ],
    }),
    linkopts = select({
        ":linux": [
            "-lpthread",
        ],
        ":osx_x86_64": [
            "-lpthread",
        ],
        ":windows_x86_64": [],
    }),
    deps = [
        ":algorithm",
        ":atomic",
        ":bind",
        ":chrono",
        ":config",
        ":core",
        ":date_time",
        ":detail",
        ":enable_shared_from_this",
        ":exception",
        ":function",
        ":io",
        ":lexical_cast",
        ":smart_ptr",
        ":system",
        ":tuple",
        ":type_traits",
    ],
)

boost_library(
    name = "tokenizer",
    hdrs = [
        "boost/token_functions.hpp",
        "boost/token_iterator.hpp",
    ],
    deps = [
        ":assert",
        ":config",
        ":detail",
        ":iterator",
        ":mpl",
        ":throw_exception",
    ],
)

boost_library(
    name = "tribool",
    hdrs = [
        "boost/logic/tribool.hpp",
        "boost/logic/tribool_fwd.hpp",
    ],
    deps = [
        ":config",
        ":detail",
    ],
)

boost_library(
    name = "tti",
    deps = [
        ":config",
        ":function_types",
        ":mpl",
        ":preprocessor",
        ":type_traits",
    ],
)

boost_library(
    name = "type",
    deps = [
        ":core",
    ],
)

boost_library(
    name = "type_index",
    deps = [
        ":container_hash",
        ":core",
        ":functional",
        ":throw_exception",
    ],
)

boost_library(
    name = "type_traits",
    hdrs = [
        "boost/aligned_storage.hpp",
    ],
    deps = [
        ":config",
        ":core",
        ":mpl",
        ":static_assert",
    ],
)

boost_library(
    name = "typeof",
    deps = [
        ":config",
        ":detail",
        ":mpl",
        ":preprocessor",
        ":type_traits",
        ":utility",
    ],
)

boost_library(
    name = "tuple",
    deps = [
        ":ref",
    ],
)

boost_library(
    name = "unordered",
    hdrs = [
        "boost/unordered_map.hpp",
        "boost/unordered_set.hpp",
    ],
    deps = [
        ":assert",
        ":config",
        ":container",
        ":detail",
        ":functional",
        ":iterator",
        ":limits",
        ":move",
        ":preprocessor",
        ":smart_ptr",
        ":swap",
        ":throw_exception",
        ":tuple",
        ":type_traits",
        ":utility",
    ],
)

boost_library(
    name = "utility",
    hdrs = [
        "boost/compressed_pair.hpp",
        "boost/next_prior.hpp",
    ],
)

boost_library(
    name = "uuid",
    deps = [
        ":assert",
        ":config",
        ":core",
        ":detail",
        ":io",
        ":random",
        ":serialization",
        ":static_assert",
        ":throw_exception",
        ":tti",
        ":type_traits",
    ],
)

boost_library(
    name = "variant",
    deps = [
        ":call_traits",
        ":config",
        ":detail",
        ":functional",
        ":math",
        ":static_assert",
        ":type_index",
        ":type_traits",
        ":utility",
    ],
)

boost_library(
    name = "version",
)

boost_library(
    name = "visit_each",
)

boost_library(
    name = "timer",
    deps = [
        ":cerrno",
        ":chrono",
        ":config",
        ":cstdint",
        ":io",
        ":limits",
        ":system",
        ":throw_exception",
    ],
)

boost_library(
    name = "numeric",
)

boost_library(
    name = "test",
    exclude_src = glob(["libs/test/src/*.cpp"]),
    deps = [
        ":algorithm",
        ":assert",
        ":bind",
        ":call_traits",
        ":config",
        ":core",
        ":current_function",
        ":detail",
        ":exception",
        ":function",
        ":io",
        ":iterator",
        ":limits",
        ":mpl",
        ":numeric_conversion",
        ":optional",
        ":preprocessor",
        ":smart_ptr",
        ":static_assert",
        ":timer",
        ":type",
        ":type_traits",
        ":utility",
        ":version",
    ],
)

boost_library(
    name = "proto",
    deps = [
        ":fusion",
    ],
)

boost_library(
    name = "phoenix",
    deps = [
        ":proto",
    ],
)

BOOST_LOG_CFLAGS = select({
    ":linux_arm": [
    ],
    ":x86_64": [
        "-msse4.2",
    ],
})

BOOST_LOG_DEFINES = [
    "BOOST_LOG_WITHOUT_WCHAR_T",
    "BOOST_LOG_USE_STD_REGEX",
    "BOOST_LOG_WITHOUT_DEFAULT_FACTORIES",
    "BOOST_LOG_WITHOUT_SETTINGS_PARSERS",
    "BOOST_LOG_WITHOUT_DEBUG_OUTPUT",
    "BOOST_LOG_WITHOUT_EVENT_LOG",
    "BOOST_LOG_WITHOUT_SYSLOG",
]

BOOST_LOG_DEPS = [
    ":asio",
    ":date_time",
    ":filesystem",
    ":interprocess",
    ":locale",
    ":parameter",
    ":phoenix",
    ":random",
    ":spirit",
    ":system",
    ":thread",
    ":variant",
]

BOOST_LOG_SSSE3_DEP = select({
    ":linux_arm": [
    ],
    ":x86_64": [
        ":log_dump_ssse3",
    ],
})

cc_library(
    name = "log_dump_ssse3",
    srcs = ["libs/log/src/dump_ssse3.cpp"] + hdr_list("log"),
    hdrs = [],
    copts = ["-msse4.2"],
    defines = BOOST_LOG_DEFINES,
    includes = [],
    licenses = ["notice"],
    linkopts = [],
    visibility = ["//visibility:public"],
    deps = BOOST_LOG_DEPS,
)

boost_library(
    name = "log",
    copts = BOOST_LOG_CFLAGS,
    defines = BOOST_LOG_DEFINES,
    exclude_src = [
        "libs/log/src/dump_avx2.cpp",
        "libs/log/src/dump_ssse3.cpp",
    ],
    deps = BOOST_LOG_DEPS + BOOST_LOG_SSSE3_DEP,
)

boost_library(
    name = "statechart",
)

boost_library(
    name = "winapi",
    visibility = ["//visibility:private"],
)

boost_library(
    name = "xpressive",
    deps = [
        ":config",
        ":core",
        ":type_traits",
    ],
)

boost_library(
    name = "property_map",
    deps = [
        ":config",
        ":core",
        ":function",
        ":lexical_cast",
        ":type_traits",
    ],
)

boost_library(
    name = "graph",
    hdrs = [
        "boost/pending/container_traits.hpp",
        "boost/pending/detail/property.hpp",
        "boost/pending/indirect_cmp.hpp",
        "boost/pending/property.hpp",
        "boost/pending/queue.hpp",
        "boost/pending/relaxed_heap.hpp",
    ],
    deps = [
        ":algorithm",
        ":conversion",
        ":fusion",
        ":intrusive_ptr",
        ":lexical_cast",
        ":parameter",
        ":property_map",
        ":property_tree",
        ":proto",
        ":tti",
        ":typeof",
        ":unordered",
        ":xpressive",
    ],
)

boost_library(
    name = "lambda",
)
