load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "compressed_tuple",
    deps = [
        "//absl/utility",
    ],
)

cc_library(
    name = "fixed_array",
    deps = [
        ":compressed_tuple",
        "//absl/algorithm",
        "//absl/base:config",
        "//absl/base:core_headers",
        "//absl/base:dynamic_annotations",
        "//absl/base:throw_delegate",
        "//absl/memory",
    ],
)

cc_library(
    name = "inlined_vector_internal",
    deps = [
        ":compressed_tuple",
        "//absl/base:core_headers",
        "//absl/memory",
        "//absl/meta:type_traits",
        "//absl/types:span",
    ],
)

cc_library(
    name = "inlined_vector",
    deps = [
        ":inlined_vector_internal",
        "//absl/algorithm",
        "//absl/base:core_headers",
        "//absl/base:throw_delegate",
        "//absl/memory",
    ],
)

cc_library(
    name = "flat_hash_map",
    deps = [
        ":container_memory",
        ":hash_function_defaults",
        ":raw_hash_map",
        "//absl/algorithm:container",
        "//absl/memory",
    ],
)

cc_library(
    name = "flat_hash_set",
    deps = [
        ":container_memory",
        ":hash_function_defaults",
        ":raw_hash_set",
        "//absl/algorithm:container",
        "//absl/base:core_headers",
        "//absl/memory",
    ],
)

cc_library(
    name = "node_hash_map",
    deps = [
        ":container_memory",
        ":hash_function_defaults",
        ":node_hash_policy",
        ":raw_hash_map",
        "//absl/algorithm:container",
        "//absl/memory",
    ],
)

cc_library(
    name = "node_hash_set",
    deps = [
        ":hash_function_defaults",
        ":node_hash_policy",
        ":raw_hash_set",
        "//absl/algorithm:container",
        "//absl/memory",
    ],
)

cc_library(
    name = "container_memory",
    deps = [
        "//absl/base:config",
        "//absl/memory",
        "//absl/meta:type_traits",
        "//absl/utility",
    ],
)

cc_library(
    name = "hash_function_defaults",
    deps = [
        "//absl/base:config",
        "//absl/hash",
        "//absl/strings",
        "//absl/strings:cord",
    ],
)

cc_library(
    name = "hash_policy_traits",
    deps = ["//absl/meta:type_traits"],
)

cc_library(
    name = "hashtable_debug",
    deps = [
        ":hashtable_debug_hooks",
    ],
)

cc_library(
    name = "hashtable_debug_hooks",
    deps = [
        "//absl/base:config",
    ],
)

cc_library(
    name = "hashtablez_sampler",
    linkopts = ["-labsl_hashtablez_sampler"],
    deps = [
        "//absl/base",
        "//absl/base:core_headers",
        "//absl/base:exponential_biased",
        "//absl/debugging:stacktrace",
        "//absl/memory",
        "//absl/synchronization",
        "//absl/utility",
    ],
)

cc_library(
    name = "node_hash_policy",
    deps = ["//absl/base:config"],
)

cc_library(
    name = "raw_hash_map",
    deps = [
        ":container_memory",
        ":raw_hash_set",
        "//absl/base:throw_delegate",
    ],
)

cc_library(
    name = "common",
    deps = [
        "//absl/meta:type_traits",
        "//absl/types:optional",
    ],
)

cc_library(
    name = "raw_hash_set",
    linkopts = ["-labsl_raw_hash_set"],
    deps = [
        ":common",
        ":compressed_tuple",
        ":container_memory",
        ":hash_policy_traits",
        ":hashtable_debug_hooks",
        ":hashtablez_sampler",
        ":layout",
        "//absl/base:config",
        "//absl/base:core_headers",
        "//absl/base:endian",
        "//absl/memory",
        "//absl/meta:type_traits",
        "//absl/numeric:bits",
        "//absl/utility",
    ],
)

cc_library(
    name = "layout",
    deps = [
        "//absl/base:config",
        "//absl/base:core_headers",
        "//absl/meta:type_traits",
        "//absl/strings",
        "//absl/types:span",
        "//absl/utility",
    ],
)

cc_library(
    name = "btree",
    deps = [
        ":common",
        ":compressed_tuple",
        ":container_memory",
        ":layout",
        "//absl/base:core_headers",
        "//absl/base:throw_delegate",
        "//absl/memory",
        "//absl/meta:type_traits",
        "//absl/strings",
        "//absl/strings:cord",
        "//absl/types:compare",
        "//absl/utility",
    ],
)
