"""Build definitions for the XLA exhaustive tests."""

load("//xla/tests:build_defs.bzl", "xla_test")
load("//xla/tsl:package_groups.bzl", "DEFAULT_LOAD_VISIBILITY")

def register_extension_info(**_kwargs):
    pass

visibility(DEFAULT_LOAD_VISIBILITY)

def exhaustive_xla_test(name, srcs, partitions, tags, **kwargs):
    """Special exhasutive wrapper for xla_test macro.

    Allows splitting what is conceptually one test across multiple test binaries to get binary-level
    parallelism when exeucting. The tests can then share the majority of source code and be
    interacted with as if they were a single build target.

    Args:
        name: Name for the target. For each partition, the partition key is appended to this for the
            eventual `xla_test` invocation.
        srcs: Shared sources for all partitions. Partitions can specify additional sources as the
            value of the partitions mapping.
        partitions: A dict[str, list[str]] of partition names to additional sources for that
            partition. Sources can be anything accepted by `xla_test`.
        tags: `xla_test` tags property that will be passed through unchanged.
        **kwargs: Any additional arguments not mentioned above that will be passed to each `xla_test`
            invocation.

    Expands to:
        `xla_test` invocation for each partition called `name + "_" + partition.key()`. And
        additional `test_suite` called `name` is also generated as a way to invoke all partitions at
        once.
    """

    if not partitions:
        fail("partitions must be non-empty.")

    test_target_names = []
    for (partition_suffix, additional_partition_srcs) in partitions.items():
        target_name = name + "_" + partition_suffix
        xla_test(
            name = target_name,
            srcs = srcs + additional_partition_srcs,
            tags = tags,
            **kwargs
        )
        test_target_names.append(target_name)

    # Build magic to allow the original name to trigger all generated xla_test targets.
    native.test_suite(
        name = name,
        tests = test_target_names,
        tags = tags + ["-broken", "-manual"],
    )

register_extension_info(
    extension = exhaustive_xla_test,
    # Needs to be kept up-to-date on all partition names defined in the invocations.
    #
    # For some reason, manually specifying the expansion targets like (cpu|cpu_.*|...) is required
    # for build tools.
    label_regex_for_dep = "{extension_name}_(f16_and_smaller|f32_and_smaller|f32|f64)_(cpu|cpu_.*|gpu|gpu_.*)",
)
