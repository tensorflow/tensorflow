load("//xla/tsl:tsl.bzl", "internal_visibility")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = internal_visibility([":friends"]),
    licenses = ["notice"],
)

package_group(
    name = "friends",
    includes = [
        "//xla:friends",
    ],
)

alias(
    name = "tile",
    actual = "//xla/codegen/tiling:tile",
)

alias(
    name = "tiled_hlo",
    actual = "//xla/codegen/tiling:tiled_hlo",
)

alias(
    name = "tiling_space",
    actual = "//xla/codegen/tiling:tiling_space",
)

alias(
    name = "tiling_space_utils",
    actual = "//xla/codegen/tiling:tiling_space_utils",
)

alias(
    name = "tile_propagation",
    actual = "//xla/codegen/tiling:tile_propagation",
)

alias(
    name = "scheduling",
    actual = "//xla/codegen/tiling:scheduling",
)

alias(
    name = "reshape_analysis",
    actual = "//xla/codegen/tiling:reshape_analysis",
)

alias(
    name = "test_utils",
    actual = "//xla/codegen/tiling:test_utils",
)
