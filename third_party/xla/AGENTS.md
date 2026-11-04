# AI Assistant Guidelines for OpenXLA

Welcome to the OpenXLA (`openxla/xla`) repository.

*   **Core XLA compiler guidelines**: When generating, reviewing, or modifying
    code under [`xla/`](xla/), follow the detailed C++, MLIR, testing, and Bazel
    guidelines in [`xla/AGENTS.md`](xla/AGENTS.md).
*   **Preparing and reviewing Pull Requests**: Before opening or reviewing a PR,
    follow the PR scoping, description format, compatibility rules, and pre-PR
    self-review checklist in [`docs/contributing.md`](docs/contributing.md) (as
    well as [`docs/avoiding_ifdefs.md`](docs/avoiding_ifdefs.md) and
    [`docs/gpu_aot_compatibility.md`](docs/gpu_aot_compatibility.md) when
    applicable).
*   [`build_tools/`](build_tools/) contains CI and developer tooling,
    [`docs/`](docs/) the user-facing documentation, and
    [`third_party/`](third_party/) vendored dependencies (including
    `third_party/shardy/temporary.patch` and
    `third_party/stablehlo/temporary.patch`) and Bazel repository rules.
