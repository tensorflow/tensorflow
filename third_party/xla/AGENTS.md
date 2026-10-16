# AI Assistant Guidelines for OpenXLA

Welcome to the OpenXLA (`openxla/xla`) repository.

*   **Core XLA compiler guidelines**: When generating, reviewing, or modifying
    code under [`xla/`](xla/), follow the detailed C++, MLIR, testing, and
    Bazel guidelines in [`xla/AGENTS.md`](xla/AGENTS.md).
*   **Code review and compatibility skills**: Domain-specific review skills
    (`gpu-aot-compatibility-review`, `cuda-api-review`, `avoid-ifdefs-review`,
    and `cpu-abi-compatibility-review`) live in
    [`.agents/skills/`](.agents/skills/) (and are symlinked in
    [`.claude/skills/`](.claude/skills/)).
*   [`build_tools/`](build_tools/) contains CI and developer tooling,
    [`docs/`](docs/) the user-facing documentation, and
    [`third_party/`](third_party/) vendored dependencies and Bazel repository
    rules.
