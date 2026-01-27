# Setting Up Developer Environment

## Setting up LSP with clangd

### Background

Editors such as Emacs, Vim, or VS Code support features like code navigation,
code completion, inline compiler error messages, and others, through
[LSP](https://en.wikipedia.org/wiki/Language_Server_Protocol), the Language
Server Protocol. A common language server with LSP support is
[clangd](https://clangd.llvm.org), which relies on the presence of
`compile_commands.json`, a JSON file with a record of the compile commands for
each file in a project.

### How do I generate `compile_commands.json` for XLA source code?

Use the
[build_tools/lint/generate_compile_commands.py](https://github.com/openxla/xla/blob/main/build_tools/lint/generate_compile_commands.py)
script. The following invocation from XLA repo root generates a
`compile_commands.json` file in place: `bazel aquery "mnemonic(CppCompile,
//xla/...)" --output=jsonproto | python3
build_tools/lint/generate_compile_commands.py`

## Build Cleaner

XLA CI inside Google runs additional checks to verify that all targets correctly
list all dependencies in the BUILD files, which are not enabled by default in
OSS Bazel CI. Running following commands before sending PR to XLA team will
massively speedup the time it takes to merge the PR as otherwise you will need
some Googler to make this fixes for you internally, or a few more rounds of PR
reviews to fix them based on the Googlers feedback.

### Layering Check

Building with `--features=layering_check` makes sure that you don't accidentally
include a header via transitive dependency without listing it in your target
dependencies

### Remove unused dependencies from BUILD files

Install
[Buildozer](https://github.com/bazelbuild/buildtools/blob/main/buildozer/README.md)
tool:

```
sudo curl -fsSL -o /usr/bin/buildozer https://github.com/bazelbuild/buildtools/releases/download/6.0.0/buildozer-linux-amd64
sudo chmod 755 /usr/bin/buildozer
```

Install [Bant](https://github.com/hzeller/bant) tool:

```
# To some writable directory that does not require root access
bazel build -c opt //bant && install -D --strip bazel-bin/bant/bant ~/bin/bant

# For a system directory that requires root-access
sudo install -D --strip bazel-bin/bant/bant /usr/local/bin/bant
```

Use `bant` to generate buildozer commands to remove unused deps:

```
bant dwyu //xla/core/collectives:symmetric_memory
```

if you feel lucky, you can execute them directly:

```
. <(bant dwyu //xla/core/collectives:symmetric_memory)
```
