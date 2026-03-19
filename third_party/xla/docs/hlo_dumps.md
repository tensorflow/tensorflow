# Dump HLO Computations

An HLO dump is a textual representation of the HLO modules at different stages
of the computation. It is useful for debugging, and you often need to include it
in bug reports. This is typically a human-readable **text file** that lists the
HLO instructions and their properties. Sometimes, HLO modules are dumped as:

-   **HloProto:** Protocol buffer files, which are a more structured,
    machine-readable format.
-   **HloSnapshot**: HLO module plus its inputs. For replaying HLOs, you
    sometimes require the actual inputs fed to a given computation rather than
    random data.

You can use XLA flags to specify and get dumps. In most cases, you can set it
with an environment variable. JAX also offers a programmatic way to print the
HLO dump.

## Local Execution

### Using Environment Variables

You can set the `XLA_FLAGS` environment variable with the necessary flags to get
dumps. This works for JAX, TensorFlow, and PyTorch/XLA.

To dump HLO modules and other debugging information to a specific directory, run
your program with the `--xla_dump_to` flag:

```shell
XLA_FLAGS="--xla_dump_to=DIRECTORY_PATH"
```

For example, you can use `/tmp` or `/tmp/xladump` as the paths.

By default, this dumps HLO modules as text, at the very beginning and end of the
optimization pipeline.

You can also explicitly specify the format:

1.  Text dumps

```shell
XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=DIRECTORY_PATH"
```

1.  HLO protos

```shell
XLA_FLAGS="--xla_dump_hlo_as_proto --xla_dump_to=DIRECTORY_PATH"
```

1.  HLO Snapshots

```shell
XLA_FLAGS="--xla_dump_hlo_snapshots --xla_dump_to=DIRECTORY_PATH"
```

1.  Graph render with graphviz server (only works well for small graphs)

```shell
XLA_FLAGS="--xla_dump_hlo_as_url --xla_dump_to=DIRECTORY_PATH"
```

1.  Graph render to HTML file (only works well for small graphs)

```shell
XLA_FLAGS="--xla_dump_hlo_as_html --xla_dump_to=DIRECTORY_PATH"
```

For larger graphs, you can use `interactive_graphviz` to visualize parts of the
graph.

**Note:** If `--xla_dump_to` is not specified but another dumping flag is
specified, it will dump to stdout. But the dump will not include binary data,
e.g., proto files, to stdout.

## Dump Specific Intermediate Passes

In addition to the standard pre-optimized / final-optimized HLOs, you can also
dump the state of HLOs after a particular compiler pass.

```shell
XLA_FLAGS="--xla_dump_hlo_pass_re=regex --xla_dump_to=DIRECTORY_PATH"
```

HLO modules will be dumped for the passes whose names match the regular
expression (regex). For example, you can observe the HLOs resulting from passes
related to SPMD partitioning with:

```shell
XLA_FLAGS="--xla_dump_to=DIRECTORY_PATH --xla_dump_hlo_pass_re=spmd|propagation"
```

To dump the result after every XLA pass (this will result in a lot of files),
you can set:

```shell
XLA_FLAGS="--xla_dump_to=DIRECTORY_PATH --xla_dump_hlo_pass_re=.*"
```

### JAX-specific Options

#### Programmatically in JAX

Instead of passing flags or environment variables, you can also programmatically
dump HLO using JAX’s `lower` and `compile` APIs.

Locally fetch the unoptimized original lowered HLO with:

```python
jax.jit(f).lower(*args).as_text('hlo')
```

For dumping to files during HLO compilation passes, specify:

```python
compilation_args = {
    'xla_dump_to': DIRECTORY_PATH,
    'xla_dump_hlo_pass_re': 'spmd|propagation', # or some other pass filter
    ...
    }

jax.jit(f).lower(*args).compile(compilation_args)
```

#### Dump jaxprs

[`jaxpr`s](https://docs.jax.dev/en/latest/jaxpr.html) are JAX's intermediate
representation for program traces. To dump this, set the environment variables:

```shell
JAX_DUMP_IR_TO="DIRECTORY_PATH" JAX_DUMP_IR_MODES=jaxpr
```

Learn more in JAX documentation on
[Exporting and serializing staged-out computations: Debugging](https://docs.jax.dev/en/latest/export/export.html#debugging).

## Google Colab

### Environment variables

In the first executed cell of your notebook (because environment variables and
command-line flags are usually only processed once, e.g., at module-import time
or XLA backend initialization time), add the `XLA_FLAGS` detailed above with
`os.environ`, for example:

```python
import os
os.environ['XLA_FLAGS'] = "--xla_dump_to=DIRECTORY_PATH"
```

This will dump the computation to `DIRECTORY_PATH`, for example `/tmp`. On
Colab, navigate to the "Files" browser in the left sidebar, to view and access
this directory.

You can use all the flags mentioned in the Local Execution section.

### JAX-specific options

Similar to local execution; for live, interactive introspection you can directly
print a computation’s pre-optimized HLO:

```python
def f(x):
    return jax.numpy.sin(jax.numpy.cos(x))

c = jax.jit(f).lower(3.).compiler_ir('hlo')

print(c.as_hlo_text())
```

You can also directly print a computation’s optimized HLO:

```python
def optimized_HLO(f, *args, platform=None):
    print(jax.jit(f).lower(*args).compile().as_text())

def f(x):
    return jax.numpy.sin(jax.numpy.cos(x))

optimized_HLO(f, 1.0)
```

#### Dumping All/Small Computations

If you want to see everything in a dump including all small compilations, set
the JAX environment variable:

```shell
JAX_COMPILER_DETAILED_LOGGING_MIN_OPS=0
```

#### Mosaic

Mosaic is a compiler for the Pallas TPU backend, and the experimental Pallas GPU
backend. To dump mosaic computation, set the following flag:

```shell
--xla_mosaic_dump_to=/tmp/mosaic_dumps
```

Or, set TPU init arguments as an environment variable:

```shell
export LIBTPU_INIT_ARGS="--xla_mosaic_dump_to=/tmp/mosaic_dumps"
```

Check out the
[JAX documentation on Pallas and Mosaic](https://docs.jax.dev/en/latest/pallas/index.html)
to learn more.

## More with HLO Dumps

### Finding the right computation

Usually, many computations get dumped. The dumped files are explicitly named
with the JAX, Tensorflow, or PyTorch/XLA "computation name” that are called out
in the logs, making it easy to identify the relevant HLO files. For example:

```
1624325116260738.module_0065.pmap__unnamed_wrapped_function_.186875.before_optimizations.txt
```

Otherwise, you can use `ripgrep` to quickly identify which module holds
particular symbols or computations.

**Tip:** Include the 3 dumped before/after/buffer-assignment files of interest
in your bug reports.

### HLO Conversion

A tool called `hlo-opt` that can translate between HLOProto and text formats.
It's useful in cases where you have one format, but need the other for
debugging.

Learn to use it:
[XLA Tooling documentation: hlo-opt](tools.md#hlo-opt-convert-hlo-module-formats).

### Replay

You can run (replay) the dumped computations on a specified XLA backend with
fake data or input snapshots. This is a convenient way to reproduce, iterate,
and debug issues in XLA.

The following commands use fake data. If you have saved HLO Snapshots, you can
pass those in instead, and the data from the snapshot will be used. To still use
fake data while running the snapshot, pass the flag `--force_fake_data`.

CPU backend:

```shell
bazel run -c opt //xla/hlo/tools:run_hlo_module -- --platform=cpu
 /tmp/xladump/module_4561.before_optimizations.txt
```

GPU backend:

```shell
bazel run -c opt //xla/hlo/tools:run_hlo_module -- --platform=CUDA
 /tmp/xladump/module_4561.before_optimizations.txt
```
