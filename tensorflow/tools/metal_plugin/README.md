# tensorflow-metal-plugin

A Metal GPU backend for TensorFlow on Apple silicon, built as an out-of-tree
PluggableDevice. It loads into a stock TensorFlow wheel and adds
`/physical_device:GPU:0`.

This is the out-of-tree form of the backend proposed in
[tensorflow/tensorflow#126254](https://github.com/tensorflow/tensorflow/pull/126254).
The sources are the same; the only difference is this repository exports
`SE_InitPlugin` and `TF_InitKernel` from a shared object, where the in-tree
form hands the same function pointers to `RegisterPluggableDevicePlugin`.

## Status

Working, and every op it registers has been run on a real GPU and checked.
One significant limitation is not this project's to fix: see
[What a released TensorFlow cannot do](#what-a-released-tensorflow-cannot-do).

`make sweep` calls all 356 registered ops through TensorFlow's own dispatch,
once on the GPU and once on the CPU with identical inputs, with soft placement
off so that a missing kernel raises rather than answering from the host:

| | |
| --- | --- |
| Verified against the CPU kernel, or against a property where there is no CPU kernel | 323 |
| Removed from TensorFlow, so no device can run them | 19 |
| Need kernel C API entry points a released TensorFlow does not export | 14 |
| **Unaccounted for** | **0** |

Every op is also run twice and required to give the same answer, which is how
an inverse transform that rewrote its own input was caught. The sweep
separately enumerates every registration TensorFlow holds for these ops and
rejects any that is duplicated or that constrains an attribute the op does not
have, since either makes an op unusable while looking registered.

Two of the nineteen announce themselves differently, complaining that a
kernel constrains an attribute the node lacks: TensorFlow's own CPU
registrations for `TopK` and `TileGrad` constrain `index_type` and
`Tmultiples`, which their op defs do not have. That is true and is not why
they cannot run. Both are deprecated in their op def, `TopK` from GraphDef
version 7 and `TileGrad` from version 3, so nothing can call them either way.

Verified on an Apple M4 Max, macOS 26.6, against the stock
`tensorflow==2.20.0` wheel for Python 3.12:

```
before: ['/physical_device:CPU:0']
after : ['/physical_device:CPU:0', '/physical_device:GPU:0']
Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0
```

`MatMul`, `Conv2D`, `Softmax`, `Relu`, `MaxPool2D` and `ReduceSum` match the
CPU kernels with soft placement disabled, so a missing GPU kernel raises
instead of quietly producing a correct answer on the wrong device.

## Install

```
pip install tensorflow
pip install tensorflow-metal-plugin
```

The two commands are in that order for a reason, and the second one fails
without the first. There is no prebuilt wheel: a PluggableDevice is compiled
against the TensorFlow it will be loaded into, and it records that
TensorFlow's location in its own load path, so there is nothing to build
against until TensorFlow is installed. Installing both in a single
`pip install` does not work either, since pip builds this package before it
installs the dependency.

The shared object is built at install time against the
TensorFlow of the interpreter doing the installing, and lands in
`site-packages/tensorflow-plugins`, which TensorFlow scans at import. Nothing
has to be loaded by hand:

```python
>>> import tensorflow as tf
>>> tf.config.list_physical_devices()
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'),
 PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

Verified on a clean environment with `tensorflow==2.21.0`, Python 3.12, macOS
26.6 on an M4 Max.

## Training works, and what it costs today

`model.fit(optimizer="adam")` trains and the loss goes down. Getting there
needed a correction worth stating plainly, because it changes the speed.

TensorFlow's own kernels for resource variables reach a tensor through its
data pointer. On a unified memory device that pointer is host-addressable, so
those kernels read and write device memory from the host with no idea that GPU
work is in flight against it. A plugin is supposed to implement those ops
itself, through `tensorflow/c/kernels_experimental.h`, and order them on its
own stream. Since 2.20.0 no shipped binary defines those entry points
([#126374](https://github.com/tensorflow/tensorflow/issues/126374)), so the
ops fall back to the host and race.

The symptom was the worst kind: an optimiser read a slot variable mid-write,
took the square root of whatever was there, and produced `nan` weights with no
error raised. `model.fit` reported `[nan, nan, nan]` and carried on.

While those entry points are missing, every Metal kernel waits for the GPU
before returning, which closes the window. It is announced in a warning at
load, and `TF_METAL_SYNCHRONOUS` forces it either way. When the entry points
come back the plugin returns to running asynchronously with no change here.

## Is it faster than the CPU

Sometimes, and by how much depends entirely on the shape of the work. Measured
on an M4 Max against TensorFlow 2.21.0, median of ten runs each, both devices
in the same process on the same data, waiting for the device before stopping
the clock:

Two columns of speedup, because the wait above costs most of it. "Today" is
what you get from a released TensorFlow; "async" is the same machine with
`TF_METAL_SYNCHRONOUS=0`, which is what the plugin does once
[#126374](https://github.com/tensorflow/tensorflow/issues/126374) is fixed.

| | GPU today | CPU | today | async |
| --- | ---: | ---: | ---: | ---: |
| MatMul 2048x2048 | 3.80 ms | 14.42 ms | **3.8x** | 6.5x |
| Conv2D, batch 64, 64x64x32 to 64 | 4.58 ms | 10.91 ms | **2.4x** | 3.0x |
| MatMul 1024x1024 | 1.12 ms | 2.17 ms | 1.9x | 1.5x |
| CNN training step, SGD, batch 128 | 19.46 ms | 20.61 ms | 1.1x | 1.7x |
| CNN forward, batch 128 | 5.53 ms | 6.00 ms | 1.1x | 1.5x |
| MatMul 512x512 | 0.44 ms | 0.36 ms | 0.8x | 0.9x |
| ReduceSum 4096x4096 | 0.44 ms | 0.34 ms | 0.8x | 0.8x |
| Elementwise 4096x4096 | 2.88 ms | 1.54 ms | 0.5x | 0.5x |

The pattern is the ordinary one and worth stating plainly: the GPU wins where
there is arithmetic to do per byte moved, and loses where there is not. A
4096x4096 elementwise chain moves 67 MB and does three floating point
operations per element, so it is bound by memory on a machine whose CPU shares
that same memory. Small matrices lose to the cost of getting work to the
device at all.

A training step is 1.7x the CPU when the plugin can run asynchronously, and
barely ahead of it while it cannot. That gap is the cost of the missing entry
points, not of the backend.

`benchmarks/benchmark.py` reproduces the table.

## Build

Needs the macOS 15 SDK or later and a Python with TensorFlow installed. The
backend aliases an `MTLBuffer` through `MPSNDArray` with packed rows, and both
`initWithBuffer:offset:descriptor:` and `preferPackedRows` arrived in that SDK;
an older one does not declare them and the build stops rather than degrading. The
header and library paths come from that TensorFlow, so the plugin is built
against exactly the one it will be loaded into.

```
make                                  # or: make PYTHON=/path/to/venv/bin/python
make check-symbols
make test
```

Then either point TensorFlow at it directly:

```python
from tensorflow.python.framework import load_library
load_library.load_pluggable_device_library("build/libmetal_plugin.dylib")
```

or install it so that `import tensorflow` finds it:

```
make install
```

`TF_DISABLE_METAL=1` keeps the backend out of the process without
uninstalling it.

## What a released TensorFlow cannot do

Six entry points of the kernel C API are declared in the headers a released
TensorFlow ships and are exported by no binary in it:

```
TF_AssignRefVariable
TF_AssignUpdateVariable
TF_GetInputTensorFromVariable
TF_MaybeLockVariableInputMutexesInOrder
TF_ReleaseVariableInputLockHolder
TF_OpKernelConstruction_GetAttrTensorShape
TF_OpKernelContext_ForwardRefInputToRefOutput
```

Checked against `tensorflow==2.20.0` on macOS arm64: absent from
`libtensorflow_framework.2.dylib`, from `libtensorflow_cc.2.dylib`, and from
every pywrap module, and unresolvable by `dlsym` inside a live process.
`TF_AllocateOutput` and `TF_NewKernelBuilder`, from the same header set, are
exported normally, so this is not a matter of the whole C API being private.

Fifteen ops need them, and the plugin does not register those when the symbols
are missing, logging one warning instead:

| Family | Ops |
| --- | --- |
| Optimisers | `ResourceApplyAdam`, `ResourceApplyGradientDescent`, `ResourceApplyMomentum`, `ResourceApplyKerasMomentum`, `ResourceApplyRMSProp` |
| Resource gather and scatter | `ResourceGather`, `ResourceGatherNd`, `ResourceScatterUpdate`, `GatherNd` |
| Reference variables | `Assign`, `AssignAdd`, `AssignSub` |
| Parallel stacking | `ParallelConcat`, `_ParallelConcatStart`, `_ParallelConcatUpdate` |

The optimisers are the whole of that list that matters: **without them there is
no training on the GPU**, only inference and manual gradient work. They run on
the host instead, which is correct and slow.

This is a regression, not a standing limitation. All fourteen symbols of
`tensorflow/c/kernels_experimental.cc` are exported by `libtensorflow_framework`
in 2.19.1 and 2.18.1, and absent from every binary in the 2.20.0 wheel, with
none added in exchange. The headers still declare them. Filed upstream as
[tensorflow/tensorflow#126374](https://github.com/tensorflow/tensorflow/issues/126374).

So this is not something the plugin can work around, and it is not permanent
either: when those exports come back, the fifteen ops below start working here
with no change to this repository.

It is also the sharpest argument for the in-tree form, where the same code
links these functions directly and all fifteen ops work. That trade is the
subject of the discussion on
[#126254](https://github.com/tensorflow/tensorflow/pull/126254).

## Why this exists

Apple's `tensorflow-metal` last shipped 1.2.0 on 2025-01-31, publishes no
wheel past cp312, has no sdist, and its repository was archived in 2021. TF
master requires Python 3.10 or later and classifies up to cp313, so on a
current Python there is no GPU path for TensorFlow on a Mac at all.

## Op coverage

The backend registers every op TensorFlow registers for `DEVICE_GPU`, less the
five TensorRT ops that `if_tensorrt` excludes from a macOS build, and less the
fifteen above when the C API entry points they need are missing. The table of
Metal kernels with their dtypes is in
[docs/ops.md](docs/ops.md).

## Layout

```
src/plugin_init.cc                          the two exported entry points
src/tensorflow/core/common_runtime/metal/   the backend, verbatim from the
                                            TensorFlow tree
tools/                                      build probes and the symbol check
tests/                                      on-device checks against CPU
```

The backend sources keep their TensorFlow paths so that syncing them from the
tree is a copy rather than a patch. Two macros, `TF_METAL_OUT_OF_TREE` and
`TF_METAL_NO_STREAM_OPTIONS`, are the whole of what the out-of-tree build
turns on; both are no-ops in the tree.

## Licence

Apache 2.0, the same as TensorFlow.
