# SPIR-V Dialect

This document defines the SPIR-V dialect in MLIR.

[SPIR-V][SPIR-V] is the Khronos Group’s binary intermediate language for
representing graphics shaders and compute kernels. It is adopted by multiple
Khronos Group’s APIs, including Vulkan and OpenCL.

## Design Principles

SPIR-V defines a stable binary format for hardware driver consumption.
Regularity is one of the design goals of SPIR-V. All concepts are represented
as SPIR-V instructions, including declaring extensions and capabilities,
defining types and constants, defining functions, attaching additional
properties to computation results, etc. This way favors driver consumption
but not necessarily compiler transformations.

The purpose of the SPIR-V dialect is to serve as the "proxy" of the binary
format and to facilitate transformations. Therefore, it should

* Be trivial to serialize into the SPIR-V binary format;
* Stay as the same semantic level and try to be a mechanical 1:1 mapping;
* But deviate representationally if possible with MLIR mechanisms.

## Conventions

The SPIR-V dialect has the following conventions:

* The prefix for all SPIR-V types and operations are `spv.`.
* Ops that directly correspond to instructions in the binary format have
  `CamelCase` names, for example, `spv.FMul`;
* Otherwise they have `snake_case` names. These ops are mostly for defining
  the SPIR-V structure, inclduing module, function, and module-level ops.
  For example, `spv.module`, `spv.constant`.

## Module

A SPIR-V module is defined via the `spv.module` op, which has one region that
contains one block. Model-level instructions, including function definitions,
are all placed inside the block.

Compared to the binary format, we adjust how certain module-level SPIR-V
instructions are represented in the SPIR-V dialect. Notably,

* Requirements for capabilities, extensions, extended instruction sets,
  addressing model, and memory model is conveyed using `spv.module` attributes.
  This is considered better because these information are for the
  exexcution environment. It's eaiser to probe them if on the module op
  itself.
* Annotations/decoration instrutions are "folded" into the instructions they
  decorate and represented as attributes on those ops. This elimiates potential
  forward references of SSA values, improves IR readability, and makes
  querying the annotations more direct.
* Various constant instructions are represented by the same `spv.constant`
  op. Those instructions are just for constants of different types; using one
  op to represent them reduces IR verbosity and makes transformations less
  tedious.

## Types

The SPIR-V dialect reuses standard integer, float, and vector types and defines
the following dialect-specific types:

``` {.ebnf}
spirv-type ::= array-type
             | pointer-type
             | runtime-array-type
```

### Array type

This corresponds to SPIR-V [array type][ArrayType]. Its syntax is

``` {.ebnf}
element-type ::= integer-type
               | floating-point-type
               | vector-type
               | spirv-type

array-type ::= `!spv.array<` integer-literal `x` element-type `>`
```

For example,

```{.mlir}
!spv.array<4 x i32>
!spv.array<16 x vector<4 x f32>>
```

### Image type

This corresponds to SPIR-V [image_type][ImageType]. Its syntax is

``` {.ebnf}
dim ::= `1D` | `2D` | `3D` | `Cube` | <and other SPIR-V Dim specifiers...>

depth-info ::= `NoDepth` | `IsDepth` | `DepthUnknown`

arrayed-info ::= `NonArrayed` | `Arrayed`

sampling-info ::= `SingleSampled` | `MultiSampled`

sampler-use-info ::= `SamplerUnknown` | `NeedSampler` | `NoSampler`

format ::= `Unknown` | `Rgba32f` | <and other SPIR-V Image Formats...>

image-type ::= `!spv.image<` element-type `,` dim `,` depth-info `,`
                           arrayed-info `,` sampling-info `,`
                           sampler-use-info `,` format `>`
```

For example,

``` {.mlir}
!spv.image<f32, 1D, NoDepth, NonArrayed, SingleSampled, SamplerUnknown, Unknown>
!spv.image<f32, Cube, IsDepth, Arrayed, MultiSampled, NeedSampler, Rgba32f>
```

### Pointer type

This corresponds to SPIR-V [pointer type][PointerType]. Its syntax is

``` {.ebnf}
storage-class ::= `UniformConstant`
                | `Uniform`
                | `Workgroup`
                | <and other storage classes...>

pointer-type ::= `!spv.ptr<` element-type `,` storage-class `>`
```

For example,

```{.mlir}
!spv.ptr<i32, Function>
!spv.ptr<vector<4 x f32>, Uniform>
```

### Runtime array type

This corresponds to SPIR-V [runtime array type][RuntimeArrayType]. Its syntax is

``` {.ebnf}
runtime-array-type ::= `!spv.rtarray<` element-type `>`
```

For example,

```{.mlir}
!spv.rtarray<i32>
!spv.rtarray<vector<4 x f32>>
```

[SPIR-V]: https://www.khronos.org/registry/spir-v/
[ArrayType]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpTypeArray
[PointerType]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpTypePointer
[RuntimeArrayType]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpTypeRuntimeArray
[ImageType]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpTypeImage
