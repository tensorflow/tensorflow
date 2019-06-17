# SPIR-V Dialect

This document defines the SPIR-V dialect in MLIR.

[SPIR-V][SPIR-V] is the Khronos Group’s binary intermediate language for
representing graphics shaders and compute kernels. It is adopted by multiple
Khronos Group’s APIs, including Vulkan and OpenCL.

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

The namespace for all SPIR-V types and operations are `spv`.

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
