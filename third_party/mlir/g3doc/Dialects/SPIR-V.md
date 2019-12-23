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

*   Stay as the same semantic level and try to be a mechanical 1:1 mapping;
*   But deviate representationally if possible with MLIR mechanisms.
*   Be straightforward to serialize into and deserialize from the SPIR-V binary
    format.

## Conventions

The SPIR-V dialect has the following conventions:

*   The prefix for all SPIR-V types and operations are `spv.`.
*   Ops that directly mirror instructions in the binary format have `CamelCase`
    names that are the same as the instruction opnames (without the `Op`
    prefix). For example, `spv.FMul` is a direct mirror of `OpFMul`. They will
    be serialized into and deserialized from one instruction.
*   Ops with `snake_case` names are those that have different representation
    from corresponding instructions (or concepts) in the binary format. These
    ops are mostly for defining the SPIR-V structure. For example, `spv.module`
    and `spv.constant`. They may correspond to zero or more instructions during
    (de)serialization.
*   Ops with `_snake_case` names are those that have no corresponding
    instructions (or concepts) in the binary format. They are introduced to
    satisfy MLIR structural requirements. For example, `spv._module_end` and
    `spv._merge`. They maps to no instructions during (de)serialization.

## Module

A SPIR-V module is defined via the `spv.module` op, which has one region that
contains one block. Model-level instructions, including function definitions,
are all placed inside the block. Functions are defined using the builtin `func`
op.

Compared to the binary format, we adjust how certain module-level SPIR-V
instructions are represented in the SPIR-V dialect. Notably,

*   Requirements for capabilities, extensions, extended instruction sets,
    addressing model, and memory model is conveyed using `spv.module`
    attributes. This is considered better because these information are for the
    execution environment. It's easier to probe them if on the module op
    itself.
*   Annotations/decoration instructions are "folded" into the instructions they
    decorate and represented as attributes on those ops. This eliminates
    potential forward references of SSA values, improves IR readability, and
    makes querying the annotations more direct.
*   Types are represented using MLIR standard types and SPIR-V dialect specific
    types. There are no type declaration ops in the SPIR-V dialect.
*   Various normal constant instructions are represented by the same
    `spv.constant` op. Those instructions are just for constants of different
    types; using one op to represent them reduces IR verbosity and makes
    transformations less tedious.
*   Normal constants are not placed in `spv.module`'s region; they are localized
    into functions. This is to make functions in the SPIR-V dialect to be
    isolated and explicit capturing.
*   Global variables are defined with the `spv.globalVariable` op. They do not
    generate SSA values. Instead they have symbols and should be referenced via
    symbols. To use a global variables in a function block, `spv._address_of` is
    needed to turn the symbol into a SSA value.
*   Specialization constants are defined with the `spv.specConstant` op. Similar
    to global variables, they do not generate SSA values and have symbols for
    reference, too. `spv._reference_of` is needed to turn the symbol into a SSA
    value for use in a function block.

## Types

The SPIR-V dialect reuses standard integer, float, and vector types and defines
the following dialect-specific types:

```
spirv-type ::= array-type
             | pointer-type
             | runtime-array-type
```

### Array type

This corresponds to SPIR-V [array type][ArrayType]. Its syntax is

```
element-type ::= integer-type
               | floating-point-type
               | vector-type
               | spirv-type

array-type ::= `!spv.array<` integer-literal `x` element-type `>`
```

For example,

```mlir
!spv.array<4 x i32>
!spv.array<16 x vector<4 x f32>>
```

### Image type

This corresponds to SPIR-V [image type][ImageType]. Its syntax is

```
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

```
!spv.image<f32, 1D, NoDepth, NonArrayed, SingleSampled, SamplerUnknown, Unknown>
!spv.image<f32, Cube, IsDepth, Arrayed, MultiSampled, NeedSampler, Rgba32f>
```

### Pointer type

This corresponds to SPIR-V [pointer type][PointerType]. Its syntax is

```
storage-class ::= `UniformConstant`
                | `Uniform`
                | `Workgroup`
                | <and other storage classes...>

pointer-type ::= `!spv.ptr<` element-type `,` storage-class `>`
```

For example,

```mlir
!spv.ptr<i32, Function>
!spv.ptr<vector<4 x f32>, Uniform>
```

### Runtime array type

This corresponds to SPIR-V [runtime array type][RuntimeArrayType]. Its syntax is

```
runtime-array-type ::= `!spv.rtarray<` element-type `>`
```

For example,

```mlir
!spv.rtarray<i32>
!spv.rtarray<vector<4 x f32>>
```

### Struct type

This corresponds to SPIR-V [struct type][StructType]. Its syntax is

```
struct-member-decoration ::= integer-literal? spirv-decoration*
struct-type ::= `!spv.struct<` spirv-type (`[` struct-member-decoration `]`)?
                     (`, ` spirv-type (`[` struct-member-decoration `]`)?
```

For Example,

```
!spv.struct<f32>
!spv.struct<f32 [0]>
!spv.struct<f32, !spv.image<f32, 1D, NoDepth, NonArrayed, SingleSampled, SamplerUnknown, Unknown>>
!spv.struct<f32 [0], i32 [4]>
```

## Function

A SPIR-V function is defined using the builtin `func` op. `spv.module` verifies
that the functions inside it comply with SPIR-V requirements: at most one
result, no nested functions, and so on.

## Operations

Operation documentation is written in each op's Op Definition Spec using
TableGen. A markdown version of the doc can be generated using `mlir-tblgen
-gen-doc`.

## Control Flow

SPIR-V binary format uses merge instructions (`OpSelectionMerge` and
`OpLoopMerge`) to declare structured control flow. They explicitly declare a
header block before the control flow diverges and a merge block where control
flow subsequently converges. These blocks delimit constructs that must nest, and
can only be entered and exited in structured ways.

In the SPIR-V dialect, we use regions to mark the boundary of a structured
control flow construct. With this approach, it's easier to discover all blocks
belonging to a structured control flow construct. It is also more idiomatic to
MLIR system.

We introduce a `spv.selection` and `spv.loop` op for structured selections and
loops, respectively. The merge targets are the next ops following them. Inside
their regions, a special terminator, `spv._merge` is introduced for branching to
the merge target.

### Selection

`spv.selection` defines a selection construct. It contains one region. The
region should contain at least two blocks: one selection header block and one
merge block.

*   The selection header block should be the first block. It should contain the
    `spv.BranchConditional` or `spv.Switch` op.
*   The merge block should be the last block. The merge block should only
    contain a `spv._merge` op. Any block can branch to the merge block for early
    exit.

```
               +--------------+
               | header block |                 (may have multiple outgoing branches)
               +--------------+
                    / | \
                     ...


   +---------+   +---------+   +---------+
   | case #0 |   | case #1 |   | case #2 |  ... (may have branches between each other)
   +---------+   +---------+   +---------+


                     ...
                    \ | /
                      v
               +-------------+
               | merge block |                  (may have multiple incoming branches)
               +-------------+
```

For example, for the given function

```c++
void loop(bool cond) {
  int x = 0;
  if (cond) {
    x = 1;
  } else {
    x = 2;
  }
  // ...
}
```

It will be represented as

```mlir
func @selection(%cond: i1) -> () {
  %zero = spv.constant 0: i32
  %one = spv.constant 1: i32
  %two = spv.constant 2: i32
  %x = spv.Variable init(%zero) : !spv.ptr<i32, Function>

  spv.selection {
    spv.BranchConditional %cond, ^then, ^else

  ^then:
    spv.Store "Function" %x, %one : i32
    spv.Branch ^merge

  ^else:
    spv.Store "Function" %x, %two : i32
    spv.Branch ^merge

  ^merge:
    spv._merge
  }

  // ...
}

```

### Loop

`spv.loop` defines a loop construct. It contains one region. The region should
contain at least four blocks: one entry block, one loop header block, one loop
continue block, one merge block.

*   The entry block should be the first block and it should jump to the loop
    header block, which is the second block.
*   The merge block should be the last block. The merge block should only
    contain a `spv._merge` op. Any block except the entry block can branch to
    the merge block for early exit.
*   The continue block should be the second to last block and it should have a
    branch to the loop header block.
*   The loop continue block should be the only block, except the entry block,
    branching to the loop header block.

```
    +-------------+
    | entry block |           (one outgoing branch)
    +-------------+
           |
           v
    +-------------+           (two incoming branches)
    | loop header | <-----+   (may have one or two outgoing branches)
    +-------------+       |
                          |
          ...             |
         \ | /            |
           v              |
   +---------------+      |   (may have multiple incoming branches)
   | loop continue | -----+   (may have one or two outgoing branches)
   +---------------+

          ...
         \ | /
           v
    +-------------+           (may have multiple incoming branches)
    | merge block |
    +-------------+
```

The reason to have another entry block instead of directly using the loop header
block as the entry block is to satisfy region's requirement: entry block of
region may not have predecessors. We have a merge block so that branch ops can
reference it as successors. The loop continue block here corresponds to
"continue construct" using SPIR-V spec's term; it does not mean the "continue
block" as defined in the SPIR-V spec, which is "a block containing a branch to
an OpLoopMerge instruction’s Continue Target."

For example, for the given function

```c++
void loop(int count) {
  for (int i = 0; i < count; ++i) {
    // ...
  }
}
```

It will be represented as

```mlir
func @loop(%count : i32) -> () {
  %zero = spv.constant 0: i32
  %one = spv.constant 1: i32
  %var = spv.Variable init(%zero) : !spv.ptr<i32, Function>

  spv.loop {
    spv.Branch ^header

  ^header:
    %val0 = spv.Load "Function" %var : i32
    %cmp = spv.SLessThan %val0, %count : i32
    spv.BranchConditional %cmp, ^body, ^merge

  ^body:
    // ...
    spv.Branch ^continue

  ^continue:
    %val1 = spv.Load "Function" %var : i32
    %add = spv.IAdd %val1, %one : i32
    spv.Store "Function" %var, %add : i32
    spv.Branch ^header

  ^merge:
    spv._merge
  }
  return
}
```

### Block argument for Phi

There are no direct Phi operations in the SPIR-V dialect; SPIR-V `OpPhi`
instructions are modelled as block arguments in the SPIR-V dialect. (See the
[Rationale][Rationale] doc for "Block Arguments vs Phi nodes".) Each block
argument corresponds to one `OpPhi` instruction in the SPIR-V binary format. For
example, for the following SPIR-V function `foo`:

```spirv
  %foo = OpFunction %void None ...
%entry = OpLabel
  %var = OpVariable %_ptr_Function_int Function
         OpSelectionMerge %merge None
         OpBranchConditional %true %true %false
 %true = OpLabel
         OpBranch %phi
%false = OpLabel
         OpBranch %phi
  %phi = OpLabel
  %val = OpPhi %int %int_1 %false %int_0 %true
         OpStore %var %val
         OpReturn
%merge = OpLabel
         OpReturn
         OpFunctionEnd
```

It will be represented as:

```mlir
func @foo() -> () {
  %var = spv.Variable : !spv.ptr<i32, Function>

  spv.selection {
    %true = spv.constant true
    spv.BranchConditional %true, ^true, ^false

  ^true:
    %zero = spv.constant 0 : i32
    spv.Branch ^phi(%zero: i32)

  ^false:
    %one = spv.constant 1 : i32
    spv.Branch ^phi(%one: i32)

  ^phi(%arg: i32):
    spv.Store "Function" %var, %arg : i32
    spv.Return

  ^merge:
    spv._merge
  }
  spv.Return
}
```

## Serialization and deserialization

The serialization library provides two entry points, `mlir::spirv::serialize()`
and `mlir::spirv::deserialize()`, for converting a MLIR SPIR-V module to binary
format and back.

The purpose of this library is to enable importing SPIR-V binary modules to run
transformations on them and exporting SPIR-V modules to be consumed by execution
environments. The focus is transformations, which inevitably means changes to
the binary module; so it is not designed to be a general tool for investigating
the SPIR-V binary module and does not guarantee roundtrip equivalence (at least
for now). For the latter, please use the assembler/disassembler in the
[SPIRV-Tools][SPIRV-Tools] project.

A few transformations are performed in the process of serialization because of
the representational differences between SPIR-V dialect and binary format:

*   Attributes on `spv.module` are emitted as their corresponding SPIR-V
    instructions.
*   `spv.constant`s are unified and placed in the SPIR-V binary module section
    for types, constants, and global variables.
*   `spv.selection`s and `spv.loop`s are emitted as basic blocks with `Op*Merge`
    instructions in the header block as required by the binary format.

Similarly, a few transformations are performed during deserialization:

*   Instructions for execution environment requirements will be placed as
    attributes on `spv.module`.
*   `OpConstant*` instructions are materialized as `spv.constant` at each use
    site.
*   `OpPhi` instructions are converted to block arguments.
*   Structured control flow are placed inside `spv.selection` and `spv.loop`.

[SPIR-V]: https://www.khronos.org/registry/spir-v/
[ArrayType]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpTypeArray
[ImageType]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpTypeImage
[PointerType]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpTypePointer
[RuntimeArrayType]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpTypeRuntimeArray
[StructType]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#Structure
[SPIRV-Tools]: https://github.com/KhronosGroup/SPIRV-Tools
[Rationale]: https://github.com/tensorflow/mlir/blob/master/g3doc/Rationale.md#block-arguments-vs-phi-nodes
