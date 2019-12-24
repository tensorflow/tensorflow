# SPIR-V Dialect

This document describes the design of the SPIR-V dialect in MLIR. It lists
various design choices we made for modeling different SPIR-V mechanisms, and
their rationale.

This document also explains in a high-level manner how different components are
organized and implemented in the code and gives steps to follow for extending
them.

This document assumes familiarity with SPIR-V. [SPIR-V][Spirv] is the Khronos
Group’s binary intermediate language for representing graphics shaders and
compute kernels. It is adopted by multiple Khronos Group’s APIs, including
Vulkan and OpenCL. It is fully defined in a
[human-readable specification][SpirvSpec]; the syntax of various SPIR-V
instructions are encoded in a [machine-readable grammar][SpirvGrammar].

## Design Guidelines

SPIR-V is a binary intermediate language that serves dual purpose: on one side,
it is an intermediate language to represent graphics shaders and compute kernels
for high-level languages to target; on the other side, it defines a stable
binary format for hardware driver consumption. As a result, SPIR-V has design
principles pertain to not only intermediate language, but also binary format.
For example, regularity is one of the design goals of SPIR-V. All concepts are
represented as SPIR-V instructions, including declaring extensions and
capabilities, defining types and constants, defining functions, attaching
additional properties to computation results, etc. This way favors binary
encoding and decoding for driver consumption but not necessarily compiler
transformations.

### Dialect design principles

The main objective of the SPIR-V dialect is to be a proper intermediate
representation (IR) to facilitate compiler transformations. While we still aim
to support serializing to and deserializing from the binary format for various
good reasons, the binary format and its concerns play less a role in the design
of the SPIR-V dialect: when there is a trade-off to be made between favoring IR
and supporting binary format, we lean towards the former.

On the IR aspect, the SPIR-V dialect aims to model SPIR-V at the same semantic
level. It is not intended to be a higher level or lower level abstraction than
the SPIR-V specification. Those abstractions are easily outside the domain of
SPIR-V and should be modeled with other proper dialects so they can be shared
among various compilation paths. Because of the dual purpose of SPIR-V, SPIR-V
dialect staying at the same semantic level as the SPIR-V specification also
means we can still have straightforward serailization and deserailization for
the majority of functionalities.

To summarize, the SPIR-V dialect follows the following design principles:

*   Stay as the same semantic level as the SPIR-V specification by having
    one-to-one mapping for most concepts and entities.
*   Adopt SPIR-V specification's syntax if possible, but deviate intentionally
    to utilize MLIR mechanisms if it results in better representation and
    benefits transformation.
*   Be straightforward to serialize into and deserialize from the SPIR-V binary
    format.

SPIR-V is designed to be consumed by hardware drivers, so its representation is
quite clear, yet verbose for some cases. Allowing representational deviation
gives us the flexibility to reduce the verbosity by using MLIR mechanisms.

### Dialect scopes

SPIR-V supports multiple execution environments, specified by client APIs.
Notable adopters include Vulkan and OpenCL. It follows that the SPIR-V dialect
should support multiple execution environments if to be a proper proxy of SPIR-V
in MLIR systems. The SPIR-V dialect is designed with these considerations: it
has proper support for versions, extensions, and capabilities and is as
extensible as SPIR-V specification.

## Conventions

The SPIR-V dialect adopts the following conventions for IR:

*   The prefix for all SPIR-V types and operations are `spv.`.
*   All instructions in an extended instruction set are further qualified with
    the extended instruction set's prefix. For example, all operations in the
    GLSL extended instruction set is has the prefix of `spv.GLSL.`.
*   Ops that directly mirror instructions in the specification have `CamelCase`
    names that are the same as the instruction opnames (without the `Op`
    prefix). For example, `spv.FMul` is a direct mirror of `OpFMul` in the
    specification. Such an op will be serialized into and deserialized from one
    SPIR-V instruction.
*   Ops with `snake_case` names are those that have different representation
    from corresponding instructions (or concepts) in the specification. These
    ops are mostly for defining the SPIR-V structure. For example, `spv.module`
    and `spv.constant`. They may correspond to one or more instructions during
    (de)serialization.
*   Ops with `_snake_case` names are those that have no corresponding
    instructions (or concepts) in the binary format. They are introduced to
    satisfy MLIR structural requirements. For example, `spv._module_end` and
    `spv._merge`. They maps to no instructions during (de)serialization.

(TODO: consider merging the last two cases and adopting `spv.mlir.` prefix for
them.)

## Module

A SPIR-V module is defined via the `spv.module` op, which has one region that
contains one block. Model-level instructions, including function definitions,
are all placed inside the block. Functions are defined using the builtin `func`
op.

We choose to model a SPIR-V module with a dedicated `spv.module` op based on the
following considerations:

*   It maps cleanly to a SPIR-V module in the specification.
*   We can enforce SPIR-V specific verification that is suitable to be performed
    at the module-level.
*   We can attach additional model-level attributes.
*   We can control custom assembly form.

The `spv.module` op's region cannot capture SSA values from outside, neither
implicitly nor explicitly. The `spv.module` op's region is closed as to what ops
can appear inside: apart from the builtin `func` op, it can only contain ops
from the SPIR-V dialect. The `spv.module` op's verifier enforces this rule. This
meaningfully guarantees that a `spv.module` can be the entry point and boundary
for serialization.

### Module-level operations

SPIR-V binary format defines the following [sections][SpirvLogicalLayout]:

1.  Capabilities required by the module.
1.  Extensions required by the module.
1.  Extended instructions sets required by the module.
1.  Addressing and memory model specification.
1.  Entry point specifications.
1.  Execution mode declarations.
1.  Debug instructions.
1.  Annotation/decoration instructions.
1.  Type, constant, global variables.
1.  Function declarations.
1.  Function definitions.

Basically, a SPIR-V binary module contains multiple module-level instructions
followed by a list of functions. Those module-level instructions are essential
and they can generate result ids referenced by functions, notably, declaring
resource variables to interact with the execution environment.

Compared to the binary format, we adjust how these module-level SPIR-V
instructions are represented in the SPIR-V dialect:

#### Use MLIR attributes for metadata

*   Requirements for capabilities, extensions, extended instruction sets,
    addressing model, and memory model is conveyed using `spv.module`
    attributes. This is considered better because these information are for the
    execution environment. It's easier to probe them if on the module op itself.
*   Annotations/decoration instructions are "folded" into the instructions they
    decorate and represented as attributes on those ops. This eliminates
    potential forward references of SSA values, improves IR readability, and
    makes querying the annotations more direct. More discussions can be found in
    the [`Decorations`](#decorations) section.

#### Model types with MLIR custom types

*   Types are represented using MLIR standard types and SPIR-V dialect specific
    types. There are no type declaration ops in the SPIR-V dialect. More
    discussions can be found in the [Types](#types) section later.

#### Unify and localize constants

*   Various normal constant instructions are represented by the same
    `spv.constant` op. Those instructions are just for constants of different
    types; using one op to represent them reduces IR verbosity and makes
    transformations less tedious.
*   Normal constants are not placed in `spv.module`'s region; they are localized
    into functions. This is to make functions in the SPIR-V dialect to be
    isolated and explicit capturing. Constants are cheap to duplicate given
    attributes are uniqued in `MLIRContext`.

#### Adopt symbol-based global variables and specialization constant

*   Global variables are defined with the `spv.globalVariable` op. They do not
    generate SSA values. Instead they have symbols and should be referenced via
    symbols. To use a global variables in a function block, `spv._address_of` is
    needed to turn the symbol into a SSA value.
*   Specialization constants are defined with the `spv.specConstant` op. Similar
    to global variables, they do not generate SSA values and have symbols for
    reference, too. `spv._reference_of` is needed to turn the symbol into a SSA
    value for use in a function block.

The above choices enables functions in the SPIR-V dialect to be isolated and
explicit capturing.

#### Disallow implicit capturing in functions

*   In SPIR-V specification, functions support implicit capturing: they can
    reference SSA values defined in modules. In the SPIR-V dialect functions are
    defined with `func` op, which disallows implicit capturing. This is more
    friendly to compiler analyses and transformations. More discussions can be
    found in the [Function](#function) section later.

### Model entry points and execution models as normal ops

*   A SPIR-V module can have multiple entry points. And these entry points refer
    to the function and interface variables. It’s not suitable to model them as
    `spv.module` op attributes. We can model them as normal ops of using symbol
    references.
*   Similarly for execution modes, which are coupled with entry points, we can
    model them as normal ops in `spv.module`'s region.

## Decorations

Annotations/decorations provide additional information on result ids. In SPIR-V,
all instructions can generate result ids, including value-computing and
type-defining ones.

For decorations on value result ids, we can just have a corresponding attribute
attached to the operation generating the SSA value. For example, for the
following SPIR-V:

```spirv
OpDecorate %v1 RelaxedPrecision
OpDecorate %v2 NoContraction
...
%v1 = OpFMul %float %0 %0
%v2 = OpFMul %float %1 %1
```

We can represent them in the SPIR-V dialect as:

```mlir
%v1 = "spv.FMul"(%0, %0) {RelaxedPrecision: unit} : (f32, f32) -> (f32)
%v2 = "spv.FMul"(%1, %1) {NoContraction: unit} : (f32, f32) -> (f32)
```

This approach benefits transformations. Essentially those decorations are just
additional properties of the result ids (and thus their defining instructions).
In SPIR-V binary format, they are just represented as instructions. Literally
following SPIR-V binary format means we need to through def-use chains to find
the decoration instructions and query information from them.

For decorations on type result ids, notice that practically, only result ids
generated from composite types (e.g., `OpTypeArray`, `OpTypeStruct`) need to be
decorated for memory layouting purpose (e.g., `ArrayStride`, `Offset`, etc.);
scalar/vector types are required to be uniqued in SPIR-V. Therefore, we can just
encode them directly in the dialect-specific type.

## Types

Theoretically we can define all SPIR-V types using MLIR extensible type system,
but other than representational purity, it does not buy us more. Instead, we
need to maintain the code and invest in pretty printing them. So we prefer to
use builtin/standard types if possible.

The SPIR-V dialect reuses standard integer, float, and vector types:

Specification                        | Dialect
:----------------------------------: | :-------------------------------:
`OpTypeBool`                         | `i1`
`OpTypeInt <bitwidth>`               | `i<bitwidth>`
`OpTypeFloat <bitwidth>`             | `f<bitwidth>`
`OpTypeVector <scalar-type> <count>` | `vector<<count> x <scalar-type>>`

Similarly, `mlir::NoneType` can be used for SPIR-V `OpTypeVoid`; builtin
function types can be used for SPIR-V `OpTypeFunction` types.

The SPIR-V dialect and defines the following dialect-specific types:

```
spirv-type ::= array-type
             | image-type
             | pointer-type
             | runtime-array-type
             | struct-type
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

```mlir
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

```mlir
!spv.struct<f32>
!spv.struct<f32 [0]>
!spv.struct<f32, !spv.image<f32, 1D, NoDepth, NonArrayed, SingleSampled, SamplerUnknown, Unknown>>
!spv.struct<f32 [0], i32 [4]>
```

## Function

In SPIR-V, a function construct consists of multiple instructions involving
`OpFunction`, `OpFunctionParameter`, `OpLabel`, `OpFunctionEnd`.

```spirv
// int f(int v) { return v; }
%1 = OpTypeInt 32 0
%2 = OpTypeFunction %1 %1
%3 = OpFunction %1 %2
%4 = OpFunctionParameter %1
%5 = OpLabel
%6 = OpReturnValue %4
     OpFunctionEnd
```

This construct is very clear yet quite verbose. It is intended for driver
consumption. There is little benefit to literally replicate this construct in
the SPIR-V dialect. Instead, we reuse the builtin `func` op to express functions
more concisely:

```mlir
func @f(%arg: i32) -> i32 {
  "spv.ReturnValue"(%arg) : (i32) -> (i32)
}
```

A SPIR-V function can have at most one result. It cannot contain nested
functions or non-SPIR-V operations. `spv.module` verifies these requirements.

A major difference between the SPIR-V dialect and the SPIR-V specification for
functions is that the former are isolated and require explicit capturing, while
the latter allow implicit capturing. In SPIR-V specification, functions can
refer to SSA values (generated by constants, global variables, etc.) defined in
modules. The SPIR-V dialect adjusted how constants and global variables are
modeled to enable isolated functions. Isolated functions are more friendly to
compiler analyses and transformations. This also enables the SPIR-V dialect to
better utilize core infrastructure: many functionalities in the core
infrastructure requires ops to be isolated, e.g., the
[greedy pattern rewriter][GreedyPatternRewriter] can only act on ops isolated
from above.

(TODO: create a dedicated `spv.fn` op for SPIR-V functions.)

## Operations

In SPIR-V, instruction is a generalized concept; a SPIR-V module is just a
sequence of instructions. Declaring types, expressing computations, annotating
result ids, expressing control flows and others are all in the form of
instructions.

We only discuss instructions expressing computations here, which can be
represented via SPIR-V dialect ops. Module-level instructions for declarations
and definitions are represented differently in the SPIR-V dialect as explained
earlier in the [Module-level operations](#module-level-operations) section.

An instruction computes zero or one result from zero or more operands. The
result is a new result id. An operand can be a result id generated by a previous
instruction, an immediate value, or a case of an enum type. We can model result
id operands and results with MLIR SSA values; for immediate value and enum
cases, we can model them with MLIR attributes.

For example,

```spirv
%i32 = OpTypeInt 32 0
%c42 = OpConstant %i32 42
...
%3 = OpVariable %i32 Function 42
%4 = OpIAdd %i32 %c42 %c42
```

can be represented in the dialect as

```mlir
%0 = "spv.constant"() { value = 42 : i32 } : () -> i32
%1 = "spv.Variable"(%0) { storage_class = "Function" } : (i32) -> !spv.ptr<i32, Function>
%2 = "spv.IAdd"(%0, %0) : (i32, i32) -> i32
```

Operation documentation is written in each op's Op Definition Spec using
TableGen. A markdown version of the doc can be generated using `mlir-tblgen
-gen-doc`.

### Ops from extended instruction sets

Analogically extended instruction set is a mechanism to import SPIR-V
instructions within another namespace. [`GLSL.std.450`][GlslStd450] is an
extended instruction set that provides common mathematical routines that should
be supported. Instead of modeling `OpExtInstImport` as a separate op and use a
single op to model `OpExtInst` for all extended instructions, we model each
SPIR-V instruction in an extended instruction set as a separate op with the
proper name prefix. For example, for

```spirv
%glsl = OpExtInstImport "GLSL.std.450"

%f32 = OpTypeFloat 32
%cst = OpConstant %f32 ...

%1 = OpExtInst %f32 %glsl 28 %cst
%2 = OpExtInst %f32 %glsl 31 %cst
```

we can have

```mlir
%1 = "spv.GLSL.Log"(%cst) : (f32) -> (f32)
%2 = "spv.GLSL.Sqrt(%cst) : (f32) -> (f32)
```

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

## Shader interface (ABI)

SPIR-V itself is just expressing computation happening on GPU device. SPIR-V
programs themselves are not enough for running workloads on GPU; a companion
host application is needed to manage the resources referenced by SPIR-V programs
and dispatch the workload. For the Vulkan execution environment, the host
application will be written using Vulkan API. Unlike CUDA, the SPIR-V program
and the Vulkan application are typically authored with different front-end
languages, which isolates these two worlds. Yet they still need to match
_interfaces_: the variables declared in a SPIR-V program for referencing
resources need to match with the actual resources managed by the application
regarding their parameters.

Still using Vulkan as an example execution environment, there are two primary
resource types in Vulkan: buffers and images. They are used to back various uses
that may differ regarding the classes of operations (load, store, atomic) to be
performed. These uses are differentiated via descriptor types. (For example,
uniform storage buffer descriptors can only support load operations while
storage buffer descriptors can support load, store, and atomic operations.)
Vulkan uses a binding model for resources. Resources are associated with
descriptors and descriptors are further grouped into sets. Each descriptor thus
has a set number and a binding number. Descriptors in the application
corresponds to variables in the SPIR-V program. Their parameters must match,
including but not limited to set and binding numbers.

Apart from buffers and images, there is other data that is set up by Vulkan and
referenced inside the SPIR-V program, for example, push constants. They also
have parameters that require matching between the two worlds.

The interface requirements are external information to the SPIR-V compilation
path in MLIR. Besides, each Vulkan application may want to handle resources
differently. To avoid duplication and to share common utilities, a SPIR-V shader
interface specification needs to be defined to provide the external requirements
to and guide the SPIR-V compilation path.

### Shader interface attributes

The SPIR-V dialect defines [a few attributes][MlirSpirvAbi] for specifying these
interfaces:

*   `spv.entry_point_abi` is a struct attribute that should be attached to the
    entry function. It contains:
    *   `local_size` for specifying the local work group size for the dispatch.
*   `spv.interface_var_abi` is a struct attribute that should be attached to
    each operand and result of the entry function. It contains:
    *   `descriptor_set` for specifying the descriptor set number for the
        corresponding resource variable.
    *   `binding` for specifying the binding number for the corresponding
        resource variable.
    *   `storage_class` for specifying the storage class for the corresponding
        resource variable.

The SPIR-V dialect provides a [`LowerABIAttributesPass`][MlirSpirvPasses] for
consuming these attributes and create SPIR-V module complying with the
interface.

## Serialization and deserialization

Although the main objective of the SPIR-V dialect is to act as a proper IR for
compiler transformations, being able to serialize to and deserialize from the
binary format is still very valuable for many good reasons. Serialization
enables the artifacts of SPIR-V compilation to be consumed by a execution
environment; deserialization allows us to import SPIR-V binary modules and run
transformations on them. So serialization and deserialization is supported from
the very beginning of the development of the SPIR-V dialect.

The serialization library provides two entry points, `mlir::spirv::serialize()`
and `mlir::spirv::deserialize()`, for converting a MLIR SPIR-V module to binary
format and back. The [Code organization](#code-organization) explains more about
this.

Given that the focus is transformations, which inevitably means changes to the
binary module; so serialization is not designed to be a general tool for
investigating the SPIR-V binary module and does not guarantee roundtrip
equivalence (at least for now). For the latter, please use the
assembler/disassembler in the [SPIRV-Tools][SpirvTools] project.

A few transformations are performed in the process of serialization because of
the representational differences between SPIR-V dialect and binary format:

*   Attributes on `spv.module` are emitted as their corresponding SPIR-V
    instructions.
*   Types are serialized into `OpType*` instructions in the SPIR-V binary module
    section for types, constants, and global variables.
*   `spv.constant`s are unified and placed in the SPIR-V binary module section
    for types, constants, and global variables.
*   Attributes on ops, if not part of the op's binary encoding, are emitted as
    `OpDecorate*` instructions in the SPIR-V binary module section for
    decorations.
*   `spv.selection`s and `spv.loop`s are emitted as basic blocks with `Op*Merge`
    instructions in the header block as required by the binary format.
*   Block arguments are materialized as `OpPhi` instructions at the beginning of
    the corresponding blocks.

Similarly, a few transformations are performed during deserialization:

*   Instructions for execution environment requirements (extensions,
    capabilities, extended instruction sets, etc.) will be placed as attributes
    on `spv.module`.
*   `OpType*` instructions will be converted into proper `mlir::Type`s.
*   `OpConstant*` instructions are materialized as `spv.constant` at each use
    site.
*   `OpVariable` instructions will be converted to `spv.globalVariable` ops if
    in module-level; otherwise they will be converted into `spv.Variable` ops.
*   Every use of a module-level `OpVariable` instruction will materialize a
    `spv._address_of` op to turn the symbol of the corresponding
    `spv.globalVariable` into an SSA value.
*   Every use of a `OpSpecConstant` instruction will materialize a
    `spv._reference_of` op to turn the symbol of the corresponding
    `spv.specConstant` into an SSA value.
*   `OpPhi` instructions are converted to block arguments.
*   Structured control flow are placed inside `spv.selection` and `spv.loop`.

## Conversions

(TODO: expand this section)

## Code organization

We aim to provide multiple libraries with clear dependencies for SPIR-V related
functionalities in MLIR so developers can just choose the needed components
without pulling in the whole world.

### The dialect

The code for the SPIR-V dialect resides in a few places:

*   Public headers are placed in [include/mlir/Dialect/SPIRV][MlirSpirvHeaders].
*   Libraries are placed in [lib/Dialect/SPIRV][MlirSpirvLibs].
*   IR tests are placed in [test/Dialect/SPIRV][MlirSpirvTests].
*   Unit tests are placed in [unittests/Dialect/SPIRV][MlirSpirvUnittests].

The whole SPIR-V dialect is exposed via multiple headers for better
organization:

*   [SPIRVDialect.h][MlirSpirvDialect] defines the SPIR-V dialect.
*   [SPIRVTypes.h][MlirSpirvTypes] defines all SPIR-V specific types.
*   [SPIRVOps.h][MlirSPirvOps] defines all SPIR-V operations.
*   [Serialization.h][MlirSpirvSerialization] defines the entry points for
    serialization and deserialization.

The dialect itself, including all types and ops, is in the `MLIRSPIRV` library.
Serialization functionalities are in the `MLIRSPIRVSerialization` library.

### Op definitions

We use [Op Definition Spec][ODS] to define all SPIR-V ops. They are written in
TableGen syntax and placed in various `*Ops.td` files in the header directory.
Those `*Ops.td` files are organized according to the instruction categories used
in the SPIR-V specification, for example, an op belonging to the "Atomics
Instructions" section is put in the `SPIRVAtomicOps.td` file.

`SPIRVOps.td` serves as the master op definition file that includes all files
for specific categories.

`SPIRVBase.td` defines common classes and utilities used by various op
definitions. It contains the TableGen SPIR-V dialect definition, SPIR-V
versions, known extensions, various SPIR-V enums, TableGen SPIR-V types, and
base op classes, etc.

Many of the contents in `SPIRVBase.td`, e.g., the opcodes and various enums, and
all `*Ops.td` files can be automatically updated via a Python script, which
queries the SPIR-V specification and grammar. This greatly reduces the burden of
supporting new ops and keeping updated with the SPIR-V spec. More details on
this automated development can be found in the
[Automated development flow](#automated-development-flow) section.

### Dialect conversions

The code for conversions from other dialects to the SPIR-V dialect also resides
in a few places:

*   From GPU dialect: headers are at
    [include/mlir/Conversion/GPUTOSPIRV][MlirGpuToSpirvHeaders]; libraries are
    at [lib/Conversion/GPUToSPIRV][MlirGpuToSpirvLibs].
*   From standard dialect: headers are at
    [include/mlir/Conversion/StandardTOSPIRV][MlirStdToSpirvHeaders]; libraries
    are at [lib/Conversion/StandardToSPIRV][MlirStdToSpirvLibs].

These dialect to dialect conversions have their dedicated libraries,
`MLIRGPUToSPIRVTransforms` and `MLIRStandardToSPIRVTransforms`, respectively.

There are also common utilities when targeting SPIR-V from any dialect:

*   [include/mlir/Dialect/SPIRV/Passes.h][MlirSpirvPasses] contains SPIR-V
    specific analyses and transformations.
*   [include/mlir/Dialect/SPIRV/SPIRVLowering.h][MlirSpirvLowering] contains
    type converters and other utility functions.

These common utilities are implemented in the `MLIRSPIRVTransforms` library.

## Contribution

All kinds of contributions are highly appreciated! :) We have GitHub issues for
tracking the [dialect][GitHubDialectTracking] and
[lowering][GitHubLoweringTracking] development. You can find todo tasks there.
The [Code organization](#code-organization) section gives an overview of how
SPIR-V related functionalities are implemented in MLIR. This section gives more
concrete steps on how to contribute.

### Automated development flow

One of the goals of SPIR-V dialect development is to leverage both the SPIR-V
[human-readable specification][SpirvSpec] and
[machine-readable grammar][SpirvGrammar] to auto-generate as much contents as
possible. Specifically, the following tasks can be automated (partially or
fully):

*   Adding support for a new operation.
*   Adding support for a new SPIR-V enum.
*   Serialization and deserialization of a new operation.

We achieve this using the Python script
[`gen_spirv_dialect.py`][GenSpirvUtilsPy]. It fetches the human-readable
specification and machine-readable grammar directly from the Internet and
updates various SPIR-V `*.td` files in place. The script gives us an automated
flow for adding support for new ops or enums.

Afterwards, we have SPIR-V specific `mlir-tblgen` backends for reading the Op
Definition Spec and generate various components, including (de)serialization
logic for ops. Together with standard `mlir-tblgen` backends, we auto-generate
all op classes, enum classes, etc.

In the following subsections, we list the detailed steps to follow for common
tasks.

### Add a new op

To add a new op, invoke the `define_inst.sh` script wrapper in utils/spirv.
`define_inst.sh` requires a few parameters:

```sh
./define_inst.sh <filename> <base-class-name> <opname>
```

For example, to define the op for `OpIAdd`, invoke

```sh
./define_inst.sh SPIRVArithmeticOps.td ArithmeticBinaryOp OpIAdd
```

where `SPIRVArithmeticOps.td` is the filename for hosting the new op and
`ArithmeticBinaryOp` is the direct base class the newly defined op will derive
from.

Similarly, to define the op for `OpAtomicAnd`,

```sh
./define_inst.sh SPIRVAtomicOps.td AtomicUpdateWithValueOp OpAtomicAnd
```

Note that the generated SPIR-V op definition is just a best-effort template; it
is still expected to be updated to have more accurate traits, arguments, and
results.

The generated op will automatically gain the logic for (de)serialization.
However, tests still need to be coupled with the change to make sure no
surprises. Serialization tests live in test/Dialect/SPIRV/Serialization.

### Add a new enum

To add a new enum, invoke the `define_enum.sh` script wrapper in utils/spirv.
`define_enum.sh` expects the following parameters:

```sh
./define_enum.sh <enum-class-name>
```

For example, to add the definition for SPIR-V storage class in to
`SPIRVBase.td`:

```sh
./define_enum.sh StorageClass
```

### Add a new conversion

(TODO: add details for this section)

[Spirv]: https://www.khronos.org/registry/spir-v/
[SpirvSpec]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html
[SpirvLogicalLayout]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#_a_id_logicallayout_a_logical_layout_of_a_module
[SpirvGrammar]: https://raw.githubusercontent.com/KhronosGroup/SPIRV-Headers/master/include/spirv/unified1/spirv.core.grammar.json
[GlslStd450]: https://www.khronos.org/registry/spir-v/specs/1.0/GLSL.std.450.html
[ArrayType]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpTypeArray
[ImageType]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpTypeImage
[PointerType]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpTypePointer
[RuntimeArrayType]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#OpTypeRuntimeArray
[StructType]: https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#Structure
[SpirvTools]: https://github.com/KhronosGroup/SPIRV-Tools
[Rationale]: https://github.com/tensorflow/mlir/blob/master/g3doc/Rationale.md#block-arguments-vs-phi-nodes
[ODS]: https://github.com/tensorflow/mlir/blob/master/g3doc/OpDefinitions.md
[GreedyPatternRewriter]: https://github.com/tensorflow/mlir/blob/master/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp
[MlirSpirvHeaders]: https://github.com/tensorflow/mlir/tree/master/include/mlir/Dialect/SPIRV
[MlirSpirvLibs]: https://github.com/tensorflow/mlir/tree/master/lib/Dialect/SPIRV
[MlirSpirvTests]: https://github.com/tensorflow/mlir/tree/master/test/Dialect/SPIRV
[MlirSpirvUnittests]: https://github.com/tensorflow/mlir/tree/master/unittests/Dialect/SPIRV
[MlirGpuToSpirvHeaders]: https://github.com/tensorflow/mlir/tree/master/include/mlir/Conversion/GPUToSPIRV
[MlirGpuToSpirvLibs]: https://github.com/tensorflow/mlir/tree/master/lib/Conversion/GPUToSPIRV
[MlirStdToSpirvHeaders]: https://github.com/tensorflow/mlir/tree/master/include/mlir/Conversion/StandardToSPIRV
[MlirStdToSpirvLibs]: https://github.com/tensorflow/mlir/tree/master/lib/Conversion/StandardToSPIRV
[MlirSpirvDialect]: https://github.com/tensorflow/mlir/blob/master/include/mlir/Dialect/SPIRV/SPIRVDialect.h
[MlirSpirvTypes]: https://github.com/tensorflow/mlir/blob/master/include/mlir/Dialect/SPIRV/SPIRVTypes.h
[MlirSpirvOps]: https://github.com/tensorflow/mlir/blob/master/include/mlir/Dialect/SPIRV/SPIRVOps.h
[MlirSpirvSerialization]: https://github.com/tensorflow/mlir/blob/master/include/mlir/Dialect/SPIRV/Serialization.h
[MlirSpirvBase]: https://github.com/tensorflow/mlir/blob/master/include/mlir/Dialect/SPIRV/SPIRVBase.td
[MlirSpirvPasses]: https://github.com/tensorflow/mlir/blob/master/include/mlir/Dialect/SPIRV/Passes.h
[MlirSpirvLowering]: https://github.com/tensorflow/mlir/blob/master/include/mlir/Dialect/SPIRV/SPIRVLowering.h
[MlirSpirvAbi]: https://github.com/tensorflow/mlir/blob/master/include/mlir/Dialect/SPIRV/SPIRVLowering.td
[GitHubDialectTracking]: https://github.com/tensorflow/mlir/issues/302
[GitHubLoweringTracking]: https://github.com/tensorflow/mlir/issues/303
[GenSpirvUtilsPy]: https://github.com/tensorflow/mlir/blob/master/utils/spirv/gen_spirv_dialect.py
