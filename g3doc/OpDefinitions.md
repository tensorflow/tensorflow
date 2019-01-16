# Operation definitions

## Motivation

MLIR allows pluggable dialects, and dialects contain, among others, a list of
operations. This open and extensible ecosystem leads to "stringly" IRs problem,
e.g., repetitive string comparisons during optimization and analysis passes,
unintuitive accessor methods (e.g., generic/error prone `GetOperand(3)` vs
`GetStride()`) with more generic return types, op verification is

1.  best case: a central string-to-verification-function map,
1.  middle case: duplication of verification across the code base, or
1.  worst case: no verification functions verbose constructors without ability
    to specify defaults.

The fix is to support op descriptions, which (in one central place per dialect)
contain everything you need to know about the op, its invariants, properties,
etc. This description is also used to generate helper functions and classes to
allow analysis/builder/verification/parsing/printing.

## Requirements

The op description should as declarative as possible to allow a wide range of
tools to work with them and query methods generated from them. In particular
this means specifying properties, constraints and shape inference information in
a way that is easily analyzable (e.g., avoid opaque calls to C++ functions where
possible).

We considered the approaches of several contemporary systems and focused on
requirements that were desirable:

*   Ops registered using a registry separate from C++ code.
    *   All ops need not be registered; unknown ops are allowed in MLIR. The
        ability of the compiler to optimize those ops or graphs containing those
        ops is constrained.
    *   The current proposal does not include a runtime op description, but it
        does not preclude such description, which could be added.
    *   The op registry is essential for generating C++ classes that make
        manipulating ops, verifying correct construction etc. in C++ easier by
        providing a typed representation and accessors.
*   The op registry will be defined in
    [TableGen](https://llvm.org/docs/TableGen/index.html) and be used to
    generate C++ classes and utility functions
    (builder/verifier/parser/printer).
    *   TableGen is a modelling specification language used by LLVM's backends
        and fits in well with trait based modelling. This is an implementation
        decision and there are alternative ways of doing this. But the
        specification language is good for the requirements of modelling the
        traits (as seen from usage in LLVM processor backend modelling) and easy
        to extend, so a practical choice.
*   MLIR allows both defined and undefined ops.
    *   Defined ops should have fixed semantics and could have a corresponding
        reference implementation defined using, for example, EDSC.
    *   Dialects are under full control of the dialect owner and normally live
        with the framework of the dialect.
*   The op's properties (e.g., commutative) are modelled along with the op in
    the registry.
    *   The set of properties will be fixed and geared towards
        optimization/analysis passes (e.g., `HasSideEffects`).
*   The op's operand/return type constraints are modelled along with the op in
    the registry (see [Type constraints](#type-constraints) discussion below).
*   Behavior of the op is documented along with the op by, following TensorFlow,
    a summary and a description. The description is written in markdown and
    extracted for inclusion in the generated LangRef section of the dialect.
*   The verbose form of printing and parsing is available as normal, but a
    custom parser and printer can either be specified or automatically generated
    from an optional string representation showing the mapping of the "assembly"
    string to operands/type.
    *   Parser-level remappings (e.g., `eq` to enum) will be supported as part
        of the parser generation.
*   Matching patterns are specified separately from the op description.
    *   Contrasted with LLVM there is no "base" set of ops that every backend
        needs to be aware of. Instead there are many different dialects and the
        transformations/legalizations between these dialects form a graph of
        transformations.
*   Reference implementation may be provided along with the op definition.

    *   The reference implementation may be in terms of either standard ops or
        other reference implementations.

        TODO: document expectation if the dependent op's definition changes.

## Operation definition

As an example of the proposal to declare an operation (say `tf.Add)`. In general
one would create a helper classes in the modelling of the operations that would
abstract out common functionality (e.g., `TF_BinaryOp`).

```tablegen
def TF_AddOp : Op<"tf.Add", [NoSideEffect]>,
               Arguments<(ins TF_Tensor:$lhs, TF_Tensor:$rhs)>,
               Results<[TF_Tensor]>,
               Traits<["BroadcastableOperandsAndResultType"]> {
  let summary = "Addition operator";

  let description = [{
    Returns lhs + rhs element-wise.

    The inputs and result must be of the same elemental type.
  }];

  let reference = [{
    auto ivs = makeBindables(lhsShape.size());
    block = edsc::Block({
      ForNest(ivs, 0, lhsShape, 1, {
        result[ivs] = lhs[ivs] + rhs[ivs]
      })});
    }
  }];
}
```

Operation definitions consists of:

1.  Operation name (`opName`).

    This is a unique identifier used to distinguish this operation vs all others
    defined in MLIR. This is the equivalent of the mnemonic in assembly
    language. Operations are within dialects which effectively namespace
    operations. The C++ class generated for the operation is based on the
    definition in the TableGen's file's name. E.g., `TF_AddOp` above would
    result in a C++ class called `AddOp` generated in the namespace `TF`.

1.  Summmary and description.

    These are human readable documentation for the operation. Documentation of
    the operations can be generated from the same source of truth as the
    operation.

1.  Arguments (`arguments`).

    This is a list of operands (optionally named) and named attributes used to
    generate builder, accessor functions and verification.

    1.  Operands.

        These are the results of other operations and mostly only known at
        runtime. They can have a fixed type or correspond to set of possible
        types. See [Type constraints](type-constraints) specification below.

    1.  Attributes.

        These are compile time constant values of the operation.

    1.  Natural attributes.

        These attributes affect the behavior of the operations (e.g., padding
        for convolution);

    1.  Derived attributes.

        These attributes are not needed to define the operation but are instead
        derived from attributes of the operation. E.g., the output shape of
        type. This is mostly used for convenience interface generation or
        interaction with other frameworks/translation.

1.  Return types.

    The type of the value(s) returned by the operation.

1.  Properties.

    Properties of the operations. There are different classes of operation
    properties that are hard-coded into `mlir::OperationInst`. Currently
    supported is `Commutative` and `NoSideEffect`.

1.  Traits.

    Operations can optionally have additional traits. This is an open set at
    present but include traits such as `SameOperandsAndResultType`.

    TODO: Should there be a differentiation between traits and properties?
    Preference to merging these.

1.  Reference description.

    The description of the operation is encoded as C++ builder using EDSC. This
    is still under active discussion and will be fleshed out post prototyping.

1.  Custom builder method (`builder`).

    This is used to generate additional convenience builder methods. For example
    when defining a C++ builder method that has default values. There are two
    builder automatically generated based on the arguments and returns types
    (see op_base.td).

1.  Custom printer method.

    The custom printer to invoke when producing the short form output.

1.  Custom verifier code.

    Additional verification to perform in addition to those generated due to
    operands, attributes, properties and traits.

1.  hasCanonicalizer and hasConstantFolder.

    These boolean fields indicate whether canonicalization patterns or
    constant folding have been defined for this operation.

### For custom parsing and printing

In the operation definition the user can specify custom functions to print or
parse the operation.

FIXME: Autogenerating printing/parsing has not been prototyped, and potentially
just being able to specify custom printer/parser methods are sufficient.

The short form/custom emitter form of the operation is specified using a string
with matching operation name, operands and attributes. With the ability to
express additional information that needs to be parsed to build the operation:

```tablegen
tfl.Add $lhs, $rhs {fused_activation_function:
                   $fused_activation_function }: ${type(self)}
```

1.  The output is never shown in the "mnemonics" string as that is fixed form
    and cannot be altered.

1.  Custom parsing of ops may include some punctuation (e.g., parenthesis).

1.  The operands/results are added to the created operation in the order that
    they are shown in the input and output dags.

1.  The `${type(self)}` operator is used to represent the type of the operator.
    The type of operands can also be queried.

1.  Attributes names are matched to the placeholders in the mnemonic strings.
    E.g., attribute axis is matched with `$axis`. Custom parsing for attribute
    type can be defined along with the attribute definition.

1.  The information in the short form should be sufficient to invoke the builder
    generated. That may require being able to propagate information (e.g., the
    `$lhs` has the same type as the result).

Printing is effectively the inverse of the parsing function generated with the
mnemonic string serving as a template.

## Type constraints

Constraints are along (at least) three axis: 1) elemental type, 2) rank
(including static or dynamic), 3) dimensions. While some ops have no compile
time fixed shape (e.g., output shape is dictated by data) we could still have
some knowledge of constraints/bounds in the system for that op (e.g., the output
of a `tf.where` is at most the size of the input data). And so there are
additional valuable constraints that could be captured even without full
knowledge.

Initially the shape inference will be declaratively specified using:

*   Constraint on the operands of an operation directly. For example
    constraining the input type to be tensor/vector elements or that the
    elemental type be of a specific type (e.g., output of sign is of elemental
    type `i1`) or class (e.g., float like).
*   Constraints on an operands of an operation. For example, enabling specifying
    equality constraints on type/constituents of a type (shape and elemental
    type) between operands and results (e.g., the output type of an add is the
    same as those of the input operands).

In general there is an input/output transfer function which maps the inputs to
the outputs (e.g., given input X and Y [or slices thereof] with these sizes, the
output is Z [or this slice thereof]). Such a function could be used to determine
the output type (shape) for given input type (shape).

But shape functions are determined by attributes and could be arbitrarily
complicated with a wide-range of specification possibilities. Equality
relationship are common (e.g., the elemental type of the output matches the
primitive type of the inputs, both inputs have exactly the same type [primitive
type and shape]) and so these should be easy to specify. Algebraic relationships
would also be common (e.g., a concat of `[n,m]` and `[n,m]` matrix along axis 0
is `[n+n, m]` matrix), while some ops only have defined shapes under certain
cases (e.g., matrix multiplication of `[a,b]` and `[c,d]` is only defined if
`b == c`). As ops are also verified, the shape inference need only specify rules
for the allowed cases (e.g., shape inference for matmul can ignore the case
where `b != c`), which would simplify type constraint specification.

Instead of specifying an additional mechanism to specify a shape transfer
function, the reference implementation of the operation will be used to derive
the shape function. The reference implementation is general and can support the
arbitrary computations needed to specify output shapes.

## Attribute definition

An attribute is a compile time known constant of an operation. Attributes are
required to be known to construct an operation (e.g., the padding behavior is
required to fully define the `conv2d` op). Attributes are defined as having a
storage type (corresponding to a derived class of `mlir::Attribute`), a return
type (that corresponds to the C++ type to use in the generation of the helper
accessors) as well as method to convert between the internal storage and the
helper method. Derived attributes are a special class of attributes that do not
have storage but are instead calculated based on the operation and its
attributes.

As with types, attributes can have a set of condition that need to be satisfied
(e.g., attribute has to be floating point, has to be nonnegative, has to be in a
range). This is true both in the specification of operations as well as matching
rules (see [DAG rewrites](op-dag-pattern-rewrites)).

# Rewrite pattern description

MLIR aims to support many graph transformations across multiple levels of
representation using declarative patterns. These patterns can be expressed using
TableGen as well as dynamically (TBD).

## Op DAG pattern rewrites

The most direct pattern supported in MLIR is rewrites of the form `(dag of
operations) -> (dag of operations)` along with constraints (on operands and
operations). The matchers require both dialects being matched between to be
included in the same TableGen file. Hence pattern matching is normally defined
in either a separate file that imports both. Matchers are defined in terms of
the TableGen instances rather than mnemonics to allow for better error checking
and verification generation.

```tablegen
def : Pat<(TF_LeakyReluOp $arg, F32Attr:$a), (TFL_LeakyReluOp $arg, $a)>;
def : Pat<(TF_ReluOp (TF_AddOp $lhs, $rhs)), (TFL_AddOp $lhs, $rhs, TFL_AF_Relu)>;
def : Pat<(TF_BiasAddOp F32Tensor:$l, F32Tensor:$r),
          (TFL_AddOp $l, $r, TFL_AF_None)>;
```

In the above examples it was shown how to construct matching rules between two
dialects (TensorFlow and TensorFlowLite), showing matching arguments (attributes
and operands) as well as matching a DAG pattern of multiple input operations to
single output.

1.  Matchers can be partially specified on the input (e.g., not all arguments
    contrained) and so multiple matchers can match the same set of nodes. The
    most discriminative matcher (as determined by the number of
    constrained/matching terms) will be selected, if two patterns are equally
    discriminative then an error will be reported.

1.  Matchers between dialects have to be completely specified on the output
    (i.e., there can be no unspecified attributes of the op generated).

1.  Operands and attributes can be further constrained from the op definition
    (e.g., bias add rule only matches the case where both Tensors have F32
    elements).

    1.  Attributes can be transformed by transform rules to produce an attribute
        of a type different than the type matched.

TODO: Add constraints on the matching rules.

TODO: Describe the generation of benefit metric given pattern.
