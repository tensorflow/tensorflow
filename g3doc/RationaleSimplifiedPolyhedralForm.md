# MLIR: The case for a <em>simplified</em> polyhedral form

MLIR embraces polyhedral compiler techniques for their many advantages
representing and transforming dense numerical kernels, but it uses a form that
differs significantly from other polyhedral frameworks.

**Disclaimer / Warning**

This document is a very early design proposal (which has since been accepted)
that explored the tradeoffs of using this simplified form vs the tranditional
polyhedral schedule list form. At some point, this document could be dusted off
and written as a proper academic paper, but until now, it is better to included
it in this crufty form than not to. Beware that this document uses archaic
syntax and should not be considered a canonical reference to modern MLIR.

## Introduction

This document discusses general goals of the project, introduces context and the
two alternatives, then talks about the tradeoffs of these designs. Written by
Chris Lattner.

## General goals of an IR, and goals of mlfunc's specifically

Our currently planned representation for MLIR consists of two kinds of
functions: an LLVM-like "CFG Function" and an "ML Function": a function
represented in multidimensional loop form. The idea is that a CFG function is
capable of full generality for expressing arbitrary computation, but is awkward
for loop transformations. In contrast, mlfunc's are limited (e.g. to control
flow involving loop nests over affine spaces) but these limitations make it much
easier to transform and analyze, particularly for the set of computations in a
machine learning kernel.

The design of an intermediate representations is an optimization problem, which
makes intentional tradeoffs that aim to make certain kinds of compiler
transformations simple. After all, it is "possible" to do almost any
transformation on any IR: we could theoretically do loop transformations on
assembly language. OTOH, such transformations would take too long to write,
would be fragile due to irrelevant changes, would be difficult to maintain, and
difficult to make target independent. Performing transformations on the "right
level" of IR makes it much easier to do analysis and transformation of code, and
can make them faster by reducing the size of the IR, and eliminating
possibilities that would have otherwise have to be considered.

This is the reason we're interested in adding polyhedral techniques to an IR in
the first place: though our base "CFG function" representation is fully capable
of expressing any computation, it is "too" expressive. The limitations imposed
by polyhedral techniques (e.g. on affine loop bounds and array subscripts)
define a closed algebra that can represent an interesting range of
transformations and their compositions, and because of their simplicity, we can
perform (e.g.) dependence analysis more efficiently and more reliably.

This raises an important question that this document examines: given we are
introducing a redundant and limited way to express code and transformations,
exactly what form is best to perform the analyses and transformations we want?

We explore two different design points that are capable of expressing the same
class of affine loop computations, but which use different representational
forms. These forms trade off verbosity, ease of transformation, and ease of
analysis in interesting ways.

## Context: Traditional Polyhedral Form

We started by discussing a representation that uses the traditional polyhedral
schedule set + domain representation, e.g. consider C-like code like:

```c
  void simple_example(...) {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
         float tmp = X[i,j]    // S1
         A[i,j] = tmp + 1      // S2
         B[i,j] = tmp * 42     // S3
       }
    }
  }
```

The polyhedral representation doesn't care about the actual computation, so we
will abstract them into S1/S2/S3 in the discussion below. Originally, we planned
to represent this with a classical form like (syntax details are not important
and probably slightly incorrect below):

```
  mlfunc @simple_example(... %N) {
    %tmp = call @S1(%X, %i, %j)
      domain: (0 <= %i < %N), (0 <= %j < %N)
      schedule: (i, j, 0)

    call @S2(%tmp, %A, %i, %j)
      domain: (0 <= %i < %N), (0 <= %j < %N)
      schedule: (i, j, 1)

    call @S3(%tmp, %B, %i, %j)
      domain: (0 <= %i < %N), (0 <= %j < %N)
      schedule: (i, j, 2)
  }
```

In this design, an mlfunc is an unordered bag of instructions whose execution
order is fully controlled by their schedule.

However, we recently agreed that a more explicit schedule tree representation is
a better fit for our needs, because it exposes important structure that will
make analyses and optimizations more efficient, and also makes the scoping of
SSA values more explicit. This leads us to a representation along the lines of:

```
  mlfunc @simple_example(... %N) {
    d0/d1 = mlspace
    for S1(d0), S2(d0), S3(d0) {
      for S1(d1), S2(d1), S3(d1) {

        %tmp = call @S1(%X, d0, d1)      ;; S1
          domain: (0 <= d0 < %N), (0 <= d1 < %N)

        call @S2(%tmp, %A, d0, d1)      ;; S2
          domain: (0 <= d0 < %N), (0 <= d1 < %N)

        call @S3(%tmp, %B, d0, d1)      ;; S3
          domain: (0 <= d0 < %N), (0 <= d1 < %N)
      }
    }
  }
```

This change makes the nesting structure of the loops an explicit part of the
representation, and makes lexical ordering within a loop significant
(eliminating the constant 0/1/2 of schedules).

It isn't obvious in the example above, but the representation allows for some
interesting features, including the ability for instructions within a loop nest
to have non-equal domains, like this - the second instruction ignores the outer
10 points inside the loop:

```
  mlfunc @reduced_domain_example(... %N) {
    d0/d1 = mlspace
    for S1(d0), S2(d0) {
      for S1(d1), S2(d1) {
        %tmp = call @S1(%X, d0, d1)    ;; S1
          domain: (0 <= d0 < %N), (0 <= d1 < %N)

        call @S2(%tmp, %A, d0, d1)      ;; S2
          domain: (10 <= d0 < %N-10), (10 <= d1 < %N-10)
      }
    }
  }
```

It also allows schedule remapping within the instruction, like this example that
introduces a diagonal skew through a simple change to the schedules of the two
instructions:

```
  mlfunc @skewed_domain_example(... %N) {
    d0/d1 = mlspace
    for S1(d0), S2(d0+d1) {
      for S1(d0+d1), S2(d1) {
        %tmp = call @S1(%X, d0, d1)    ;; S1
          domain: (0 <= d0 < %N), (0 <= d1 < %N)

        call @S2(%tmp, %A, d0, d1)      ;; S2
          domain: (0 <= d0 < %N), (0 <= d1 < %N)
      }
    }
  }
```

This form has great power, and the polyhedral code generator (which lowers from
an mlfunc to a cfgfunc representation) handles this power so things that
introduce loop transformations don't have to explicitly manipulate the looping
structure.

## Proposal: Simplified Polyhedral Form

This document proposes and explores the idea of going one step further, moving
all of the domain and schedule information into the "schedule tree". In this
form, we would have a representation where all instructions inside of a given
for-loop are known to have the same domain, which is maintained by the loop. In
the simplified form, we also have an "if" instruction that takes an affine
condition.

Our simple example above would be represented as:

```mlir
  mlfunc @simple_example(... %N) {
    affine.for %i = 0 ... %N step 1 {
      affine.for %j = 0 ... %N step 1 {
        // identity noop in this case, but can exist in general.
        %0,%1 = affine.apply #57(%i, %j)

        %tmp = call @S1(%X, %0, %1)

        call @S2(%tmp, %A, %0, %1)

        call @S3(%tmp, %B, %0, %1)
      }
    }
  }
```

The example with the reduced domain would be represented with an if instruction:

```mlir
  mlfunc @reduced_domain_example(... %N) {
    affine.for %i = 0 ... %N step 1 {
      affine.for %j = 0 ... %N step 1 {
        // identity noop in this case, but can exist in general.
        %0,%1 = affinecall #57(%i, %j)

        %tmp = call @S1(%X, %0, %1)

        if (10 <= %i < %N-10), (10 <= %j < %N-10) {

          %2,%3 = affine.apply(%i, %j)    // identity noop in this case

          call @S2(%tmp, %A, %2, %3)
        }
      }
    }
  }
```

These IRs represent exactly the same information, and use a similar information
density. The 'traditional' form introduces an extra level of abstraction
(schedules and domains) that make it easy to transform instructions at the
expense of making it difficult to reason about how those instructions will come
out after code generation. With the simplified form, transformations have to do
parts of code generation inline with their transformation: instead of simply
changing a schedule to **(i+j, j)** to get skewing, you'd have to generate this
code explicitly (potentially implemented by making polyhedral codegen a library
that transformations call into):

```mlir
mlfunc @skewed_domain_example(... %N) {
  affine.for %t1 = 0 ... 2*N-2 step 1 {
    affine.for %t2 = max(0, t1-N+1) ... min(N, t1) step 1 {
      (%i, %j) = (%t1-%t2, %t2)
      ...
    }
  }
}
```

## Evaluation

Both of these forms are capable of expressing the same class of computation:
multidimensional loop nests with affine loop bounds and affine memory
references. That said, they pose very different tradeoffs in other ways.

### Commonality: can express same computation

Both of these can express the same sorts of computation, e.g. kernels written in
one form are representable in the other form in all cases.

### Commonality: dependence analysis

These representations both use affine functions for data layout mapping and
access subscripts, and dependence analysis works the same way.

### Commonality: difficulty of determining optimal transformation series

One major challenge in performance of optimization of this sort of code is
choosing the ordering and behavior of various loop transformations that get
applied. There are non-local effects of every decision, and neither
representation helps solve this inherently hard problem.

### Commonality: compactness of IR

In the cases that are most relevant to us (hyper rectangular spaces) these forms
are directly equivalent: a traditional instruction with a limited domain (e.g.
the "reduced_domain_example" above) ends up having one level of ML 'if' inside
its loops. The simplified form pays for this by eliminating schedules and
domains from the IR. Both forms allow code duplication to reduce dynamic
branches in the IR: the traditional approach allows instruction splitting, the
simplified form supports instruction duplication.

It is important to point out that the traditional form wins on compactness in
the extreme cases: e.g. the loop skewing case. These cases will be rare in
practice for our workloads, and are exactly the cases that downstream
transformations want to be explicit about what they are doing.

### Simplicity of code generation

A key final stage of an mlfunc is its conversion to a cfg function, which is
required as part of lowering to the target machine. The simplified form has a
clear advantage here: the IR has a direct correspondence to the structure of the
generated code.

In contrast, the traditional form has significant complexity in the lowering
process to a CFG function, because the verbosity not imbued in the IR needs to
come out during code generation. Code generation from ISL shows that it is
possible to do this, but it is a non-trivial transformation.

### Ease of transformation

An advantage for the traditional form is that it is easier to perform certain
transformations on it: skewing and tiling are just transformations on the
schedule of the instructions in question, it doesn't require changing the loop
structure.

In practice, the simplified form requires moving the complexity of code
generation into the transformations themselves - this is sometimes trivial,
sometimes involved. The author believes that this should be possible by making
the code generation algorithms themselves be library functions that
transformations call into, instead of an opaque block that happens at the end of
the mlfunc processing.

Also, the sorts of transformations performed today by XLA (including tiling,
padding, unrolling, and other rectangular transformations) should be easy enough
to implement on either representation. The only cases that are a challenge are
more advanced cases like skewing, e.g. for DMA data movement generation.

### Ease of analysis: Cost models

The simplified form is much easier for analyses and transformations to build
cost models for (e.g. answering the question of "how much code bloat will be
caused by unrolling a loop at this level?"), because it is easier to predict
what target code will be generated. With the traditional form, these analyses
will have to anticipate what polyhedral codegen will do to a set of instructions
under consideration: something that is non-trivial in the interesting cases in
question (see "Cost of code generation").

### Cost of code generation

State of the art polyhedral code generation is
[expensive and complicated](https://lirias.kuleuven.be/bitstream/123456789/497238/1/toplas-astgen.pdf),
sometimes exponential time complexity. We expect that most machine learning
workloads will be hyper-rectangular, and thus it should be easy to specialize in
important cases. That said, the traditional polyhedral representation makes it
very easy to introduce complicated and expensive schedules, and provides no way
to understand and project a cost model for using them. All downstream clients of
the IR need to be prepared to handle the full generality of IR that may come to
them.

The simplified form defines this away: the concepts in the IR remain simple, and
the code much more directly reflects the cost model for lowering to CFG
functions and machine code. This is expected to be very important in the late
stages of a code generator for an accelerator.

### SSA in ML Functions

We agree already that values defined in an mlfunc can include scalar values and
they are defined based on traditional dominance. In the simplified form, this is
very simple: arguments and induction variables defined in for-loops are live
inside their lexical body, and linear series of instructions have the same "top
down" dominance relation that a basic block does.

In the traditional form though, this is not the case: it seems that a lot of
knowledge about how codegen will emit the code is necessary to determine if SSA
form is correct or not. For example, this is invalid code:

```
  %tmp = call @S1(%X, %0, %1)
    domain: (10 <= %i < %N), (0 <= %j < %N)
    schedule: (i, j)

  call @S2(%tmp, %A, %0, %1)
    domain: (0 <= %i < %N), (0 <= %j < %N)
    schedule: (i, j)
```

Because `%tmp` isn't defined on some iterations of the %i loop.

This matters because it makes the verifier more complicated, but more
significantly, it means that load promotion and other optimizations that will
produce SSA form will need to be aware of this and be able to model what codegen
does.

An emergent property of this that we discussed recently is that PHI nodes in
mlfunc's (if we support them) will also have to have domains.

### Lack of redundancy in IR

The traditional form has multiple encodings for the same sorts of behavior: you
end up having bits on `affine.for` loops to specify whether codegen should use
"atomic/separate" policies, unroll loops, etc. Instructions can be split or can
generate multiple copies of their instruction because of overlapping domains,
etc.

This is a problem for analyses and cost models, because they each have to reason
about these additional forms in the IR.

### Suitability to purpose: lowering to machine code

One of the main drivers for this work is lowering to low-level accelerator code,
including two-dimensional vectorization, insertion of DMAs, and other
utilization of the matrix accelerator units. In the author's opinion, the extra
compactness of the traditional form is a negative for this purpose: reasoning
about the generated machine code will require understanding the mapping from
mlfunc to lowered code, which means that it must understand what code generation
will do.

In the simplified form, the effect of "code generation" is always obvious from
the IR itself, which should make it easier to perform vectorization to target
instructions and other analyses we need to perform.

## Third Alternative: two different levels of mlfunc

One hybrid alternative is to support both the traditional and simplified forms
of mlfunc in our IR.

The stages could look like this, for example:

1.  Early performance transformations could be done on the traditional form.
1.  Partial code generation lowers to the simplified form
1.  Target specific lowering phases for tiling, and vectorization and other 2D
    transforms that don't benefit much from the traditional form could be run.
1.  Final codegen to a cfg func can be done when all of the instructions are
    replaced with ones valid on the target.

While this is possible, it isn't clear what would justify the complexity of this
approach. Unless there is a super compelling reason for this, it would be nice
to not do this. **Update:** we discussed this as a design team and agreed that
this wouldn't be a good way to go.
