# MLIR: Incremental Application to Graph Algorithms in ML Frameworks

The existing documentation about MLIR focuses on long term vision, how its
pieces fit together, and the benefits of modular and composable infrastructure
in the vast and distant future. While this viewpoint appeals to some, it causes
concern for others who are more concerned about the "here and now" - why does it
make sense to make a "revolutionary" change when any individual problem can be
fixed in place?

This document explains that adoption of MLIR to solve graph based problems
_isn't_ a revolutionary change: it is an incremental series of steps which build
on each other, each of which delivers local value. This document also addresses
some points of confusion that keep coming up.

One note: even though a major advantage of MLIR is that it can span the full
spectrum from graph algorithms down to low-level code generation, this document
focuses on the use of MLIR for **graph-level algorithms**. MLIR will also unlock
exciting code generation opportunities (particularly given its novel approach to
integrating state of the art polyhedral techniques), but issues that touch on
MLIR's relationship to XLA, Eigen, etc, are out of scope for this particular
doc.

This document uses TensorFlow as the example given that it is the focus of our
immediate work, but we believe that the same viewpoint could be useful for
people working in the context of other ML frameworks that may consider adopting
MLIR in the future.

### How is MLIR relevant?

MLIR is an overloaded acronym which unpacks as "Multi-Level Intermediate
Representation". Its high-level purpose is to provide mechanics for describing
and transforming programs and computations in a flexible way. It provides common
compiler infrastructure for things like constant folding, dead code elimination,
graph rewriting, and others - which are independent of the representational
choices picked by a given dialect (e.g. its concurrency semantics). It was built
with a specific focus on compile time and memory efficiency, accurate
propagation of source location information (important for reporting high quality
errors and warnings) and is designed for testability.

TensorFlow has numerous subsystems (some of which are proprietary, e.g.
Tensor-RT, nGraph, CoreML, etc) as well as translation layers between these
different subsystems, and these translation layers face similar challenges. ((As
an aside, the internals of each of these subsystems could often benefit from
MLIR infrastructure, but that isn't a focus of this doc.))

A key observation that MLIR makes is that these subsystems often have two things
going on: they are both particular data structures and encodings (e.g. HLO
graphs, TF-Lite's flat buffer format, TensorFlow's Graph format, the ONNX
abstraction, etc) as well as an abstraction of computation (a specific way of
modeling a convolution, a set of supported operations etc).

MLIR uses a standard IR (i.e., a set of data structures) for representing these
computations - this allows a huge amount of shared infrastructure across these
problem domains. MLIR then allows the definition of domain-specific "dialects"
that describe the set of operations that are legal and supported for a given
application. This means that the actual translations between data structures are
kept as simple as possible - and are thus relatively easy to make "correct".
This allows the common compiler infrastructure to handle the mapping problems
and the other issues within the domain.

MLIR's design is directly informed by the experience of building (and then
living with) intermediate representations like the LLVM IR, LLVM SelectionDAG,
the LLVM machine instruction representation, Swift SIL IR, and learns new
lessons from TensorFlow and XLA HLO, as well as learning from building countless
research and production systems on top of them. Our goal is to drag the state of
the art in compilers forward, not to merely apply a few well-known techniques to
the machine learning domain.

### What does adoption mean?

The point of this document is not to advocate for rewriting any particular
subsystem in TensorFlow - indeed, the burden required to justify a rewrite is
high, and often very specific to that subsystem. That said, there are several
subsystems that are about to get rewritten or substantially revised anyway, so
we use those as examples to concretely describe the benefits that MLIR provides
in these cases and what it will take. The subsystems discussed are:

1.  the TF Lite TOCO translator, which we need to improve error
    reporting/reliability issues and generalize it to support more ops, and
1.  the TF/XLA bridge which needs to improve usability by merging some of its
    usage models, support dynamic shapes and generalize guest subsystem support
    to Tensor-RT and nGraph.
1.  Grappler is another subsystem that is likely to get substantial revisions in
    the future, and would definitely benefit from the MLIR framework, but there
    are no known plans to do that work at this point, so we don't discuss it
    further.

Adopting MLIR for these works the same way - and, in fact, the work to support
TF Lite is mostly a subset of the larger work to support the functionality of
the TF/XLA bridge. TF Lite and the TF/XLA bridge include several compiler passes
(things like encapsulate, functionalize control flow, lowering of ops, fusion,
constant folding, shape inference, etc).

MLIR supports converting from TensorFlow Graphs to MLIR and back, which means
that we can start by putting in a no-op translation to MLIR and back into the
pipeline, and verify that nothing breaks. Then we can work on replacing the
compiler transformations one by one by reimplementing them (with the improved
algorithms that we're planning).

This is a development plan, we wouldn't actually ship a TensorFlow that just
uses MLIR for a single pass. In practice, we'll have the MLIR flag gated under
an option, build out a replacement for an entire subsystem (e.g. the TOCO
translator) and when the time is right, we'll do A/B comparisons and eventually
make a switch and phase out the old code over time.

## What benefit does MLIR provide?

The adoption plan above might sound like it only makes things worse in the
immediate term - we have two implementations of the same functionality, we are
dividing our efforts, etc. In order for this to be worth it, we should have a
good sense that we are building towards an improved future that will make
customers and TensorFlow engineers happier when it lands. Here we describe a few
of the benefits that MLIR provides, in no particular order:

### A Lossless Human Editable Textual Representation

The MLIR in-memory data structure has a human readable and writable format, as
well as [a specification](LangRef.md) for that format - built just like any
other programming language. Important properties of this format is that it is
compact, easy to read, and lossless. You can dump an MLIR program out to disk
and munge around with it, then send it through a few more passes.

If you haven't worked with a system that works this way, it is hard to overstate
how big of a deal this in practice: it means that you can call `foo->dump()` on
an IR object to see its full contents, it means you can diff the IR before and
after a change, delta reduce IR files, and many other things.

### A Graph Verification Pass

Like many other popular compiler infrastructures, MLIR provides infrastructure
and implementation for a "verifier" which checks that the IR is well formed. The
MLIR verifier is a simple framework that makes it easy to provide a single
source of truth for those correctness properties and is general across all
Dialects (e.g. TF Graph, TF Lite flat buffer, XLA HLO, etc).

A verifier pass is sort of like a 'super assertion' that catches mistakes in
program transformations early, making you as an engineer more productive, making
the product more reliable, and making it easier to track down bugs when they
appear - because the verifier can be run at any time, either as a compiler pass
or with a single function call.

While MLIR provides a well considered infrastructure for IR verification, and
has simple checks for existing TensorFlow operations, there is a lot that should
be added here and lots of opportunity to get involved!

### Designed for Testability

There are many aspects of this in MLIR, but we'll focus on compiler
transformations since they are the easiest to understand. Compiler
transformations are modeled as subclasses of the `Pass` C++ class, which are
driven by an `mlir-opt` tool. When combined with a lossless textual
representation, it becomes really easy to write unit tests for compiler
transformations, for example, this is a simple test that shows "x-x" is being
turned into zero:

```mlir
  // RUN: mlir-opt %s -canonicalize | FileCheck %s
  cfgfunc @test_subi_zero_cfg(i32) -> i32 {
  ^bb0(%arg0: i32):
    %y = subi %arg0, %arg0 : i32
    return %y: i32
  }
  // CHECK-LABEL: cfgfunc @test_subi_zero_cfg
  // CHECK-NEXT: bb0(%arg0: i32):
  // CHECK-NEXT: %c0_i32 = constant 0 : i32
  // CHECK-NEXT: return %c0
```

The "CHECK" comments are interpreted by the
[LLVM FileCheck tool](https://llvm.org/docs/CommandGuide/FileCheck.html), which
is sort of like a really advanced grep. This test is fully self contained: it
feeds the input into the [canonicalize pass](Canonicalization.md), and checks
that the output matches the CHECK lines. See the `test/Transforms` directory for
more examples. In contrast, standard unit testing exposes the API of the
underlying framework to lots and lots of tests (making it harder to refactor and
move the API), typically requires a lot more code, and exacerbates issues with
link time. For examples, see
[the TEST_F functions in TensorFlow's testsuite](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/grappler/optimizers/arithmetic_optimizer_test.cc).

MLIR has been pervasively designed with this sort of design by testability,
allowing us to put in place a culture that expects every behavior changing
commit to include a test case, and for these test cases to be stable and
reliable over time, since they are testing exactly what they are supposed to.
End to end integration tests are still super useful for some things of course!

### Infrastructure for Warnings and Error Diagnostics and Location Tracking

MLIR benefits from the lessons learned from building other compilers - including
Clang which
[[set the standard](http://blog.llvm.org/2010/04/amazing-feats-of-clang-error-recovery.html)](http://blog.llvm.org/2010/04/amazing-feats-of-clang-error-recovery.html)
for quality of implementation in C/C++ compiler diagnostics. Drawing from this
experience (and fixing mistakes in LLVM), MLIR requires that operations and
functions carry abstract location information, that transformations propagate
this information, and provides standardized mechanisms to emit errors and
warnings, as well as for clients to hook into them to capture and report them in
custom ways.

Why is this important? In practice, many graph-to-graph translators can fail
(e.g. TF Lite when an unsupported op is used) and it is important to be able to
report the error up through to the user in the most precise way possible, in
order for it to be actionable. This includes tracking rewrites through fusions
and fissions of ops, mapping back into language / API specific domains, etc.

More selfishly for infrastructure hackers, this is a huge boon because it means
that it is easy to write good tests for this: the testing tools for MLIR capture
the diagnostics produced by passes (using the standard diagnostic hooks) and
check that they match the expected diagnostics in the testcase. For example, to
test the dependence analysis infra in the code generator, Andy Davis wrote a
simple pass that checks dependencies and emits them as "notes", allowing him to
write tests like this:

```mlir {.mlir}
  // RUN: mlir-opt %s -memref-dependence-check -verify
  mlfunc @different_memrefs() {
    %m.a = alloc() : memref<100xf32>
    %m.b = alloc() : memref<100xf32>
    %c0 = constant 0 : index
    %c1 = constant 1.0 : f32
    store %c1, %m.a[%c0] : memref<100xf32>
    // expected-note@-1 {{dependence from memref access 0 to access 1 = false}}
    %v0 = load %m.b[%c0] : memref<100xf32>
    return
  }
```

Note that a major limitation of this is that MLIR suffers from a problem of
"garbage in, garbage out": if the input locations to MLIR are imprecise, then
there is nothing that it can do to recover them. There is work underway in
TensorFlow/Python to improve the situation, and Swift for TensorFlow already has
perfect location tracking due to its design.

### Shape Information Captured in the IR

In TensorFlow Graphs, each op takes and returns values using a very simple type
system (TF_DataType) in which each value is a tensor of unknown rank and
dimensions. At the same time, many graphs have static shapes easily knowable for
wide swaths of the computation, and even dynamically shaped operations often
have statically knowable dimensions. Many analyses and transformations benefit
and use this information when available, but because TensorFlow graphs don't
capture this (e.g. serialize it to proto), passes have to recompute it on demand
with ShapeRefiner.

The [MLIR Tensor Type](g3doc/LangRef.md#tensor-type) directly captures shape
information, so you can have things like:

```mlir {.mlir}
  %x = tf.Add %x, %y : tensor<128 x 8 x ? x f32>
```

Capturing this in the IR is expected to speed up transformations (avoiding
recomputing the same info over and over again) which therefore makes it
practical to apply stronger shape analysis algorithms. It also makes it easier
to work with the IR, because on-the-side representations can get out of date,
and the API is easier to work with from an ergonomics perspective.

### Unified Graph Rewriting Infrastructure

This is still a work in progress, but we have sightlines towards a
[general rewriting infrastructure](GenericDAGRewriter.md) for transforming DAG
tiles into other DAG tiles, using a declarative pattern format. DAG to DAG
rewriting is a generalized solution for many common compiler optimizations,
lowerings, and other rewrites and having an IR enables us to invest in building
a single high quality implementation.

Declarative pattern rules are preferable to imperative C++ code for a number of
reasons: they are more compact, easier to reason about, can have checkers
written against them, and new tools can be built that inspect and manipulate the
declarative patterns in interesting ways - e.g. applying theorem provers to
them. It will be exciting to see this ecosystem develop as the infrastructure
matures.

### Clarified Semantics for TensorFlow Operations

One of the challenging things about working with TensorFlow is that there are
many invariants and behaviors that need to be preserved and known about when
working with Graphs, and these can be difficult to reason about and lead to
bugs. Things like 'dead values', Switch and Merge nodes, concurrency semantics,
nodes that execute even when passed a dead value, multiple device program
representation - etc... all add complexities that can make it challenging to
reason about whether a transformation or analysis is correct in general. Even
something as simple as constant folding or transforming integer `x-x` into `0`
is non-trivial because you need to consider control dependence edges.

One of our major goals for the TensorFlow dialect of MLIR is to sort out these
situations and upgrade existing TensorFlow graphs to semantics that are easier
to reason about. The solutions to these problems are all still being debated,
but those discussions have already yielded a lot of potential answers:
introducing a `tf_dead_or<x>` types for switch/merge, modeling of TF operations
using futures/async semantics etc. None of these particular battles are critical
or important for MLIR to succeed (because of its "meta" nature, the abstraction
decisions of any given dialect are up for it to decide), but each one that works
out will make it easier to work with and transform TensorFlow operations. We
expect these issues to get nailed down in the next couple of months when MLIR
effort moves beyond TF Lite / TOCO support. The discussions that are happening
now are super valuable and making progress.

### Ergonomics

A minor-in-theory, but important-in-practice point is that MLIR is designed to
make it easy, memory efficient, and less error prone to transform code than
other systems. `TensorFlow::Graph` has implementation issues where the same
information is stored redundantly in different places (which must be manually
kept up to date), has somewhat unusual representation of certain constructs
(e.g. the function library, which makes it very difficult to add or remove
functions, e.g. during interprocedural transformations), and stores information
in the graph that is used by the executor, but isn't necessary for program
transformation.

TensorFlow has made a lot of progress in this area over the years, and there are
lots of ideas about further improvements in the future, we are happy that MLIR
addresses these needs (making it much easier to implement correct program
transformations) today, and are committed to pushing hard to make it better.

### Compile Time Performance and Memory Use

MLIR has been designed to be memory and compile-time efficient in its algorithms
and data structures, using immutable and uniqued structures, low level
bit-packing, and other well known techniques to avoid unnecessary heap
allocations, and allow simple and safe multithreaded optimization of MLIR
programs. There are other reasons to believe that the MLIR implementations of
common transformations will be more efficient than the Python and C++
TensorFlow::Graph implementations of the same things, given the current
implementation details of TensorFlow.

That said, this is very much a theory at this point. When the new implementation
of various subsystems are available, we will see what happens in practice: there
will be no reason to speculate - we can measure.

## Common Questions and Concerns

Here we address some frequently asked questions and concerns.

### Isn't MLIR a big dependency to take on?

We've heard that at least some people are concerned that MLIR is a "big"
dependency to take on, and could result in large code size. Here are some key
points MLIR:

1.  The entire MLIR codebase is a pretty small C++ code base in absolute terms
    compared to what goes into a modern ML framework.
1.  Like LLVM, MLIR is designed as a set of libraries that clients can link in
    or ignore as they wish. For example, the transformations in MLIR kept
    separate from the core IR abstractions, and dialect specific code (e.g.
    TensorFlow, TF-Lite, XLA, etc) is all independently selectable by the build
    system. Clients that don't care about XLA don't link in that code, whether
    they are a TF-Lite system or a client that is completely unrelated to
    TensorFlow.
1.  MLIR's only third party dependency is on LLVM, but it doesn't depend on LLVM
    IR or any other heavy dependency - it just depends on LLVM's support library
    which provides efficient hash tables and other
    [memory efficient data structures that the STL does not](http://llvm.org/docs/ProgrammersManual.html#picking-the-right-data-structure-for-a-task).
    There have been discussions about splitting this set of libraries out to its
    own subproject in LLVM that the LLVM IR project depends on. This would be
    great for MLIR as well as other LLVM subprojects.
1.  TensorFlow and many other frameworks already use LLVM - if so, MLIR would
    not be pulling in an additional dependency at all.

### How does MLIR represent {control flow, concurrency, …} semantics in TensorFlow?

MLIR provides a dialect that is an isomorphic 1-1 mapping between TensorFlow
graphs and MLIR, as well as a pretty complete translator back and forth (the
only known gap is that a few TF_DataType enums aren't handled yet). MLIR is a
"Multi-Level IR", which allows it to represent code with different abstraction
levels, so the ability to faithfully represent TensorFlow code in a completely
backwards compatible way (even if there are some historical warts!) is critical.

In *addition* to the isomorphic mapping, we are actively working on efforts to
raise the abstraction level for working with TensorFlow graphs in MLIR. Doing so
would make it even easier to write TensorFlow transformations than it is today,
and would provide a path to migrating TF 1.x graphs forward into the TF 2.x
world. For example, because MLIR has an extensible type system, we can directly
model whether it is impossible for a Tensor value to be a "dead" value - similar
to the use of optional types in modern programming languages.

These discussions occasionally cause confusion because there are several issues
being mixed up into one:

*   What are the current semantics of TensorFlow graphs, and what invariants can
    we rely on?
*   What should the semantics be in TensorFlow 2.0?
*   What do programs rely on in practice, and if it is unfriendly, can we
    migrate it?
*   Can we find a way to make it so transforms don't have to worry about the
    complexities of Switch/Merge, by using higher level control flow
    representations? (tentative answer: yes)
*   How should MLIR represent async vs sync operations, what invariants are
    provided, how does this dovetail with control flow?
*   When is it safe and beneficial to perform optimizations that might reduce
    parallelism?

All of these questions have a "conservative/safe fallback": we can continue
providing exactly the same abstractions that TensorFlow always has. That said,
we are trying hard to level-up the representation (taking advantage of the
"Multi-Level" part of MLIR) because doing so will make it much much easier to
write analyses and transformations than it currently is in TensorFlow.

### Non Goals

It is important to point out things that MLIR does not aim to do. For example,
there is no runtime component to MLIR: the TensorFlow executor, the TF Lite
FlatBuffer interpreter, or other existing runtime should be used as-is.

Another non-goal is that MLIR currently doesn't support a stable binary
encoding. We will certainly add this at some point, but existing formats should
be used for serialization and distribution in the meantime.
