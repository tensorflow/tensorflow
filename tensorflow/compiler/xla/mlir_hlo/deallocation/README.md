# MLIR-HLO deallocation and buffer reuse passes

MLIR-HLO deallocation is an alternative to the upstream buffer-deallocation and
buffer-hoisting passes.

The core concept is that of *ownership*, i.e. for each allocation, we track an
*ownership indicator* that can be moved around. These indicators can be
understood as a `std::unique_ptr` or alternatively a ref-counted pointer with a
maximum count of 1. At the end of a block, an ownership indicator must either
be yielded or the underlying alloc must be freed. In practice, it is not always
known whether a particular alloc is owned by the current block. Therefore, we
must also be able to represent empty ownership indicators (i.e., null pointers).

## Usage

This is the recommended and supported pass pipeline to use these passes:

1.  `hlo-split-alloc-tensors`
1.  `one-shot-bufferize` with `create-deallocs=0`
1.  `hlo-deallocate`
1.  `hlo-deallocation-simplification`
1.  `hlo-buffer-reuse`
1.  `hlo-deallocation-simplification`
1.  `hlo-deallocation-to-scf`
1.  (...)
1.  `convert-deallocation-ops-to-llvm`

It is possible to use just the deallocation pass or just buffer-reuse, but the
former isn't recommended because the output will be inefficient. The latter will
work as long as the invariants assumed by this code are maintained (in
particular, there should be no unranked memrefs in the input IR, since as
described above, the code here assigns special meaning to those).

## "ABI"

As long as the IR contains only a single function, there shouldn't be any sharp
edges here. If there are multiple functions, it is important to pay attention to
the ABI assumed here:

1.  Function arguments are always owned by the caller.
1.  Function results are always owned by the caller **and do not alias with any
    function arguments**. In other words, function results are always freshly
    allocated buffers. Function arguments may alias each other.

Warning: The second condition here is particularly important - if a function
returns one of its arguments, the deallocation pass will silently introduce a
double free.

This restriction could be lifted by introducing ownership indicators for
function arguments, but as of March 2023, this is not done.

## The deallocation pass

The deallocation pass assumes that:

1.  The input IR was fully bufferized (i.e., no tensors are left in the
    program).
1.  No `dealloc`s, `alloca`s or `realloc`s exist yet.
1.  No `memrefs` with distinct element types alias (strict aliasing; in
    particular, no `xla_cpu.memref_element_cast` ops should exist at this point)

The basic deallocation algorithm works mostly locally within blocks. It
transforms the input IR op by op, keeping track of memref alias information as
it goes. For each op, it produces the following information: 1) which allocs
were released by the parent block (i.e., are no longer owned by it; more on that
in the section on transferring ownership), 2) which new allocs are now owned by
the parent block. For example, when processing an `alloc` op, nothing is
released, and the result of the op is now owned by the block. It also keeps
track of aliasing information. Conservatively, it is assumed that all inputs
alias all compatible outputs.

When transforming a block, it is not possible to know in general whether
`memref` arguments are owned by it or by some ancestor. Therefore, we introduce
ownership indicator arguments (`!deallocation.ownership`) for each `memref`
argument. Inside the block, `allocs` and alias sets are tracked as described
above. At the end of the block, we must reconcile these memrefs and potentially
owned allocs. We can do this separately for those that are yielded from the
block and those that aren't.

For `memrefs` (or rather sets of `memrefs` that potentially alias) that aren't
yielded, we must free the corresponding `alloc` if we own it. In general, we
can't know statically whether that's the case, so we use the `retain` op, which
frees non-null allocs [^1] that are no longer needed. To find the place to
insert the op, we simply traverse the block backwards, starting from the
terminator, and look for the last op that contains any reference to a memref
from the alias set.

```
  // Free %alloc_0 and %alloc_1 iff they are non-null.
  deallocation.retain() of(%alloc_0, %alloc_1)
      : (!deallocation.ownership, !deallocation.ownership) -> ()
```

For `memrefs` that are yielded, we also insert retain ops, but this time, we
must retain allocs if we own them. The `retain` ops look like this:

```
  // Check if %yielded_memref aliases with any of %a, %b or %c. If it does,
  // return the corresponding memref. Free the others if they are non-null.
  %maybe_owned = deallocation.retain(%yielded_memref) of(%a, %b, %c)
      : (!deallocation.ownership, !deallocation.ownership, !deallocation.ownership)
      -> (!deallocation.ownership)
```

To understand where such ops come from, consider the following code:

```
  %result = scf.if %cond -> memref<2xi32> {
    scf.yield %some_alloc : memref<2xi32>
  } else {
    %new_alloc = memref.alloc() : memref<2xi32>
    scf.yield %new_alloc : memref<2xi32>
  }
```

Whether the parent block owns the alloc that backs `%result` depends on which
branch was taken. Therefore, after transforming the block, the `if` will look
like this:

```
  %result, %result_ownership = scf.if %cond -> memref<2xi32> {
    %null = deallocation.null
    scf.yield %some_alloc, %null : memref<2xi32>, !deallocation.ownership
  } else {
    %new_alloc = memref.alloc() : memref<2xi32>
    %new_alloc_owned = deallocation.own %new_alloc : memref<2x32>
    scf.yield %new_alloc, %new_alloc_owned : memref<2xi32>, !deallocation.ownership
  }
```

`%result_ownership` is nonnull iff `%result` is owned by the parent block. If
`%result` is yielded, the corresponding retain op would be:

```
  %yielded_result_ownership = deallocation.retain(%result) of(%result_ownership)
```

However, here we can statically determine that this always results in
`%result_ownership`, so the `retain` op will not be emitted.

### Loops and if: `RegionBranchOpInterface`

RegionBranchOpInterface ops mostly follow what was described above for blocks,
but there are two interesting things about them:

1.  Regions with multiple predecessors
1.  Transferring ownership to the op

*Multiple predecessors*. In `scf.while`, and `scf.if`, some regions have
multiple predecessors (in the case of `while`, the `before` region, in the case
of `if`, the parent region). As it turns out, no special logic is required to
handle this - the regions will always yield the same types of memrefs, and
therefore the added ownership indicators will also have the same types.

*Transfer of ownership*. If a `memref` operand of a loop has no further uses
after the loop, we can transfer the ownership indicator for the operand to the
loop. Note that this does not necessarily mean ownership is actually
transferred - the ownership indicator may be null.

#### Implicit capture / implicit transfer of ownership

Consider the following program, which conditionally reallocates a memref:

```
%alloc = memref.alloc(%size) : memref<?xi32>
scf.for %i = %lb to %ub step %step iter_args(%arg0 = %alloc) {
  %should_grow, %new_size = "dummy.check_capacity"(%arg0)
    : (memref<?xi32>) -> (i1, index)
  %mem = scf.if %should_grow {
    %0 = memref.realloc %arg0(%new_size) : memref<?xi32> -> memref<?xi32>
    scf.yield %0 : memref<?xi32>
  } else {
    scf.yield %arg0 : memref<?xi32>
  }
  "dummy.use"(%mem) : (memref<?xi32>) -> ()
  scf.yield %mem : memref<?xi32>
}
```

`%arg0` is owned by the loop, but it must not be deallocated at the end of the
loop body - otherwise, we'd run into a double free when it is reallocated.

We solve this by defining implicit captures, or implicit transfer of ownership.
`memref.realloc` ops are considered to implicitly capture and release their
operand. There are a couple of restrictions to this:

1.  Only ops owned by the parent block can be implicitly captured.
1.  Implicit capture is only allowed in `scf.if` ops. This rule may be applied
    recursively.
1.  The implicit capture must be the last use of the captured value across all
    execution paths.
1.  Implied by the previous rule: Implicit capture is not allowed in `scf.if`
    ops that do not have an else branch.

To illustrate these restrictions, we can look at some IR that violates them:

```
%alloc = memref.alloc()
scf.if %cond {
  %0 = memref.realloc %alloc  // invalid
}
```

This IR contains an implicit capture inside an `scf.if` without an `else`
branch. Since `%alloc` is only freed if `%cond` is true, there must be some
further use of `%alloc`, which is invalid. To make this valid, the following IR
should be emitted instead:

```
%alloc = memref.alloc()
%0 = scf.if %cond {
  %1 = memref.realloc %alloc
  scf.yield %1
} else {
  scf.yield %alloc
}
```

Note that `scf.yield %alloc` is executed no execution path that also executes
the `realloc`, so condition 3 is not violated.

An example that violates condition 1:

```
%alloc = memref.alloc()
scf.for %i = %lb to %ub step %step {
  scf.if ... {
    %0 = memref.realloc %alloc  // invalid
  } else {
    ...
  }
}
```

`%alloc` cannot be implicitly captured here, since there is no chain of ancestor
`scf.if` ops to its definition. To make this valid, turn `%alloc` into an
`iter_arg`:

```
%alloc = memref.alloc()
%0 = scf.for %i = %lb to %ub step %step iter_args(%arg0 = %alloc) {
  %1 = scf.if ... {
    %2 = memref.realloc %alloc
  } else {
    ...
  }
  scf.yield %1
}
```

## Ops in the deallocation dialect

### The `null` op

Creates a null pointer.

### The `own` op

Declares ownership of an alloc and returns an ownership indicator. This is
lowered to an extraction of the alloc's base pointer.

### The `retain` op

Takes a list of memrefs and a list of ownership indicator. For each memref,
returns the ownership (alloc) that it was derived from (if present). Each alloc
is returned at most once. Alloc that are not returned are freed.

Some retain ops can be simplified to a no op (e.g. if there's only one alloc
and one memref, and they're the same). Others can be rewritten to memref.dealloc
(if we know that the alloc is non-null and there is no memref). This is done by
the `deallocation-simplification` pass. In the general case, we lower `retain`
to a sequence of `scf.if` ops. This lowering has a code size of
`O(|allocs| * |memrefs|)`. In practice, we haven't yet observed any large retain
ops where this becomes a problem, but we expect that a better lowering will be
necessary eventually, for example by emitting a library call. For details, see
the section on the deallocation-to-scf pass.

### The `get_buffer` op

Returns the memref's base pointer as an index.

## The buffer reuse pass

The buffer reuse pass is intended to be run after the deallocation pass and
assumes that the code has the structure that the pass guarantees (in particular,
unranked memref == ownership indicator). For best results, the IR should be
canonicalized first.

### Loop simplification

As a preprocessing step, this pass transforms `retain` ops that operate on the
result of loops. Consider the following IR:

```
%alloc1 = memref.alloc() : memref<4xi32>
%alloc2 = memref.alloc() : memref<4xi32>
%0:4 = scf.while(%arg0 = %alloc1, $arg1 = %alloc2) {
  scf.condition(%cond) %arg1, %arg0
do {
  (...)
  scf.yield %arg0, %arg1
}
memref.dealloc %0#0 : memref<4xi32>
memref.dealloc %0#1 : memref<4xi32>
```

`%0#0` and `%0#1` are `%alloc1` and `%alloc2`, in some order. Since there is no
further use of these allocs and they are all deallocated, we can rewrite the
operands to `%alloc1` and `%alloc2`, even though we don't know which one is
which.

The purpose of this preprocessing step is to allow more buffer reuse, which
requires `dealloc`/`alloc` pairs to work.

### Buffer reuse

Buffer reuse coalesces `dealloc`/`alloc` pairs:

```
memref.dealloc %alloc : memref<100xi32>
(...)
%alloc_1 = memref.alloc() : memref<100xi32>
```

Instead of deallocating and allocating, we replace all uses of `%alloc_1` with
`%alloc`. Currently, we only do this for immediate `dealloc`/`alloc` pairs with
no other `alloc`/`dealloc` ops in between. So in the example above, if `(...)`
included any other allocation or deallocation, no reuse would occur.

### Copy elision

Another simple transformation eliminates `alloc`/`copy`/`dealloc` patterns:

```
%a = memref.alloc() : memref<100xi32>
(... 1)  // no uses of %a
memref.copy %b, %a : memref<100xi32> to memref<100xi32>
memref.dealloc %b : memref<100xi32>
(... 2)  // uses of %a
```

Since `%a` is completely overwritten with `%b`, which is deallocated immediately
afterwards, we can remove the allocation of `%a` and replace its uses with `%b`.

```
(... 1)  // potential uses of %b
(... 2)  // all uses of %a replaced with %b
```

Note: This pattern could be generalized to only look at copy ops and the uses of
its operand, leaving the elimination of the allocation and deallocation to other
patterns. As of March 2023, this is not done.

### Hoisting

The second transformation implemented in this pass is buffer hoisting. This
simply looks for allocs that happen in each iteration of a loop and moves them
out of the loop:

```
scf.for %i = %c0 to %c1000 step %c1 {
  %foo = memref.alloc() : memref<100xi32>
  (...)
  memref.dealloc %foo : memref<100xi32>
}
```

Since the contents of a freshly allocated memref are undefined, this can be
transformed as follows:

```
%foo = memref.alloc() : memref<100xi32>
scf.for %i = %c0 to %c1000 step %c1 {
  (...)
}
memref.dealloc %foo : memref<100xi32>
```

The same transformation applies for while loops, with the caveat that it may
increase peak heap usage in that case.

### Double buffering

Double buffering can be considered a variant of hoisting. It is useful in cases
where use ranges of buffers overlap, preventing simple hoisting. Consider the
following IR (ownership indicator omitted for clarity):

```
%0 = scf.for %i = %c0 to %c1000 step %c1 iter_args(%arg = %alloc)
    -> memref<100xi32> {
  %tmp = memref.alloc() : memref<100xi32>
  "some.op"(%tmp, %arg) : (memref<100xi32>, memref<100xi32>) -> ()
  memref.dealloc %arg : memref<100xi32>
  scf.yield %tmp : memref<100xi32>
}
memref.dealloc %0 : memref<100xi32>
```

The live ranges of `%alloc` and `%tmp` overlap, so we can't do straightforward
hoisting here. However, we only need two distinct buffers at any given time, so
instead, we introduce an additional iter arg for the temporary buffer, hoist and
swap in each iteration:

```
%tmp = memref.alloc() : memref<100xi32>
%0, %1 = scf.for %i = %c0 to %c1000 step %c1
    iter_args(%arg = %alloc, %tmp_ = %tmp) -> memref<100xi32> {
  "some.op"(%tmp_, %arg) : (memref<100xi32>, memref<100xi32>) -> ()
  scf.yield %tmp_, %arg : memref<100xi32>, memref<100xi32>
}
memref.dealloc %1 : memref<100xi32>
memref.dealloc %0 : memref<100xi32>
```

Note that the presence of a deallocation of `%arg` inside the loop implies no
further uses of `%alloc` after the loop. So, similarly to the case described in
the section on loop simplification, it doesn't matter which alloc is in `%0` and
which one is in `%1`.

Double buffering works analogously for `while` loops, with the exception that
buffers have to be plumbed through the before region.

Note: as of March 2023, double buffering allocations in `while` loops is only
implemented for the `after` region.

## The split-alloc-tensors pass

This pass is a helper pass to improve the behavior of the other passes when used
together with `one-shot-bufferize`. The purpose of this pass is to prevent
accidental buffer reuse by `one-shot-bufferize` by ensuring each `alloc_tensor`
is used only once, thereby minimizing the sizes of live ranges and enabling the
buffer reuse pass to work optimally.

## The deallocation-to-scf pass

As described previously, most `deallocation.retain` ops are eliminated either by
canonicalization or by `buffer-reuse`. `deallocation-to-scf` lowers the ones
that remain to sequences of `scf.if` ops.

Note: the size of the emitted code is in `O(|allocs| * |memrefs|)`. In pratice,
there may be pathological cases where the code gets too large. While we haven't
observed this yet, an alternative lowering may therefore be desirable.

[^1]: `memref.dealloc` happens to tolerate null inputs as well, but at this
    point of the pipeline, we assume that the argument is always non-null,
    because 1) this behavior isn't documented 2) it simplifies analysis in
    subsequent passes.
