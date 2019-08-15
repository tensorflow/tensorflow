# Introduction to MLIR Interfaces

MLIR is generic and very extensible, it allows for opaquely representing many
different dialects that have their own operations, attributes, types, etc. This
allows for dialects to be very expressive in their semantics, and allows for
MLIR to capture many different levels of abstraction. The downside to this is
that transformations and analyses must be extremely conservative about the
operations that they encounter, and must special case the different dialects
that they support. To combat this, MLIR provides the concept of `interfaces`.

## Motivation

Interfaces provide a generic way of interacting with the IR. The goal is to be
able to express transformations/analyses in terms of these interfaces without
encoding specific knowledge about the exact operation or dialect involved. This
will make the compiler more extensible by being able to add new dialects and
operations in a decoupled way with respect to the implementation of
transformations/analyses.

### Dialect Interfaces

Dialect interfaces are generally useful for transformation passes, or analyses,
that want to opaquely operate on operations, even across *across* dialects.
These interfaces generally involve wide coverage over the entire dialect, and
are only used for a handful of transformations/analyses. In these cases,
registering the interface directly on each operation is overly complex and
cumbersome. The interface is not core to the operation, just to the specific
transformation. An example of where this type of interface would be used is
inlining. Inlining generally queries high level information about the operations
within a dialect, like legality and cost modeling, that often is not specific to
one operation.

A dialect interface can be defined by inheriting from the CRTP base class
`DialectInterfaceBase::Base`. This class provides the necessary utilities for
registering an interface with the dialect so that it can be looked up later.
Once the interface has been defined, dialects can override it using dialect
specific information. The interfaces defined by a dialect are registered in a
similar mechanism to Attributes, Operations, Types, etc.

```c++
/// Define an Inlining interface to allow for dialects to opt-in.
class DialectInlinerInterface :
    public DialectInterface::Base<DialectInlinerInterface> {
public:
  /// Returns true if the given region 'src' can be inlined into the region
  /// 'dest' that is attached to an operation registered to the current dialect.
  /// 'valueMapping' contains any remapped values from within the 'src' region.
  /// This can be used to examine what values will replace entry arguments into
  /// the 'src' region for example.
  virtual bool isLegalToInline(Region *dest, Region *src,
                               BlockAndValueMapping &valueMapping) const {
    return false;
  }
};

/// Override the inliner interface to add support for inlining std operations.
struct AffineInlinerInterface : public DialectInlinerInterface {
  /// Affine structures have specific inlining constraints.
  bool isLegalToInline(Region *dest, Region *src,
                       BlockAndValueMapping &valueMapping) const final {
    ...
  }
};

/// Register the interface with the dialect.
AffineOpsDialect::AffineOpsDialect(MLIRContext *context) ... {
  addInterfaces<AffineInlinerInterface>();
}
```

Once registered, these interfaces can be opaquely queried from the dialect by
the transformation/analysis that wants to use them:

```c++
Dialect *dialect = ...;
if (auto *interface = dialect->getInterface<DialectInlinerInterface>())
    ... // The dialect provides this interface.
```

#### DialectInterfaceCollections

An additional utility is provided via DialectInterfaceCollection. This CRTP
class allows for collecting all of the dialects that have registered a given
interface within the context.

```c++
class InlinerInterface : public
    DialectInterfaceCollection<DialectInlinerInterface> {
  /// The hooks for this class mirror the hooks for the DialectInlinerInterface,
  /// with default implementations that call the hook on the interface for a
  /// given dialect.
  virtual bool isLegalToInline(Region *dest, Region *src,
                               BlockAndValueMapping &valueMapping) const {
    auto *handler = getInterfaceFor(dest->getContainingOp());
    return handler ? handler->isLegalToInline(dest, src, valueMapping) : false;
  }
};

MLIRContext *ctx = ...;
InlinerInterface interface(ctx);
if(!interface.isLegalToInline(...))
   ...
```
