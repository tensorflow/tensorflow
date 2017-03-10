# TensorFlow Data Versioning: GraphDefs and Checkpoints

As described in
@{$version_semantics#compatibility-for-graphs-and-checkpoints$Compatibility for Graphs and Checkpoints},
TensorFlow marks each kind of data with version information in order to maintain
backward compatibility. This document provides additional details about the
versioning mechanism, and how to use it to safely change data formats.

## Backward and partial forward compatibility

The two core artifacts exported from and imported into TensorFlow are
checkpoints (serialized variable states) and `GraphDef`s (serialized computation
graphs). Any approach to versioning these artifacts must take into account the
following requirements:

*   **Backward compatibility** to support loading `GraphDefs` created with older
    versions of TensorFlow.
*   **Forward compatibility** to support scenarios where the producer of a
    `GraphDef` is upgraded to a newer version of TensorFlow before the consumer.
*   Enable evolving TensorFlow in incompatible ways. For example, removing Ops,
    adding attributes, and removing attributes.

For `GraphDef`s, backward compatibility is enforced within a major version. This
means functionality can only be removed between major versions. Forward
compatibility is enforced within Patch releases (1.x.1 -> 1.x.2, for example).


In order to achieve backward and forward compatibility as well as know when to
enforce changes in formats, the serialized representations of graphs and
variable state need to have metadata that describes when they were produced. The
sections below detail the TensorFlow implementation and guidelines for evolving
`GraphDef` versions.

### Independent data version schemes

There are data versions for `GraphDef`s and checkpoints. Both data formats
evolve at different rates, and also at different speeds than the version of
TensorFlow. Both versioning systems are defined in
[`core/public/version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/version.h).
Whenever a new version is added a note is added to the header detailing what
changed and the date.

### Data, producers, and consumers

This section discusses version information for **data**, binaries that produce
data (**producers**), and binaries that consume data (**consumers**):

*   Producer binaries have a version (`producer`) and a minimum consumer version
    that they are compatible with (`min_consumer`).
*   Consumer binaries have a version (`consumer`) and a minimum producer version
    that they are compatible with (`min_producer`).
*   Each piece of versioned data has a [`VersionDef
    versions`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/versions.proto)
    field which records the `producer` that made the data, the `min_consumer`
    that it is compatible with, and a list of `bad_consumers` versions that are
    disallowed.

By default, when a producer makes some data, the data inherits the producer's
`producer` and `min_consumer` versions. `bad_consumers` can be set if specific
consumer versions are known to contain bugs and must be avoided. A consumer can
accept a piece of data if

*   `consumer` >= data's `min_consumer`
*   data's `producer` >= consumer's `min_producer`
*   `consumer` not in data's `bad_consumers`

Since both producers and consumers come from the same TensorFlow code base,
[`core/public/version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/version.h)
contains a main binary version which is treated as either `producer` or
`consumer` depending on context and both `min_consumer` and `min_producer`
(needed by producers and consumers, respectively). Specifically,

*   For `GraphDef` versions, we have `TF_GRAPH_DEF_VERSION`,
    `TF_GRAPH_DEF_VERSION_MIN_CONSUMER`, and
    `TF_GRAPH_DEF_VERSION_MIN_PRODUCER`.
*   For checkpoint versions, we have `TF_CHECKPOINT_VERSION`,
    `TF_CHECKPOINT_VERSION_MIN_CONSUMER`, and
    `TF_CHECKPOINT_VERSION_MIN_PRODUCER`.

### Evolving GraphDef versions

This section presents examples of using this versioning mechanism to make
changes to the `GraphDef` format.

**Adding a new Op:**

1.  Add the new Op to both consumers and producers at the same time, and do not
    change any `GraphDef` versions. This type of change is automatically
    backward compatible, and does not impact forward compatibility plan since
    existing producer scripts will not suddenly use the new functionality.

**Adding a new Op and switching existing Python wrappers to use it:**

1.  Implement new consumer functionality and increment the binary version.
2.  If it is possible to make the wrappers use the new functionality only in
    cases that did not work before, the wrappers can be updated now.
3.  Change Python wrappers to use the new functionality. Do not increment
    `min_consumer`, since models which do not use this Op should not break.

**Removing an Op or restricting the functionality of an Op:**

1.  Fix all producer scripts (not TensorFlow itself) to not use the banned Op or
    functionality.
2.  Increment the binary version and implement new consumer functionality that
    bans the removed Op or functionality for GraphDefs at the new version and
    above. If possible, make TensorFlow stop producing `GraphDefs` with the
    banned functionality. This can be done with
    [`REGISTER_OP(...).Deprecated(deprecated_at_version,
    message)`](https://github.com/tensorflow/tensorflow/blob/b289bc7a50fc0254970c60aaeba01c33de61a728/tensorflow/core/ops/array_ops.cc#L1009).
3.  Wait for a major release for backward compatibility purposes.
4.  Increase `min_producer` to the GraphDef version from (2) and remove the
    functionality entirely.

**Changing the functionality of an Op:**

1.  Add a new similar Op named `SomethingV2` or similar and go through the
    process of adding it and switching existing Python wrappers to use it (may
    take 3 weeks if forward compatibility is desired).
2.  Remove the old Op (Can only take place with a major version change due to
    backward compatibility).
3.  Increase `min_consumer` to rule out consumers with the old Op, add back the
    old Op as an alias for `SomethingV2`, and go through the process to switch
    existing Python wrappers to use it.
4.  Go through the process to remove `SomethingV2`.

**Banning a single consumer version that cannot run safely:**

1.  Bump the binary version and add the bad version to `bad_consumers` for all
    new GraphDefs. If possible, add to `bad_consumers` only for GraphDefs which
    contain a certain Op or similar.
2.  If existing consumers have the bad version, push them out as soon as
    possible.
