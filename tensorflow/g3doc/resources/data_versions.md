# TensorFlow Data Versioning: GraphDefs and Checkpoints

As described in [Compatibility for Graphs and Checkpoints](versions.md#graphs),
TensorFlow marks each kind of data with version information in order to maintain
backwards compatibility even across major releases in some cases.

This document describes the versioning mechanism in more detail, and explains
how to use it to change data formats safely.

## Goals: backwards and partial forwards compatibility

Consider the case of TensorFlow graphs serialized via the GraphDef protobuf.  We
have a number of competing constraints:

* We would like to be able to evolve TensorFlow in eventually incompatible ways:
  removing ops, adding or removing attrs, etc.
* GraphDefs produced by TensorFlow may live for months after they are generated,
  so we want **backwards compatibility**: new versions of TensorFlow should be
  able to read old data.
* Sometimes a producer of a GraphDef is upgraded to a new version of TensorFlow
  before the consumer of that data is updated, so we would like **forwards
  compatibility**: new versions of TensorFlow should generate GraphDefs readable
  by older versions of TensorFlow.  Unfortunately, forwards compatibility is
  much more intrusive than backwards compatibility, so we support it only in
  limited situations within Google and across *patch* releases for open source.

For GraphDefs, we support backwards compatibility for 6 months and forwards
compatibility for 3 weeks in limited situations.  For backwards compatibility,
this means that we can only remove functionality 6 months after we stop
producing data using that functionality.  Similarly, in the limited situations
where we support forwards compatibility, we can add functionality only 3 weeks
after TensorFlow can consume data using that functionality.

In order to implement these semantics, we need to know when data is produced so
that we can know when to enforce changes in formats.  The versioning system
described below achieves that goal in a manner that supports both backwards and
forwards compatibility (when they apply).

For checkpoints, we have no plans to make either backwards or forwards
incompatible changes, but still attach versions to checkpoints in case we ever
do have to make a change.

## Each type of data has separate version scheme

Since different data formats evolve at different rates, we have a separate
integer versioning scheme for each kind of data, and these schemes are separate
from the overall version of TensorFlow.

For now, there are data versions for GraphDefs (serialized computation graphs)
and checkpoints (serialized variable state).  Both versioning schemes are
defined in
[`core/public/version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/version.h).
Whenever a new version is added, a note should be made in that header recording
what changed and when.

## Data, producers, and consumers

In the discussion below, we consider version information for **data**, binaries
that produce that data (**producers**), and binaries that consume that data
(**consumers**):

* Producer binaries have a version (`producer`) and a minimum consumer version
  that they are compatible with (`min_consumer`).
* Consumer binaries have a version (`consumer`) and a minimum producer version
  that they are compatible with (`min_producer`).
* Each piece of versioned data has a [`VersionDef
  versions`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/versions.proto)
  field which records the `producer` that made the data, the `min_consumer` that
  it is compatible with, and a list of `bad_consumers` versions that are
  disallowed.

By default, when a producer makes some data, the data inherits the producer's
`producer` and `min_consumer` versions.  `bad_consumers` can be set if specific
consumer versions are known to contain bugs and must be avoided.  A consumer
can accept a piece of data if

* `consumer` >= data's `min_consumer`
* data's `producer` >= consumer's `min_producer`
* `consumer` not in data's `bad_consumers`

Since both producers and consumers come from the same TensorFlow code base,
[`core/public/version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/version.h)
contains a main binary version which is treated as either `producer` or
`consumer` depending on context and both `min_consumer` and `min_producer`
(needed by producers and consumers, respectively).  Specifically,

* For GraphDef versions, we have `TF_GRAPH_DEF_VERSION`,
  `TF_GRAPH_DEF_VERSION_MIN_CONSUMER`, and `TF_GRAPH_DEF_VERSION_MIN_PRODUCER`.
* For checkpoint versions, we have `TF_CHECKPOINT_VERSION`,
  `TF_CHECKPOINT_VERSION_MIN_CONSUMER`, and
  `TF_CHECKPOINT_VERSION_MIN_PRODUCER`.

## Evolving GraphDef versions

We now discuss examples of using this versioning mechanism to make various
changes to the GraphDef format.  Our goal is to be backwards compatible for six
months, which means that data produced by TensorFlow at time `T` must be
consumable by TensorFlow at time `T + 6 months`.  If forwards compatibility is
desired, the data must be consumable at time `T - 3 weeks`.

**Adding a new op:**

1. Add the new op to both consumers and producers at the same time, and do not
   change any GraphDef versions.  This type of change is automatically backwards
   compatible, and is outside our forwards compatibility plan since existing
   producer scripts will not suddenly use the new functionality.

**Adding a new op and switching existing Python wrappers to use it:**

1. Implement new consumer functionality and increment the binary version.
2. If it is possible to make the wrappers use the new functionality only in
   cases that did not work before, the wrappers can be updated now.
3. If forwards compatibility is necessary, wait 3 weeks.
4. Change Python wrappers to use the new functionality.  Do not increment
   `min_consumer`, since models which do not use this op should not break.

**Removing an op or restricting the functionality of an op:**

1. Fix all producer scripts (not TensorFlow itself) to not use the banned op or
   functionality.
2. Increment the binary version and implement new consumer functionality that
   bans the removed op or functionality for GraphDefs at the new version and
   above.  If possible, make TensorFlow stop producing GraphDefs with the banned
   functionality.  This can be done with
   [`REGISTER_OP(...).Deprecated(deprecated_at_version, message)`](
   https://github.com/tensorflow/tensorflow/blob/b289bc7a50fc0254970c60aaeba01c33de61a728/tensorflow/core/ops/array_ops.cc#L1009).
3. Wait 6 months for backwards compatibility purposes.
4. Increase `min_producer` to the GraphDef version from (2) and remove the
   functionality entirely.

**Changing the functionality of an op:**

1. Add a new similar op named `SomethingV2` or similar and go through the
   process of adding it and switching existing Python wrappers to use it (may
   take 3 weeks if forwards compatibility is desired).
2. Remove the old op (takes 6 months due to backwards compatibility).
3. Increase `min_consumer` to rule out consumers with the old op, add back the
   old op as an alias for `SomethingV2`, and go through the process to switch
   existing Python wrappers to use it (may take 3 weeks).
4. Go through the process to remove `SomethingV2`.

**Banning a single consumer version that cannot run safely:**

1. Bump the binary version and add the bad version to `bad_consumers` for all
   new GraphDefs.  If possible, add to `bad_consumers` only for GraphDefs which
   contain a certain op or similar.
2. If existing consumers have the bad version, push them out as soon as
   possible.
