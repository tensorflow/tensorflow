# GraphDef Regularization

This directory contains the code for TensorFlow GraphDef regularization,
sometimes referred to as "canonicalization".

## What does it mean to "regularize" a GraphDef?

The TensorFlow GraphDef is the representation of TensorFlow programs. It shares
a lot of the characteristics of an
[intermediate representation](https://en.wikipedia.org/wiki/Intermediate_representation)
or IR. A single TensorFlow program can produce different GraphDefs, depending on
the device, platform, TF version, runtime state, etc.

"Regularization" is the process of removing this non-determinism from the
GraphDef.

## Interesting Problems

GraphDef regularization helps us answer a variety of interesting questions:

- [Graph Isomorphism](https://en.wikipedia.org/wiki/Graph_isomorphism): Do two
  GraphDefs describe the same program?
- [Graph Fingerprinting](https://github.com/tensorflow/community/pull/415): How
  can we can uniquely identify a graph using a much shorter fingerprint?

## Algorithms

- `simple_delete`: An algorithm that deletes parts of the GraphDef that are not
   deterministic.

## Testing
- TODO(b/239046865): Complete this section.

## Contributions Welcome

If you would like to contribute to the GraphDef regularization library, please
send us a pull request. We welcome collaboration!
