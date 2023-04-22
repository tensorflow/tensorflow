# AutoGraph reference

[Index](index.md)

## Introduction

### Terminology

Typically, AutoGraph operates by converting a function into a new function with
new semantics.

```
def f(...):
  ...

converted_f = tf.autograph.to_graph(f)
```

In TensorFlow 2, AutoGraph is used to convert functions before they are being
traced, when using the `tf.function` API. For example, the following code:

```
def f(...):
  ...

graph_f = tf.function(f)
```

is roughly equivalent to:

```
converted_f = tf.autograph.to_graph(f)
graph_f = tf.function(autograph=False)(converted_f)
```

For the remainder of this document:

 * **converted functions** are functions converted by AutoGraph
 * **graph functions** are functions compiled with `tf.function`

Graph functions are usually also converted, unless specified otherwise.

### Safe behavior

The semantics described below can be summarized as:

 1. code should either produce the same results as running it in Eager mode, or
   fail with error.
 2. TF 1.* graph code should produce the same graph as running it directly in
   graph mode.

### Python semantics

In general, AutoGraph does not change the semantics of error-free Python.
In special circumstances, AutoGraph may also preserve the semantics of code that
raises error as well. Semantics are preserved in the sense of the
[as-if rule](https://en.wikipedia.org/wiki/As-if_rule).

More specifically, code that would not be legal TensorFlow Graph code may become
legal when converted.

For example, consider the following function and the corresponding converted
function:

```
def f(x):
  if x > 0:
    return x
  return -x

converted_f = tf.autograph.to_graph(f)
```

If `f` executes without error, then the `converted_f` has identical results:

```
>>> f(3)                               # Runs without error
3
>>> converted_f(3)                     # Identical result
3
```

### TensorFlow Eager semantics

If a function is called with `Tensor` arguments in Eager execution mode, it has
certain semantics that are different the Graph execution semantics.

For example, the function below produces different results when executed in
Eager and Graph modes:

```
def f(x):
  if x > 0:
    return x
  return -x

# Use tf.function to run code as a TensorFlow Graph.
unconverted_graph_f = tf.function(f, autograph=False)
graph_f = tf.function(f, autograph=True)
```

```
>>> f(tf.constant(3))                      # Valid in Eager
<tf.Tensor ... numpy=3>

>>> unconverted_graph_f(tf.constant(3))    # Error - not legal Graph code
TypeError: Using a `tf.Tensor` ...
```

AutoGraph transfers the Eager semantics to the graph function:

```
>>> graph_f(tf.constant(3))                # Valid in AutoGraph
<tf.Tensor ... numpy=3>
```

### Subset of Eager

AutoGraph currently supports only a subset of Eager. For example, list operation
semantics are not supported:

```
def f(n):
  l = []
  for i in tf.range(n):
    l.append(i)
  return l

converted_f = tf.autograph.to_graph(f)
```

```
>>> f(3)                               # Valid in Eager
[1, 2, 3]
>>> converted_f(3)                     # Not valid in AutoGraph (by default)
<<error>>
```

### Experimental features

AutoGraph supports additional semantics which are not yet stable. This includes
for example support for list semantics applied to `TensorArray`:

```
def f(n):
  l = tf.TensorArray(...)
  for i in tf.range(n):
    l.append(i)
  return l

converted_f = tf.autograph.to_graph(f)
experimental_converted_f = tf.autograph.to_graph(
    f, experimental_autograph_options=tf.autograph.experimental.Feature.LISTS)
```

```
>>> converted_f(3)                     # Not valid in AutoGraph (by default)
<<error>>
>>> experimental_converted_f(3)        # Valid
<<TensorArray object>>
```
