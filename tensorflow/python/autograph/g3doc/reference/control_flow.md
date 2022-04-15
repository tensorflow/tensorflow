# AutoGraph reference

[Index](index.md)

## Control flow

AutoGraph rewrites all control flow statements with specialized AutoGraph
function calls. These function calls are capable of executing the corresponding
control flow statement using Python semantics for effects outside the Python
interpreter itself (see the [Introduction](intro.md)).

### Dispatch rules

Key Point: Only statements that are conditioned on, or iterate over, a
TensorFlow object such as `tf.Tensor`, are converted into TensorFlow ops.

As described in the [Introduction](intro.md), AutoGraph aims to preserve the
semantics of valid Python code. If a control flow statement runs in graph
execution without raising an error, then AutoGraph will also execute it as
normal Python control flow. Statements which would normally raise an error, for
example an `if` statement using a `bool` `Tensor` as condition, are converted to
TensorFlow control flow ops.

#### Analogy with compile-time constants and code optimization

From the perspective of a TensorFlow graph, non-Tensor values, for example an
integer or a NumPy array, are _constants_: they do not change value while the
graph executes.

For example, in the graph below, the condition is always `True` (it is
invariant):

```
x = 1
y = tf.cond(x > 0, lambda: 3 * x, lambda 5 * x)
```

That is equivalent to the code below:

```
x = 1
y = 3 * x
```

In the example above, we've optimized away the conditional on a constant
condition. The AutoGraph dispatch rules have the same effect: anything that is
not a TensorFlow object is a compile-time constant for TensorFlow, and can be
optimized away. For this reason, you can usually mix Python and TensorFlow
computation and it will transparently have the expected result even
when only some computations are executed in the graph.

<!-- TODO(mdan): This is actually a limitation (a very subtle one) -->
Caution: The assumption of invariant code made above is not true if the
TensorFlow graph had callbacks into the Python code. If you modify data
from within a `tf.py_function`, then the code outside a `tf.py_function`
will have unpredictable behavior if it depends on the same data.

For example, the `tf.cond` that runs as part of the `if` statement below will
miss the update made by `f`:

```
n = [10]
def f():
  n[0] = 20
  return 0
tf.py_function(f, (), (tf.int32,))
if tf.equal(n[0], 10):
  tf.print('n is 10')
```

```
n is 10
```

### Compound symbols

AutoGraph usually handles basic symbols:

```
if a < 0:
  a = -a
```

```
a = tf.cond(a < 0, lambda: -a, lambda: a)
```

But it can also handle complex symbols in many cases. For example, if we treat
`a.b` as a symbol in the code below, then we can use it as if it were a basic
symbol name:

```
if a.b < 0
  a.b = -a.b
```

```
a.b = tf.cond(a.b < 0, lambda: -a.b, lambda: a.b)
```

This is useful in methods, which can operate on properties of `self`, as well as
working directly on more complex object structures or collections.

Caution: There are certain [limitations](limitations.md) around using Python
collections and object mutation. When in doubt, place the values you work
with into local variables and operate on those.

### Effects of the tracing process

#### All Python code paths are executed during tracing

When constructing a graph, TensorFlow _traces_ the code. The tracing of control
flow requires visiting _every possible code path_ (usually once).

Note: In rare cases, the runtime may decide to trace some code paths several
times. For example, the condition of a `while` statement may be executed twice,
first with a temporary graph, to determine whether it evaluates to a
`tf.Tensor`, then if it is a `tf.Tensor`, it's executed a second time in the
proper graph.

In other words, tracing executes both branches of an if statement. Similarly,
the body of loops is executed once (even if the loop would otherwise not iterate
at all).

This explains why inserting `print` statements in an `if` statement produces
this output:

```
print('before if')
if tf.constant(True):
  print('true branch')
else:
  print('false branch')
print('after if')
```

```
before if
true branch
false branch
after if
```

Note: Control flow that is not executed as a TensorFlow graph is not traced. Its
body will execute as expected.

Example of code that runs as regular Python code:

```
print('before if')
if True:  # Condition not a Tensor, running normally
  print('true branch')
else:
  print('false branch')
print('after if')
```

```
before if
true branch
after if
```

#### Python values modified in TensorFlow control flow become Tensors

If a symbol is modified in a TensorFlow control flow statement, then it becomes
a `tf.Tensor`, even if it started off as a Python primitive value.

For example, the conditional below will run as a `tf.cond` (its condition is a
`tf.Tensor`), which in turn will cause `i` to become a `tf.Tensor`.

```
i = 0
if tf.greater(i, 0):
  i = 1
# i is now a Tensor
```

### `if` statements

`if` statements whose condition is a `tf.Tensor` are executed as TensorFlow
conditionals by converting them to `tf.cond`:

```
if tf.random.uniform(()) > 0.5:
  x = 1
else:
  x = 2
```

`if` statements whose condition is not a `tf.Tensor` are executed as normal
Python:

```
if np.random.uniform() > 0.5:
  x = 1
else:
  x = 2
```

`if` statements executed as TensorFlow conditionals are subject to restrictions
(see [limitations](limitations.md)). All symbols affected by the statement and
used thereafter must be:

 * of a data type understood by TensorFlow
 * defined in both branches
 * of consistent dtypes in both branches, for TensorFlow entities
 * of consistent structure in both branches, for static collections (such as
   lists or tuples)

### `while` statements

`while` statements whose condition is a `tf.Tensor` are executed as TensorFlow
loops by converting them to `tf.while_loop`:

```
x = 0
while tf.random.uniform(()) > 0.5:
  x = x + 1
```

`while` statements whose condition is not a `tf.Tensor` are executed as normal
Python:

```
x = 0
while np.random.uniform() > 0.5:
  x = x + 1
```

`while` statements executed as TensorFlow loops are subject to restrictions
(see [limitations](limitations.md)). All symbols affected by the statement and
used thereafter must be:

 * of a data type understood by TensorFlow
 * defined before the loop
 * of consistent dtype at the beginning and the end of the loop,
   for TensorFlow entities
 * either of consistent shape at the beginning and the end of the loop,
   for TensorFlow entities, or declared in `shape_invariants`
 * of consistent structure  at the beginning and the end of the loop, for
   static collections (such as lists or tuples)

Caution: A `while` loop whose condition is a Python scalar will execute as
normal Python. If you intended to run the loop as a TensorFlow loop, the loop
will replicate its body in the graph (it is unrolled). To avoid that, make sure
its condition is converted to a `tf.Tensor`, using for instance `tf.constant`.

For example, the following loop is unrolled, even though the list contains
`tf.Tensor` values, because the type of `l` is a Python `list`:

```
l = [tf.constant(1), tf.constant(2), tf.constant(3)]
for i in l:
  tf.print(i)  # This is unrolled - three `tf.print`s are built in the graph. 
```

If you wish for the loop to run as a TensorFlow loop, stack the loop:

```
l = [tf.constant(1), tf.constant(2), tf.constant(3)]
for i in tf.stack(l):
  tf.print(i)  # This runs as a TensorFlow loop.
```

<!-- TODO(mdan): List this under limitations -->

Caution: A loop in which the type of the condition changes across iterations, in
a way that would influence the way the loop is executed, is not allowed in
AutoGraph.

For example, the loop below will generate an error, because after the first
iteration, `i` becomes a tf.Tensor:

```
i = 0
while i < 10:  # `i < 10` is a Python bool - run as normal while loop
  i = tf.constant(1)  # Error -- `i < 10` would now be a `tf.Tensor`
```

### `for` statements

`for` statements that iterate over a `tf.Tensor` are executed as TensorFlow
loops by converting them to a `tf.while_loop` which iterates over the first
dimension (equivalent to NumPy):

```
for i in tf.constant(((1, 2), (3, 4))):
  tf.print('iteration:', i)
```

```
iteration: [1, 2]
iteration: [3, 4]
```

Note: If possible, AutoGraph will also set the `maximum_iteration` parameter
of the `tf.while_loop`.

`for` statements that iterate over the output of a `tf.range` are executed as
TensorFlow loops by converting them to a `tf.while_loop` which uses the
arguments passed to the `tf.range`:

```
for i in tf.range(3):
  tf.print('iteration:', i)
```

`for` statements that iterate over a `tf.data.Dataset` and which do not contain
`break` or `return` statements are executed as TensorFlow loops by converting
them to `tf.data.Dataset.reduce` ops:

```
for i in tf.data.Dataset.range(3):
  tf.print('iteration:', i)
```

`for` statements that iterate over a _distributed dataset_ and which do not
contain `break` or `return` statements are executed as TensorFlow loops by
converting them to the dataset's `reduce` ops:

```
for i in tf.distribute.OneDeviceStrategy('cpu').experimental_distribute_dataset(
    tf.data.Dataset.range(3)):
  tf.print('iteration:', i)
```

`for` statements that iterate over a `tf.data.Dataset` and which contain
`break` or `return` statements are executed as TensorFlow loops by converting
them to a combination of `tf.data.Dataset.scan`, `tf.data.Dataset.take_while`
and `tf.data.Dataset.reduce` ops:

```
for i in tf.data.Dataset.range(3):
  tf.print('iteration:', i)
  break
```

```
iteration: 1
```

`for` statements that iterate over a `tf.data.Dataset` _iterator_ are executed
as TensorFlow loops by converting them to a combination of `tf.while_loop`,
and `tf.cond` ops:

```
for i in iter(tf.data.Dataset.range(3)):
  tf.print('iteration:', i)
```

`for` statements that iterate over a type different from any of the above are
executed as normal Python:

```
for i in [1, 2, 3]:
  print('iteration:', i)
```

Caution: A `for` loop over a `list` or `tuple` of `tf.Tensor` is considered to
iterate over a Python `list` (or respectively `tuple`), therefore will be
executed as normal Python. If you intended to run it as a TensorFlow loop,
use `tf.stack` or `tf.concat`.

Caution: A `for` loop over a Python `range` will execute as normal Python.
If you intended to run it as a TensorFlow loop, use `tf.range`.

Note: AutoGraph may output a warning when it believes that you are unrolling
a loop inefficiently. However, the warning thresholds are very conservative.
The warning is only printed when
[__debug__](https://docs.python.org/3/library/constants.html#__debug__) is
`True`.

Note: If `__debug__` is `True`, AutoGraph limits the number of iterations in
normal Python loops to prevent infinite loops and raise an error if the limits
are exceeded. However, the iteration limits are very large and may take a while
to trigger an error.

### `break` statements

Code blocks in which `break` statements are used are rewritten with equivalent
code that uses extra control booleans and conditionals. The control booleans are
used directly in `while` loops. In the case of `for` loops, the AutoGraph
corresponding operator accepts an `extra_test` argument which is similar to
the conditional of a while loop, and which contains the control boolean.

For example, the `while` loop below is rewritten as (showing the output of the
`break` transformation only):

```
while i < 10:
  if i > 3:
    break
  i += 1
```

```
break_ = False
while i < 10 and not break_:
  if i > 3:
    break_ = True
    continue  # The continue statement is also rewritten in a subsequent pass
  i += 1
```

Another example shows how the control boolean is used in the overload of a `for`
loop (showing portions of the final output):

```
for i in range(10):
  if i > 3:
    break
```

```
break_ = False
...
def extra_test(break_):
  return ag__.not_(break_)
# break_ becomes a loop variable.
break_, = ag__.for_stmt(range(10), extra_test, ..., (break_,))
```

Mixing Tensor-dependent `break` and Python-dependent loops is disallowed:

```
@tf.function
def buggy_while_py_true_tf_break(x):
  while True:   # python conditional
    if tf.equal(x, 0): # tensor break
      break
    x -= 1
  return x

# Raises OperatorNotAllowedInGraphError: using a `tf.Tensor` as a Python `bool` is not allowed
# buggy_while_true_tf_break(5)
```

### `continue` statements

Code blocks in which `continue` statements are used are rewritten with
equivalent code that uses extra control booleans and conditionals, similar to
how `break` is handled.

For example, the `for` loop below is rewritten as (showing the output of the
`continue` transformation only):

```
for i in range(10):
  if i > 3:
    continue
```

```
for i in range(10):
  continue_ = False
  if i > 3:
    continue_ = True
  if not continue_:
    i += 1
```

Notice that unlike `break`, `continue` statements are local to the loop and do
not influence the number of iterations.

### `return` statements

`return` statements are also rewritten using control symbols, in a manner
similar to how `break` is converted. In the case of `return` statements, an
additional symbol keeps track of the return value.

Depending on the structure of the code, the return value might be undefined
in parts of the code (for example on code paths in which no return statement
has executed). AutoGraph keeps track of this by using a special value.
This special value is converted to `None` (the default return value) upon
exiting the function.

Caution: TensorFlow control flow doe not support undefined values, and an
undefined return value is no exception. Therefore, AutoGraph will raise an
error for TensorFlow control flow in which the return value is not known for
all code paths.

For example, the following code raises an error because the return value would
be undefined when the random number would be less than 0.5:

```
if tf.random.uniform(()) > 0.5:
  return 1
```

```
ValueError: A value must also be returned from the else branch.
```

An example of rewriting a `while` (showing the output of the `return`
transformation only):

```
def f():
  while i < 10:
    if i > 3:
      return 1
    i += 1
```

```
def f():
  do_return = False
  retval_ = ag__.UndefinedReturnValue()
  while i < 10 and not do_return:
    if i > 3:
      do_return = True
      retval_ = 1
    if not do_return:
      i += 1
  return ag__.retval(retval_)  # Transforms any UndefinedReturnValue to None
```

Note: AutoGraph performs an additional code normalization in which an `if`
statement with no `else` branch contains a `return` statement it is rewritten as
an `if-else` statement in which the code that follows the statement is moved
under the `else` branch.

Example (showing the normalization only):

```
def f():
  if i > 3:
    return 1
  i += 1
```

```
def f():
  if i > 3:
    return 1
  else:
   i += 1
```


