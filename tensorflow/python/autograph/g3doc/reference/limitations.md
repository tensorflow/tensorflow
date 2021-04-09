# AutoGraph reference

[Index](index.md)

## Limitations

When AutoGraph is applied to normal Python code, you should expect no change
in functionality.
However, when applied to TensorFlow control flow (for example, an if statement
with a `tf.Tensor` condition), there are certain limitations. This section
describes these limitations and practices that will allow you to avoid them.

Key Term: Python variables refer to Python symbols (or symbols for short) and
should not be confused with TensorFlow variables.

Key Term: A TensorFlow loop variable (or loop variable for short) refers to a
value (typically a `tf.Tensor`) modified by a loop. See `tf.while_loop`.

### Undefined and None values in TensorFlow

TensorFlow does not support undefined or `None` values. All tensors must have
a value.

Example:

```
x = tf.cond(
    tf.random.uniform(()) > 0.5,
    lambda: tf.constant(1),
    lambda: None)  # Error -- a Tensor cannot be None
```

The same restriction carries over in AutoGraph. If a variable is created inside
control flow, and used after, then it must be defined before the control flow
statement:

```
if tf.random.uniform(()) > 0.5:
  x = tf.constant(1)
else:
  x = None
tf.print(x)  # Error -- x may be None here
```

For this reason, AutoGraph forbids variables to be defined in only one branch
of a TensorFlow conditional, if the variable is used afterwards:

```
del x
if tf.random.uniform(()) > 0.5:
  x = tf.constant(1)
else:
  pass
tf.print(x)  # Error -- x may be undefined here
```

Note that if the variable is not used after the control flow statement, then it
is considered local to the control flow block, and is not subject to these
restrictions.

```
del x
if tf.random.uniform(()) > 0.5:
  x = tf.constant(1)  # Okay -- x does not need to be returned from the TF cond
else:
  pass
```

Similarly, variables must usually be defined before a TensorFlow loop.

The most common example that is not allowed is a loop which initializes some
accumulator variable in the first iteration:

```
del x
for i in tf.range(100):  # Error -- x must be defined before the loop
  if i == 0:
    x = tf.constant(1)
  else:
    x = x + 1
tf.print(x)
```

When the variable is only used inside the loop and does not depend on previous
iterations, then it's ok to only be initialized inside the loop.

```
del x
while tf.random.uniform(()) > 0.5:  # Okay -- x is not used after the loop
  x = tf.constant(1)
```

* New in TF 2.4 *

As long as it doesn't depend on previous iterations, the variable may also be
used after the loop, however in that case the loop must execute at least one
iteration, and will raise a runtime error otherwise.

```
del x
for i in tf.range(10):  # Okay -- x does not depend on previous iterations
  x = tf.constant(1)
tf.print(x)
```

```
del x
while tf.constant(False):  # Error -- loop must initialize x!
  x = tf.constant(1)
tf.print(x)
```

Avoid these limitations by defining a default value before the control flow
statement:

```
x = tf.constant()
if tf.random.uniform(()) > 0.5:
  x = tf.constant(1)
tf.print(x)  # Okay -- x is either 0 or 1
```

Note: `None` values and undefined symbols are allowed in Eager control flow,
because Eager execution uses Python control flow, rather than TensorFlow
control flow ops.

#### Special case: creating Tensors in a loop

* New in TF 2.4 *

A very common use-case is to run a training loop that creates some outputs:

```
for i in tf.range(num_steps):
  outputs = train(next(data_iterator))
```

Often times these outputs can be nested structures of Tensors, which makes them
impractical to initialize ahead of the loop.

To help with this use-case, AutoGraph lets you run such loops, under certain
conditions:

 * outputs must be a Tensor, Python numeric, or a structure of these
 * outputs must not depend on the value from a previous iteration; in other
   words, the outputs may only appear to the left of an assignment operation
 * the loop must run at least one iteration

If the type of outputs is not recognized, then the usual
"outputs must be defined before the loop" is raised at graph construction.

AutoGraph also inserts a `tf.Assert` statement that raises a runtime error
if the loop did not execute at least one iteration.

### Indirect modifications and hidden side effects in TensorFlow control flow

Key Point: We recommend using a functional programming style, immutable Python
collections, TensorFlow ops and collections. Only TensorFlow objects should be
used for side effects.

#### AutoGraph analyzes code to detect modifications to Python objects

Note: Modifications to TensorFlow objects, such as `tf.Variable`, are tracked
using a different mechanism (automatic control dependencies) which does not
rely on code analysis.

One of the most important functions of AutoGraph is to rewrite Python control
flow statements into equivalent TensorFlow ops. This process requires "wiring"
variables covered by these control flow statements into the respective ops.

The examples below use a `while` loop, but the same notions extend to all
control flow such as `if` and `for` statements.

In the example below, `x` needs to become a loop variable of the
corresponding `tf.while_loop':

```
while x > 0:
  x = x - 1
```
```
x = tf.while_loop(..., loop_vars=(x,)
```

TF control ops support only a limited set of types for loop variables. At the
same time, the efficiency of TensorFlow graphs is influenced by the number of
loop variables, so we don't want to create them unnecessarily. AutoGraph pulls
symbols through loop variables only if necessary to minimize the number of
loop variables.

Note: If a symbol refers to a nested structure, such as a `dict` of `dict`s,
the entire structure is mapped to multiple loop variables - TensorFlow
automatically unpacks it.

For example, the symbol 'y' below is not wired through the `tf.while_loop`'s
`loop_vars` because it is not affected by the `while` loop:

```
y = 0
while x > 0:
  x = x - 1
print(y)
```
```
x = tf.while_loop(..., loop_vars=(x,)  # y does not need to be a loop variable
```

AutoGraph uses static analysis to determine which symbols are modified by the
code, in order to transform them into control flow variables. Static analysis
is generally performed on single functions - Python's dynamic nature limits its
effectiveness across functions.

#### Modifications of Python objects are not detected across functions

Note: Modifications to TensorFlow objects, such as `tf.Variable`, are tracked
using a different mechanism (automatic control dependencies). Modifications
to `tf.Variable` objects are correctly handled even when called in other
functions.

Because static analysis is limited to single functions, modifications that are
performed in other functions are not visible to AutoGraph:

```
def change_y():
  global y
  y = y + 1

while x > 0:
  change_y()  # Problem -- change made to y is not visible here!
```

This can be easily remedied using a functional programming style - writing
functions that use argument for all their inputs and return values for all their
outputs:

```
def change(y):
  y = y + 1
  return y

while x > 0:
  y = change(y)  # Okay -- y can now be properly tracked!
```

As noted before, this limitation does not apply to most TensorFlow objects,
although it is still a good idea to use functional programming style for
better code readability:

```
def change(y_var):
  y_var.assign_add(1)

y = tf.Variable(1)
while x > 0:
  change(y)  # This is still okay -- TensorFlow side effects are robust.
```

Keep in mind however that certain types like `tf.TensorArray` don't support
side effects and must have their result assigned, otherwise they may raise an
error:

```
def change(ta):
  ta.write(0, 1)  # Incorrect use of TensorArray - will raise an error
```

In other words, `tf.TensorArray` must be handled using functional programming
style:

```
def change(ta):
  ta = ta.write(0, 1)  # Modifications create a new TensorArray efficiently.
  return ta

ta = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
while x > 0:
  # TensorArray must be handled using functional programming style.
  ta = change(ta)
```

#### Modifications of Python objects are not detected in methods

A special case of hidden side effects are methods, which are commonly used
to change the value of objects:

```
class MyClass(object):
  def change(self):
    self.y += 1

c = MyClass()
while x > 0:
  c.change()  # Problem -- modification to c.y is not visible here!
```

This can be addressed in a number of ways.

One possibility is to operate directly on the object properties:

```
c = MyClass()
while x > 0:
  c.y += 1  # Okay -- c.y can now be properly tracked!
```

Another possibility is to rely on immutable objects with value semantics. This
may lead to many temporary objects when executing eagerly, but their number is
greatly reduced in `@tf.function`:

```
class MyClass(collections.namedtuple('MyClass', ('y',))):
  def change(self):
    new_y = self.y + 1
    return MyClass(new_y)

c = MyClass()
while x > 0:
  c = c.change()  # Okay -- c is now a loop var.
```

It is also recommended to use a functional programming style with such immutable
objects - that is, all arguments are inputs, all changes are return values:

```
def use_my_class(c: MyClass) -> MyClass:
  new_c = c.change()
  return new_c
```

Don't worry about creating a few extra objects - they are only used at trace
time, and don't exist at graph execution.

Note: TensorFlow control flow does not currently support arbitrary Python
objects, but it does support basic collection objects such as `list`, `dict`,
`tuple`, `namedtuple` and their subclasses. Design your objects as subclasses
of [namedtuple](https://docs.python.org/3/library/collections.html#collections.namedtuple),
or other types that [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest/map_structure)
recognizes.

#### Variables closed over by lambda functions

AutoGraph assumes that variables that local functions close over may be used
anywhere in the parent function, because in general it is possible to hide a
function call in almost any Python statement). For this reason, these variables
are accounted within TensorFlow loops.

For example, the following code correctly captures `a` in the TensorFlow loop
variables:

```
a = 0
def f():
  tf.print(a)
for i in tf.range(3):
  a = i
f()  # Prints 2
```

An consequence is that these variables must be defined before the loop (see
Undefined and None values above). So the following code will raise an error,
even if the variable is never used after the loop:

```
def f():
  tf.print(a)
for i in tf.range(3):  # Error -- `a` must be defined before the loop.
  a = i
```

However, lambda functions are handled differently, for reasons of backward
compatibility. Lambda functions are assumed to be used in the statement where
they are used, or at least in the same block.

```
a = 0
foo(lambda: a)  # This lambda is not expected to be called anywhere else.
for i in tf.range(3):  # Okay -- `a` is local to the loop.
  a = i
```

Due to that reason, the following code will not work as expected for TensorFlow
loops.

```
a = 0
l = lambda: tf.print(a)
for i in tf.range(3):
  a = i  # `a` is considered local to the loop
l()  # Prints 0!
```

Note that none of these restrictions only apply to TensorFlow loops; Python
loops correctly handle closures in all cases.

### Python collections in TensorFlow control flow

Key Point: Use TensorFlow collection classes instead of Python collections.
Python collections are okay to use when they represent a fixed structure (that
is, `list`s don't change length, `dict`s don't add or remove keys).

#### Modifying Python collections in TensorFlow control flow is not allowed

One of the advantages of eager execution is that you may use the usual Python
collections, like `list` or `dict` to hold `tf.Tensor` values. However, these
are generally not compatible with TensorFlow control flow. Specialized
collections like `tf.TensorArray` are required.

Consider the following example:

```
def fn():
  l = []

  def loop_cond(i):
    return i < 10

  def loop_body(i):
    i = i + 1
    l.append(i)
    return i,

  tf.while_loop(
      cond=loop_cond,
      body=loop_body,
      loop_vars=(0,))

  return l
```

This code works in eager execution, which does not use the TensorFlow runtime
for the `tf.while_loop`:

```
fn()
```

However, it does not work in graph execution, because TensorFlow uses special
mechanisms to ensure the computations are correctly sequenced in the dataflow
graph:

```
tf.function(fn)()  # Error -- illegal tensor capture!
```

The equivalent AutoGraph code raises the same error:

```
l = []
for i in tf.range(10):
  l.append(i)  # Error -- illegal tensor capture!
```

Instead, use the specialized `tf.TensorArray` class:

```
l = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
for i in tf.range(10):
  l = l.write(l.size(), i)  # Okay
```

#### Python collections of fixed structure are allowed TensorFlow control flow

An exception from the previous rule is made by Python collections that are
static, that is, they don't grow in size for the duration of the computation.

Caution: Use functional programming style when manipulating static collections.

Examples:

```
static_list = [tf.constant(3)]
while d.prop > 0:
  static_list[0] -= 1  # Okay -- static_list does not change structure
```
```
static_object = MyClass()
static_object.field = tf.constant(3)
while static_object.field > 0:
  static_object.field -= 1  # Okay -- static_object does not change structure
```
```
static_dict = {'field': tf.constant(3)}
while static_dict['field'] > 0:
  static_dict['field'] -= 1  # Okay -- static_dict does not change structure
```

However, remember to use functional programming style when these collections
are used inside control flow.

#### Python collections of fixed structure with dynamic index

A more subtle error occurs when the collection is static, but is accessed in a
dynamic way, that is with a key that is not constant.

For example:

```
d = {'a': tf.constant(3)}
for i in tf.range(10):
  for key in d:
    d[key] += i  # Problem -- accessing `dict` using non-constant key
```

The code above will raises an "illegal capture" error. To remedy it, write it
in functional programming style:

```
d = {'a': tf.constant(3)}
for i in tf.range(10):
  d = {key: value + i for key, value in d.items()}  # Okay
```

### Shape and dtype consistency in TensorFlow control flow

Unlike Python, TensorFlow has limited support for dynamic typing. This means
that tensors must maintain consistent shapes and dtypes across control flow
paths.

Note: In general, these restrictions do not apply in control flow in Eager
execution, because Eager execution uses Python control flow, rather than
TensorFlow control flow ops.

#### Mixing dynamic computations and static shapes

Key Point: Use `.shape` on tensors of static shape, and `.shape.rank` on
tensors of static rank; only use `tf.shape` and `tf.rank` when the shape or
rank is dynamic.

TensorFlow has optional static types and shapes: the shape of tensors may be
static (e.g. `my_tensor.shape=(3, 3)` denotes a three by three matrix) or
dynamic (e.g. `my_tensor.shape=(None, 3)` denotes a matrix with a dynamic
number of rows and three columns. When the shapes are dynamic, you can still
query it at runtime by using the `tf.shape()` function.

Note: `tf.shape` always returns a tensor.

For static shapes, TensorFlow will perform additional shape verifications at
graph construction time, that is, during tracing. These static shape
verifications are useful because they work like a compiler for example, errors
are caught early, before execution even begins.

For example:

```
x = tf.constant([1, 2, 3])
x[4]  # Tracing error! 4 is out of bounds.
```

To avoid tracing errors, you can add static shape verifications, which help
make your code more robust:

```
if x.shape[0] > 4:
  val = x[4]
else:
  val = some_default_value
```

In the snippet above, the code is protected against index-out-of-bounds
errors. The code is also efficient because the verification `x.shape[0] > 4`
will not be included in the graph.

But what happens if you try to perform the index verifications using dynamic
control flow? You might expect that the code works in the same way:

```
val = tf.cond(
  x.shape[0] >= 4,
  lambda: x[4],
  lambda: some_default_value)
```

However, TensorFlow will not let you write code that could result in an error,
even if that code appeared in a branch of a `tf.cond` statement that would
never execute. Remember that the shape of `x` is `(3,)`, so TensorFlow performs
static shape verification.

This can lead to surprising behavior when using `tf.shape` on tensors with
static shape in TensorFlow:

```
x = tf.constant((1, 2, 3))
if tf.shape(x)[0] > 4:
  val = x[4]  # Error at tracing: 4 is out of bounds!
else:
  val = some_default_value
```

Because `tf.shape` always evaluates to a Tensor, the `if` statement above is
converted by AutoGraph into a `tf.cond`, which performs static shape
verification of both branches.

What if you need to write code which can handle both static and dynamic
shapes? There are a few options in this case:

A first option is to always work with dynamic shapes, for instance by
using `input_signature` in `tf.function`. Many shape and shape-related checks
are skipped when the shape is dynamic:

```
@tf.function(input_signature=(tf.TensorSpec(shape=(None,))))
def f(x):  # x now has dynamic shape
  if tf.shape(x)[0] >= 3:  # Builds a tf.cond
    val = x[4]  # Okay, bounds checks are skipped when the shape is dynamic
  else:
    val = some_default_value
```

A second option is to first verify whether the shape is static or dynamic.
This can be done at tracing time, allowing to use Python `if` to only trace
the code that is suitable for the situation:

```
if x.shape[0] is None:  # Python bool, does not use tf.cond
  # ... use x.shape here ...
else:
  # ... use tf.shape(x) here ...
```

#### Consistency of dtype

The dtypes across all code paths must be consistent in conditionals and loops.

For example, if a `tf.cond` (and correspondingly, an AutoGraph `if`) sets a
tensor value conditionally, then that tensor must have the same shape and dtype
in both branches of the conditional.

Example of illegal dtype change in a conditional:

```
x = tf.cond(
    tf.random.uniform(()) > 0.5,
    lambda: tf.constant(1, dtype=tf.int32),
    lambda: tf.constant(1, dtype=tf.float32))  # Error -- inconsistent dtypes: int32, float32
```

The same restriction in AutoGraph code:

```
if tf.random.uniform(()) > 0.5:
  x = tf.constant(1, dtype=tf.int32)
else:
  x = tf.constant(1, dtype=tf.float32)  # Error -- inconsistent dtypes: int32, float32
```

Example of illegal dtype change in a loop:

```
# This won't work - "x" changes dtype inside the loop.
x = tf.while_loop(
    lambda _: tf.random.uniform(()) > 0.5,
    lambda x: tf.constant(1, dtype=tf.float32),
    loop_vars=(tf.constant(1, dtype=tf.int32),))  # Error -- inconsistent dtypes: int32, float32
```

The same restriction in AutoGraph code:

```
x = tf.constant(0, dtype=tf.int32)
while tf.random.uniform(()) > 0.5:
  x = tf.constant(0, dtype=tf.float32)   # Error -- inconsistent dtypes: int32, float32
```

#### Consistency of shape

The shapes across all code paths must be consistent in loops only. When tensors
do need to change shape across iterations, use `shape_invariants`.

Note: Shapes are allowed to be inconsistent in conditionals. The result will be
a partially dynamic shape.

In a `tf.while_loop` (and correspondingly, an AutoGraph `while` or `for` loop)
all loop variables must maintain consistent shape and dtype across iterations.
That is, every loop variable must have the same shape at the end of the loop
body as it had at the beginning of the loop body.

Example of illegal shape change in a loop:

```
def loop_body(x):  # x.shape is ()
  return tf.constant((1, 2, 3))  # Error -- inconsistent shapes: (), (3,)

x = tf.while_loop(
    lambda _: tf.random.uniform(()) > 0.5,
    loop_body,
    loop_vars=(tf.constant(1,))
```

The same restriction in AutoGraph code:

```
x = tf.constant(1,)
while tf.random.uniform(()) > 0.5:
  x = tf.constant((1, 2, 3))  # Error -- inconsistent shapes: (), (3,)
```

### Access to source code

Key point: AutoGraph can only handle functions whose source code can be
accessed at runtime.

Almost all Python functions allow access to their source code. However, a few
exceptions exist:

 * functions created in the Python interactive shell
 * functions with native bindings (these do not have Python source code)
 * functions created dynamically, using `exec` or `eval`

Use
[inspect.findsource](https://docs.python.org/3/library/inspect.html#inspect.findsource)
to quickly diagnose whether the source code is available for a function.

For example:

```
import inspect

def simple_function():
  return 1

# If this raises an error, then AutoGraph prints a warning.
# If it returns source code, then AutoGraph should work as well.
inspect.findsource(simple_function)
```

#### Source code of lambda functions

##### TF 2.4 and newer

Key Point: When nesting lambda functions, use distinguishing argument names
to avoid parse errors.

The Python runtime exposes the source code of lambda functions, however it
may omit parts of the actual body, or include surrounding code. This may make it
impossible to parse the exact source code of the lambda function (see
https://github.com/tensorflow/tensorflow/issues/39832).

AutoGraph uses alternate methods to parse the source code more robustly, but
in rare cases it may be unable to distinguish between nested lambda functions
of identical signatures.

Example:

```
l = lambda x: lambda x: x + 1
```

AutoGraph raises an error for the code above because the parser cannot
distinguish between the two function signatures. To work around this limitation,
use distinct argument names:

```
l = lambda outer_x: lambda inner_x: inner_x + 1
```

##### Before TF 2.3 and older

In older versions of TensorFlow, the loading code for lambda functions is not
robust. Follow the guidance below to avoid errors.

Important: Declare lambda functions on single lines to make sure their source
code loads correctly.

The Python runtime exposes the source code of lambda functions, however it
may omit parts of the actual body, or include surrounding code. This may make it
impossible to parse the exact source code of the lambda function.

For example, consider the declaration of a lambda function below:

```
foo = (
    lambda y: lambda x: x * y
    - y
)
```

The Python runtime will report the following source code for `foo`:

```
>>> inspect.getsource(foo)
'    lambda y: lambda x: x*y \n'
```

In other cases, the source code it returns is not valid Python code, resulting
in an error:

```
foo = (
 'bar',
 lambda: x)
```

The reported source code contains an invalid token `)`:

```
>>> inspect.getsource(foo[1])
' lambda: x)\n'
```

This shortcoming can be avoided by declaring the lambda in a single assignment
or return value, and avoiding placing it inside parentheses which could cause
auto-formatting tools to break it into multiple lines:


```
# Good - single assignment
my_lambda = lambda: x

# Good - single return
return lambda x, y: x*y - y
```

```
# Bad - wrapped in parentheses
my_lambda = (lambda x, y: x * y - y)

# Bad - inlined in another expression
foo(lambda x, y: x + y, bar)
```
