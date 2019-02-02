# Capabilities and Limitations

TF AutoGraph converts Eager Python code into TensorFlow graph-mode code. For example, users write code with `if` and `while` and AutoGraph automatically converts it into the equivalent `tf.cond`, and `tf.while_loop`.

Python is a large language, so hoping to convert arbitrary Python code directly to TF graphs is overly ambitious. However, the Python code written to metaprogram TF graphs is in practice a restricted subset. We aim to support as much of this subset as possible. The table below lays out what we currently handle, what we hope to support, and what we have no plans to support.

# Python Language Support Status

Note: as more complex features in TensorFlow are made more accessible using AutoGraph, we expect to come across use cases that haven't been tried before, some of which might reveal rare bugs. If we do find any such bugs, we may add additional restrictions for the affected configurations, until those bugs are resolved.

Construct                   | Supported now? | Plan to support? | Notes
:-------------------------- | :------------: | :--------------: | :----
If statement                | Yes            |                  | Converts to `tf.cond`. If variables are created in one branch that don’t exist in another, which is inexpressible in TF, we throw a clear error.
For statement               | Yes            |                  | We will specialize `for` loops with unknown and known lengths, as well as for loops over TF datasets. Converts to `tf.while_loop`, with an additional `maximum_iterations` hint, if that is known. Creating variables inside the loop that are used later outside the loop is not supported, as the loop may have no iterations.
While statement             | Yes            |                  | Converts to `tf.while_loop`. Creating variables inside the loop is not supported, as the loop may have no iterations.
Continue and break          | Yes            |                  | Converts to boolean flags and extra predicates in loop tests.
Composition of control flow | Yes            |                  | Arbitrary composition of `if`, `while`, `for`, `break`, and `continue`, along with other supported language elements, is supported and tested.
Iterators                   | Some           | Yes              | Not all iterators supported, but we plan to support everything that can be desugared, such as `enumerate` and `zip`.
Multiple return values      | Yes            |                  | We desugar them into variables, boolean flags and conditionals so that the function has a single return value at the end, and provide a clear error if we are unable to do so.
Print expression            | Yes            |                  | Wrapped in `PyFunc`, and given proper control dependencies. Optional support for using tf.Log when py_func is undesirable exists.
Static function calls       | Yes            |                  | Non-recursive function calls
Nested call trees           | Yes            |                  | For example, `f` calls `g` which calls `h`, all of which need conversion.
Recursive function calls    | No             | Maybe            | Based on available support in TF. Currently `function.Defun` is the best candidate, but it is not reentrant.
Python built-ins            | Some           | Yes              | `print`, `len`, `range`, `xrange`, `int`, `float` are supported, and we plan to support or clearly error on all [Python built-ins](https://docs.python.org/3/library/functions.html).
List operations             | Yes            |                  | We convert list creation, append, pop and indexing to their TF TensorArray equivalents. However, we do need some extra type hints to fully convert correctly. We hope to remove this limitation.
Function variables          | Yes            |                  | e.g. `f_new = f_orig; f_new()`
Lambda functions            | No             | Yes              | Planned feature.
Classes                     | Yes            |                  | Classes can be converted all at once, or method-by-method. Some limitations exist around static and class methods.
Subclasses                  | Yes            |                  | Subclassing library objects like tf.keras.Model is also supported.
Dynamic types               | Some           |                  | `o = C1() if foo else C2(); o.bar()`. Some scenarios where types are data-dependent may not be supported. We will raise a meaningful error in that case.
Dynamic code / exec         | No             |                  |
Reflection                  | No             |                  |
Try / Except                | No             | No               | No current sane TF equivalent.
Global variables            | Restricted     |                  | In general, we only support read-only access to arguments or variables defined outside the converted code. A few exceptions include TensorFlow library code.
Functions with side effects | Some           |                  | Side effects are allowed, under certain circumstances.
Collections                 | Some           | Yes              | We currently support lists. There are currently no TF equivalents of dictionaries or tuples.
List Comprehensions         | Yes            |                  | We desugar `ListComp` into the appropriate combination of `For` and `If` statements. Other comprehensions are currently very low priority.
Custom context managers     | No             | Yes              | Currently low priority. Left unconverted currently.
Generators                  | No             | Maybe            | Could be achievable using queues; very low priority.
Assertions                  | Yes            |                  | As `tf.Assert`
Deletion                    | Yes            | Maybe            | Currently unconverted. If new semantics are required for `del`, we are able to add it in.
Inline imports              | No             | Yes              | For example, `import numpy as np; np.eye(3)`. Currently low priority.
Async                       | No             | No               |

## Extra capabilities

 - We liberally add name scopes to generated functions
 - Operations get decent default names everywhere (planned)
 - Statements that have no output values are given correct control dependencies. For example, `for i in range(n): print(i)` will have control dependencies to ensure the `print` statements are executed serially.

