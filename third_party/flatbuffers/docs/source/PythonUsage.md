Use in Python    {#flatbuffers_guide_use_python}
=============

## Before you get started

Before diving into the FlatBuffers usage in Python, it should be noted that the
[Tutorial](@ref flatbuffers_guide_tutorial) page has a complete guide to general
FlatBuffers usage in all of the supported languages (including Python). This
page is designed to cover the nuances of FlatBuffers usage, specific to
Python.

You should also have read the [Building](@ref flatbuffers_guide_building)
documentation to build `flatc` and should be familiar with
[Using the schema compiler](@ref flatbuffers_guide_using_schema_compiler) and
[Writing a schema](@ref flatbuffers_guide_writing_schema).

## FlatBuffers Python library code location

The code for the FlatBuffers Python library can be found at
`flatbuffers/python/flatbuffers`. You can browse the library code on the
[FlatBuffers GitHub page](https://github.com/google/flatbuffers/tree/master/
python).

## Testing the FlatBuffers Python library

The code to test the Python library can be found at `flatbuffers/tests`.
The test code itself is located in [py_test.py](https://github.com/google/
flatbuffers/blob/master/tests/py_test.py).

To run the tests, use the [PythonTest.sh](https://github.com/google/flatbuffers/
blob/master/tests/PythonTest.sh) shell script.

*Note: This script requires [python](https://www.python.org/) to be
installed.*

## Using the FlatBuffers Python library

*Note: See [Tutorial](@ref flatbuffers_guide_tutorial) for a more in-depth
example of how to use FlatBuffers in Python.*

There is support for both reading and writing FlatBuffers in Python.

To use FlatBuffers in your own code, first generate Python classes from your
schema with the `--python` option to `flatc`. Then you can include both
FlatBuffers and the generated code to read or write a FlatBuffer.

For example, here is how you would read a FlatBuffer binary file in Python:
First, import the library and the generated code. Then read a FlatBuffer binary
file into a `bytearray`, which you pass to the `GetRootAsMonster` function:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.py}
    import MyGame.Example as example
    import flatbuffers

    buf = open('monster.dat', 'rb').read()
    buf = bytearray(buf)
    monster = example.GetRootAsMonster(buf, 0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now you can access values like this:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.py}
    hp = monster.Hp()
    pos = monster.Pos()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Support for Numpy arrays

The Flatbuffers python library also has support for accessing scalar
vectors as numpy arrays. This can be orders of magnitude faster than
iterating over the vector one element at a time, and is particularly
useful when unpacking large nested flatbuffers. The generated code for
a scalar vector will have a method `<vector name>AsNumpy()`. In the
case of the Monster example, you could access the inventory vector
like this:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.py}
    inventory = monster.InventoryAsNumpy()
    # inventory is a numpy array of type np.dtype('uint8')
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

instead of

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.py}
    inventory = []
    for i in range(monster.InventoryLength()):
        inventory.append(int(monster.Inventory(i)))
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Numpy is not a requirement. If numpy is not installed on your system,
then attempting to access one of the `*asNumpy()` methods will result
in a `NumpyRequiredForThisFeature` exception.

## Text Parsing

There currently is no support for parsing text (Schema's and JSON) directly
from Python, though you could use the C++ parser through SWIG or ctypes. Please
see the C++ documentation for more on text parsing.

<br>
