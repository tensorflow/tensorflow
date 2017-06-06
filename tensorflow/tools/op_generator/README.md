# Tensorflow Custom Operator Code Outline Generator

Writing a tensorflow operator requires writing fair amounts of boilerplate C++ and CUDA code.
This script generates code for the CPU and GPU version of a tensorflow operator.
More specifically, given tensorflow `inputs`, `outputs` and `attribute`s, it generates:

* C++ Header file that defines the operator class, templated on Device.
* C++ Header file that defines the CPU implementation of the operator.
* C++ Source file with Shape Function, REGISTER_OP and REGISTER_KERNEL_BUILDER constructs.
* Cuda Header that defines the GPU implementation of the operator, including a CUDA kernel.
* Cuda Source file with GPU REGISTER_KERNEL_BUILDER's for the operator.
* python unit test case, which constructs random input data, and calls the operator.
* Makefile for compiling the operator into a shared library, using g++ and nvcc.

## Requirements

The jinja2 templating engine is required, as well as a tensorflow installation for building the operator.

```bash
pip install jinja2
```

## Usage

The user should edit the `op_config.py` file and define the operator:

* inputs and optionally, their shapes.
* outputs and optionally, their outputs.
* polymorphic type attributes.
* other attributes.
* documentation.

Once complete the script can be called as follows

```bash
$ python create_op.py --project=tensorflow --library=custom MyCustomOperator
```

to create the following directory structure

```bash
$ tree custom/
custom/
├── custom_op_op_cpu.cpp
├── custom_op_op_cpu.h
├── custom_op_op_gpu.cu
├── custom_op_op_gpu.cuh
├── custom_op_op.h
├── Makefile
└── test_custom_op.py
```

The `--project` and `--library` flags specify C++ namespaces within which the operator is created. Additionally, the Makefile will created a `custom.so` that can be loaded with `tf.load_op_library('custom.so')`.


The operator inputs and their optional shapes should be specified as a list of tuples. If concrete dimensions are specified, corresponding checks will be generated in the Shape Function associated with the operator. If `None` is supplied, a shape of `(N, )` where `N=1024` is assumed.

```python
# Operator inputs and shapes
# If shape is None a default one dimensional shape of (N, ) will be given
# Shape dimensions may be None, in which case they will not be checked
op_inputs = [
    ("uvw: FT", (100, 10, 3)),
    ("lm: FT", (75, None)),
    ("frequency: FT", (32,)),
    ("mapping: int32", None),
]
```

Similarly the operator outputs and their shapes should be specified as a list of tuples. Dimensions may not be None as memory allocations for the outputs will be created in the CPU and GPU ops.

```python
# Operator outputs and shapes
# Shape dimensions should not be None
op_outputs = [
    ("complex_phase: CT", (75, 100, 10, 32))
]
```

Given these inputs and outputs, CPU and GPU operators are created with named variables corresponding to the inputs and outputs. Additionally, a CUDA kernel with the given inputs and outputs is created, as well as a shape function checking the rank and dimensions of the supplied inputs.

Next, polymorphic type attributes should be supplied. The generator will template the operators on type attributes. It will also generate concrete permutations of REGISTER_KERNEL_BUILDER for both the CPU and GPU op using the actual types supplied in the type attributes (float, double, complex64 and complex128) below.

```python
# Attributes specifying polymorphic types
op_type_attrs = [
    "FT: {float, double} = DT_FLOAT",
    "CT: {complex64, complex128} = DT_COMPLEX64"]
```

Other attributes may be specified (and will be output in the REGISTER_OP) directive, but are not catered for automatically by the generator code as the range of attribute behaviour is complex.

```python
# Any other attributes
op_other_attrs = [
    "iterations: int32 >= 2",
]
```

Finally operator documentation may also be supplied.

```python
# Operator documentation
op_doc = """Custom Operator"""
```
