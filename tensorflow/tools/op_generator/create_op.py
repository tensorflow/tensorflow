import argparse
from collections import namedtuple
import itertools
import os
import re

import jinja2
from tensorflow.python.framework.dtypes import (
    _STRING_TO_TF,
    _TYPE_TO_STRING,
    _TF_TO_NP)

parser = argparse.ArgumentParser()
parser.add_argument('op_name')
parser.add_argument('-p', '--project', default='project')
parser.add_argument('-l', '--library', default='library')
args = parser.parse_args()

LIBRARY = args.library
PROJECT = args.project

# Set a shape for our variables
N = 1024
var_shape = (N, )

FIRST_CAP_RE = re.compile('(.)([A-Z][a-z]+)')
ALL_CAP_RE = re.compile('([a-z0-9])([A-Z])')

# Convert CamelCase op names to snake case
def camel_to_snake_case(name):
    s1 = FIRST_CAP_RE.sub(r'\1_\2', name)
    return ALL_CAP_RE.sub(r'\1_\2', s1).lower()

# Derive a C++ header guard from the header name
def header_guard(header_name):
    guard_str = header_name.replace('.', '_')
    return ''.join([LIBRARY, '_', guard_str]).upper()

def parse_inout(s, shape):
    var, type_ = tuple(c.strip() for c in s.split(":"))

    if "*" in type_:
        raise ValueError("Failed to parse '{}'. "
            "List lengths are not yet supported".format(s))

    TF_TYPES = _TYPE_TO_STRING.values()
    tf_type = "tensorflow::" + type_ if type_ in TF_TYPES else type_
    np_type = ("np." + _TF_TO_NP[_STRING_TO_TF[type_]].__name__
                    if type_ in _STRING_TO_TF else type_)

    shape = var_shape if shape is None else shape

    return var, type_, tf_type, np_type, shape

def parse_attr_type(s):
    var, type = tuple(c.strip() for c in s.split(":"))

    split = type.split("=")
    default = None if len(split) > 1 else split[1].strip()
    types = split[0].strip()

    if types.startswith("{") and types.endswith("}"):
        types = tuple(c.strip() for c in types[1:-1].split(","))
    else:
        types = tuple(types,)

    TF_TYPES = _TYPE_TO_STRING.values()
    tf_types = tuple("tensorflow::" + t if t in TF_TYPES else t for t in types)
    np_types = ("np." + _TF_TO_NP[_STRING_TO_TF[t]].__name__
                    if t in _STRING_TO_TF else type_ for t in types)

    return s, var, types, tf_types, np_types, default

def strip_and_split(s, sep):
    return (c.strip() for c in s.split(sep))

InOut = namedtuple("InOut", ["name", "type",
    "tf_type", "np_type", "shape"])
Attr = namedtuple("Attr", ["original", "name", "types",
    "tf_types", "np_types", "default"])

from op_config import (op_inputs, op_outputs,
    op_type_attrs, op_other_attrs, op_doc)

# Parse input ops
op_inputs = [InOut(*parse_inout(i, s)) for i, s in op_inputs]

# Parse output ops
op_outputs = [InOut(*parse_inout(o, s)) for o, s in op_outputs]

# Parse type constrained attrs
op_type_attrs = [Attr(*parse_attr_type(a)) for a in op_type_attrs]

type_constraints = [[t for t in a.np_types]for a in op_type_attrs]

# Permute the type constraints
op_type_perms = itertools.product(*(a.tf_types for a in op_type_attrs))
op_type_perms = [list(p) for p in op_type_perms]

# Snake case python version of the operator
py_op_name = camel_to_snake_case(args.op_name)

# Create dictionary with variables required for creating the templates
D = {
    'op_name' : args.op_name,
    'py_op_name' : py_op_name,
    'project' : PROJECT,
    'library' : LIBRARY,
    'shared_library' : ''.join([LIBRARY, '.so']),
}

D.update({
    'op_inputs' : op_inputs,
    'op_outputs' : op_outputs,
    'op_type_attrs' : op_type_attrs,
    'op_other_attrs' : op_other_attrs,
    'op_type_perms' : op_type_perms,
    'type_constraints' : type_constraints,
    'op_doc' : op_doc,
})

# Filenames
D.update({
    'main_header_file' : ''.join([py_op_name, '_op.h']),
    'cpp_header_file' : ''.join([py_op_name, '_op_cpu.h']),
    'cpp_source_file' : ''.join([py_op_name, '_op_cpu.cpp']),
    'cuda_header_file' : ''.join([py_op_name, '_op_gpu.cuh']),
    'cuda_source_file' : ''.join([py_op_name, '_op_gpu.cu']),
    'python_test_file' : ''.join(['test_', py_op_name, '.py']),
    'makefile' : 'Makefile',
})

# C++ header guards
D.update({
    'main_header_guard' : header_guard(D['main_header_file']),
    'cpp_header_guard' : header_guard(D['cpp_header_file']),
    'cuda_header_guard' : header_guard(D['cuda_header_file']),
})

NB = '_namespace_begin'
NE = '_namespace_stop'

# C++ namespace
D.update({
    'project_namespace_start' : ''.join([PROJECT, NB]).upper(),
    'project_namespace_stop' : ''.join([PROJECT, NE]).upper(),
    'op_namespace_start' : ''.join([PROJECT, '_', py_op_name, NB]).upper(),
    'op_namespace_stop' : ''.join([PROJECT, '_', py_op_name, NE]).upper(),
})

# CUDA kernel
D.update({
    'kernel_name' : ''.join([LIBRARY, '_', py_op_name]),
})


jinja_loader = jinja2.FileSystemLoader('templates')
jinja_env = jinja2.Environment(loader=jinja_loader,
    trim_blocks=False, lstrip_blocks=False)

# Create a filter for formatting a list
jinja_env.filters['format_list'] = lambda l, p: [p % s for s in l]

# Create library directory if it does not exist
if not os.path.exists(LIBRARY):
    os.makedirs(LIBRARY)

def render(template, output):
    """ Hook to render template file to output """
    with open(os.path.join(LIBRARY, D[output]), 'w') as f:
        header_template = jinja_env.get_template(template)
        f.write(header_template.render(**D))

render('main_header.j2', 'main_header_file')
render('cpp_header.j2', 'cpp_header_file')
render('cpp_source.j2', 'cpp_source_file')
render('cuda_header.j2', 'cuda_header_file')
render('cuda_source.j2', 'cuda_source_file')
render('test_source.j2', 'python_test_file')
render('Makefile.j2', 'makefile')
