# Utilities to help with wrapping of TF namespace
# This module helps obtain list of all gen op modules
# And will generate the correct order of wrapping Python op module
# (not all orders work because modules include each other)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ["python_op_module_list_sorted", "gen_op_module_list",
           "python_op_module_dep"]

from collections import defaultdict
from itertools import takewhile, count

gen_op_module_list = ["gen_array_ops","gen_candidate_sampling_ops","gen_control_flow_ops","gen_ctc_ops","gen_data_flow_ops","gen_functional_ops","gen_image_ops","gen_io_ops","gen_linalg_ops","gen_logging_ops","gen_math_ops","gen_nn_ops","gen_parsing_ops","gen_random_ops","gen_script_ops","gen_sparse_ops","gen_state_ops","gen_string_ops","gen_user_ops"]

python_op_module_dep =  {"array_ops": ["logging_ops"],
                      "check_ops": ["array_ops","control_flow_ops","logging_ops","math_ops"],
                      "clip_ops": ["array_ops","constant_op","math_ops","nn_ops"],
                      "constant_op": [],
                      "control_flow_ops": ["array_ops","constant_op","logging_ops","math_ops", "tensor_array_ops"],
                      "data_flow_ops": ["array_ops","control_flow_ops"],
                      "functional_ops": ["array_ops","constant_op","control_flow_ops","tensor_array_ops"],
                      "init_ops": ["array_ops","constant_op","math_ops","nn_ops","random_ops"],
                      "io_ops": [],
                      "linalg_ops": [],
                      "logging_ops": [],
                      "math_ops": ["array_ops","state_ops"],
                      "nn_ops": ["array_ops","math_ops","random_ops"],
                      "parsing_ops": ["array_ops","constant_op","control_flow_ops","logging_ops","math_ops"],
                      "random_ops": ["array_ops","control_flow_ops","logging_ops","math_ops"],
                      "session_ops": ["array_ops"],
                      "sparse_ops": ["array_ops","math_ops"],
                      "state_ops": [],
                      "tensor_array_ops": ["math_ops", "constant_op", "array_ops"]}



# from http://stackoverflow.com/a/15039202/419116

def _sort_topologically(graph):
    levels_by_name = {}
    names_by_level = defaultdict(set)

    def walk_depth_first(name):
        if name in levels_by_name:
            return levels_by_name[name]
        children = graph.get(name, None)
        level = 0 if not children else (1 + max(walk_depth_first(lname) for lname in children))
        levels_by_name[name] = level
        names_by_level[level].add(name)
        return level

    for name in graph:
        walk_depth_first(name)

    return list(takewhile(lambda x: x is not None, (names_by_level.get(i, None) for i in count())))


def python_op_module_list_sorted():
  """Returns list of Python op module in topological sort order"""

  levels = _sort_topologically(python_op_module_dep)
  import_order = []
  for level in levels:
    import_order.extend(level)
  return import_order
