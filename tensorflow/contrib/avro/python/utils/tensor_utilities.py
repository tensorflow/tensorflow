from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np


def is_list_of_lists(lists):
    """
    Checks is this list contains as elements all lists.

    :param lists: The possibly lists.
    :return: True if list contains all elements as lists otherwise false.
    """
    return all(isinstance(l, list) for l in lists)


def to_list(tensor_or_list):
    """
    Convert the tensor which could be a numpy ND array into a list of lists, one list per dimension. If a list is passed
    nothing happens.

    :param tensor_or_list: A tensor or list.
    :return: A tensor is converted into a list.
    """
    if isinstance(tensor_or_list, np.ndarray):
        return tensor_or_list.tolist()
    return tensor_or_list


def flatten_lists(lists):
    """
    Flattens all list elements in this list.

    :param lists: List of lists.
    :return: Flatten list.
    """
    return [item for sublist in lists for item in sublist] if is_list_of_lists(lists) else lists


def get_end_indices(lists):
    """
    Gets the end indices for all 0D, 1D, 2D tensors.

    :param lists: The possibly lists.
    :return: If a scalar [1].
             If a list [1]*len(lists)
             If a list of lists [len(list[0]) len(list[0])+len(list[1]) ...]
    """
    lists = to_list(lists)
    if is_list_of_lists(lists):
        # Get the length of each row
        indices = [len(row) for row in lists]
    elif isinstance(lists, list):
        # Each row is a scalar and has length 1
        indices = [1]*len(lists)
    else:
        # Single row with one scalar
        indices = [1]
    cum_index = 0
    for index in range(len(indices)):
        cum_index += indices[index]
        indices[index] += cum_index
    return indices


def fill_in_fixed_len_sequence(lists, end_indices, n_elements_per_batch, default_value):
    """
    Fills in lists for fixed length sequence with default value.  Here the 'default_value' is a scalar.

    Notice, the 'default_value' are only used if we need to fill in values.  This is important because if we do not
    need to fill in values the default_values can be None.

    NOTE: This only works for 0D, 1D and 2D lists.

    :param lists: The lists representing a 2D array.
    :param end_indices: The end indices.
    :param n_elements_per_batch: The number of elements in a batch/row.
    :param default_value: The default value. Can be None.
    :return: Lists with all rows have 'n_elements_per_batch' values with defaults filled in.
    """
    return fill_in_fixed_len(lists=lists, end_indices=end_indices, n_elements_per_batch=n_elements_per_batch,
                             default_values=[default_value]*n_elements_per_batch)


def fill_in_fixed_len(lists, end_indices, n_elements_per_batch, default_values):
    """
    Fill in lists for fixed length with default values.  Here 'default_values' is a list or list of lists.

    Notice, the 'default_value' are only used if we need to fill in values.  This is important because if we do not
    need to fill in values the default_values can be None.

    NOTE: This ONLY works for 0D, 1D and 2D lists.

    :param lists: The lists.
    :param end_indices: The end indicies.
    :param n_elements_per_batch: The number of elements per batch/row.
    :param default_values: The default values. There must 'n_elements_per_batch' many default values.
    :return: Lists with all rows have 'n_elements_per_batch' values with defaults filled in.
    """
    total = 0
    for i_row, end_index in enumerate(end_indices):
        n_elems = end_index - total
        total = end_index
        if n_elems < n_elements_per_batch:
            lists[i_row] += default_values[n_elems:n_elements_per_batch]
    return lists


def get_n_elements_per_batch(lists):
    """
    Get the number of elements in a batch.

    NOTE: This ONLY works for 0D, 1D and 2D lists.

    :param lists: The lists that represent the tensor.
    :return: The number of elements in a batch.
    """
    lists = to_list(lists)
    return len(lists[0]) if isinstance(lists, list) and is_list_of_lists(lists) else 1
