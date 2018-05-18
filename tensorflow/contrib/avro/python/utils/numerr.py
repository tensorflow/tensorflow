from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
from tensorflow.contrib.avro.python.utils.tensor_utilities import to_list, flatten_lists


ALMOST_EQUALS_THRESHOLD = 5.96e-8  # Threshold for almost equality applied to floating point values


def relative_error_for_value(value_is, value_be):
    """
    For numeric values we define the relative error as:
        relative-error = | value_is - value_be | / (eps + |value_be|).

    For non-numeric values we define the relative error as:
        relative-error = value_is != value_be

    :param value_is: The actual value.
    :param value_be: The intended value.
    :return: The relative error.
    """
    if isinstance(value_be, (int, long, float, complex)):
        return abs(value_is-value_be) / (sys.float_info.epsilon+abs(value_be))
    else:
        return float(value_is != value_be)


def relative_error_for_dense_tensor(tensor_is, tensor_be):
    """
    Computes the relative error for the fixed length tensors.

    This method currently works only for 2D tensors, where the 1st dimension is typically the batch.

    :param tensor_is: The actual values.
    :param tensor_be: The intended values.
    :return: The maximum relative error over all values in tensors 'tensor_is' and 'tensor_be'.
    """
    list_is = flatten_lists(to_list(tensor_is))
    list_be = flatten_lists(to_list(tensor_be))

    return relative_error_for_list(list_is, list_be)


def relative_error_for_var_len_tensor(tensor_is, tensor_be):
    """
    Computes the relative error for the variable length tensors.
    :param tensor_is: The actual values.
    :param tensor_be: The intended values.
    :return: The maximum relative error over all values in tensors 'tensor_is' and 'tensor_be'.
    """
    err = 0.0
    # For these sparse tensors we need to select the proper row/column as defined by the indices.
    for index, value_is in zip(tensor_is.indices, tensor_is.values):
        row, col = index[0], index[1]
        value_be = tensor_be[row][col]
        err = max(err, relative_error_for_value(value_is, value_be))
    return err


def relative_error_for_sparse_tensor(tensor_is, tensor_be, key_be):
    """
    Computes the relative error for sparse tensors.
    :param tensor_is: The actual values.
    :param tensor_be: The intended values.
    :param key_be: The key with the 'index_key' and 'value_key' that were used to define indices and values in the
    sparse tensor.
    :return: The maximum relative error over all indices and values of the sparse tensors 'tensor_is' and 'tensor_be'.
    """
    err = 0.0
    i_offset = 0
    for i_row_be, tensor_row in enumerate(tensor_be):
        # Note that the sparse_tensor function returns values sorted by index from smallest to largest
        # Build this index by using all index values for this row
        tensors = flatten_lists(tensor_row)  # Filters these return an additional list that need to be flattened
        index = [record[key_be.index_key] for record in tensors]
        index = [i[0] for i in sorted(enumerate(index), key=lambda ii:ii[1])]
        # Pull records according to the sorted index
        for i_col_be, i_record in enumerate(index):
            record = tensors[i_record]
            i_linear = i_offset + i_col_be
            i_row_is, index_is = tensor_is.indices[i_linear]
            value_is = tensor_is.values[i_linear]
            index_be = record[key_be.index_key]
            value_be = record[key_be.value_key]
            err = max(err, float(i_row_is != i_row_be))
            err = max(err, float(index_is != index_be))
            err = max(err, relative_error_for_value(value_is, value_be))
        i_offset += len(tensors)
    return err


def relative_error_for_list(list_is, list_be):
    """
    Returns the maximum relative error.

    :param list_is: The list with the actual values.
    :param list_be: The list with the intended values.
    :return: The error value.
    """
    err = 0.0
    for value_is, value_be in zip(list_is, list_be):
        err = max(err, relative_error_for_value(value_is, value_be))
    return err


def almost_equal_value(value_is, value_be, th=ALMOST_EQUALS_THRESHOLD):
    """
    Checks for almost equality between values.

    :param value_is: The actual value.
    :param value_be: The intended value.
    :param th: The threshold applied to the relative error between the actual value and intended value.
    :return: True if the relative error is below threshold otherwise false.
    """
    return relative_error_for_value(value_is, value_be) < th


def almost_equal_list(list_is, list_be, th=ALMOST_EQUALS_THRESHOLD):
    """
    Checks for almost equal of values in the lists.
    :param list_is: Actual values.
    :param list_be: Intended values.
    :param th: The threshold applied to the relative error.
    :return: True if the relative error is below threshold otherwise false.
    """
    return relative_error_for_list(list_is, list_be) < th


def almost_equal_dense_tensor(tensor_is, tensor_be, th=ALMOST_EQUALS_THRESHOLD):
    """
    Compares fixed length tensors for almost equal.
    :param tensor_is: Actual fixed length tensor.
    :param tensor_be: Intended fixed length tensor.
    :param th: Threshold for almost equal.
    :return: Returns true if the relative error is below threshold, otherwise false.
    """
    return relative_error_for_dense_tensor(tensor_is, tensor_be) < th


def almost_equal_var_len_tensor(tensor_is, tensor_be, th=ALMOST_EQUALS_THRESHOLD):
    """
    Compares variable length tensors for almost equal.
    :param tensor_is: Actual variable length tensor.
    :param tensor_be: Intended variable length tensor.
    :param th: Threshold for almost equal.
    :return: Returns true if the relative error is below threshold, otherwise false.
    """
    return relative_error_for_var_len_tensor(tensor_is, tensor_be) < th


def almost_equal_sparse_tensor(tensor_is, tensor_be, key_be, th=ALMOST_EQUALS_THRESHOLD):
    """
    Compares sparse tensors for almost equal.
    :param tensor_is: Actual sparse tensor.
    :param tensor_be: Intended sparse tensor.
    :param key_be: The key for the indices, values, and dense shape for the 'tensor_be' sparse tensor.
    :param th: Threshold for almost equal.
    :return: Returns true if the relative error is below the threshold, otherwise false.
    """
    return relative_error_for_sparse_tensor(tensor_is, tensor_be, key_be) < th
