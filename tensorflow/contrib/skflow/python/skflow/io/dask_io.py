"""Methods to allow dask.DataFrame."""
#  Copyright 2015-present The Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


try:
    import dask.dataframe as dd
    allowed_classes = (dd.Series, dd.DataFrame)
    HAS_DASK = True
except ImportError:
    HAS_DASK = False


def _add_to_index(df, start):
    """Make a new dask.dataframe where we add these values to the
    index of each subdataframe.
    """
    df = df.copy()
    df.index = df.index + start
    return df


def _get_divisions(df):
    """Number of rows in each sub-dataframe"""
    lengths = df.map_partitions(len).compute()
    divisions = np.cumsum(lengths).tolist()
    divisions.insert(0, 0)
    return divisions


def _construct_dask_df_with_divisions(df):
    """Construct the new task graph and make a new dask.dataframe around it"""
    divisions = _get_divisions(df)
    name = 'csv-index' + df._name
    dsk = {(name, i): (_add_to_index, (df._name, i), divisions[i]) for i in range(df.npartitions)}
    from toolz import merge
    if isinstance(df, dd.DataFrame):
        return dd.DataFrame(merge(dsk, df.dask), name, df.columns, divisions)
    elif isinstance(df, dd.Series):
        return dd.Series(merge(dsk, df.dask), name, df.name, divisions)


def extract_dask_data(data):
    """Extract data from dask.Series or dask.DataFrame for predictors"""
    if isinstance(data, allowed_classes):
        return _construct_dask_df_with_divisions(data)
    else:
        return data


def extract_dask_labels(labels):
    """Extract data from dask.Series for labels"""
    if isinstance(labels, dd.DataFrame):
        ncol = labels.columns
    elif isinstance(labels, dd.Series):
        ncol = labels.name
    if isinstance(labels, allowed_classes):
        if len(ncol) > 1:
            raise ValueError('Only one column for labels is allowed.')
        return _construct_dask_df_with_divisions(labels)
    else:
        return labels
