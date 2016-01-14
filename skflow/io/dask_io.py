"""Methods to allow dask.DataFrame."""
#  Copyright 2015-present Scikit Flow Authors. All Rights Reserved.
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

try:
    import dask.dataframe as dd
    HAS_DASK = True
except ImportError:
    HAS_DASK = False

def _add_to_index(df, start):
    """Make a new dask.dataframe where we add these values to the 
    index of each subdataframe. 
    """
    if not isinstance(df, dd.DataFrame):
        raise ValueError("df must be a dask.dataframe")
    df = df.copy()
    df.index = df.index + start
    return df

def _get_divisions(df):
    """Number of rows in each sub-dataframe"""
    if not isinstance(df, dd.DataFrame):
        raise ValueError("df must be a dask.dataframe")
    lengths = df.map_partitions(len).compute()
    divisions = np.cumsum(lengths).tolist()
    divisions.insert(0, 0)
    return divisions

def extract_dask_data(data):
    """Extract data from dask.Series for predictors"""
    if isinstance(data, dd.Series):
        data.divisions = tuple(range(len(data.divisions)))
    return data

def extract_dask_labels(labels):
    """Extract data from dask.Series for labels"""
    if isinstance(labels, dd.Series):
        if len(labels.columns) > 1:
            raise ValueError('Only one column for labels is allowed.')
        labels.divisions = tuple(range(len(labels.divisions)))
    else:
        return labels

