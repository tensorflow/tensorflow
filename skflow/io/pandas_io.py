"""Methods to allow pandas.DataFrame."""
# pylint: disable=unused-import
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

PANDAS_DTYPES = {'int8': 'int', 'int16': 'int', 'int32': 'int', 'int64': 'int',\
'uint8': 'int', 'uint16': 'int', 'uint32': 'int', 'uint64': 'int', 'float16': 'float',\
'float32': 'float', 'float64': 'float', 'bool': 'i'}

def extract_pandas_data(data):
    """Extract data from pandas.DataFrame for predictors"""
    if not isinstance(data, pd.DataFrame):
        return data

    if all(dtype.name in PANDAS_DTYPES for dtype in data.dtypes):
        return data.values.astype('float')
    else:
        raise ValueError('Data types for data must be int, float, or bool.')

def extract_pandas_labels(labels):
    """Extract data from pandas.DataFrame for labels"""
    if isinstance(labels, pd.DataFrame): # pandas.Series also belongs to DataFrame
        if len(labels.columns) > 1:
            raise ValueError('Only one column for labels is allowed.')

        if all(dtype.name in PANDAS_DTYPES for dtype in labels.dtypes):
            return labels.values.astype('float')
        else:
            raise ValueError('Data types for labels must be int, float, or bool.')
    else:
        return labels

