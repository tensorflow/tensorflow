PANDAS_DTYPES = {'int8': 'int', 'int16': 'int', 'int32': 'int', 'int64': 'int',
'uint8': 'int', 'uint16': 'int', 'uint32': 'int', 'uint64': 'int', 'float16': 'float',
'float32': 'float', 'float64': 'float', 'bool': 'i'}

def extract_pandas_data(data):
	"""Extract data from pandas.DataFrame"""
	if not isinstance(data, DataFrame):
		return data

	if all(dtype.name in PANDAS_DTYPES for dtype in data.dtypes):
		return data.values.astype('float')
	else:
		raise ValueError('Data types for data must be int, float, or bool.')




