# Operator inputs and shapes
# If shape is None a default one dimension shape of (N, ) will be given
# Shape dimensions may be None
op_inputs = [
    ("uvw: FT", (100, 10, 3)),
    ("lm: FT", (75, None)),
    ("frequency: FT", (32,)),
    ("mapping: int32", None),
]

# Operator outputs and shapes
# Shape dimensions should not be None
op_outputs = [
    ("complex_phase: CT", (75, 100, 10, 32))
]

# Attributes specifying polymorphic types
op_type_attrs = [
"FT: {float, double} = DT_FLOAT",
    "CT: {complex64, complex128} = DT_COMPLEX64"]

# Any other attributes
op_other_attrs = []

# Operator documentation
op_doc = """Custom Operator"""
