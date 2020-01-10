"""Switch between Google or open source dependencies."""
# Switch between Google and OSS dependencies
USE_OSS = True

# Per-dependency switches determining whether each dependency is ready
# to be replaced by its OSS equivalence.
# TODO(danmane,mrry,opensource): Flip these switches, then remove them
OSS_APP = True
OSS_FLAGS = True
OSS_GFILE = True
OSS_GOOGLETEST = True
OSS_LOGGING = True
OSS_PARAMETERIZED = True
