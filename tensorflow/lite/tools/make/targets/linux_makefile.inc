# Settings for Linux.
ifeq ($(TARGET), linux)
  CXXFLAGS += \
    -fPIC \
    -DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK \
    -pthread
  # TODO(petewarden): In the future we may want to add architecture-specific
  # flags like -msse4.2
	LIBS += -ldl
endif
