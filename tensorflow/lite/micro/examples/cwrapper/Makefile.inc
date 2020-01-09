TENSORFLOW_LIB_SRCS := tensorflow/lite/micro/examples/cwrapper/micro_api.cc 

TENSORFLOW_LIB_HDRS := tensorflow/lite/micro/examples/cwrapper/micro_api.h

# Builds a standalone library.
$(eval $(call microlite_test,cwrapper,\
$(TENSORFLOW_LIB_SRCS),$(TENSORFLOW_LIB_HDRS)))
