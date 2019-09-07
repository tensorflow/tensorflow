SRCS =  c_api.cc c_api_experimental.cc
OBJS = $(subst .cc,.o,$(subst .cxx,.o,$(subst .cpp,.o,$(SRCS))))
TENSORFLOW_ROOT = $(shell go env GOPATH)/src/github.com/tensorflow/tensorflow
CXXFLAGS = -DTF_COMPILE_LIBRARY -I$(TENSORFLOW_ROOT) -I$(TENSORFLOW_ROOT)/tensorflow/lite/tools/make/downloads/flatbuffers/include -fPIC
TARGET = libtensorflowlite_c

ifeq ($(OS),Windows_NT)
	OS_ARCH = windows_x86_64
	TARGET_SHARED := $(TARGET).dll
else
	TARGET_SHARED := $(TARGET).so
	ifeq ($(shell uname -m),x86_64)
		OS_ARCH = linux_x86_64
	else
	ifeq ($(shell uname -m),armv6l)
		OS_ARCH = linux_armv6l
	else
		OS_ARCH = rpi_armv7l
	endif
	endif
endif

LDFLAGS += -L$(TENSORFLOW_ROOT)/tensorflow/lite/tools/make/gen/$(OS_ARCH)/lib
LIBS = -ltensorflow-lite

.SUFFIXES: .cpp .cxx .o

all : $(TARGET_SHARED)

$(TARGET_SHARED) : $(OBJS)
		g++ -shared -o $@ $(OBJS) $(LDFLAGS) $(LIBS)
.cxx.o :
		g++ -std=c++14 -c $(CXXFLAGS) -I. $< -o $@
.cpp.o :
		g++ -std=c++14 -c $(CXXFLAGS) -I. $< -o $@
clean :
		rm -f *.o $(TARGET_SHARED)
