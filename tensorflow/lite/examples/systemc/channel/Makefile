SYSTEMC_HOME    = $(HOME)/Development/xilinx/systemc/systemc-2.3.1/install
TARGET_ARCH     = linux64

SYSTEMC_INC_DIR = $(SYSTEMC_HOME)/include
SYSTEMC_LIB_DIR = $(SYSTEMC_HOME)/lib-$(TARGET_ARCH)

# LIBTOOL Setting
#SHELL           = /bin/bash
#LIBTOOL_DIR     = $(HOME)/anaconda3/bin/
#LIBTOOL         = $(SHELL) $(LIBTOOL_DIR)/libtool

GXX             = g++
GCC             = gcc
LD              = g++

FLAGS           = -g -Wall -I$(SYSTEMC_INC_DIR) 

LDFLAGS         = -static -L$(SYSTEMC_LIB_DIR) -lsystemc -lm -lpthread 

SRCS = main.cpp
OBJS=$(SRCS:.cpp=.o)
INCS=stack_if.h  stack.h producer.h consumer.h

# main
main: $(OBJS)
	g++ -o $@ $(OBJS) $(LDFLAGS)

main.o : main.cpp $(INCS)
	g++ -Wall -c $(FLAGS) $<

clean:
	rm -f main *.o *.lo
	rm -rf .libs
	rm -f *.vcd
