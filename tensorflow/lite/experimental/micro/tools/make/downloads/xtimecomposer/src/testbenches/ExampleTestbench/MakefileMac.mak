TOOLS_ROOT = ../../..
include $(TOOLS_ROOT)/src/MakefileMac.mak

OBJS = ExampleTestbench.o
LIBS = $(TOOLS_ROOT)/lib/libxsidevice.so

all: $(BINDIR)/ExampleTestbench

$(BINDIR)/ExampleTestbench: $(OBJS)
	$(CPP) $(OBJS) -o $(BINDIR)/ExampleTestbench $(LIBS) $(INCDIRS) $(EXTRALIBS)

%.o: %.cpp
	$(CPP) $(CPPFLAGS) -c $< -o $@ -I$(TOOLS_ROOT)/include

clean: 
	rm -rf $(OBJS)
	rm -rf $(BINDIR)/ExampleTestbench.*
