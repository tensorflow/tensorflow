TOOLS_ROOT = ../../..
include $(TOOLS_ROOT)/src/MakefileUnix.mak

OBJS = ExampleTestbench.o

all: $(BINDIR)/ExampleTestbench

$(BINDIR)/ExampleTestbench: $(OBJS)
	$(CPP) $(OBJS) -o $(BINDIR)/ExampleTestbench -L$(TOOLS_ROOT)/lib $(LIBS) -lxsidevice $(INCDIRS) $(EXTRALIBS)

%.o: %.cpp
	$(CPP) $(CPPFLAGS) -c $< -o $@ -I$(TOOLS_ROOT)/include

clean: 
	rm -rf $(OBJS)
	rm -rf $(BINDIR)/ExampleTestbench.*
