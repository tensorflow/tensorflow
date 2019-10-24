TOOLS_ROOT = ../../..
include $(TOOLS_ROOT)/src/MakefileMac.mak

OBJS = ExamplePlugin.o

all: $(DLLDIR)/ExamplePlugin.so

$(DLLDIR)/ExamplePlugin.so: $(OBJS)
	$(CCPP) $(OBJS) -dynamiclib -o $(DLLDIR)/ExamplePlugin.so $(EXTRALIBS)

%.o: %.cpp
	$(CPP) $(CPPFLAGS) -c $< -o $@ -I$(TOOLS_ROOT)/include

clean: 
	rm -rf $(OBJS)
	rm -rf $(DLLDIR)/ExamplePlugin.*
