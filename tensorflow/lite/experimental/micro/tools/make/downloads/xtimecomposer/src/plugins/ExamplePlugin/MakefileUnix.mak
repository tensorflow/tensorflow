TOOLS_ROOT = ../../..
include $(TOOLS_ROOT)/src/MakefileUnix.mak

OBJS = ExamplePlugin.o

all: $(DLLDIR)/ExamplePlugin$(DLLEXT)

$(DLLDIR)/ExamplePlugin$(DLLEXT): $(OBJS)
	$(CCPP) $(OBJS) -shared -o $(DLLDIR)/ExamplePlugin$(DLLEXT) $(LIBS) $(EXTRALIBS)

%.o: %.cpp
	$(CPP) $(CPPFLAGS) -c $< -o $@ -I$(TOOLS_ROOT)/include

clean: 
	rm -rf $(OBJS)
	rm -rf $(DLLDIR)/ExamplePlugin.*
