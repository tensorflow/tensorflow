TOOLS_ROOT = ../../..
!INCLUDE $(TOOLS_ROOT)/src/MakefilePc.mak

OBJS = ExampleTestbench.obj
LIBS = $(TOOLS_ROOT)/lib/xsidevice.lib

all: $(BINDIR)/ExampleTestbench.exe

"$(BINDIR)/ExampleTestbench.exe": $(OBJS)
    @echo Linking...
    $(LINK32) @<<
    $(EXE32_FLAGS) /out:"$(BINDIR)/ExampleTestbench.exe" $(OBJS) $(LIBS)
<<

.cpp{}.obj::
    $(CPP) @<<
    $(CFLAGS) -I$(TOOLS_ROOT)/include $<
<<

clean:
    -@rm $(OBJS) *.idb *.pdb 2> NUL
    -@rm $(DLLDIR)/ExampleTestbench.* 2> NUL
 