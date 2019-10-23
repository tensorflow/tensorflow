TOOLS_ROOT = ../../..
!INCLUDE $(TOOLS_ROOT)/src/MakefilePc.mak

OBJS = ExamplePlugin.obj

all: $(DLLDIR)/ExamplePlugin.dll

"$(DLLDIR)/ExamplePlugin.dll": $(OBJS)
    $(LINK32) $(LINK32_LIBS) /DLL /nologo /out:"$(DLLDIR)/ExamplePlugin.dll" @<<
    $(LINKFLAGS) $(OBJS)
<<

.cpp{}.obj::
    $(CPP) @<<
    $(CFLAGS) -I$(TOOLS_ROOT)/include $<
<<

clean:
    -@rm $(OBJS) *.idb *.pdb 2> NUL
    -@rm $(DLLDIR)/ExamplePlugin.* 2> NUL
 