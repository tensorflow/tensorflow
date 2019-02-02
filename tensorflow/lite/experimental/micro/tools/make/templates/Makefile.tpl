SRCS := \
%{SRCS}%

OBJS := \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(SRCS)))

INCLUDES := \
-I. \
-I./third_party/gemmlowp \
-I./third_party/flatbuffers/include

CXXFLAGS += %{CXX_FLAGS}%

LDFLAGS += %{LINKER_FLAGS}%

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

%{EXECUTABLE}% : $(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) \
	-o $@

all: %{EXECUTABLE}%
