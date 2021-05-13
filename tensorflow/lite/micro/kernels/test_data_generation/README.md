# Background

As a Custom operator, detection_postprocess is using Flexbuffers library. In the
unit test there is a need to use flexbuffers::Builder since the operator itself
use flexbuffers::Map. However flexbuffers::Builder can not be used for most
targets (basically only on X86), since it is using std::vector and std::map.
Therefore the flexbuffers::Builder data is pregenerated on X86.

# How to generate new data:

~~~
    ```g++ -I../../../micro/tools/make/downloads/flatbuffers/include generate_flexbuffers_data.cc && ./a.out > ../flexbuffers_generated_data.cc```

~~~
