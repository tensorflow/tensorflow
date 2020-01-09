This is a simple project that can be used to generate 
a library file that can be included in your c projects.

Modify the micro_api.cc to suit your specific application. 

To generate the library file

`make lib`

you can link to libtensorflow-microlite.a and micro_api.h to use your model 
in your project

you can use the parse_micro_ops_function.py  in order to pull out of a tf_lite model
only the necessary functions. 

`fill_template_file(tf_lite_model)`

which will overwwrite the micro_api.cc file with a file containing only the operations
that need to be loaded by the micro_mutable_ops_resolver. You can then build a much
smaller library file to include in your project.








