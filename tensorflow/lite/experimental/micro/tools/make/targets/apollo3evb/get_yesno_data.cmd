file ../../gen/apollo3evb_cortex-m4/bin/preprocessor_test
target remote localhost:2331
load ../../gen/apollo3evb_cortex-m4/bin/preprocessor_test
monitor reset
break preprocessor_test.cc:35
break preprocessor_test.cc:51
c
dump verilog value yes_calculated_data.txt yes_calculated_data
c
dump verilog value no_calculated_data.txt no_calculated_data
