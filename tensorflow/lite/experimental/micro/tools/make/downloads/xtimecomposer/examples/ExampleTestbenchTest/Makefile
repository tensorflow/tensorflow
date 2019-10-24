
all: build

#
# Compile the example
#
build:
	xcc -target=XS1-L16A-128-FB324-A10 ExampleTestbenchTest.xc -o ExampleTestbenchTest.xe

#
# Connect the pins together that map to XS1_PORT_1A and XS1_PORT_1B
# NOTE: Requires the testbench to have been built (src/testbenches/ExampleTestbench)
#
run:
	ExampleTestbench --connect 0 X0D00 0 X0D01 ExampleTestbenchTest.xe

#
# This will give a list of the pins which correspond to the ports
#
list:
	xsim --dump-ios ExampleTestbenchTest.xe
