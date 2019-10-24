
all: build

#
# Compile the example
#
build:
	xcc -target=XS1-L16A-128-FB324-A10 ExamplePluginTest.xc -o ExamplePluginTest.xe

#
# Connect the pins together that map to XS1_PORT_1A and XS1_PORT_1B
# NOTE: Requires the plugin to have been built (src/plugins/ExamplePlugin)
#
run:
	xsim --max-cycles 100000 --plugin ExamplePlugin.dll "0 X0D00 0 X0D01" ExamplePluginTest.xe

#
# This will give a list of the pins which correspond to the ports
#
list:
	xsim --dump-ios ExamplePluginTest.xe
