#!/bin/sh

# Testing C# on Linux using Mono.

mcs -debug -out:./fbnettest.exe \
  ../../net/FlatBuffers/*.cs ../MyGame/Example/*.cs ../MyGame/*.cs ../union_vector/*.cs \
  FlatBuffersTestClassAttribute.cs FlatBuffersTestMethodAttribute.cs Assert.cs FlatBuffersExampleTests.cs Program.cs ByteBufferTests.cs FlatBufferBuilderTests.cs FlatBuffersFuzzTests.cs FuzzTestData.cs Lcg.cs TestTable.cs
mono --debug ./fbnettest.exe
rm fbnettest.exe
rm Resources/monsterdata_cstest.mon
rm Resources/monsterdata_cstest_sp.mon

# Repeat with unsafe versions

mcs -debug -out:./fbnettest.exe \
  -unsafe -d:UNSAFE_BYTEBUFFER \
  ../../net/FlatBuffers/*.cs ../MyGame/Example/*.cs ../MyGame/*.cs ../union_vector/*.cs\
  FlatBuffersTestClassAttribute.cs FlatBuffersTestMethodAttribute.cs Assert.cs FlatBuffersExampleTests.cs Program.cs ByteBufferTests.cs FlatBufferBuilderTests.cs FlatBuffersFuzzTests.cs FuzzTestData.cs Lcg.cs TestTable.cs
mono --debug ./fbnettest.exe
rm fbnettest.exe
rm Resources/monsterdata_cstest.mon
rm Resources/monsterdata_cstest_sp.mon

