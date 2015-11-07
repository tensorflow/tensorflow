"""Tests for tensorflow.python.ops.op_def_library."""

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import types
from tensorflow.python.ops.op_def_library import OpDefLibrary
from tensorflow.python.platform import googletest


# NOTE(mrry): Dummy shape registrations for ops used in the tests.
ops.RegisterShape("Attr")(None)
ops.RegisterShape("AttrBool")(None)
ops.RegisterShape("AttrBoolList")(None)
ops.RegisterShape("AttrDefault")(None)
ops.RegisterShape("AttrEmptyListDefault")(None)
ops.RegisterShape("AttrEnum")(None)
ops.RegisterShape("AttrEnumList")(None)
ops.RegisterShape("AttrFloat")(None)
ops.RegisterShape("AttrListDefault")(None)
ops.RegisterShape("AttrListMin")(None)
ops.RegisterShape("AttrMin")(None)
ops.RegisterShape("AttrShape")(None)
ops.RegisterShape("AttrShapeList")(None)
ops.RegisterShape("Binary")(None)
ops.RegisterShape("ComplexStruct")(None)
ops.RegisterShape("InPolymorphicTwice")(None)
ops.RegisterShape("MixedStruct")(None)
ops.RegisterShape("NInPolymorphicTwice")(None)
ops.RegisterShape("NInTwice")(None)
ops.RegisterShape("NInTwoTypeVariables")(None)
ops.RegisterShape("NIntsIn")(None)
ops.RegisterShape("NIntsOut")(None)
ops.RegisterShape("NIntsOutDefault")(None)
ops.RegisterShape("NPolymorphicIn")(None)
ops.RegisterShape("NPolymorphicOut")(None)
ops.RegisterShape("NPolymorphicOutDefault")(None)
ops.RegisterShape("NPolymorphicRestrictIn")(None)
ops.RegisterShape("NPolymorphicRestrictOut")(None)
ops.RegisterShape("OutT")(None)
ops.RegisterShape("OutTypeList")(None)
ops.RegisterShape("OutTypeListRestrict")(None)
ops.RegisterShape("Polymorphic")(None)
ops.RegisterShape("PolymorphicDefaultOut")(None)
ops.RegisterShape("PolymorphicOut")(None)
ops.RegisterShape("RefIn")(None)
ops.RegisterShape("RefOut")(None)
ops.RegisterShape("ReservedAttr")(None)
ops.RegisterShape("ReservedInput")(None)
ops.RegisterShape("Restrict")(None)
ops.RegisterShape("Simple")(None)
ops.RegisterShape("SimpleStruct")(None)
ops.RegisterShape("TypeList")(None)
ops.RegisterShape("TypeListRestrict")(None)
ops.RegisterShape("TypeListTwice")(None)


class OpDefLibraryTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self._lib = OpDefLibrary()
    self._g = ops.Graph()
    self._default_graph_controller = self._g.as_default()
    self._default_graph_controller.__enter__()
    self._add_op("name: 'Simple' input_arg { name: 'a' type: DT_INT32 } "
                 "output_arg { name: 'out' type: DT_FLOAT }")
    self._add_op("name: 'OutT' output_arg { name: 'a' type_attr: 'T' } "
                 "attr { name: 'T' type: 'type' }")

  def tearDown(self):
    self._default_graph_controller.__exit__(None, None, None)

  def _add_op(self, ascii):
    op_def = op_def_pb2.OpDef()
    text_format.Merge(ascii, op_def)
    self._lib.add_op(op_def)

  def Tensor(self, t, name="in"):
    return self._lib.apply_op("OutT", T=t, name=name)

  def testNoRegisteredOpFails(self):
    with self.assertRaises(RuntimeError) as cm:
      self._lib.apply_op("unknown", g=self._g)
    self.assertEqual(cm.exception.message, "Unrecognized Op name unknown")

  def testAddOpValidation(self):
    with self.assertRaises(TypeError) as cm:
      self._add_op("name: 'MissingTypeAttr' "
                   "input_arg { name: 'a' type_attr: 'T' } ")
    self.assertEqual(cm.exception.message,
                     "Inconsistent OpDef for 'MissingTypeAttr', "
                     "missing attr 'T'")

    with self.assertRaises(TypeError) as cm:
      self._add_op("name: 'BadTypeAttr' "
                   "output_arg { name: 'a' type_attr: 'T' } "
                   "attr { name: 'T' type: 'int' }")
    self.assertEqual(
        cm.exception.message,
        "Attr 'T' of 'BadTypeAttr' used as a type_attr but has type int")

    with self.assertRaises(TypeError) as cm:
      self._add_op("name: 'MissingNumberAttr' "
                   "input_arg { name: 'a' type: DT_INT32 number_attr: 'N' } ")
    self.assertEqual(cm.exception.message,
                     "Inconsistent OpDef for 'MissingNumberAttr', "
                     "missing attr 'N'")

    with self.assertRaises(TypeError) as cm:
      self._add_op("name: 'BadNumberAttr' "
                   "output_arg { name: 'a' type: DT_INT32 number_attr: 'N' } "
                   "attr { name: 'N' type: 'type' }")
    self.assertEqual(
        cm.exception.message,
        "Attr 'N' of 'BadNumberAttr' used as a number_attr but has type type")

    with self.assertRaises(TypeError) as cm:
      self._add_op("name: 'TwoTypesA' "
                   "input_arg { name: 'a' type: DT_INT32 type_attr: 'T' } "
                   "attr { name: 'T' type: 'type' }")
    self.assertEqual(cm.exception.message,
                     "Arg 'a' of 'TwoTypesA' must have one type field not 2")

    with self.assertRaises(TypeError) as cm:
      self._add_op("name: 'TwoTypesB' "
                   "input_arg { name: 'a' type: DT_INT32 type_list_attr: 'T' } "
                   "attr { name: 'T' type: 'list(type)' }")
    self.assertEqual(cm.exception.message,
                     "Arg 'a' of 'TwoTypesB' must have one type field not 2")

    with self.assertRaises(TypeError) as cm:
      self._add_op("name: 'ThreeTypes' "
                   "input_arg { name: 'a' type: DT_INT32 type_attr: 'T' "
                   "type_list_attr: 'U' } "
                   "attr { name: 'T' type: 'type' } "
                   "attr { name: 'U' type: 'list(type)' }")
    self.assertEqual(cm.exception.message,
                     "Arg 'a' of 'ThreeTypes' must have one type field not 3")

    with self.assertRaises(TypeError) as cm:
      self._add_op("name: 'NoTypes' output_arg { name: 'a' } ")
    self.assertEqual(cm.exception.message,
                     "Arg 'a' of 'NoTypes' must have one type field not 0")

  def testSimple(self):
    out = self._lib.apply_op("Simple", a=3)
    self.assertEquals(types.float32, out.dtype)
    self.assertProtoEquals("""
      name: 'Simple' op: 'Simple' input: 'Simple/a'
      """, out.op.node_def)

    out = self._lib.apply_op("Simple", a=4)
    self.assertProtoEquals("""
      name: 'Simple_1' op: 'Simple' input: 'Simple_1/a'
      """, out.op.node_def)

    out = self._lib.apply_op("Simple", a=5, name="named")
    self.assertProtoEquals("""
      name: 'named' op: 'Simple' input: 'named/a'
      """, out.op.node_def)

    out = self._lib.apply_op("Simple", a=[[1, 2, 3], [4, 5, 6]], name="two_d")
    self.assertProtoEquals("""
      name: 'two_d' op: 'Simple' input: 'two_d/a'
      """, out.op.node_def)

  def testSimpleFailures(self):
    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("Simple", a="Bad string")
    self.assertEqual(cm.exception.message,
                     "Expected int32, got 'Bad string' instead.")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("Simple", a=self.Tensor(types.string))
    self.assertEqual(cm.exception.message,
                     "Input 'a' of 'Simple' Op has type string "
                     "that does not match expected type of int32.")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("Simple", a=6, extra="bogus")
    self.assertEqual(cm.exception.message,
                     "apply_op() got unexpected keyword arguments: extra")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("Simple", a=6, extra1="bogus", extra2="also_bogus")
    self.assertEqual(cm.exception.message,
                     "apply_op() got unexpected keyword arguments: extra1, "
                     "extra2")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("Simple")
    self.assertEqual(cm.exception.message, "No argument for input a")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("Simple", wrong=7)
    self.assertEqual(cm.exception.message, "No argument for input a")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("Simple", a=[self.Tensor(types.int32)])
    self.assertStartsWith(cm.exception.message, "Expected int32, got")

  def testReservedInput(self):
    self._add_op("name: 'ReservedInput' "
                 "input_arg { name: 'input' type: DT_INT32 } ")
    op = self._lib.apply_op("ReservedInput", input_=7, name="x")
    self.assertProtoEquals("""
      name: 'x' op: 'ReservedInput' input: 'x/input'
      """, op.node_def)

  def testPolymorphic(self):
    self._add_op("name: 'Polymorphic' "
                 "input_arg { name: 'a' type_attr: 'T' } "
                 "output_arg { name: 'out' type_attr: 'T' } "
                 "attr { name: 'T' type: 'type' }")

    out = self._lib.apply_op("Polymorphic", a=7, name="p")
    self.assertEquals(types.int32, out.dtype)
    self.assertProtoEquals("""
      name: 'p' op: 'Polymorphic' input: 'p/a'
      attr { key: 'T' value { type: DT_INT32 } }
      """, out.op.node_def)

    out = self._lib.apply_op("Polymorphic", a="s", name="q")
    self.assertEquals(types.string, out.dtype)
    self.assertProtoEquals("""
      name: 'q' op: 'Polymorphic' input: 'q/a'
      attr { key: 'T' value { type: DT_STRING } }
      """, out.op.node_def)

    out = self._lib.apply_op("Polymorphic", a=["s", "t", "u"], name="r")
    self.assertEquals(types.string, out.dtype)
    self.assertProtoEquals("""
      name: 'r' op: 'Polymorphic' input: 'r/a'
      attr { key: 'T' value { type: DT_STRING } }
      """, out.op.node_def)

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("Polymorphic", a="s", T=types.string)
    self.assertEqual(cm.exception.message,
                     "Should not specify value for inferred attr 'T'.")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("Polymorphic", a=[self.Tensor(types.bool)])
    self.assertEqual(cm.exception.message,
                     "List of Tensors when single Tensor expected")

  def testPolymorphicOut(self):
    self._add_op("name: 'PolymorphicOut' "
                 "output_arg { name: 'out' type_attr: 'T' } "
                 "attr { name: 'T' type: 'type' }")

    out = self._lib.apply_op("PolymorphicOut", T=types.int32, name="p")
    self.assertEquals(types.int32, out.dtype)
    self.assertProtoEquals("""
      name: 'p' op: 'PolymorphicOut'
      attr { key: 'T' value { type: DT_INT32 } }
      """, out.op.node_def)

    out = self._lib.apply_op("PolymorphicOut", T=types.bool, name="q")
    self.assertEquals(types.bool, out.dtype)
    self.assertProtoEquals("""
      name: 'q' op: 'PolymorphicOut'
      attr { key: 'T' value { type: DT_BOOL } }
      """, out.op.node_def)

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("PolymorphicOut")
    self.assertEqual(cm.exception.message,
                     "No argument for attr T")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("PolymorphicOut", T=None)
    self.assertEqual(cm.exception.message,
                     "Expected DataType for argument 'T' not None.")

  def testPolymorphicDefaultOut(self):
    self._add_op("name: 'PolymorphicDefaultOut' "
                 "output_arg { name: 'out' type_attr: 'T' } "
                 "attr { name: 'T' type: 'type' "
                 "  default_value { type: DT_STRING } }")

    out = self._lib.apply_op("PolymorphicDefaultOut", T=None, name="p")
    self.assertEquals(types.string, out.dtype)
    self.assertProtoEquals("""
      name: 'p' op: 'PolymorphicDefaultOut'
      attr { key: 'T' value { type: DT_STRING } }
      """, out.op.node_def)

    out = self._lib.apply_op("PolymorphicDefaultOut", T=types.bool,
                            name="q")
    self.assertEquals(types.bool, out.dtype)
    self.assertProtoEquals("""
      name: 'q' op: 'PolymorphicDefaultOut'
      attr { key: 'T' value { type: DT_BOOL } }
      """, out.op.node_def)

  def testBinary(self):
    self._add_op("name: 'Binary' "
                 "input_arg { name: 'a' type_attr: 'T' } "
                 "input_arg { name: 'b' type_attr: 'T' } "
                 "output_arg { name: 'out' type_attr: 'T' } "
                 "attr { name: 'T' type: 'type' }")

    out = self._lib.apply_op("Binary", a=8, b=9, name="b")
    self.assertEquals(types.int32, out.dtype)
    self.assertProtoEquals("""
      name: 'b' op: 'Binary' input: 'b/a' input: 'b/b'
      attr { key: 'T' value { type: DT_INT32 } }
      """, out.op.node_def)

    out = self._lib.apply_op("Binary", a="left", b="right", name="c")
    self.assertEquals(types.string, out.dtype)
    self.assertProtoEquals("""
      name: 'c' op: 'Binary' input: 'c/a' input: 'c/b'
      attr { key: 'T' value { type: DT_STRING } }
      """, out.op.node_def)

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("Binary", a="left", b=12)
    self.assertEqual(cm.exception.message,
                     "Expected string, got 12 instead.")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("Binary", a=self.Tensor(types.string),
                        b=self.Tensor(types.int32))
    self.assertEqual(cm.exception.message,
                     "Input 'b' of 'Binary' Op has type int32 "
                     "that does not match type string of argument 'a'.")

  def testRestrict(self):
    self._add_op("name: 'Restrict' "
                 "input_arg { name: 'a' type_attr: 'T' } "
                 "output_arg { name: 'out' type_attr: 'T' } "
                 "attr { name: 'T' type: 'type' allowed_values { list { "
                 "  type: DT_STRING type: DT_BOOL } } }")

    out = self._lib.apply_op("Restrict", a="foo", name="g")
    self.assertEquals(types.string, out.dtype)
    self.assertProtoEquals("""
      name: 'g' op: 'Restrict' input: 'g/a'
      attr { key: 'T' value { type: DT_STRING } }
      """, out.op.node_def)

    out = self._lib.apply_op("Restrict", a=True, name="h")
    self.assertEquals(types.bool, out.dtype)
    self.assertProtoEquals("""
      name: 'h' op: 'Restrict' input: 'h/a'
      attr { key: 'T' value { type: DT_BOOL } }
      """, out.op.node_def)

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("Restrict", a=17)
    self.assertEqual(cm.exception.message,
                     "DataType int32 for attr 'T' "
                     "not in list of allowed values: "
                     "string, bool")

  def testTypeList(self):
    self._add_op("name: 'TypeList' "
                 "input_arg { name: 'a' type_list_attr: 'T' } "
                 "attr { name: 'T' type: 'list(type)' }")

    op = self._lib.apply_op("TypeList", a=["foo"], name="z")
    self.assertProtoEquals("""
      name: 'z' op: 'TypeList' input: 'z/a_0'
      attr { key: 'T' value { list { type: DT_STRING } } }
      """, op.node_def)

    op = self._lib.apply_op("TypeList", a=[True, 12], name="y")
    self.assertProtoEquals("""
      name: 'y' op: 'TypeList' input: 'y/a_0' input: 'y/a_1'
      attr { key: 'T' value { list { type: DT_BOOL type: DT_INT32 } } }
      """, op.node_def)

    op = self._lib.apply_op("TypeList", a=[], name="empty")
    self.assertProtoEquals("""
      name: 'empty' op: 'TypeList' attr { key: 'T' value { list { } } }
      """, op.node_def)

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("TypeList", a=17)
    self.assertStartsWith(cm.exception.message,
                          "Expected list for 'a' "
                          "argument to 'TypeList' Op, not ")

  def testTypeListTwice(self):
    self._add_op("name: 'TypeListTwice' "
                 "input_arg { name: 'a' type_list_attr: 'T' } "
                 "input_arg { name: 'b' type_list_attr: 'T' } "
                 "attr { name: 'T' type: 'list(type)' }")

    op = self._lib.apply_op("TypeListTwice", a=["foo", True], b=["bar", False],
                           name="z")
    self.assertProtoEquals("""
      name: 'z' op: 'TypeListTwice'
      input: 'z/a_0' input: 'z/a_1' input: 'z/b_0' input: 'z/b_1'
      attr { key: 'T' value { list { type: DT_STRING type: DT_BOOL } } }
      """, op.node_def)

    op = self._lib.apply_op("TypeListTwice", a=[], b=[], name="empty")
    self.assertProtoEquals("""
      name: 'empty' op: 'TypeListTwice' attr { key: 'T' value { list { } } }
      """, op.node_def)

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("TypeListTwice", a=["foo", True], b=["bar", 6])
    self.assertEqual(cm.exception.message,
                     "Input 'b' of 'TypeListTwice' Op has type list of "
                     "string, int32 that does not match type list "
                     "string, bool of argument 'a'.")

  def testOutTypeList(self):
    self._add_op("name: 'OutTypeList' "
                 "output_arg { name: 'out' type_list_attr: 'T' } "
                 "attr { name: 'T' type: 'list(type)' }")

    out, = self._lib.apply_op("OutTypeList", T=[types.float32], name="x")
    self.assertEquals(types.float32, out.dtype)
    self.assertProtoEquals("""
      name: 'x' op: 'OutTypeList'
      attr { key: 'T' value { list { type: DT_FLOAT } } }
      """, out.op.node_def)

    out1, out2 = self._lib.apply_op("OutTypeList",
                                   T=[types.int32, types.bool],
                                   name="w")
    self.assertEquals(types.int32, out1.dtype)
    self.assertEquals(types.bool, out2.dtype)
    self.assertProtoEquals("""
      name: 'w' op: 'OutTypeList'
      attr { key: 'T' value { list { type: DT_INT32 type: DT_BOOL } } }
      """, out1.op.node_def)

    out = self._lib.apply_op("OutTypeList", T=[], name="empty")
    self.assertEqual([], out)

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("OutTypeList", T=types.int32)
    self.assertEqual(cm.exception.message, "Expected list for attr T")

  def testTypeListRestrict(self):
    self._add_op("name: 'TypeListRestrict' "
                 "input_arg { name: 'a' type_list_attr: 'T' } "
                 "attr { name: 'T' type: 'list(type)' allowed_values { list { "
                 "  type: DT_STRING type: DT_BOOL } } }")

    op = self._lib.apply_op("TypeListRestrict", a=["foo", False], name="v")
    self.assertProtoEquals("""
      name: 'v' op: 'TypeListRestrict' input: 'v/a_0' input: 'v/a_1'
      attr { key: 'T' value { list { type: DT_STRING type: DT_BOOL } } }
      """, op.node_def)

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("TypeListRestrict", a=[True, 12])
    self.assertEqual(cm.exception.message,
                     "DataType int32 for attr 'T' "
                     "not in list of allowed values: string, bool")

  def testOutTypeListRestrict(self):
    self._add_op("name: 'OutTypeListRestrict' "
                 "output_arg { name: 'out' type_list_attr: 't' } "
                 "attr { name: 't' type: 'list(type)' allowed_values { list { "
                 "  type: DT_STRING type: DT_BOOL } } }")

    out1, out2 = self._lib.apply_op("OutTypeListRestrict",
                                   t=[types.bool, types.string],
                                   name="u")
    self.assertEquals(types.bool, out1.dtype)
    self.assertEquals(types.string, out2.dtype)
    self.assertProtoEquals("""
      name: 'u' op: 'OutTypeListRestrict'
      attr { key: 't' value { list { type: DT_BOOL type: DT_STRING } } }
      """, out1.op.node_def)

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("OutTypeListRestrict",
                        t=[types.string, types.int32])
    self.assertEqual(cm.exception.message,
                     "DataType int32 for attr 't' "
                     "not in list of allowed values: string, bool")

  def testAttr(self):
    self._add_op("name: 'Attr' attr { name: 'a' type: 'int' }")
    op = self._lib.apply_op("Attr", a=12, name="t")
    self.assertProtoEquals("""
      name: 't' op: 'Attr' attr { key: 'a' value { i: 12 } }
      """, op.node_def)

    op = self._lib.apply_op("Attr", a=tensor_shape.Dimension(13), name="u")
    self.assertProtoEquals("""
      name: 'u' op: 'Attr' attr { key: 'a' value { i: 13 } }
      """, op.node_def)

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("Attr", a="bad")
    self.assertEqual(cm.exception.message,
                     "Expected int for argument 'a' not 'bad'.")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("Attr", a=[12])
    self.assertEqual(cm.exception.message,
                     "Expected int for argument 'a' not [12].")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("Attr", a=None)
    self.assertEqual(cm.exception.message,
                     "Expected int for argument 'a' not None.")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("Attr")
    self.assertEqual(cm.exception.message, "No argument for attr a")

  def testAttrFloat(self):
    self._add_op("name: 'AttrFloat' attr { name: 'a' type: 'float' }")

    op = self._lib.apply_op("AttrFloat", a=1.2, name="t")
    self.assertProtoEquals("""
      name: 't' op: 'AttrFloat' attr { key: 'a' value { f: 1.2 } }
      """, op.node_def)

    op = self._lib.apply_op("AttrFloat", a=12, name="u")
    self.assertProtoEquals("""
      name: 'u' op: 'AttrFloat' attr { key: 'a' value { f: 12 } }
      """, op.node_def)

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("AttrFloat", a="bad")
    self.assertEqual(cm.exception.message,
                     "Expected float for argument 'a' not 'bad'.")

  def testAttrBool(self):
    self._add_op("name: 'AttrBool' attr { name: 'a' type: 'bool' }")

    op = self._lib.apply_op("AttrBool", a=True, name="t")
    self.assertProtoEquals("""
      name: 't' op: 'AttrBool' attr { key: 'a' value { b: true } }
      """, op.node_def)

    op = self._lib.apply_op("AttrBool", a=False, name="u")
    self.assertProtoEquals("""
      name: 'u' op: 'AttrBool' attr { key: 'a' value { b: false } }
      """, op.node_def)

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("AttrBool", a=0)
    self.assertEqual(cm.exception.message,
                     "Expected bool for argument 'a' not 0.")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("AttrBool", a=1)
    self.assertEqual(cm.exception.message,
                     "Expected bool for argument 'a' not 1.")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("AttrBool", a=[])
    self.assertEqual(cm.exception.message,
                     "Expected bool for argument 'a' not [].")

  def testAttrBoolList(self):
    self._add_op("name: 'AttrBoolList' attr { name: 'a' type: 'list(bool)' }")

    op = self._lib.apply_op("AttrBoolList", a=[True, False, True], name="t")
    self.assertProtoEquals("""
      name: 't' op: 'AttrBoolList'
      attr { key: 'a' value { list { b: true b: false b:true } } }
      """, op.node_def)

    op = self._lib.apply_op("AttrBoolList", a=[], name="u")
    self.assertProtoEquals("""
      name: 'u' op: 'AttrBoolList' attr { key: 'a' value { list { } } }
      """, op.node_def)

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("AttrBoolList", a=[0])
    self.assertEqual(cm.exception.message,
                     "Expected bool for argument 'a' not 0.")

  def testAttrMin(self):
    self._add_op("name: 'AttrMin' attr { name: 'a' type: 'int' "
                 "has_minimum: true minimum: 5 }")
    op = self._lib.apply_op("AttrMin", a=12, name="s")
    self.assertProtoEquals("""
      name: 's' op: 'AttrMin' attr { key: 'a' value { i: 12 } }
      """, op.node_def)

    with self.assertRaises(ValueError) as cm:
      self._lib.apply_op("AttrMin", a=2)
    self.assertEqual(cm.exception.message,
                     "Attr 'a' of 'AttrMin' Op passed 2 less than minimum 5.")

  def testAttrListMin(self):
    self._add_op("name: 'AttrListMin' attr { name: 'a' type: 'list(int)' "
                 "has_minimum: true minimum: 2 }")

    op = self._lib.apply_op("AttrListMin", a=[1, 2], name="r")
    self.assertProtoEquals("""
      name: 'r' op: 'AttrListMin'
      attr { key: 'a' value { list { i: 1 i: 2 } } }
      """, op.node_def)

    with self.assertRaises(ValueError) as cm:
      self._lib.apply_op("AttrListMin", a=[17])
    self.assertEqual(cm.exception.message,
                     "Attr 'a' of 'AttrListMin' Op "
                     "passed list of length 1 less than minimum 2.")

  def testAttrEnum(self):
    self._add_op("name: 'AttrEnum' "
                 "attr { name: 'a' type: 'string' "
                 "  allowed_values { list { s: 'apples' s: 'oranges' } } }")

    op = self._lib.apply_op("AttrEnum", a="oranges", name="e")
    self.assertProtoEquals("""
      name: 'e' op: 'AttrEnum' attr { key: 'a' value { s: 'oranges' } }
      """, op.node_def)

    with self.assertRaises(ValueError) as cm:
      self._lib.apply_op("AttrEnum", a="invalid")
    self.assertEqual(cm.exception.message,
                     'Attr \'a\' of \'AttrEnum\' Op '
                     'passed string \'invalid\' not in: '
                     '"apples", "oranges".')

  def testAttrEnumList(self):
    self._add_op("name: 'AttrEnumList' "
                 "attr { name: 'a' type: 'list(string)' "
                 "  allowed_values { list { s: 'apples' s: 'oranges' } } }")

    op = self._lib.apply_op("AttrEnumList", a=["oranges", "apples"], name="f")
    self.assertProtoEquals("""
      name: 'f' op: 'AttrEnumList'
      attr { key: 'a' value { list { s: 'oranges' s: 'apples' } } }
      """, op.node_def)

    with self.assertRaises(ValueError) as cm:
      self._lib.apply_op("AttrEnumList", a=["apples", "invalid", "oranges"])
    self.assertEqual(cm.exception.message,
                     'Attr \'a\' of \'AttrEnumList\' Op '
                     'passed string \'invalid\' not '
                     'in: "apples", "oranges".')

  def testAttrShape(self):
    self._add_op("name: 'AttrShape' attr { name: 'a' type: 'shape' }")

    op = self._lib.apply_op("AttrShape", a=[5], name="s1")
    self.assertProtoEquals("""
      name: 's1' op: 'AttrShape'
      attr { key: 'a' value { shape { dim { size: 5 } } } }
      """, op.node_def)

    op = self._lib.apply_op("AttrShape", a=(4, 3, 2), name="s2")
    self.assertProtoEquals("""
      name: 's2' op: 'AttrShape'
      attr { key: 'a' value {
        shape { dim { size: 4 } dim { size: 3 } dim { size: 2 } } } }
      """, op.node_def)

    op = self._lib.apply_op(
        "AttrShape", a=tensor_shape.TensorShape([3, 2]), name="s3")
    self.assertProtoEquals("""
      name: 's3' op: 'AttrShape'
      attr { key: 'a' value {
        shape { dim { size: 3 } dim { size: 2 } } } }
      """, op.node_def)

    op = self._lib.apply_op("AttrShape", a=[], name="s4")
    self.assertProtoEquals("""
      name: 's4' op: 'AttrShape' attr { key: 'a' value { shape { } } }
      """, op.node_def)

    shape = tensor_shape_pb2.TensorShapeProto()
    shape.dim.add().size = 6
    shape.dim.add().size = 3
    op = self._lib.apply_op("AttrShape", a=shape, name="s5")
    self.assertProtoEquals("""
      name: 's5' op: 'AttrShape'
      attr { key: 'a' value { shape { dim { size: 6 } dim { size: 3 } } } }
      """, op.node_def)

    # TODO(josh11b): Re-enable this test once we stop promoting scalars to shapes.
    # with self.assertRaises(TypeError) as cm:
    #   self._lib.apply_op("AttrShape", a=5)
    # self.assertEqual(cm.exception.message,
    #                  "Don't know how to convert 5 to a TensorShapeProto for "
    #                  "argument 'a'")

    with self.assertRaises(ValueError) as cm:
      self._lib.apply_op("AttrShape", a="ABC")

  def testAttrShapeList(self):
    self._add_op("name: 'AttrShapeList' attr { name: 'a' type: 'list(shape)' }")

    op = self._lib.apply_op("AttrShapeList", a=[[3, 2], [6, 5, 4]], name="sl")
    self.assertProtoEquals("""
      name: 'sl' op: 'AttrShapeList'
      attr { key: 'a' value { list {
        shape { dim { size: 3 } dim { size: 2 } }
        shape { dim { size: 6 } dim { size: 5 } dim { size: 4 } } } } }
      """, op.node_def)

    op = self._lib.apply_op("AttrShapeList", a=[], name="esl")
    self.assertProtoEquals("""
      name: 'esl' op: 'AttrShapeList' attr { key: 'a' value { list { } } }
      """, op.node_def)

  def testAttrDefault(self):
    self._add_op("name: 'AttrDefault' "
                 "attr { name: 'a' type: 'string' "
                 "  default_value { s: 'banana' } }")

    op = self._lib.apply_op("AttrDefault", a=None, name="d")
    self.assertProtoEquals("""
      name: 'd' op: 'AttrDefault' attr { key: 'a' value { s: 'banana' } }
      """, op.node_def)

    op = self._lib.apply_op("AttrDefault", a="kiwi", name="c")
    self.assertProtoEquals("""
      name: 'c' op: 'AttrDefault' attr { key: 'a' value { s: 'kiwi' } }
      """, op.node_def)

  def testAttrListDefault(self):
    self._add_op("name: 'AttrListDefault' "
                 "attr { name: 'a' type: 'list(int)' "
                 "  default_value { list { i: 5 i: 15 } } }")

    op = self._lib.apply_op("AttrListDefault", a=None, name="b")
    self.assertProtoEquals("""
      name: 'b' op: 'AttrListDefault'
      attr { key: 'a' value { list { i: 5 i: 15 } } }
      """, op.node_def)

    op = self._lib.apply_op("AttrListDefault", a=[3], name="a")
    self.assertProtoEquals("""
      name: 'a' op: 'AttrListDefault'
      attr { key: 'a' value { list { i: 3 } } }
      """, op.node_def)

    op = self._lib.apply_op("AttrListDefault", a=[], name="empty")
    self.assertProtoEquals("""
      name: 'empty' op: 'AttrListDefault'
      attr { key: 'a' value { list { } } }
      """, op.node_def)

  def testAttrEmptyListDefault(self):
    self._add_op("name: 'AttrEmptyListDefault' "
                 "attr { name: 'a' type: 'list(float)' "
                 "       default_value { list { } } }")

    op = self._lib.apply_op("AttrEmptyListDefault", a=None, name="b")
    self.assertProtoEquals("""
      name: 'b' op: 'AttrEmptyListDefault'
      attr { key: 'a' value { list { } } }
      """, op.node_def)

    op = self._lib.apply_op("AttrEmptyListDefault", a=[3], name="a")
    self.assertProtoEquals("""
      name: 'a' op: 'AttrEmptyListDefault'
      attr { key: 'a' value { list { f: 3 } } }
      """, op.node_def)

    op = self._lib.apply_op("AttrEmptyListDefault", a=[], name="empty")
    self.assertProtoEquals("""
      name: 'empty' op: 'AttrEmptyListDefault'
      attr { key: 'a' value { list { } } }
      """, op.node_def)

  def testReservedAttr(self):
    self._add_op("name: 'ReservedAttr' "
                 "attr { name: 'range' type: 'int' } ")
    op = self._lib.apply_op("ReservedAttr", range_=7, name="x")
    self.assertProtoEquals("""
      name: 'x' op: 'ReservedAttr' attr { key: 'range' value { i: 7 } }
      """, op.node_def)

  def testNIntsIn(self):
    self._add_op("name: 'NIntsIn' "
                 "input_arg { name: 'a' type: DT_INT32 number_attr: 'N' } "
                 "attr { name: 'N' type: 'int' has_minimum: true minimum: 2 }")

    op = self._lib.apply_op("NIntsIn", a=[1, 2], name="n")
    self.assertProtoEquals("""
      name: 'n' op: 'NIntsIn' input: 'n/a_0' input: 'n/a_1'
      attr { key: 'N' value { i: 2 } }
      """, op.node_def)

    op = self._lib.apply_op("NIntsIn", a=[5, 4, 3, 2, 1], name="o")
    self.assertProtoEquals("""
      name: 'o' op: 'NIntsIn'
      input: 'o/a_0' input: 'o/a_1' input: 'o/a_2' input: 'o/a_3' input: 'o/a_4'
      attr { key: 'N' value { i: 5 } }
      """, op.node_def)

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("NIntsIn", a=["foo", "bar"])
    self.assertEqual(cm.exception.message,
                     "Tensors in list passed to 'a' of 'NIntsIn' Op have types "
                     "[string, string] that do not match expected type int32.")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("NIntsIn", a=[self.Tensor(types.string),
                                      self.Tensor(types.string)])
    self.assertEqual(cm.exception.message,
                     "Tensors in list passed to 'a' of 'NIntsIn' Op have "
                     "types [string, string] that do not match expected type "
                     "int32.")

    with self.assertRaises(ValueError) as cm:
      self._lib.apply_op("NIntsIn", a=[99])
    self.assertEqual(cm.exception.message,
                     "List argument 'a' to 'NIntsIn' Op "
                     "with length 1 shorter than "
                     "minimum length 2.")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("NIntsIn", a=[38, "bar"])
    self.assertEqual(cm.exception.message,
                     "Tensors in list passed to 'a' of 'NIntsIn' Op have types "
                     "[int32, string] that do not match expected type int32.")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("NIntsIn", a=[self.Tensor(types.int32),
                                      self.Tensor(types.string)])
    self.assertEqual(cm.exception.message,
                     "Tensors in list passed to 'a' of 'NIntsIn' Op "
                     "have types [int32, string] that do not match expected "
                     "type int32.")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("NIntsIn", a=17)
    self.assertStartsWith(cm.exception.message,
                          "Expected list for 'a' argument "
                          "to 'NIntsIn' Op, not ")

  def testNPolymorphicIn(self):
    self._add_op("name: 'NPolymorphicIn' "
                 "input_arg { name: 'a' type_attr: 'T' number_attr: 'N' } "
                 "attr { name: 'T' type: 'type' } "
                 "attr { name: 'N' type: 'int' has_minimum: true minimum: 2 }")

    op = self._lib.apply_op("NPolymorphicIn", a=[1, 2], name="n")
    self.assertProtoEquals("""
      name: 'n' op: 'NPolymorphicIn' input: 'n/a_0' input: 'n/a_1'
      attr { key: 'T' value { type: DT_INT32 } }
      attr { key: 'N' value { i: 2 } }
      """, op.node_def)

    op = self._lib.apply_op("NPolymorphicIn", a=[5, 4, 3, 2, 1], name="o")
    self.assertProtoEquals("""
      name: 'o' op: 'NPolymorphicIn'
      input: 'o/a_0' input: 'o/a_1' input: 'o/a_2' input: 'o/a_3' input: 'o/a_4'
      attr { key: 'T' value { type: DT_INT32 } }
      attr { key: 'N' value { i: 5 } }
      """, op.node_def)

    op = self._lib.apply_op("NPolymorphicIn", a=["foo", "bar"], name="p")
    self.assertProtoEquals("""
      name: 'p' op: 'NPolymorphicIn' input: 'p/a_0' input: 'p/a_1'
      attr { key: 'T' value { type: DT_STRING } }
      attr { key: 'N' value { i: 2 } }
      """, op.node_def)

    op = self._lib.apply_op("NPolymorphicIn",
                           a=[1, self.Tensor(types.float32, name="x")],
                           name="q")
    self.assertProtoEquals("""
      name: 'q' op: 'NPolymorphicIn' input: 'q/a_0' input: 'x'
      attr { key: 'T' value { type: DT_FLOAT } }
      attr { key: 'N' value { i: 2 } }
      """, op.node_def)

    with self.assertRaises(ValueError) as cm:
      self._lib.apply_op("NPolymorphicIn", a=[99])
    self.assertEqual(cm.exception.message,
                     "List argument 'a' to 'NPolymorphicIn' Op with length 1 "
                     "shorter than minimum length 2.")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("NPolymorphicIn", a=[38, "bar"])
    self.assertEqual(cm.exception.message,
                     "All tensors passed to 'a' of 'NPolymorphicIn' "
                     "Op must have the same type.")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("NPolymorphicIn",
                        a=[38, self.Tensor(types.string)])
    self.assertEqual(cm.exception.message,
                     "Tensors in list passed to 'a' of 'NPolymorphicIn' Op "
                     "have types [int32, string] that don't all match.")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("NPolymorphicIn",
                        a=["abcd", self.Tensor(types.int32)])
    self.assertEqual(cm.exception.message,
                     "Tensors in list passed to 'a' of 'NPolymorphicIn' Op "
                     "have types [string, int32] that don't all match.")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("NPolymorphicIn", a=17)
    self.assertStartsWith(cm.exception.message,
                          "Expected list for 'a' argument "
                          "to 'NPolymorphicIn' Op, not ")

  def testNPolymorphicRestrictIn(self):
    self._add_op("name: 'NPolymorphicRestrictIn' "
                 "input_arg { name: 'a' type_attr: 'T' number_attr: 'N' } "
                 "attr { name: 'T' type: 'type' allowed_values { "
                 "  list { type: DT_STRING type: DT_BOOL } } } "
                 "attr { name: 'N' type: 'int' has_minimum: true minimum: 2 }")

    op = self._lib.apply_op("NPolymorphicRestrictIn", a=["foo", "bar"],
                            name="p")
    self.assertProtoEquals("""
      name: 'p' op: 'NPolymorphicRestrictIn' input: 'p/a_0' input: 'p/a_1'
      attr { key: 'T' value { type: DT_STRING } }
      attr { key: 'N' value { i: 2 } }
      """, op.node_def)

    op = self._lib.apply_op("NPolymorphicRestrictIn", a=[False, True, False],
                           name="b")
    self.assertProtoEquals("""
      name: 'b' op: 'NPolymorphicRestrictIn'
      input: 'b/a_0' input: 'b/a_1' input: 'b/a_2'
      attr { key: 'T' value { type: DT_BOOL } }
      attr { key: 'N' value { i: 3 } }
      """, op.node_def)

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("NPolymorphicRestrictIn", a=[1, 2])
    self.assertEqual(cm.exception.message,
                     "DataType int32 for attr 'T' "
                     "not in list of allowed values: string, bool")

  def testNInTwice(self):
    self._add_op("name: 'NInTwice' "
                 "input_arg { name: 'a' type: DT_INT32 number_attr: 'N' } "
                 "input_arg { name: 'b' type: DT_STRING number_attr: 'N' } "
                 "attr { name: 'N' type: 'int' has_minimum: true minimum: 0 }")

    op = self._lib.apply_op("NInTwice", a=[1, 2], b=["one", "two"], name="n")
    self.assertProtoEquals("""
      name: 'n' op: 'NInTwice'
      input: 'n/a_0' input: 'n/a_1' input: 'n/b_0' input: 'n/b_1'
      attr { key: 'N' value { i: 2 } }
      """, op.node_def)

    op = self._lib.apply_op("NInTwice", a=[], b=[], name="o")
    self.assertProtoEquals("""
      name: 'o' op: 'NInTwice' attr { key: 'N' value { i: 0 } }
      """, op.node_def)

    with self.assertRaises(ValueError) as cm:
      self._lib.apply_op("NInTwice", a=[1, 2, 3], b=["too short"])
    self.assertEqual(cm.exception.message,
                     "List argument 'b' to 'NInTwice' Op "
                     "with length 1 must match "
                     "length 3 of argument 'a'.")

  def testNInPolymorphicTwice(self):
    self._add_op("name: 'NInPolymorphicTwice' "
                 "input_arg { name: 'a' type_attr: 'T' number_attr: 'N' } "
                 "input_arg { name: 'b' type_attr: 'T' number_attr: 'N' } "
                 "attr { name: 'T' type: 'type' } "
                 "attr { name: 'N' type: 'int' has_minimum: true minimum: 0 }")

    op = self._lib.apply_op("NInPolymorphicTwice", a=[1, 2], b=[3, 4], name="n")
    self.assertProtoEquals("""
      name: 'n' op: 'NInPolymorphicTwice'
      input: 'n/a_0' input: 'n/a_1' input: 'n/b_0' input: 'n/b_1'
      attr { key: 'T' value { type: DT_INT32 } }
      attr { key: 'N' value { i: 2 } }
      """, op.node_def)

    with self.assertRaises(ValueError) as cm:
      self._lib.apply_op("NInPolymorphicTwice", a=[1, 2, 3], b=[5])
    self.assertEqual(cm.exception.message,
                     "List argument 'b' to 'NInPolymorphicTwice' Op "
                     "with length 1 "
                     "must match length 3 of argument 'a'.")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("NInPolymorphicTwice", a=[1, 2], b=["one", "two"])
    self.assertEqual(cm.exception.message,
                     "Tensors in list passed to 'b' of 'NInPolymorphicTwice' "
                     "Op have types [string, string] that do not match type "
                     "int32 inferred from earlier arguments.")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("NInPolymorphicTwice",
                        a=[self.Tensor(types.int32)],
                        b=[self.Tensor(types.string)])
    self.assertEqual(cm.exception.message,
                     "Tensors in list passed to 'b' of "
                     "'NInPolymorphicTwice' Op have types [string] that do not "
                     "match type int32 inferred from earlier arguments.")

  def testNInTwoTypeVariables(self):
    self._add_op("name: 'NInTwoTypeVariables' "
                 "input_arg { name: 'a' type_attr: 'S' number_attr: 'N' } "
                 "input_arg { name: 'b' type_attr: 'T' number_attr: 'N' } "
                 "attr { name: 'S' type: 'type' } "
                 "attr { name: 'T' type: 'type' } "
                 "attr { name: 'N' type: 'int' has_minimum: true minimum: 0 }")

    op = self._lib.apply_op("NInTwoTypeVariables", a=[1, 2], b=[True, False],
                           name="n")
    self.assertProtoEquals("""
      name: 'n' op: 'NInTwoTypeVariables'
      input: 'n/a_0' input: 'n/a_1' input: 'n/b_0' input: 'n/b_1'
      attr { key: 'S' value { type: DT_INT32 } }
      attr { key: 'T' value { type: DT_BOOL } }
      attr { key: 'N' value { i: 2 } }
      """, op.node_def)

    op = self._lib.apply_op("NInTwoTypeVariables", a=[1, 2], b=[3, 4], name="o")
    self.assertProtoEquals("""
      name: 'o' op: 'NInTwoTypeVariables'
      input: 'o/a_0' input: 'o/a_1' input: 'o/b_0' input: 'o/b_1'
      attr { key: 'S' value { type: DT_INT32 } }
      attr { key: 'T' value { type: DT_INT32 } }
      attr { key: 'N' value { i: 2 } }
      """, op.node_def)

    op = self._lib.apply_op("NInTwoTypeVariables",
                           a=[self.Tensor(types.int32, name="q")],
                           b=[self.Tensor(types.string, name="r")],
                           name="p")
    self.assertProtoEquals("""
      name: 'p' op: 'NInTwoTypeVariables' input: 'q' input: 'r'
      attr { key: 'S' value { type: DT_INT32 } }
      attr { key: 'T' value { type: DT_STRING } }
      attr { key: 'N' value { i: 1 } }
      """, op.node_def)

    with self.assertRaises(ValueError) as cm:
      self._lib.apply_op("NInTwoTypeVariables", a=[1, 2, 3], b=["5"])
    self.assertEqual(cm.exception.message,
                     "List argument 'b' to 'NInTwoTypeVariables' Op "
                     "with length 1 "
                     "must match length 3 of argument 'a'.")

  def testInPolymorphicTwice(self):
    self._add_op("name: 'InPolymorphicTwice' "
                 "input_arg { name: 'a' type_attr: 'T' number_attr: 'N' } "
                 "input_arg { name: 'b' type_attr: 'T' number_attr: 'M' } "
                 "attr { name: 'T' type: 'type' } "
                 "attr { name: 'N' type: 'int' has_minimum: true minimum: 0 } "
                 "attr { name: 'M' type: 'int' has_minimum: true minimum: 0 } ")

    op = self._lib.apply_op("InPolymorphicTwice", a=[8], b=[3, 4, 5], name="n")
    self.assertProtoEquals("""
      name: 'n' op: 'InPolymorphicTwice'
      input: 'n/a_0' input: 'n/b_0' input: 'n/b_1' input: 'n/b_2'
      attr { key: 'T' value { type: DT_INT32 } }
      attr { key: 'N' value { i: 1 } }
      attr { key: 'M' value { i: 3 } }
      """, op.node_def)

    op = self._lib.apply_op("InPolymorphicTwice", a=[8], b=[], name="o")
    self.assertProtoEquals("""
      name: 'o' op: 'InPolymorphicTwice' input: 'o/a_0'
      attr { key: 'T' value { type: DT_INT32 } }
      attr { key: 'N' value { i: 1 } }
      attr { key: 'M' value { i: 0 } }
      """, op.node_def)

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("InPolymorphicTwice", a=[], b=[3, 4, 5])
    self.assertEqual(cm.exception.message,
                     "Don't know how to infer type variable from empty input "
                     "list passed to input 'a' of 'InPolymorphicTwice' Op.")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("InPolymorphicTwice", a=[1, 2], b=["one", "two"])
    self.assertEqual(cm.exception.message,
                     "Tensors in list passed to 'b' of 'InPolymorphicTwice' Op "
                     "have types [string, string] that do not match type int32 "
                     "inferred from earlier arguments.")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("InPolymorphicTwice",
                        a=[self.Tensor(types.int32)],
                        b=[self.Tensor(types.string)])
    self.assertEqual(cm.exception.message,
                     "Tensors in list passed to 'b' of 'InPolymorphicTwice' "
                     "Op have types [string] that do not match type int32 "
                     "inferred from earlier arguments.")

  def testNIntsOut(self):
    self._add_op("name: 'NIntsOut' "
                 "output_arg { name: 'a' type: DT_INT32 number_attr: 'N' } "
                 "attr { name: 'N' type: 'int' has_minimum: true minimum: 2 }")

    out1, out2 = self._lib.apply_op("NIntsOut", N=2, name="n")
    self.assertEquals(types.int32, out1.dtype)
    self.assertEquals(types.int32, out2.dtype)
    self.assertProtoEquals("""
      name: 'n' op: 'NIntsOut' attr { key: 'N' value { i: 2 } }
      """, out1.op.node_def)

    out1, out2, out3, out4, out5 = self._lib.apply_op(
        "NIntsOut", N=5, name="o")
    self.assertEquals(types.int32, out1.dtype)
    self.assertEquals(types.int32, out2.dtype)
    self.assertEquals(types.int32, out3.dtype)
    self.assertEquals(types.int32, out4.dtype)
    self.assertEquals(types.int32, out5.dtype)
    self.assertProtoEquals("""
      name: 'o' op: 'NIntsOut' attr { key: 'N' value { i: 5 } }
      """, out5.op.node_def)

    with self.assertRaises(ValueError) as cm:
      self._lib.apply_op("NIntsOut", N=1)
    self.assertEqual(cm.exception.message,
                     "Attr 'N' of 'NIntsOut' Op passed 1 less than minimum 2.")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("NIntsOut", N=[3])
    self.assertEqual(cm.exception.message,
                     "Expected int for argument 'N' not [3].")

  def testNIntsOutDefault(self):
    self._add_op("name: 'NIntsOutDefault' "
                 "output_arg { name: 'a' type: DT_INT32 number_attr: 'N' } "
                 "attr { name: 'N' type: 'int' has_minimum: true minimum: 2"
                 "  default_value { i:3 } }")

    out1, out2, out3 = self._lib.apply_op(
        "NIntsOutDefault", N=None, name="z")
    self.assertEquals(types.int32, out1.dtype)
    self.assertEquals(types.int32, out2.dtype)
    self.assertEquals(types.int32, out3.dtype)
    self.assertProtoEquals("""
      name: 'z' op: 'NIntsOutDefault' attr { key: 'N' value { i: 3 } }
      """, out1.op.node_def)

    out1, out2 = self._lib.apply_op("NIntsOutDefault", N=2, name="y")
    self.assertEquals(types.int32, out1.dtype)
    self.assertEquals(types.int32, out2.dtype)
    self.assertProtoEquals("""
      name: 'y' op: 'NIntsOutDefault' attr { key: 'N' value { i: 2 } }
      """, out2.op.node_def)

  def testNPolymorphicOut(self):
    self._add_op("name: 'NPolymorphicOut' "
                 "output_arg { name: 'a' type_attr: 'T' number_attr: 'N' } "
                 "attr { name: 'T' type: 'type' } "
                 "attr { name: 'N' type: 'int' has_minimum: true minimum: 2 }")

    out1, out2 = self._lib.apply_op("NPolymorphicOut", N=2,
                                   T=types.int32, name="n")
    self.assertEquals(types.int32, out1.dtype)
    self.assertEquals(types.int32, out2.dtype)
    self.assertProtoEquals("""
      name: 'n' op: 'NPolymorphicOut'
      attr { key: 'T' value { type: DT_INT32 } }
      attr { key: 'N' value { i: 2 } }
      """, out1.op.node_def)

    out1, out2, out3 = self._lib.apply_op(
        "NPolymorphicOut", T=types.string, N=3, name="o")
    self.assertEquals(types.string, out1.dtype)
    self.assertEquals(types.string, out2.dtype)
    self.assertEquals(types.string, out3.dtype)
    self.assertProtoEquals("""
      name: 'o' op: 'NPolymorphicOut'
      attr { key: 'T' value { type: DT_STRING } }
      attr { key: 'N' value { i: 3 } }
      """, out3.op.node_def)

    with self.assertRaises(ValueError) as cm:
      self._lib.apply_op("NPolymorphicOut", N=1, T=types.string)
    self.assertEqual(cm.exception.message,
                     "Attr 'N' of 'NPolymorphicOut' Op "
                     "passed 1 less than minimum 2.")

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("NPolymorphicOut", N=3, T=[types.string])
    self.assertEqual(
        cm.exception.message,
        "Expected DataType for argument 'T' not [tf.string].")

  def testNPolymorphicOutDefault(self):
    self._add_op("name: 'NPolymorphicOutDefault' "
                 "output_arg { name: 'a' type_attr: 'T' number_attr: 'N' } "
                 "attr { name: 'T' type: 'type'"
                 "  default_value { type: DT_BOOL } } "
                 "attr { name: 'N' type: 'int' has_minimum: true minimum: 2 "
                 "  default_value { i: 2 } }")

    out1, out2 = self._lib.apply_op(
        "NPolymorphicOutDefault", N=None, T=None, name="r")
    self.assertEquals(types.bool, out1.dtype)
    self.assertEquals(types.bool, out2.dtype)
    self.assertProtoEquals("""
      name: 'r' op: 'NPolymorphicOutDefault'
      attr { key: 'T' value { type: DT_BOOL } }
      attr { key: 'N' value { i: 2 } }
      """, out1.op.node_def)

    out1, out2, out3 = self._lib.apply_op(
        "NPolymorphicOutDefault", N=3, T=None, name="s")
    self.assertEquals(types.bool, out1.dtype)
    self.assertEquals(types.bool, out2.dtype)
    self.assertEquals(types.bool, out3.dtype)
    self.assertProtoEquals("""
      name: 's' op: 'NPolymorphicOutDefault'
      attr { key: 'T' value { type: DT_BOOL } }
      attr { key: 'N' value { i: 3 } }
      """, out1.op.node_def)

    out1, out2 = self._lib.apply_op(
        "NPolymorphicOutDefault", N=None, T=types.int32, name="t")
    self.assertEquals(types.int32, out1.dtype)
    self.assertEquals(types.int32, out2.dtype)
    self.assertProtoEquals("""
      name: 't' op: 'NPolymorphicOutDefault'
      attr { key: 'T' value { type: DT_INT32 } }
      attr { key: 'N' value { i: 2 } }
      """, out1.op.node_def)

    out1, out2, out3 = self._lib.apply_op(
        "NPolymorphicOutDefault", N=3, T=types.int32, name="u")
    self.assertEquals(types.int32, out1.dtype)
    self.assertEquals(types.int32, out2.dtype)
    self.assertEquals(types.int32, out3.dtype)
    self.assertProtoEquals("""
      name: 'u' op: 'NPolymorphicOutDefault'
      attr { key: 'T' value { type: DT_INT32 } }
      attr { key: 'N' value { i: 3 } }
      """, out1.op.node_def)

  def testNPolymorphicRestrictOut(self):
    self._add_op("name: 'NPolymorphicRestrictOut' "
                 "output_arg { name: 'a' type_attr: 'T' number_attr: 'N' } "
                 "attr { name: 'T' type: 'type' allowed_values { "
                 "  list { type: DT_STRING type: DT_BOOL } } } "
                 "attr { name: 'N' type: 'int' has_minimum: true minimum: 2 }")

    out1, out2, out3 = self._lib.apply_op(
        "NPolymorphicRestrictOut", N=3, T=types.bool, name="u")
    self.assertEquals(types.bool, out1.dtype)
    self.assertEquals(types.bool, out2.dtype)
    self.assertEquals(types.bool, out3.dtype)
    self.assertProtoEquals("""
      name: 'u' op: 'NPolymorphicRestrictOut'
      attr { key: 'T' value { type: DT_BOOL } }
      attr { key: 'N' value { i: 3 } }
      """, out1.op.node_def)

    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("NPolymorphicRestrictOut", N=2, T=types.int32)
    self.assertEqual(cm.exception.message,
                     "DataType int32 for attr 'T' "
                     "not in list of allowed values: string, bool")

  def testRef(self):
    self._add_op("name: 'RefIn' "
                 "input_arg { name: 'a' type_attr: 'T' is_ref: true } "
                 "attr { name: 'T' type: 'type' } ")
    self._add_op("name: 'RefOut' "
                 "output_arg { name: 'a' type_attr: 'T' is_ref: true } "
                 "attr { name: 'T' type: 'type' } ")

    out = self._lib.apply_op("RefOut", T=types.bool, name="o")
    self.assertEquals(types.bool_ref, out.dtype)
    self.assertProtoEquals("""
      name: 'o' op: 'RefOut'
      attr { key: 'T' value { type: DT_BOOL } }
      """, out.op.node_def)

    op = self._lib.apply_op("RefIn", a=out, name="i")
    self.assertProtoEquals("""
      name: 'i' op: 'RefIn' input: 'o'
      attr { key: 'T' value { type: DT_BOOL } }
      """, op.node_def)

    # Can pass ref to non-ref input.
    out = self._lib.apply_op("RefOut", T=types.int32, name="r")
    out = self._lib.apply_op("Simple", a=out, name="s")
    self.assertProtoEquals("""
      name: 's' op: 'Simple' input: 'r'
      """, out.op.node_def)

    # Can't pass non-ref to ref input.
    with self.assertRaises(TypeError) as cm:
      self._lib.apply_op("RefIn", a=2)
    self.assertEqual(cm.exception.message,
                     "Input 'a' of 'RefIn' Op requires l-value input")

  def testSpecifyDevice(self):
    with self._g.device("ADevice"):
      self._lib.apply_op("Simple", a=3)
    # We look at the whole graph here to make sure the Const op is also given
    # the specified device.
    graph_def = self._g.as_graph_def()
    self.assertEqual(len(graph_def.node), 2)
    for node in graph_def.node:
      self.assertEqual(node.device, "ADevice")

  def testStructuredOutputSingleList(self):
    self._add_op("name: 'SimpleStruct' "
                 "output_arg { name: 'a' type: DT_INT32 number_attr: 'n_a' } "
                 "attr { name: 'n_a' type: 'int' }")
    for n_a in [0, 1, 3]:
      a = self._lib.apply_op("SimpleStruct", n_a=n_a)
      self.assertTrue(isinstance(a, list))
      self.assertEqual(n_a, len(a))

  def testStructuredOutputListAndSingle(self):
    self._add_op("name: 'MixedStruct' "
                 "output_arg { name: 'a' type: DT_INT32 number_attr: 'n_a' } "
                 "output_arg { name: 'b' type: DT_FLOAT } "
                 "attr { name: 'n_a' type: 'int' }")
    for n_a in [0, 1, 3]:
      a, b = self._lib.apply_op("MixedStruct", n_a=n_a)
      self.assertTrue(isinstance(a, list))
      self.assertEqual(n_a, len(a))
      self.assertTrue(all(x.dtype == types.int32 for x in a))
      self.assertTrue(isinstance(b, ops.Tensor))
      self.assertEqual(types.float32, b.dtype)

  def testStructuredOutputMultipleLists(self):
    self._add_op("name: 'ComplexStruct' "
                 "output_arg { name: 'a' type: DT_INT32 number_attr: 'n_a' } "
                 "output_arg { name: 'b' type: DT_INT64 number_attr: 'n_b' } "
                 "output_arg { name: 'c' type_list_attr: 't_c' } "
                 "attr { name: 'n_a' type: 'int' } "
                 "attr { name: 'n_b' type: 'int' } "
                 "attr { name: 't_c' type: 'list(type)' }")
    for n_a in [0, 1, 3]:
      for n_b in [0, 1, 3]:
        for t_c in [[],
                    [types.int32],
                    [types.int32, types.float32]]:
          a, b, c = self._lib.apply_op("ComplexStruct",
                                      n_a=n_a, n_b=n_b, t_c=t_c)

          self.assertEqual(n_a, len(a))
          self.assertTrue(all(x.dtype == types.int32 for x in a))
          self.assertEqual(n_b, len(b))
          self.assertTrue(all(x.dtype == types.int64 for x in b))
          self.assertEqual(t_c, [x.dtype for x in c])


class OpDefLibraryGraphTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self._lib = OpDefLibrary()
    self._g = ops.Graph()
    self._add_op("name: 'Simple' input_arg { name: 'a' type: DT_INT32 } "
                 "output_arg { name: 'out' type: DT_FLOAT }")
    self._add_op("name: 'Binary' "
                 "input_arg { name: 'a' type_attr: 'T' } "
                 "input_arg { name: 'b' type_attr: 'T' } "
                 "output_arg { name: 'out' type_attr: 'T' } "
                 "attr { name: 'T' type: 'type' }")

  def _add_op(self, ascii):
    op_def = op_def_pb2.OpDef()
    text_format.Merge(ascii, op_def)
    self._lib.add_op(op_def)

  def testNoGraph(self):
    out = self._lib.apply_op("Simple", a=3)
    self.assertEquals(out.graph, ops.get_default_graph())

  def testDefaultGraph(self):
    with self._g.as_default():
      out = self._lib.apply_op("Simple", a=3)
      self.assertEquals(out.graph, self._g)

  def testIgnoreDefaultGraphWithGraphArgument(self):
    default_g = ops.Graph()
    with default_g.as_default():
      out = self._lib.apply_op("Simple", a=3, g=self._g)
      self.assertEquals(ops.get_default_graph(), default_g)
      self.assertEquals(out.graph, self._g)

  def testDifferentGraphFails(self):
    a = self._lib.apply_op("Simple", a=3, g=self._g)
    other_g = ops.Graph()
    b = self._lib.apply_op("Simple", a=4, g=other_g)
    with self.assertRaises(ValueError) as cm:
      self._lib.apply_op("Binary", a=a, b=b)
    self.assertTrue("must be from the same graph" in cm.exception.message)

  def testDifferentGraphFailsWithGraphArgument(self):
    other_g = ops.Graph()
    a = self._lib.apply_op("Simple", a=3, g=other_g)
    b = self._lib.apply_op("Simple", a=4, g=other_g)
    with self.assertRaises(ValueError) as cm:
      self._lib.apply_op("Binary", a=a, b=b, g=self._g)
    self.assertTrue(
        "not from the passed-in graph" in cm.exception.message)


if __name__ == "__main__":
  googletest.main()
