# -*- Python -*-

# A more robust set of lint and errorprone checks when building
# Java source to improve code consistency.

XLINT_OPTS = [
    "-Werror",
    "-Xlint:all",
    "-Xlint:-serial",
    "-Xlint:-try",
]

# errorprone checks related to basic coding style.
EP_STYLE_CHECKS = [
    "-Xep:ConstantField:ERROR",
    "-Xep:LoggerVariableCase:ERROR",
    "-Xep:MissingDefault:ERROR",
    "-Xep:MixedArrayDimensions:ERROR",
    "-Xep:MultiVariableDeclaration:ERROR",
    "-Xep:MultipleTopLevelClasses:ERROR",
    "-Xep:PackageLocation:ERROR",
    "-Xep:RemoveUnusedImports:ERROR",
    "-Xep:UnnecessaryStaticImport:ERROR",
    "-Xep:WildcardImport:ERROR",
]

# additional checks with occasional false positives, enabled
# to detect common errors.
EP_USEFUL_CHECKS = [
    "-Xep:AmbiguousMethodReference:ERROR",
    "-Xep:ArgumentSelectionDefectChecker:ERROR",
    "-Xep:AssertEqualsArgumentOrderChecker:ERROR",
    "-Xep:AssistedInjectAndInjectOnConstructors:ERROR",
    "-Xep:BadAnnotationImplementation:ERROR",
    "-Xep:BadComparable:ERROR",
    "-Xep:BindingToUnqualifiedCommonType:ERROR",
    "-Xep:BoxedPrimitiveConstructor:ERROR",
    "-Xep:CannotMockFinalClass:ERROR",
    "-Xep:ClassCanBeStatic:ERROR",
    "-Xep:ClassNewInstance:ERROR",
    "-Xep:DefaultCharset:ERROR",
    "-Xep:DoNotMock_ForTricorder:ERROR",
    "-Xep:DoubleCheckedLocking:ERROR",
    "-Xep:ElementsCountedInLoop:ERROR",
    "-Xep:EqualsHashCode:ERROR",
    "-Xep:ExpectedExceptionChecker:ERROR",
    "-Xep:Finally:ERROR",
    "-Xep:FloatingPointLiteralPrecision:ERROR",
    "-Xep:FragmentInjection:ERROR",
    "-Xep:FragmentNotInstantiable:ERROR",
    "-Xep:FunctionalInterfaceClash:ERROR",
    "-Xep:FutureReturnValueIgnored:ERROR",
    "-Xep:GetClassOnEnum:ERROR",
    "-Xep:ImmutableAnnotationChecker:ERROR",
    "-Xep:ImmutableEnumChecker:ERROR",
    "-Xep:IncompatibleModifiers:ERROR",
    "-Xep:InjectOnConstructorOfAbstractClass:ERROR",
    "-Xep:InjectScopeOrQualifierAnnotationRetention:ERROR",
    "-Xep:InputStreamSlowMultibyteRead:ERROR",
    "-Xep:IterableAndIterator:ERROR",
    "-Xep:JUnit3FloatingPointComparisonWithoutDelta:ERROR",
    "-Xep:JUnit4SuppressWithoutIgnore:ERROR",
    "-Xep:JUnitAmbiguousTestClass:ERROR",
    "-Xep:JavaLangClash:ERROR",
    "-Xep:JavaNetIdn:ERROR",
    "-Xep:JodaDurationFlagStrictParsing:ERROR",
    "-Xep:LiteralClassName:ERROR",
    "-Xep:LogicalAssignment:ERROR",
    "-Xep:MissingFail:ERROR",
    "-Xep:MissingOverride:ERROR",
    "-Xep:MisusedFormattingLogger:ERROR",
    "-Xep:MutableConstantField:ERROR",
    "-Xep:NamedParameters:ERROR",
    "-Xep:NarrowingCompoundAssignment:ERROR",
    "-Xep:NonAtomicVolatileUpdate:ERROR",
    "-Xep:NonOverridingEquals:ERROR",
    "-Xep:NullableConstructor:ERROR",
    "-Xep:NullablePrimitive:ERROR",
    "-Xep:NullableVoid:ERROR",
    "-Xep:OperatorPrecedence:ERROR",
    "-Xep:OverridesGuiceInjectableMethod:ERROR",
    "-Xep:PreconditionsInvalidPlaceholder:ERROR",
    "-Xep:PredicateIncompatibleType:ERROR",
    "-Xep:ProduceMethodShouldBeAnnotated:ERROR",
    "-Xep:ProtoFieldPreconditionsCheckNotNull:ERROR",
    "-Xep:QualifierWithTypeUse:ERROR",
    "-Xep:ReferenceEquality:ERROR",
    "-Xep:RequiredModifiers:ERROR",
    "-Xep:ShortCircuitBoolean:ERROR",
    "-Xep:SimpleDateFormatConstant:ERROR",
    "-Xep:StaticFlagUsage:ERROR",
    "-Xep:StaticGuardedByInstance:ERROR",
    "-Xep:StaticQualifiedUsingExpression:ERROR",
    "-Xep:SynchronizeOnNonFinalField:ERROR",
    "-Xep:TestExceptionChecker:ERROR",
    "-Xep:TruthConstantAsserts:ERROR",
    "-Xep:TypeParameterShadowing:ERROR",
    "-Xep:TypeParameterUnusedInFormals:ERROR",
    "-Xep:URLEqualsHashCode:ERROR",
    "-Xep:UnsafeSdkVersionCheck:ERROR",
    "-Xep:UnsynchronizedOverridesSynchronized:ERROR",
    "-Xep:WaitNotInLoop:ERROR",
]

# Additional restrictions that enforce a recommended coding style
# rather than actual errors.
EP_SUGGESTED_CHECKS = [
    "-Xep:AssertFalse:ERROR",
    "-Xep:AssistedInjectAndInjectOnSameConstructor:ERROR",
    "-Xep:AutoFactoryAtInject:ERROR",
    "-Xep:BigDecimalLiteralDouble:ERROR",
    "-Xep:EmptyTopLevelDeclaration:ERROR",
    "-Xep:MethodCanBeStatic:ERROR",
    "-Xep:NonCanonicalStaticMemberImport:ERROR",
    "-Xep:PrimitiveArrayPassedToVarargsMethod:ERROR",
    "-Xep:PrivateConstructorForNoninstantiableModuleTest:ERROR",
    "-Xep:PrivateConstructorForUtilityClass:ERROR",
    "-Xep:RedundantThrows:ERROR",
    "-Xep:StringEquality:ERROR",
    #    "-Xep:ThrowsUncheckedException:ERROR",
    "-Xep:UngroupedOverloads:ERROR",
    #    "-Xep:UnnecessaryDefaultInEnumSwitch:ERROR",
]

EP_OPTS = EP_STYLE_CHECKS + EP_USEFUL_CHECKS + EP_SUGGESTED_CHECKS

JAVACOPTS = XLINT_OPTS + EP_OPTS
