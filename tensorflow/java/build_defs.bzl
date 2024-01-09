JAVA_VERSION_OPTS = []

# A more robust set of lint and errorprone checks when building
# Java source to improve code consistency.

# The bazel errorprone plugin currently only enables default errorChecks
# https://github.com/bazelbuild/bazel/blob/97975603e5ff2247e6bb352e3afd27fea38f108d/src/java_tools/buildjar/java/com/google/devtools/build/buildjar/javac/plugins/errorprone/ErrorPronePlugin.java#L52
#
# Default errorChecks are errorprone checkers listed under ENABLED_ERRORS at
# https://github.com/google/error-prone/blob/c6f24bc387989158d99af28e7ae86755e56c5f38/core/src/main/java/com/google/errorprone/scanner/BuiltInCheckerSuppliers.java#L273
#
# Here we enable all available errorprone checks to converge on a consistent
# code style.
# https://github.com/google/error-prone/blob/c6f24bc387989158d99af28e7ae86755e56c5f38/core/src/main/java/com/google/errorprone/scanner/BuiltInCheckerSuppliers.java#L260

# This list is from ENABLED_WARNINGS in
# com/google/errorprone/scanner/BuiltInCheckerSuppliers.java
EP_ENABLED_WARNINGS = [
    "-Xep:AmbiguousMethodReference:WARN",
    "-Xep:ArgumentSelectionDefectChecker:WARN",
    "-Xep:AssertEqualsArgumentOrderChecker:WARN",
    "-Xep:BadAnnotationImplementation:WARN",
    "-Xep:BadComparable:WARN",
    "-Xep:BoxedPrimitiveConstructor:WARN",
    "-Xep:CannotMockFinalClass:WARN",
    "-Xep:ClassCanBeStatic:WARN",
    "-Xep:ClassNewInstance:WARN",
    "-Xep:DefaultCharset:WARN",
    "-Xep:DoubleCheckedLocking:WARN",
    "-Xep:ElementsCountedInLoop:WARN",
    "-Xep:EqualsHashCode:WARN",
    "-Xep:EqualsIncompatibleType:WARN",
    "-Xep:Finally:WARN",
    "-Xep:FloatingPointLiteralPrecision:WARN",
    "-Xep:FragmentInjection:WARN",
    "-Xep:FragmentNotInstantiable:WARN",
    "-Xep:FunctionalInterfaceClash:WARN",
    "-Xep:FutureReturnValueIgnored:WARN",
    "-Xep:GetClassOnEnum:WARN",
    "-Xep:ImmutableAnnotationChecker:WARN",
    "-Xep:ImmutableEnumChecker:WARN",
    "-Xep:IncompatibleModifiers:WARN",
    "-Xep:InjectOnConstructorOfAbstractClass:WARN",
    "-Xep:InputStreamSlowMultibyteRead:WARN",
    "-Xep:IterableAndIterator:WARN",
    "-Xep:JavaLangClash:WARN",
    "-Xep:JUnit3FloatingPointComparisonWithoutDelta:WARN",
    "-Xep:JUnitAmbiguousTestClass:WARN",
    "-Xep:LiteralClassName:WARN",
    "-Xep:LogicalAssignment:WARN",
    "-Xep:MissingFail:WARN",
    "-Xep:MissingOverride:WARN",
    "-Xep:MutableConstantField:WARN",
    "-Xep:NamedParameters:WARN",
    "-Xep:NarrowingCompoundAssignment:WARN",
    "-Xep:NonAtomicVolatileUpdate:WARN",
    "-Xep:NonOverridingEquals:WARN",
    "-Xep:NullableConstructor:WARN",
    "-Xep:NullablePrimitive:WARN",
    "-Xep:NullableVoid:WARN",
    "-Xep:OperatorPrecedence:WARN",
    "-Xep:OverridesGuiceInjectableMethod:WARN",
    "-Xep:PreconditionsInvalidPlaceholder:WARN",
    "-Xep:ProtoFieldPreconditionsCheckNotNull:WARN",
    "-Xep:ReferenceEquality:WARN",
    "-Xep:RequiredModifiers:WARN",
    "-Xep:ShortCircuitBoolean:WARN",
    "-Xep:SimpleDateFormatConstant:WARN",
    "-Xep:StaticGuardedByInstance:WARN",
    "-Xep:SynchronizeOnNonFinalField:WARN",
    "-Xep:TruthConstantAsserts:WARN",
    "-Xep:TypeParameterShadowing:WARN",
    "-Xep:TypeParameterUnusedInFormals:WARN",
    "-Xep:UnsynchronizedOverridesSynchronized:WARN",
    "-Xep:URLEqualsHashCode:WARN",
    "-Xep:WaitNotInLoop:WARN",
]

# This list is from DISABLED_CHECKS in
# com/google/errorprone/scanner/BuiltInCheckerSuppliers.java
EP_DISABLED_CHECKS = [
    "-Xep:AutoFactoryAtInject:WARN",
    "-Xep:AssertFalse:WARN",
    "-Xep:AssistedInjectAndInjectOnConstructors:WARN",
    "-Xep:AssistedInjectAndInjectOnSameConstructor:WARN",
    "-Xep:BigDecimalLiteralDouble:WARN",
    "-Xep:BindingToUnqualifiedCommonType:WARN",
    "-Xep:ClassName:WARN",
    "-Xep:ComparisonContractViolated:WARN",
    "-Xep:ConstantField:WARN",
    "-Xep:ConstructorInvokesOverridable:WARN",
    # False positives, disabled
    # "-Xep:ConstructorLeaksThis:WARN",
    "-Xep:DepAnn:WARN",
    "-Xep:DivZero:WARN",
    "-Xep:EmptyIfStatement:WARN",
    "-Xep:EmptySetMultibindingContributions:WARN",
    "-Xep:EmptyTopLevelDeclaration:WARN",
    "-Xep:ExpectedExceptionChecker:WARN",
    "-Xep:HardCodedSdCardPath:WARN",
    "-Xep:InjectedConstructorAnnotations:WARN",
    "-Xep:InvalidTargetingOnScopingAnnotation:WARN",
    "-Xep:IterablePathParameter:WARN",
    "-Xep:JMockTestWithoutRunWithOrRuleAnnotation:WARN",
    "-Xep:JavaxInjectOnFinalField:WARN",
    "-Xep:LockMethodChecker:WARN",
    "-Xep:LongLiteralLowerCaseSuffix:WARN",
    "-Xep:MethodCanBeStatic:WARN",
    "-Xep:MissingDefault:WARN",
    "-Xep:MixedArrayDimensions:WARN",
    "-Xep:MoreThanOneQualifier:WARN",
    "-Xep:MultiVariableDeclaration:WARN",
    "-Xep:MultipleTopLevelClasses:WARN",
    "-Xep:NoAllocationChecker:WARN",
    "-Xep:NonCanonicalStaticMemberImport:WARN",
    "-Xep:NumericEquality:WARN",
    "-Xep:PackageLocation:WARN",
    "-Xep:PrimitiveArrayPassedToVarargsMethod:WARN",
    "-Xep:PrivateConstructorForUtilityClass:WARN",
    "-Xep:PrivateConstructorForNoninstantiableModule:WARN",
    "-Xep:ProtoStringFieldReferenceEquality:WARN",
    "-Xep:QualifierOrScopeOnInjectMethod:WARN",
    "-Xep:QualifierWithTypeUse:WARN",
    "-Xep:RedundantThrows:WARN",
    "-Xep:RemoveUnusedImports:WARN",
    "-Xep:ScopeAnnotationOnInterfaceOrAbstractClass:WARN",
    "-Xep:ScopeOrQualifierAnnotationRetention:WARN",
    "-Xep:StaticQualifiedUsingExpression:WARN",
    "-Xep:StaticOrDefaultInterfaceMethod:WARN",
    "-Xep:StringEquality:WARN",
    "-Xep:TestExceptionChecker:WARN",
    # TODO: stylistic changes in code
    # "-Xep:ThrowsUncheckedException:WARN",
    # "-Xep:UngroupedOverloads:WARN",
    "-Xep:UnlockMethodChecker:WARN",
    "-Xep:UnnecessaryDefaultInEnumSwitch:WARN",
    "-Xep:UnnecessaryStaticImport:WARN",
    "-Xep:UseBinds:WARN",
    "-Xep:VarChecker:WARN",
    "-Xep:WildcardImport:WARN",
    "-Xep:WrongParameterPackage:WARN",
]

EP_OPTS = EP_ENABLED_WARNINGS + EP_DISABLED_CHECKS

JAVACOPTS = JAVA_VERSION_OPTS + EP_OPTS
