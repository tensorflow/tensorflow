#ifdef GET_SUPPORTED_OP_LIST
#undef GET_SUPPORTED_OP_LIST

// clang-format off

// Unary Elementwise Ops
lmhlo::AbsOp, lmhlo::CeilOp, lmhlo::FloorOp, lmhlo::ConvertOp, lmhlo::CosOp,
lmhlo::ExpOp, lmhlo::LogOp, lmhlo::NegOp, lmhlo::RsqrtOp, lmhlo::SqrtOp,
lmhlo::SignOp, lmhlo::TanhOp,

// Binary Elementwise Ops
lmhlo::AddOp, lmhlo::DivOp, lmhlo::MaxOp, lmhlo::MinOp, lmhlo::MulOp,
lmhlo::RemOp, lmhlo::SubOp, lmhlo::AndOp, lmhlo::OrOp, lmhlo::CompareOp,
lmhlo::PowOp,

// Ternary Elementwise Ops
lmhlo::SelectOp,

// Other Ops
lmhlo::ConstOp,
lmhlo::RealDynamicSliceOp,
lmhlo::DynamicBroadcastInDimOp,
lmhlo::BroadcastInDimOp

// clang-format on

#endif
