📝 Summary of Changes
Please provide a clear and concise summary of the changes you've made.

🎯 Justification
Explain why this change is important and which workload benefits from this
change.

🚀 Kind of Contribution
Please remove what does not apply: 🐛 Bug Fix, ⚡️ Performance Improvement,
✨ New Feature, ♻️ Cleanup, 📚 Documentation, 🧪 Tests

📊 Benchmark (for Performance Improvements)
Please measure and include speedups for one of the public HLOs in
`compiler/xla/tools/benchmarks/hlo/`.

🧪 Unit Tests:
What unit tests were added? For example, a new pass should be tested on minimal
HLO. The transformation can be tested with FileCheck tests or assertions on the
transformed HLO.

🧪 Execution Tests:
What execution tests were added? For example, a new optimization should be
tested with an end-to-end execution test triggering the optimization and
asserting correctness. Please provide test cases running with at most 2 GPUs.
