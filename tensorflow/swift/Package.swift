import PackageDescription

let package = Package(
    name: "TensorFlow",
    targets: [
        Target(name: "TensorFlow")
    ],
    dependencies: [
         .Package(url: "https://github.com/rxwei/CTensorFlow", majorVersion: 0)
    ]
)
