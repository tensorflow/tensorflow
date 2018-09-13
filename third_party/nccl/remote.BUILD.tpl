licenses(["restricted"])

package(default_visibility = ["//visibility:public"])

alias(name="LICENSE", actual = "%{target}:LICENSE")
alias(name = "nccl", actual = "%{target}:nccl")
