"""KleidiAI library."""

load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

def repo():
    tf_http_archive(
        name = "KleidiAI",
        sha256 = "6b3e6630be314a28f6ea28fed14f7109b0b7c472f1e06d2dba17ffccda3b9466",
        strip_prefix = "kleidiai-dce86647385ab2638aa5abebcb652f3e4271970d",
        urls = tf_mirror_urls("https://gitlab.arm.com/kleidi/kleidiai/-/archive/dce86647385ab2638aa5abebcb652f3e4271970d/kleidiai-dce86647385ab2638aa5abebcb652f3e4271970d.zip"),
    )
