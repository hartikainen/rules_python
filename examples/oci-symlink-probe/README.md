# OCI symlink probe

This example exercises the container packaging failure in
[`rules_python#3388`](https://github.com/bazel-contrib/rules_python/issues/3388).
It uses the enclosing `rules_python` checkout with `venvs_site_packages=yes`
and `venvs_use_declare_symlink=yes`.

The `tar.bzl` `0.10.8` dependency carries `tar-preserve-symlinks.patch`, which
preserves generated external output paths during symlink normalization.
Both `rules_oci` and `rules_img` package the same tar layer, produced with
`mtree_mutate(..., preserve_symlinks=True)`. The `//symlink_probe:load` target
also exercises native `rules_img` packaging without `tar.bzl`.

Run from this directory with Bazelisk and Docker. The images target
`linux/arm64` (other Docker hosts need emulation).

```sh
bazel build //symlink_probe:oci_tar
docker load --input bazel-bin/symlink_probe/oci_load/tarball.tar
docker run --rm --pull=never --network=none --read-only \
    --platform=linux/arm64 codex-symlink-probe:rules-oci-fixed-tar

bazel build //symlink_probe:preserved_image_tar
docker load --input bazel-bin/symlink_probe/preserved_load_docker.tar
docker run --rm --pull=never --network=none --read-only \
    --platform=linux/arm64 codex-symlink-probe:tar-bzl-0.10.8
```

The probe checks the interpreter and `absl` symlinks, imports NumPy, performs
an array computation, and checks every symlink encountered in the runfiles
tree for a relative, existing target. It does not traverse symlinked
directories or execute the generated console scripts.

To reproduce the generated-file failure, remove `single_version_override`
from `MODULE.bazel` and repeat the commands. With unpatched `tar.bzl` `0.10.8`,
the container builds, but the probe finds dangling links for generated files.
The manual `//symlink_probe:raw_tar` target separately demonstrates packaging
without symlink preservation.

To test a `tar.bzl` checkout, pass `--override_module=tar.bzl=/path/to/tar.bzl`
to either `bazel build` command. The checkout overrides the bundled patch.
