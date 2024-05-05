# hoshikasane
Align stars and stack image CLI tool.

1. Reshape target image fit to base image.
```shell
$ cargo run -- affine-convert --base <base_file_path> --target <target_file_path>
```
Output following files (now include some files for DEBUG):
- matches.tiff: match feature points between base and target image
- base.tiff: base image
- target.tiff: target image
- converted.tiff: target image which reshape fit to base image

2. Lighten composite
```shell
$ cargo run -- composite --file base.tiff,converted.tiff --output output.tiff
```