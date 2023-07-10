# karpathy-guides

This repo is for following along in Andrej Karpathy's [series](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) on ML.

## Environment

```bash
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip
$ pip3 install -r requirements.txt
```

## Jax and Metal

See https://developer.apple.com/metal/jax/

Required manually setting a MacOS X SDK to build:
```
$ python build/build.py --bazel_options=--@xla//xla/python:enable_tpu=true --bazel_options="--macos_sdk_version=13.1"
```
