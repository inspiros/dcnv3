DCNv3 [![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/inspiros/dcnv3/build_wheels.yml)](https://github.com/inspiros/dcnv3/actions) [![GitHub](https://img.shields.io/github/license/inspiros/dcnv3)](LICENSE.txt)
========

This repo contains the implementation of the **DCNv3** introduced in the paper
[InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions](https://arxiv.org/abs/2211.05778).
The official implementation is ported here with minimal changes for the purpose of testing.
Original source can be found at https://github.com/OpenGVLab/InternImage.

## Remark

Guess what? **DCNv3** is nothing like its first (https://arxiv.org/abs/1703.06211) and
second (https://arxiv.org/abs/1811.11168) predecessors.
It turned out to not even remotely a Convolution operation.

## Requirements

- `torch>=1.9.0`

## Installation

#### From TestPyPI:

```terminal
pip install --index-url https://test.pypi.org/simple/ dcnv3
```

#### From Source:

Clone this repo and execute the following command:

```terminal
pip install .
```

Or just compile the binary for inplace usage:

```terminal
python setup.py build_ext --inplace
```

## License

The code is released under the MIT-0 license. Feel free to use. See [`LICENSE.txt`](LICENSE.txt) for details.
