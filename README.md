![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)

# EMDF

Keunsoo Ko, Seunggyun Woo, and Chang-Su Kim

Official PyTorch Code for "Error-Mask-Adaptive Dynamic Filtering for Image Inpainting, IEEE Access, vol. 13, pp. 18403--18417 2025"

### Installation
Download repository:
```
    $ git clone https://github.com/keunsoo-ko/EMDF.git
```
Download [pre-trained model on Places2](https://www.dropbox.com/scl/fi/6bptz5bote44nezl2ab0u/Places2.pth?rlkey=tb9z07s2t9l5lkg8aj392ph7z&st=c327zq2e&dl=0) or [pre-trained model on CelebA](https://www.dropbox.com/scl/fi/w8drq1m488o4tya7hx8z1/CelebA.pth?rlkey=4onc3cse5fwt053bk48zyje0i&st=dxjq4aw7&dl=0)

### Usage
Run Test:
```
    $ python demo.py --ckpt your_(ckpt)file_name.pth(put downloaded model path) --path ./example/Places2 --output_path ./results
```
