# Tiger-Time-varying-Diffusion-Model-for-Point-Cloud-Generation
This is the official code for the CVPR 2024 Publication: Tiger: Time-Varying Denoising Model for 3D Point Cloud Generation with Diffusion Process

## Requirements:

Make sure the following environments are installed.
(Code tested with cuda11)

```
python
pytorch
torchvision
matplotlib
tqdm
open3d
trimesh
scipy
```

Install PyTorchEMD by
```
cd metrics/PyTorchEMD
python setup.py install
cp build/**/emd_cuda.cpython-36m-x86_64-linux-gnu.so .
```
