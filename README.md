
# :tiger:Tiger-Time-varying-Diffusion-Model-for-Point-Cloud-Generation
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


## Training:

```bash
$ python train_generation.py --category car|chair|airplane
```

Please refer to the python file for optimal training parameters.


## Reference

@inproceedings{ren2024tiger,
  title={TIGER: Time-Varying Denoising Model for 3D Point Cloud Generation with Diffusion Process},
  author={Ren, Zhiyuan and Kim, Minchul and Liu, Feng and Liu, Xiaoming},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9462--9471},
  year={2024}
}
