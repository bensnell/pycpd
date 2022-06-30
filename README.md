# Python-(LG)CPD

Pure Numpy Implementation of Landmark-Guided Coherent Point Drift Algorithm. This module is an extended version of [pycpd](https://github.com/siavashk/pycpd), which doesn't provide the option of performing CPD using landmarks as guides.

## Project Details

This is a pure numpy implementation of the [coherent point drift](https://arxiv.org/abs/0905.2635/) (CPD) algorithm by Myronenko and Song. It provides three registration methods for 2D + 3D point clouds: 
1. Rigid (+ scale) registration
2. Affine registration
3. Deformable (gaussian regularized non-rigid) registration

The CPD algorithm is a registration method for aligning two point clouds. In this method, the moving point cloud is modelled as a Gaussian Mixture Model (GMM) and the fixed point cloud are treated as observations from the GMM. The optimal transformation parameters maximze the Maximum A Posteriori (MAP) estimation that the observed point cloud is drawn from the GMM.

Each registration method also supports landmark-guided coherent point drift (LGCPD), as defined by the paper *Deformable Vessel-Based Registration Using Landmark-Guided Coherent Point Drift*. LGCPD is the same as CPD, except it takes as additional arguments a list of correponding source and target landmarks (points). This can be useful for helping to orient the source and target correctly. When included, the strength of landmarks's influence can be tuned with the positive scalar parameter `ss2`. Lower values of `ss2` indicate higher landmark influence and higher values indicate lower influence.

### Dependencies

- Numpy

### Resources

- [CPD Tutorial](http://siavashk.github.io/2017/05/14/coherent-point-drift/)

## Installation

### Using Pip

```bash
pip install git+https://github.com/bensnell/pylgcpd.git
```

### From Source

1. Clone the repository to a location, referred to as the `root` folder. For example:
   ```bash
   git clone https://github.com/bensnell/pylgcpd.git ~/dev/pylgcpd
   ```
2. Install the package:
   ```bash
   pip install .
   ```

### As a Submodule of Your Project Repo

1. Include this module as a submodule of your repo.
   ```bash
   git submodule add https://github.com/bensnell/pylgcpd.git path/to/repo/pylgcpd
2. Include this module as you would others, using relative paths to import functions:
   ```python
     from pylgcpd.pylgcpd import *
   ```

## Usage

```python
TY, _ = RigidRegistration(
  X = ...,
  Y = ...,
  X_landmarks = ..., # optional (include for LGCPD)
  Y_landmarks = ..., # optional (include for LGCPD)
  ss2 = ...          # optional (include for LGCPD)
).register()

TY, _ = AffineRegistration(
  X = ...,
  Y = ...,
  X_landmarks = ..., # optional (include for LGCPD)
  Y_landmarks = ..., # optional (include for LGCPD)
  ss2 = ...          # optional (include for LGCPD)
).register()

TY, _ = DeformableRegistration(
  X = ...,
  Y = ...,
  X_landmarks = ..., # optional (include for LGCPD)
  Y_landmarks = ..., # optional (include for LGCPD)
  alpha = ...,        
  beta = ...,
  ss2 = ...          # optional (include for LGCPD)
).register()
```

## Troubleshooting

## Roadmap

## Notes

## License

MIT License