# Python-(LG)CPD

Pure Numpy Implementation of the (Optionally Landmark-Guided) Coherent Point Drift Algorithm.

## Project Details

This is a pure numpy implementation of the coherent point drift `CPD <https://arxiv.org/abs/0905.2635/>`_ algorithm by Myronenko and Song. It provides three registration methods for point clouds: 
1. Scale and rigid registration
2. Affine registration
3. Gaussian regularized non-rigid registration

Each method also supports landmark-guided coherent point drift (LGCPD), as defined by the paper *Deformable Vessel-Based Registration Using Landmark-Guided Coherent Point Drift*. LGCPD is the same as CPD, except it takes as additional arguments a list of correponding source and target landmarks (points). This can be useful for helping to orient the source and target correctly. When include, the strength of landmarks's influence can be tuned with the positive scalar parameter `ss2`. Lower values of `ss2`

The CPD algorithm is a registration method for aligning two point clouds. In this method, the moving point cloud is modelled as a Gaussian Mixture Model (GMM) and the fixed point cloud are treated as observations from the GMM. The optimal transformation parameters maximze the Maximum A Posteriori (MAP) estimation that the observed point cloud is drawn from the GMM.

The registration methods work for 2D and 3D point clouds. For more information, please refer to my `blog <http://siavashk.github.io/2017/05/14/coherent-point-drift/>`_.

## Installation

### As a Submodule

1. Include this module as a submodule of your repo.
2. Include this module as you would others, using relative paths to import functions:
   ```python
     from pycpd.pycpd import *
   ```

### From Source

1. Clone the repository to a location, referred to as the ``root`` folder. For example:
   ```bash
   git clone https://github.com/bensnell/pycpd.git ~/dev/pycpd
   ```
2. Install the package:
   ```bash
   pip install .
   ```

## Usage

```python

TY, _ = RigidRegistration(
  X = ...,
  Y = ...,
  X_landmarks = ..., # include for LGCPD
  Y_landmarks = ..., # include for LGCPD
  ss2 = ...          # include for LGCPD
).register()

TY, _ = AffineRegistration(
  X = ...,
  Y = ...,
  X_landmarks = ..., # include for LGCPD
  Y_landmarks = ..., # include for LGCPD
  ss2 = ...          # include for LGCPD
).register()

TY, _ = DeformableRegistration(
  X = ...,
  Y = ...,
  X_landmarks = ..., # include for LGCPD
  Y_landmarks = ..., # include for LGCPD
  alpha = ...,        
  beta = ...,
  ss2 = ...          # include for LGCPD
).register()

```

## Troubleshooting

## Roadmap

## Notes

## License

MIT License