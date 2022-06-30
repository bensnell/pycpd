# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [3.0.0] - 2022-06-30

### Added
- Support for landmark-guided coherent point drift across rigid, affine and deformable registration.
- Support for normalization of input data (& denormalization of output data), in line with original Matlab implementation.

### Changed
- Name of package to `pylgcpd` to reflect its ability to perform Landmark-Guided Coherent Point Drift.

## [2.0.0] - 2020-04-05

## [1.0.5]

### Fixed
- Fixed a leftover extra transpose operation when transforming a point using rigid registration parameters.

## [1.0.4]

### Fixed
- Fixed a bug where updated variance for deformable registration was wrong.

### Removed
- Removed extra transpose operations on rotation and translation.

## [1.0.3]

### Changed
- Narrowed supported Python versions to what Travis supports for CI.

## [1.0.2]

### Fixed
- Fixed Python 3.x compatibility for inheriting from the base class.

## [1.0.0]

### Changed
- All registration algorithms now inherit from expectation_maximization_registration class to remove duplicate code.
- All functions and classes follow Python's PEP8 style.

## [0.0.4]

### Fixed
- Fixed the addition of the uniform distribution to the mixture model to account for noise and outliers.

## [0.0.3]

### Fixed
= Fixed the mutability of the moving point cloud.

## [0.0.2]

### Fixed
- Fixed the compatibility of pycpd with Python 3.x

## [0.0.1]

### Fixed
- Fixed a bug for registering 3D point clouds. Added 3D examples.

## [0.0.0]

### Added
- Initial release with rigid, affine and deformable registration methods for 2D point clouds.
