# PAWSS: Patch-based Adaptive Weighting with Segmentation and Scale
This is a C++ implementation of the tracking algorithm.
Tracking can be performed on video sequences.

## Requirements
* [OpenCV](http://opencv.org/)
* [Eigen](http://eigen.tuxfamily.org/)
* [Boost](http://www.boost.org/) filesystem system components

This code has been tested using Opencv 3.0, Boost 1.59.0 and Eigen 3.2.6.

## Compilation
CMake is used for cross-platform compilation.

**Note: make sure you compile the code in Release mode, as a Debug build will result in significantly slower performance.**

## Usage
Please see config.txt for configuration options.

## Data
The code is tested using the [Online Tracking Benchmark (OTB) dataset](https://sites.google.com/site/trackerbenchmark/benchmarks/v10). 

## License

## Acknowledgements
This code uses the Struck algorithm provided by [Sam Hare](https://github.com/samhare/struck).
