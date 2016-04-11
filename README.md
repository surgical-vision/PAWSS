# PAWSS: Patch-based Adaptive Weighting with Segmentation and Scale
This is a C++ implementation of the tracking algorithm.
Tracking can be performed on video sequences.

## Requirements
* [OpenCV](http://opencv.org/)
* [Eigen](http://eigen.tuxfamily.org/)
* [Boost](http://www.boost.org/) filesystem system components

This code has been tested using OpenCV 3.0.0, Boost 1.59.0 and Eigen 3.2.6.

**Note: We notice that if you are using OpenCV 3.1.0 built with Homebrew, VideoCapture and waitKey crashes after a while on OS X 10.11 as listed [here](https://github.com/Itseez/opencv/issues/5874). The simplest solution is reverting OpenCV back to 3.0.0.**

## Compilation
CMake is used for cross-platform compilation. For example on Unit-based systems run:
```
> mkdir build
> cd build
> cmake ..
> make
```
**Note: make sure you compile the code in Release mode, as a Debug build will result in significantly slower performance.**

## Usage
After compilation, from the top level of the repository run:
```
> build/bin/PAWSS [config-file-path]
```
If no path is given the application will attempt to use ./config.txt.

Please see config.txt for configuration options.

## Data
The code is tested using the [Online Tracking Benchmark (OTB) dataset](https://sites.google.com/site/trackerbenchmark/benchmarks/v10). 

## License

## Acknowledgements
This code uses the Struck algorithm provided by [Sam Hare](https://github.com/samhare/struck).
