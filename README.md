# surgeRy

[![CircleCI](https://circleci.com/gh/stnava/surgeRy/tree/master.svg?style=svg)](https://circleci.com/gh/stnava/surgeRy/tree/master)

documentation page [here](https://stnava.github.io/surgeRy/)

## deep learning training

Common tasks for data augmentation of images, segmentations and points.

Appropriate for cascaded models.

## Installing

The pre-release version of the package can be pulled from GitHub using the [devtools](https://github.com/r-lib/devtools) package:

```r
    # install.packages("devtools")
    devtools::install_github("stnava/surgeRy", build_vignettes=TRUE)
```

this lets you access vignettes via:

```
vignette(package='surgeRy')
```

which will list the current vignettes.

## For developers

The repository includes a Makefile to facilitate some common tasks.

### Running tests

`$ make test`. Requires the [testthat](http://testthat.r-lib.org/) package. You can also specify a specific test file or files to run by adding a "file=" argument, like `$ make test file=logging`. `testthat::test_package()` will do a regular-expression pattern match within the file names (ignoring the `test-` prefix and the `.R` file extension).

### Updating documentation

`$ make doc`. Requires the [roxygen2](https://github.com/klutometis/roxygen) package.


### References to the concepts used here

* Cephalometric Landmark Regression with Convolutional Neural Networks on 3D Computed Tomography Data ( a review paper )

* Numerical Coordinate Regression with Convolutional Neural Networks

* Human pose regression by combining indirect part detection and contextual information (section 3.2)

* Automatic 3d cephalometric annotation system using shadowed 2d image-based machine learning
