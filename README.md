# surgeRy

[![CircleCI](https://circleci.com/gh/stnava/surgeRy/tree/master.svg?style=svg)](https://circleci.com/gh/stnava/surgeRy/tree/master)

documentation page [here](https://stnava.github.io/surgeRy/)

## deep learning training

Common tasks for


## Installing

The pre-release version of the package can be pulled from GitHub using the [devtools](https://github.com/r-lib/devtools) package:

```r
    # install.packages("devtools")
    devtools::install_github("stnava/surgeRy", build_vignettes=TRUE)
```

## For developers

The repository includes a Makefile to facilitate some common tasks.

### Running tests

`$ make test`. Requires the [testthat](http://testthat.r-lib.org/) package. You can also specify a specific test file or files to run by adding a "file=" argument, like `$ make test file=logging`. `testthat::test_package()` will do a regular-expression pattern match within the file names (ignoring the `test-` prefix and the `.R` file extension).

### Updating documentation

`$ make doc`. Requires the [roxygen2](https://github.com/klutometis/roxygen) package.
