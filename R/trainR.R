# Here's a good place to put your top-level package documentation

.onLoad <- function (lib, pkgname="subtyper") {
    ## Put stuff here you want to run when your package is loaded
    invisible()
}


#' Choose files by modification date
#'
#' @param fileDataFrame a dataframe containing files required for training some model
#' @return the recommended filenames to read right now
#' @author Avants BB
#' @examples
#' mydf = NULL
#' @export
chooseTrainingFilesToRead <- function( fileDataFrame ) {
  # take the most recent of all file mod times for all rows
  extime = Sys.time()
  mytimes = rep( extime, nrow( fileDataFrame ) )
  for ( i in 1:nrow(fileDataFrame) ) {
    localtimes = rep( extime, ncol(fileDataFrame) )
    for ( j in 1:ncol(fileDataFrame) ) {
      localtimes[j] = R.utils::lastModified( as.character( fileDataFrame[i,j] ) )
    }
    mytimes[ i ] = max( localtimes )
  }
  myindex = rev(order(mytimes))[2]
  return( fileDataFrame[myindex,] )
}


#' Read augmentation data from on disk storage
#'
#' @param numpynames the names of the numpy on disk files should contain something with
#' the string mask if using maskIndex and something with the word coordconv if using CC
#'
#' @return list of arrays in order of numpynames
#' @author Avants BB
#' @examples
#' mydf = NULL
#' @export
loadNPData <- function( numpynames ) {
  np <- import("numpy")
  masknameindex = grep("mask",numpynames)
  ccnameindex = grep("coordconv",numpynames)
  doMask = length( masknameindex > 0  )
  doCC = length( ccnameindex > 0  )
  outlist = list()
  for ( x in 1:length( numpynames ) )
    outlist[[length(outlist)+1]] = np$load( numpynames[ x ] )
  return( outlist )
}


#' Generate augmentation data with on disk storage
#'
#' @param inputImageList list of lists of input images to warp.  The internal
#'          list sets contains one or more images (per subject) which are
#'          assumed to be mutually aligned.  The outer list contains
#'          multiple subject lists which are randomly sampled to produce
#'          output image list.
#' @param segmentationImageList of segmentation images corresponding to the input image list.
#' @param segmentationNumbers the integer list of values in the segmentation to model
#' @param selector subsets the inputImageList and segmentationImageList (eg to define train test splits)
#' @param addCoordConv boolean - generates another array with CoordConv data
#' @param segmentationsArePoints boolean - converts segmentations to points
#' @param maskIndex the entry within the list of lists that contains a mask
#' @param numpynames the names of the numpy on disk files should contain something with
#' the string mask if using maskIndex and something with the word coordconv if using CC
#' @param ... additional arguments passed to the ANTsRNet function dataAugmentation
#'
#' @return list of array
#' @author Avants BB
#' @importFrom ANTsRCore getCentroids iMath thresholdImage
#' @importFrom R.utils lastModified
#' @importFrom patchMatchR coordinateImages
#' @importFrom ANTsRNet dataAugmentation
#' @importFrom reticulate import
#' @examples
#' mydf = NULL
#' @export
generateDiskData  <- function(
  inputImageList,
  segmentationImageList,
  segmentationNumbers,
  selector,
  addCoordConv=0,
  segmentationsArePoints = FALSE,
  maskIndex,
  numpynames,
  ... ) {
  nClasses = length( segmentationNumbers )
  overindices = 1:length(inputImageList[[1]])
  if (  ! missing( maskIndex ) ) overindices = overindices[ -maskIndex ]
  np <- import("numpy")
  idim = dim( inputImageList[[1]][[1]] )
  myimgdim = length( idim )
  doMask = FALSE
  doCC = FALSE
  if ( addCoordConv > 0  & ! missing( maskIndex ) )
    stopifnot( length( numpynames > 3 ) )
  if ( missing( sdAffine ) ) sdAffine = 0.3
  if ( missing( noiseParameters ) ) noiseParameters = c( 0, 0.01 )
  if ( missing( sdSimulatedBiasField ) ) sdSimulatedBiasField = 0.001
  if ( missing( sdHistogramWarping ) ) sdHistogramWarping = 0.001
  if ( missing( transformType ) ) transformType = 'rigid'
  if ( missing( numberOfSimulations ) ) numberOfSimulations = 128

  X = array( dim = c( numberOfSimulations, idim, length(overindices) ) )
  if ( ! segmentationsArePoints ) {
    Y = array( dim = c( numberOfSimulations, idim, nClasses ) )
  } else {
    Y = array( dim = c( numberOfSimulations, nClasses, idim ) )
  }
  if (! missing( maskIndex ) ) {
    doMask = TRUE
    Xm = array( dim = c( numberOfSimulations, idim, 1 ) )
    stopifnot( length( numpynames > 2 ) )
    if ( length( grep("mask",numpynames) ) == 0 )
      stop( "numpynames must have a name containing the string mask" )
    masknameindex = grep("mask",numpynames)
    }
  if ( addCoordConv > 0 ) {
    doCC = TRUE
    Xcc = array( dim = c( numberOfSimulations, idim, length(idim) ) )
    stopifnot( length( numpynames > 2 ) )
    if ( length( grep("coordconv",numpynames) ) == 0 )
      stop( "numpynames must have a name containing the string coordconv" )
    ccnameindex = grep("coordconv",numpynames)
    }
  data <- dataAugmentation(
    inputImageList[selector],
    segmentationImageList[selector],
    transformType = transformType,
    numberOfSimulations = numberOfSimulations,
    sdAffine = sdAffine,
    noiseParameters = noiseParameters,
    sdSimulatedBiasField = sdSimulatedBiasField,
    sdHistogramWarping = sdHistogramWarping,
    verbose = FALSE )
  for ( k in 1:length(data$simulatedImages) ) {
    if ( addCoordConv > 0 ) {
      myccLocal = patchMatchR::coordinateImages( data$simulatedImages[[k]][[1]] * 0 + 1 )
      for ( jj in 1:myimgdim ) {
        if ( myimgdim == 2 ) Xcc[k,,,jj] = as.array( myccLocal[[jj]] )/addCoordConv
        if ( myimgdim == 3 ) Xcc[k,,,,jj] = as.array( myccLocal[[jj]] )/addCoordConv
      }
    }
    if ( ! missing( maskIndex ) ) {
      mymask = thresholdImage( data$simulatedImages[[k]][[maskIndex]], 1e-8, Inf )
      if ( myimgdim == 2 ) Xm[k,,,1] = as.array( mymask )
      if ( myimgdim == 3 ) Xm[k,,,,1] = as.array( mymask )
    }
    for ( j in overindices ) {
      temp = iMath( data$simulatedImages[[k]][[j]], "Normalize")
      if ( myimgdim == 2 ) X[k, , , j ] = as.array( temp )
      if ( myimgdim == 3 ) X[k, , , , j ] = as.array( temp )
      }
    if ( ! segmentationsArePoints ) {
      for ( j in 1:nClasses ) {
        temp = thresholdImage( data$simulatedSegmentationImages[[k]][[1]],
          segmentationNumbers[j], segmentationNumbers[j] )
        if ( myimgdim == 2 ) Y[k, , , j ] = as.array( temp )
        if ( myimgdim == 3 ) Y[k, , , , j ] = as.array( temp )
        }
      } else {
        for ( j in 1:nClasses ) {
          temp = thresholdImage( data$simulatedSegmentationImages[[k]][[1]],
            segmentationNumbers[j], segmentationNumbers[j]  )
          mypt = getCentroids( temp, clustparam=0  )
          if ( length(mypt) == 0 ) mypt = rep( NA, myimgdim )
          Y[k, j, ] = mypt
          }
        }
    }
  np$save( numpynames[1], X )
  np$save( numpynames[2], Y )
  outlist = list(X,Y)
  if ( doMask ) {
    np$save( numpynames[masknameindex], Xm )
    outlist[[length(outlist)+1]] = Xm
    }
  if ( doCC ) {
    np$save( numpynames[ccnameindex], Xcc )
    outlist[[length(outlist)+1]] = Xcc
    }
  return( outlist  )
  }
