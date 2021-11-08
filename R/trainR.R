# Here's a good place to put your top-level package documentation

.onLoad <- function (lib, pkgname="subtyper") {
    ## Put stuff here you want to run when your package is loaded
    invisible()
}


#' Choose files by modification date
#'
#' @param fileDataFrame a dataframe containing files required for training some model
#' @param random boolean that will return a random choice of files
#' @param notFirst boolean that will return a random choice of files instead
#' though not the most recently modified
#' @return the recommended filenames to read right now
#' @author Avants BB
#' @examples
#' mydf = NULL
#' @export
chooseTrainingFilesToRead <- function( fileDataFrame, random=FALSE, notFirst=FALSE ) {
  # take the most recent of all file mod times for all rows
  extime = Sys.time()
  mytimes = rep( extime, nrow( fileDataFrame ) )
  for ( i in 1:nrow(fileDataFrame) ) {
    localtimes = rep( extime, ncol(fileDataFrame) )
    for ( j in 1:ncol(fileDataFrame) ) {
      myfn = as.character( fileDataFrame[i,j] )
      stopifnot( file.exists( myfn ) )
      localtimes[j] = R.utils::lastModified( myfn )
    }
    mytimes[ i ] = max( localtimes )
  }
  indices = rev(order(mytimes))
  myindex = indices[2] # default choice
  if ( notFirst ) {
    return( fileDataFrame[ sample( indices[-1], 1 ),] )
  } else if ( random ) {
    return( fileDataFrame[ sample( indices, 1 ),] )
  }
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
  numpynames = as.character( numpynames )
  masknameindex = grep("mask",numpynames)
  ccnameindex = grep("coordconv",numpynames)
  heatmapnameindex = grep("heatmap",numpynames)
  doMask = length( masknameindex) > 0
  doCC = length( ccnameindex ) > 0
  doHM = length( heatmapnameindex ) > 0
  outlist = list()
  for ( x in 1:length( numpynames ) ) {
    if ( ! file.exists( numpynames[ x ] ) )
      stop(paste(  numpynames[ x ],"does not exist on disk" ) )
    outlist[[length(outlist)+1]] = np$load( numpynames[ x ] )
    }
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
#' @param smoothHeatMaps numeric greater than zero will cause method to return heatmaps.
#' the value passed here also sets the smoothing parameter passed to \code{smoothImage}
#' in pixel/voxel space.
#' @param numpynames the names of the numpy on disk files should contain something with
#' the string mask if using maskIndex and something with the word coordconv if using CC.
#' should include something with the word heatmaps if using heatmaps.
#' @param numberOfSimulations number of output images.  Default = 10.
#' @param referenceImage defines the spatial domain for all output images.  If
#' the input images do not match the spatial domain of the reference image, we
#' internally resample the target to the reference image.  This could have
#' unexpected consequences.  Resampling to the reference domain is performed by
#' testing using \code{antsImagePhysicalSpaceConsistency} then calling
#' \code{resampleImageToTarget} upon failure.
#' @param transformType one of the following options
#' \code{c( "translation", "rigid", "scaleShear", "affine"," deformation" ,
#'   "affineAndDeformation" )}.
#' @param noiseModel one of the following options
#'   \code{c( "additivegaussian", "saltandpepper", "shot", "speckle" )}
#' @param noiseParameters 'additivegaussian': \code{c( mean, standardDeviation )},
#'   'saltandpepper': \code{c( probability, saltValue, pepperValue) }, 'shot':
#'    scale, 'speckle': standardDeviation.  Note that the standard deviation,
#'    scale, and probability values are *max* values and are randomly selected
#'    in the range \code{[0, noise_parameter]}.  Also, the "mean", "saltValue" and
#'    pepperValue" are assumed to be in the intensity normalized range of \code{[0, 1]}.
#' @param sdSimulatedBiasField Characterize the standard deviation of the amplitude.
#' @param sdHistogramWarping Determines the strength of the bias field.
#' @param sdAffine Determines the amount of transformation based change
#'
#' @return list of array
#' @author Avants BB
#' @importFrom ANTsRCore getCentroids iMath thresholdImage
#' @importFrom ANTsRCore smoothImage antsTransformPhysicalPointToIndex
#' @importFrom R.utils lastModified
#' @importFrom patchMatchR coordinateImages
#' @importFrom ANTsRNet dataAugmentation
#' @importFrom reticulate import
#' @examples
#' library( ANTsR )
#' ilist = list( list( ri(1) ), list( ri(2) ) )
#' slist = list(
#'   thresholdImage( ilist[[1]][[1]], "Otsu",3),
#'   thresholdImage( ilist[[2]][[1]], "Otsu",3) )
#' npn = paste0(tempfile(), c('i.npy','s.npy','heatmap.npy') )
#' temp = generateDiskData( ilist, slist, c(0:3), c(TRUE,TRUE), numpynames = npn  )
#' temp = generateDiskData( ilist, slist, c(0:3), c(TRUE,TRUE),
#'   segmentationsArePoints=TRUE, numpynames = npn )
#' temp = generateDiskData( ilist, slist, c(0:3), c(TRUE,TRUE),
#'   segmentationsArePoints=TRUE, smoothHeatMaps = 3.0, numpynames = npn )
#' @export
generateDiskData  <- function(
  inputImageList,
  segmentationImageList,
  segmentationNumbers,
  selector,
  addCoordConv = TRUE,
  segmentationsArePoints = FALSE,
  maskIndex,
  smoothHeatMaps = 0,
  numpynames,
  numberOfSimulations = 16,
  referenceImage = NULL,
  transformType = 'rigid',
  noiseModel = 'additivegaussian',
  noiseParameters = c( 0.0, 0.002 ),
  sdSimulatedBiasField = 0.0005,
  sdHistogramWarping = 0.0005,
  sdAffine = 0.2  ) {
  nClasses = length( segmentationNumbers )
  overindices = 1:length(inputImageList[[1]])
  if (  ! missing( maskIndex ) ) overindices = overindices[ -maskIndex ]
  np <- import("numpy")
  idim = dim( inputImageList[[1]][[1]] )
  myimgdim = length( idim )
  doMask = FALSE
  doCC = FALSE
  if ( addCoordConv  & ! missing( maskIndex ) )
    stopifnot( length( numpynames ) > 3 )

  X = array( dim = c( numberOfSimulations, idim, length(overindices) ) )
  if ( ! segmentationsArePoints ) {
    Y = array( dim = c( numberOfSimulations, idim, nClasses ) )
  } else {
    Y = array( dim = c( numberOfSimulations, nClasses, myimgdim ) )
  }
  if ( smoothHeatMaps > 0 & segmentationsArePoints ) {
    Yh = array( 0, dim = c( numberOfSimulations, idim, nClasses ) )
    if ( length( grep("heatmap",numpynames) ) == 0 )
      stop( "numpynames must have a name containing the string heatmap" )
    heatmapnameindex = grep("heatmap",numpynames)
  }
  if (! missing( maskIndex ) ) {
    doMask = TRUE
    Xm = array( dim = c( numberOfSimulations, idim, 1 ) )
    stopifnot( length( numpynames ) > 2 )
    if ( length( grep("mask",numpynames) ) == 0 )
      stop( "numpynames must have a name containing the string mask" )
    masknameindex = grep("mask",numpynames)
    }
  if ( addCoordConv ) {
    doCC = TRUE
    Xcc = array( dim = c( numberOfSimulations, idim, length(idim) ) )
    stopifnot( length( numpynames ) > 2 )
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
    referenceImage = referenceImage,
    verbose = FALSE )
  for ( k in 1:length(data$simulatedImages) ) {
    if ( addCoordConv ) {
      myccLocal = patchMatchR::coordinateImages( data$simulatedImages[[k]][[1]] * 0 + 1 )
      for ( jj in 1:myimgdim ) {
        if ( myimgdim == 2 ) Xcc[k,,,jj] = as.array( myccLocal[[jj]] )
        if ( myimgdim == 3 ) Xcc[k,,,,jj] = as.array( myccLocal[[jj]] )
      }
    }
    if ( ! missing( maskIndex ) ) {
      mymask = thresholdImage( data$simulatedImages[[k]][[maskIndex]], 0.5, Inf )
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
          if ( smoothHeatMaps > 0 & ( length(mypt) > 0 ) ) {
            mypti = antsTransformPhysicalPointToIndex( temp, mypt[,1:myimgdim] )
            heatmap = temp * 0.0
            if ( myimgdim == 2 ) heatmap[mypti[1,1],mypti[1,2]]=1
            if ( myimgdim == 3 ) heatmap[mypti[1,1],mypti[1,2],mypti[1,3]]=1
            heatmap = smoothImage( heatmap, smoothHeatMaps, sigmaInPhysicalCoordinates=FALSE )
            heatmap = heatmap / ( max( heatmap ) + 0.0001 )
            if ( myimgdim == 2 ) Yh[k, , , j ] = as.array( heatmap )
            if ( myimgdim == 3 ) Yh[k, , , , j ] = as.array( heatmap )
            }
          if ( length(mypt) == 0 ) mypt = rep( NA, myimgdim )
          Y[k, j, ] = mypt[1:myimgdim]
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
  if ( doCC > 0 ) {
    np$save( numpynames[ccnameindex], Xcc )
    outlist[[length(outlist)+1]] = Xcc
    }
  if ( smoothHeatMaps > 0 ) {
    np$save( numpynames[heatmapnameindex], Yh )
    outlist[[length(outlist)+1]] = Yh
    }
  return( outlist  )
  }
