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
  fex = file.exists(fileDataFrame[,1])
  if ( sum( fex == TRUE )  < 1 ) stop("Training files do not exist yet")
  fileDataFrameUse = fileDataFrame[ fex, ]
  mytimes = rep( extime, nrow( fileDataFrameUse ) )
  for ( i in 1:nrow(fileDataFrameUse) ) {
    localtimes = rep( extime, ncol(fileDataFrameUse) )
    for ( j in 1:ncol(fileDataFrameUse) ) {
      myfn = as.character( fileDataFrameUse[i,j] )
      if ( ! file.exists( myfn ) ) {
        localtimes[j] = NA
      } else localtimes[j] = R.utils::lastModified( myfn )
    }
    mytimes[ i ] = max( localtimes, na.rm=T )
  }
  indices = rev(order(mytimes))
  myindex = indices[2] # default choice
  if ( notFirst ) {
    return( fileDataFrameUse[ sample( indices[-1], 1 ),] )
  } else if ( random ) {
    return( fileDataFrameUse[ sample( indices, 1 ),] )
  }
  return( fileDataFrameUse[myindex,] )
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


#' Generate segmentation-based augmentation data with on disk storage
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
#' @importFrom ANTsRCore getCentroids iMath thresholdImage antsCopyImageInfo2 cropIndices makeImage resampleImageToTarget
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
#' npn = paste0(tempfile(), c('i.npy','s.npy','heatmap.npy','coordconv.npy') )
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


#' special cropping
#'
#' cropping method specialized for our data augmenation approach
#' @param x input antsImage
#' @param pt point in physical space around which we crop
#' @param domainer vector of dimensionality equal to the image indicating the
#' size of cropped region
#' @return cropped region
#' @author Avants BB
#' @examples
#' library( ANTsR )
#' specialCrop( ri(1), c(60,66), c(32,32) )
#' @export
specialCrop <- function( x, pt, domainer=NULL ) {
  if ( is.null( domainer ) ) return( x )
  pti = round( antsTransformPhysicalPointToIndex( x, pt ) )
  xdim = dim( x )
  for ( k in 1:x@dimension ) {
    if ( pti[k] < 1 ) pti[k]=1
    if ( pti[k] > xdim[k] ) pti[k]=xdim[k]
  }
  mim = makeImage( domainer )
  ptioff = pti - round( dim( mim ) / 2 )
  domainerlo = ptioff
  domainerhi = ptioff
  loi = cropIndices( x, domainerlo, domainerhi )
  mim = antsCopyImageInfo2( mim, loi )
  resampleImageToTarget( x, mim )
}


#' Generate point set and image augmentation data with on disk storage
#'
#' This assumes that at least images and point sets are passed in - segmentation
#' images are optional.  The indexing of point sets will be whichSimulation by
#' whichPoint by dimensionality (e.g. 10 x 25 x 3 for 10 simulations of point sets
#' with 25 points in 3D.
#'
#' @param inputImageList list of lists of input images to warp.  The internal
#'          list sets contains one or more images (per subject) which are
#'          assumed to be mutually aligned.  The outer list contains
#'          multiple subject lists which are randomly sampled to produce
#'          output image list.
#' @param pointsetList list of matrices containing point sets where each matrix
#' is of size n-points by dimensionality matching to input segmentation/image data (optional)
#' @param segmentationImageList of segmentation images corresponding to the input image list (optional)
#' @param segmentationNumbers the integer list of values in the segmentation to model
#' @param selector subsets the input lists (eg to define train test splits)
#' @param maskIndex the entry within the list of lists that contains a mask
#' @param smoothHeatMaps numeric greater than zero will cause method to return heatmaps.
#' the value passed here also sets the smoothing parameter passed to \code{smoothImage}
#' in pixel/voxel space.
#' @param numpynames the names of the numpy on disk files should include something
#' with the word pointset. Also:
#' the string segmentation if using segmentations and
#' the string mask if using maskIndex and something with the word coordconv if using CC.
#' should include something with the word heatmaps if using heatmaps.
#' @param cropping a vector of size 1 plus image dimensionality where the first
#' parameter indicates which landmark to target and the trailing parameters define
#' the size of the cropping patch.  e.g. \code{c(2,32,32,64)} would crop a box
#' of size 32 by 32 by 64 around landmark 2.
#' @param numberOfSimulations number of output images/pointsets.  Default = 10.
#' @param referenceImage defines the spatial domain for all output images.  If
#' the input images do not match the spatial domain of the reference image, we
#' internally resample the target to the reference image.  This could have
#' unexpected consequences.  Resampling to the reference domain is performed by
#' testing using \code{antsImagePhysicalSpaceConsistency} then calling
#' \code{resampleImageToTarget} upon failure.
#' @param transformType one of the following options
#' \code{c( "translation", "rigid", "scaleShear", "affine" )}.  Non-invertible
#' models will not work with point sets.
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
#' library( reticulate )
#' library( ANTsR )
#' library( ANTsRNet )
#' library( surgeRy )
#' image1 <- antsImageRead( getANTsRData( "r16" ) )
#' image2 <- antsImageRead( getANTsRData( "r27" ) )
#' segmentation1 <- thresholdImage( image1, "Otsu", 3 )
#' segmentation11 = thresholdImage( segmentation1, 1, 1 )
#' segmentation12 = thresholdImage( segmentation1, 2, 2 )
#' segmentation13 = thresholdImage( segmentation1, 3, 3 )
#' segmentation11[1:128,1:256]=0
#' segmentation12[1:256,1:180]=0
#' segmentation13[1:256,1:128]=0
#' segmentation1 = segmentation11 + segmentation12* 2 + segmentation13 * 3
#' segmentation2 <- thresholdImage( image2, "Otsu", 3 )
#' segmentation21 = thresholdImage( segmentation2, 1, 1 )
#' segmentation22 = thresholdImage( segmentation2, 2, 2 )
#' segmentation23 = thresholdImage( segmentation2, 3, 3 )
#' segmentation21[1:128,1:256]=0
#' segmentation22[1:256,1:180]=0
#' segmentation23[1:256,1:128]=0
#' segmentation2 = segmentation21 + segmentation22* 2 + segmentation23 * 3
#' pts1 = getCentroids( segmentation1 )[,1:2]
#' pts2 = getCentroids( segmentation2 )[,1:2]
#' plist = list( pts1, pts2)
#' ilist = list( list( image1 ), list( image2 ) )
#' slist = list( segmentation1, segmentation2 )
#' npn = paste0(tempfile(), c('i.npy','pointset.npy','heatmap.npy','coordconv.npy','segmentation.npy') )
#' temp1 = generateDiskPointAndSegmentationData( ilist, plist, slist,
#'    segmentationNumbers = 1:3, numpynames = npn )
#' temp2 = generateDiskPointAndSegmentationData( ilist, plist,
#'    segmentationNumbers = 1:3, numpynames = npn, smoothHeatMaps = 3 )
#' temp = generateDiskPointAndSegmentationData( ilist, plist, slist,
#'    segmentationNumbers = 1:3, numpynames = npn, smoothHeatMaps = 3 )
#' locimg=as.antsImage( temp$images[1,,,1] )
#' locseg=as.antsImage( temp$segmentation[1,,,3] )
#' locseg2=as.antsImage( temp$segmentation[2,,,3] )
#' locimg2=as.antsImage( temp$images[2,,,1] )
#' # layout(matrix(1:4,nrow=1))
#' # plot(locimg,locseg)
#' # plot(locimg,as.antsImage( temp$heatmaps[1,,,3]))
#' # plot(locimg2,locseg2)
#' # plot(locimg2,as.antsImage( temp$heatmaps[2,,,3]))
#' print(getCentroids( thresholdImage(as.antsImage( temp$heatmaps[1,,,1]),0.5,1) ))
#' print(getCentroids( thresholdImage(as.antsImage( temp$heatmaps[1,,,3]),0.5,1) ))
#' print(temp$points[1,1,])
#' print(temp$points[1,3,])
#' mm = makePointsImage( temp$points[1,,], getMask( locimg ) )
#' mm2 = makePointsImage( temp$points[2,,], getMask( locimg2 ) )
#' gg1 = thresholdImage(as.antsImage( temp$heatmaps[1,,,1]),0.5,1) %>% antsCopyImageInfo2(ri(1))
#' gg2 = thresholdImage(as.antsImage( temp$heatmaps[1,,,2]),0.5,1) %>% antsCopyImageInfo2(ri(1))
#' gg3 = thresholdImage(as.antsImage( temp$heatmaps[1,,,3]),0.5,1) %>% antsCopyImageInfo2(ri(1))
#' # plot( locimg, mm )
#' # plot( locimg, gg1*1+gg2*2+gg3*3 )
#' @export
generateDiskPointAndSegmentationData  <- function(
  inputImageList,
  pointsetList,
  segmentationImageList,
  segmentationNumbers,
  selector,
  maskIndex,
  smoothHeatMaps = 0,
  numpynames,
  cropping = NULL,
  numberOfSimulations = 16,
  referenceImage = NULL,
  transformType = 'rigid',
  noiseModel = 'additivegaussian',
  noiseParameters = c( 0.0, 0.002 ),
  sdSimulatedBiasField = 0.0005,
  sdHistogramWarping = 0.0005,
  sdAffine = 0.2  ) {
  addCoordConv = TRUE
  nClasses = 0
  hasPoints = ! missing( pointsetList )
  stopifnot( hasPoints )
  stopifnot( ! missing( inputImageList ) )
  hasSeg = ! missing( segmentationImageList )
  if ( hasSeg )
    nClasses = length( segmentationNumbers )
  overindices = 1:length(inputImageList[[1]])
  if (  ! missing( maskIndex ) ) overindices = overindices[ -maskIndex ]
  np <- import("numpy")
  idim = dim( inputImageList[[1]][[1]] )
  myimgdim = length( idim )
  doMask = FALSE

  nPoints = nrow( pointsetList[[1]] )
  for ( j in 1:length(pointsetList) )
    stopifnot( nPoints ==  nrow( pointsetList[[j]] ) )

  whichPoints = 1:nPoints
  doCrop = ! is.null( cropping )
  if ( doCrop ) {
    stopifnot( length( cropping ) == (myimgdim+1) )
    idim = cropping[ -1 ]
    whichPoints = cropping[1]
    nPoints = length( whichPoints )
    }

  Ypt = array( dim = c( numberOfSimulations, nPoints, myimgdim ) )
  if ( length( grep("pointset",numpynames) ) == 0 )
    stop( "numpynames must have a name containing the string pointset" )
  pointsetnameindex = grep("pointset",numpynames)

  if ( addCoordConv  & ! missing( maskIndex ) )
    stopifnot( length( numpynames ) > 3 )

  X = array( dim = c( numberOfSimulations, idim, length(overindices) ) )
  if ( hasSeg ) {
    Y = array( dim = c( numberOfSimulations, idim, nClasses ) )
    if ( length( grep("segmentation",numpynames) ) == 0 )
      stop( "numpynames must have a name containing the string segmentation" )
    segmentationnameindex = grep("segmentation",numpynames)
  }
  if ( smoothHeatMaps > 0 & hasPoints ) {
    if ( length( grep("heatmap",numpynames) ) == 0 )
      stop( "numpynames must have a name containing the string heatmap" )
    heatmapnameindex = grep("heatmap",numpynames)
    Yh = array( 0, dim = c( numberOfSimulations, idim, nPoints ) )
  }
  if (! missing( maskIndex ) ) {
    doMask = TRUE
    if ( length( inputImageList[[1]] ) < maskIndex )
      stop("Did you pass the mask into the inputImageList?")
    Xm = array( dim = c( numberOfSimulations, idim, 1 ) )
    stopifnot( length( numpynames ) > 2 )
    if ( length( grep("mask",numpynames) ) == 0 )
      stop( "numpynames must have a name containing the string mask" )
    masknameindex = grep("mask",numpynames)
    }
  Xcc = array( dim = c( numberOfSimulations, idim, length(idim) ) )
  stopifnot( length( numpynames ) > 2 )
  if ( length( grep("coordconv",numpynames) ) == 0 )
    stop( "numpynames must have a name containing the string coordconv" )
  ccnameindex = grep("coordconv",numpynames)

  if ( hasPoints & ! hasSeg ) {
    data <- dataAugmentation(
      inputImageList[selector],
      pointsetList = pointsetList[selector],
      transformType = transformType,
      numberOfSimulations = numberOfSimulations,
      sdAffine = sdAffine,
      noiseParameters = noiseParameters,
      sdSimulatedBiasField = sdSimulatedBiasField,
      sdHistogramWarping = sdHistogramWarping,
      referenceImage = referenceImage,
      verbose = FALSE )
    }

  if ( hasPoints & hasSeg ) {
    data <- dataAugmentation(
      inputImageList[selector],
      segmentationImageList[selector],
      pointsetList = pointsetList[selector],
      transformType = transformType,
      numberOfSimulations = numberOfSimulations,
      sdAffine = sdAffine,
      noiseParameters = noiseParameters,
      sdSimulatedBiasField = sdSimulatedBiasField,
      sdHistogramWarping = sdHistogramWarping,
      referenceImage = referenceImage,
      verbose = FALSE )
    }

  for ( k in 1:length(data$simulatedImages) ) {
    myccLocal = patchMatchR::coordinateImages( data$simulatedImages[[k]][[1]] * 0 + 1 )
    if ( doCrop )
      for ( jj in 1:length( myccLocal ) )
        myccLocal[[jj]] = specialCrop(
          myccLocal[[jj]],
            data$simulatedPointsetList[[k]][cropping[1],], cropping[-1] )
    for ( jj in 1:myimgdim ) {
      if ( myimgdim == 2 ) Xcc[k,,,jj] = as.array( myccLocal[[jj]] )
      if ( myimgdim == 3 ) Xcc[k,,,,jj] = as.array( myccLocal[[jj]] )
      }
    if ( ! missing( maskIndex ) ) {
      mymask = thresholdImage( data$simulatedImages[[k]][[maskIndex]], 0.5, Inf )
      if ( doCrop )
        mymask = specialCrop( mymask,
          data$simulatedPointsetList[[k]][cropping[1],], cropping[-1] )
      if ( myimgdim == 2 ) Xm[k,,,1] = as.array( mymask )
      if ( myimgdim == 3 ) Xm[k,,,,1] = as.array( mymask )
    }
    for ( j in overindices ) {
      temp = iMath( data$simulatedImages[[k]][[j]], "Normalize")
      if ( doCrop )
        temp = specialCrop( temp,
          data$simulatedPointsetList[[k]][cropping[1],], cropping[-1] )
      if ( myimgdim == 2 ) X[k, , , j ] = as.array( temp )
      if ( myimgdim == 3 ) X[k, , , , j ] = as.array( temp )
      }
    if ( hasSeg ) {
      for ( j in 1:nClasses ) {
        temp = thresholdImage( data$simulatedSegmentationImages[[k]][[1]],
          segmentationNumbers[j], segmentationNumbers[j] )
        if ( doCrop )
          temp = specialCrop( temp,
            data$simulatedPointsetList[[k]][cropping[1],], cropping[-1] )
        if ( myimgdim == 2 ) Y[k, , , j ] = as.array( temp )
        if ( myimgdim == 3 ) Y[k, , , , j ] = as.array( temp )
        }
      }

    if ( hasPoints ) {
      mypti = data$simulatedPointsetList[[k]]
      ct = 1
      for ( j in whichPoints ) {
        myptiloc = mypti[j,]
        Ypt[k, ct, ] = myptiloc[1:myimgdim]
        ct = ct + 1
        }
      }

    if ( smoothHeatMaps > 0 & hasPoints ) {
      mypti = data$simulatedPointsetList[[k]]
      ct = 1
      for ( j in whichPoints ) {
        heatmap = data$simulatedImages[[k]][[1]] * 0.0
        if ( doCrop )
          heatmap = specialCrop( heatmap,
            data$simulatedPointsetList[[k]][cropping[1],], cropping[-1] )
        myptiloc = mypti[j,]
        myptind = round( antsTransformPhysicalPointToIndex( heatmap, myptiloc[1:myimgdim] ) )
        if ( myimgdim == 2 ) heatmap[myptind[1,1],myptind[1,2]]=1
        if ( myimgdim == 3 ) heatmap[myptind[1,1],myptind[1,2],myptind[1,3]]=1
        heatmap = smoothImage( heatmap, smoothHeatMaps, sigmaInPhysicalCoordinates=FALSE )
        heatmap = heatmap / ( max( heatmap ) + 0.0001 )
        if ( myimgdim == 2 ) Yh[k, , , ct ] = as.array( heatmap )
        if ( myimgdim == 3 ) Yh[k, , , , ct ] = as.array( heatmap )
        ct = ct + 1
        }
      }
    }
  np$save( numpynames[1], X )
  np$save( numpynames[pointsetnameindex], Ypt )
  outlist = list(X,Ypt)
  outnames = c( "images", "points")
  if ( hasSeg ) {
    np$save( numpynames[segmentationnameindex], Y )
    outlist[[length(outlist)+1]] = Y
    outnames[length(outnames)+1]='segmentation'
  }
  if ( doMask ) {
    np$save( numpynames[masknameindex], Xm )
    outlist[[length(outlist)+1]] = Xm
    outnames[length(outnames)+1]='mask'
    }
  np$save( numpynames[ccnameindex], Xcc )
  outlist[[length(outlist)+1]] = Xcc
  outnames[length(outnames)+1]='coordconv'
  if ( smoothHeatMaps > 0 ) {
    np$save( numpynames[heatmapnameindex], Yh )
    outlist[[length(outlist)+1]] = Yh
    outnames[length(outnames)+1]='heatmaps'
    }
  names( outlist ) = outnames
  return( outlist  )
  }
