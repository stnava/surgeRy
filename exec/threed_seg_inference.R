Sys.setenv("TF_NUM_INTEROP_THREADS"=12)
Sys.setenv("TF_NUM_INTRAOP_THREADS"=12)
Sys.setenv("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"=12)
library( ANTsR )
library( ANTsRNet )
library( tensorflow )
library( keras )
library( tfdatasets )
library( reticulate )
library( patchMatchR )
library( surgeRy )
np <- import("numpy")
mytype = "float32"
nlayers = 4   # for unet
########################
# this is the point unet
########################
unetLM = createUnetModel3D(
       list( NULL, NULL, NULL, 1 ),
       numberOfOutputs = 2, # number of landmarks must be known
       numberOfLayers = nlayers, # should optimize this wrt criterion
       numberOfFiltersAtBaseLayer = 32, # should optimize this wrt criterion
       convolutionKernelSize = 3, # maybe should optimize this wrt criterion
       deconvolutionKernelSize = 2,
       poolSize = 2,
       strides = 2,
       dropoutRate = 0,
       weightDecay = 0,
       additionalOptions = c( "nnUnetActivationStyle" ),
       mode = c("regression")
     )
maskinput = layer_input( list( NULL, NULL,  NULL, 1 ) )
posteriorMask <- layer_multiply(
  list( unetLM$outputs[[1]] , maskinput ), name='maskTimesPosteriors'  )
unetLM = keras_model( list( unetLM$inputs[[1]], maskinput ), posteriorMask )
unetLM = patchMatchR::deepLandmarkRegressionWithHeatmaps( unetLM, activation = 'relu', theta=0 )
load_model_weights_hdf5( unetLM, 'lm_weights_gpu1.h5' )
###########################
patchSize = c( 64, 64, 32 )
# patchSize = c( 64, 32, 32 )
whichPoint = 1
ifns = Sys.glob("images/*nocsf.nii.gz")
# inference
ww = sample( 1:length( ifns ), 1 )
print( paste( ww, ifns[ww] ) )
image = antsImageRead( ifns[ww] ) %>% iMath("Normalize")
imager = resampleImage( image, c( 88, 128, 128 ), useVoxels=TRUE )
mask = thresholdImage( image, 0.01, 1 )
maskr = thresholdImage( imager, 0.01, 1 )
mycom = getCenterOfMass( mask )
ilist = list( list( image, mask ) )
ilistr = list( list( imager, maskr ) )
plist = list( matrix(mycom,nrow=1) )
types = c("images.npy", "pointset.npy", "mask.npy", "coordconv.npy", "heatmap.npy" )
npns = paste0("numpyinference/INF",types)
nsim = 1
gg = generateDiskPointAndSegmentationData(
    inputImageList = ilistr,
    pointsetList = plist,
    maskIndex = 2,
    transformType = "scaleShear",
    noiseParameters = c(0, 0.0),
    sdSimulatedBiasField = 0.0,
    sdHistogramWarping = 0.0,
    sdAffine = 0.0, # limited augmentation for inference
    numpynames = npns,
    numberOfSimulations = nsim
    )
################################################################################
# define the physical space for the sub-image
# needs to match the augmentation code
# unetLM expects (1) image feature;  (2) mask;  (3) coordconv
with(tf$device("/cpu:0"), {
  unetp = predict( unetLM, list( gg[[1]], gg[[3]], gg[[4]] ) )
})
plist[[1]] = matrix( unetp[[2]][1,1,], nrow=1, ncol=3 ) # 1st sample, 1st point
physspace = specialCrop( ilist[[1]][[1]], plist[[1]][whichPoint,], patchSize)
print( plist[[1]] )
antsImageWrite( physspace, '/tmp/tempPS.nii.gz' )
################################################################################
################################################################################
unet = createUnetModel3D(
       list( NULL, NULL, NULL, 2 ),
       numberOfOutputs = 1, # number of landmarks must be known
       numberOfLayers = nlayers, # should optimize this wrt criterion
       numberOfFiltersAtBaseLayer = 32, # should optimize this wrt criterion
       convolutionKernelSize = 3, # maybe should optimize this wrt criterion
       deconvolutionKernelSize = 2,
       poolSize = 2,
       strides = 2,
       dropoutRate = 0,
       weightDecay = 0,
       additionalOptions = c( "nnUnetActivationStyle" ),
       mode = c("sigmoid")
     )
load_model_weights_hdf5( unet, 'nbm_weights_gpu1.h5' )
################################################################################
Xte = generateDiskPointAndSegmentationData(
  ilist,
  plist,
  numpynames = npns,
  smoothHeatMaps = 0,
  numberOfSimulations = 1,
  sdAffine = 0.0, # adjust for inference
  transformType = 'scaleShear',
  cropping=c(whichPoint,patchSize) )
preds = predict( unet, Xte[[1]] )
predseg = as.antsImage( preds[1,,,,1 ]  ) %>% antsCopyImageInfo2( physspace )

# show image and prediction
# layout(matrix(1:2,nrow=1))
antsImageWrite( ilist[[1]][[1]], '/tmp/temp.nii.gz' )
predsegdecrop = resampleImageToTarget( predseg, ilist[[1]][[1]] )
antsImageWrite( predsegdecrop, '/tmp/temps.nii.gz' )
bint = thresholdImage( predsegdecrop, 0.5, 1 )
antsImageWrite( bint, '/tmp/bint.nii.gz' )
# plot(ilist[[1]][[1]],predsegdecrop,window.overlay=c(0.1,1))
# plot(ilist[[1]][[1]],bint,window.overlay=c(0.5,1))
