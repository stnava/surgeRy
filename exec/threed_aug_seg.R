Sys.setenv("TF_NUM_INTEROP_THREADS"=12)
Sys.setenv("TF_NUM_INTRAOP_THREADS"=12)
Sys.setenv("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"=12)
# record the GPU
gpuid = Sys.getenv(x = "CUDA_VISIBLE_DEVICES")
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
smoothHeat = 0.0
downsam = 4   # downsamples images

########################
nlayers = 4   # for unet
unet = createUnetModel3D(
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
  list( unet$outputs[[1]] , maskinput ), name='maskTimesPosteriors'  )
unet = keras_model( list( unet$inputs[[1]], maskinput ), posteriorMask )
unet = patchMatchR::deepLandmarkRegressionWithHeatmaps( unet,
  activation = 'relu', theta=0 )
load_model_weights_hdf5( unet, 'lm_weights_gpu1good.h5' )
# this unet predicts points
#
# collect your images - a list of lists - multiple entries in each list are fine

types = c("images.npy", "pointset.npy", "mask.npy", "coordconv.npy",
  "heatmap.npy" )
npns = paste0("numpyinference/INF",types)

# mydf = read.csv( "manual/ba_notes.csv" )
# ifns = Sys.glob( paste0( "images/",mydf$ids, "*nocsf.nii.gz" ) )
# sfnsR = Sys.glob( paste0( "manual/",mydf$ids, "*rightNBMmanual.nii.gz" ) )
# sfnsL = Sys.glob( paste0( "manual/",mydf$ids, "*leftNBMmanual.nii.gz" ) )
ifns = Sys.glob( paste0( "images/*nocsf.nii.gz" ) )
sfnsR = Sys.glob( paste0( "evaluationResults13/*rightNBMX3PARCSRLJLF.nii.gz" ) )
sfnsL = Sys.glob( paste0( "evaluationResults13/*leftNBMX3PARCSRLJLF.nii.gz" ) )
ilist = list()
ilistFull = list()
slistL = list()
slistR = list()
plist = list()
plistu = list()
print("COLLECT DATA")
for ( k in 1:length( ifns ) ) {
  print(k)
  image = iMath( antsImageRead( ifns[k] ), "Normalize" )
  mask = thresholdImage( image, 0.01, 1.0 )
  ilistFull[[k]] = list( image, mask )
  image = antsImageRead( ifns[k] ) %>% resampleImage( c( 88, 128, 128 ), useVoxels=TRUE )
  image = iMath( image, "Normalize" )
  mask = thresholdImage( image, 0.01, 1.0 )
  segL = antsImageRead( sfnsL[k] ) %>% thresholdImage( 1, 3 )
  segR = antsImageRead( sfnsR[k] ) %>% thresholdImage( 1, 3 )
  ilist[[k]] = list( image, mask )
  slistL[[k]] = segL
  slistR[[k]] = segR
  ptmat = rbind( getCentroids( segL )[,1:3], getCentroids( segR )[,1:3] )
  plist[[k]] = ptmat
  nsim = 4
  gg = generateDiskPointAndSegmentationData(
      inputImageList = ilist[k],
      pointsetList = plist[k],
      maskIndex = 2,
      transformType = "scaleShear",
      noiseParameters = c(0, 0.0),
      sdSimulatedBiasField = 0.0,
      sdHistogramWarping = 0.0,
      sdAffine = 0.02, # limited augmentation for inference
      numpynames = npns,
      numberOfSimulations = nsim
      )
   unetp = predict( unet, list( gg[[1]], gg[[3]], gg[[4]] ) )
   mattoavg = matrix( 0, nrow=2, ncol=3 ) #
   for ( jj in 1:nsim ) mattoavg = mattoavg + unetp[[2]][jj,,]/nsim
   plistu[[k]] = mattoavg
}


# identify the number of segmentation classes
isTrain = c( rep(TRUE,length(ilist)-20), FALSE )

nFiles = 24
if ( ! exists( "uid" ) )
  uid = paste0("LM",sample(1:1000000,nFiles),"Z")
types = c("images.npy", "pointset.npy", "mask.npy", "coordconv.npy",
  "heatmap.npy", "segmentation.npy" )
nmats = length(types) # 5 arrays out
trainTestFileNames = data.frame( matrix("",nrow=nFiles,ncol=nmats*2))
colnamesTT = c(
  paste0("train",gsub(".npy","",types)),
  paste0("test",gsub(".npy","",types)) )
colnames( trainTestFileNames ) = colnamesTT
for ( k in 1:nFiles ) {
  twogroups = paste0("numpySeg/",uid[k],c("train","test"))
  npextsTr = paste0( twogroups[1], types )
  npextsTe = paste0( twogroups[2], types )
  trainTestFileNames[k,]=as.character( c(npextsTr,npextsTe) )
}
# record some of the parameters
trainTestFileNames$side = 'right'
trainTestFileNames$lowX = 88
trainTestFileNames$lowY = 128
trainTestFileNames$lowZ = 128
trainTestFileNames$patchX = 64
trainTestFileNames$patchY = 64
trainTestFileNames$patchZ = 32
trainTestFileNames$whichPoint = 2
write.csv( trainTestFileNames, "numpySeg/LMtrainttestfiles.csv", row.names=FALSE)

print("TEST DATA")
tardim = c(trainTestFileNames$patchX[1],trainTestFileNames$patchY[1],trainTestFileNames$patchZ[1])
testfilename = as.character(trainTestFileNames[1,grep("test",colnames(trainTestFileNames))])
gg = generateDiskPointAndSegmentationData(
    inputImageList = ilistFull,
    pointsetList = plistu,
    slistR,   # should match the correct side of the landmark
    cropping=c(trainTestFileNames$whichPoint[1],tardim), # just train one side first
    segmentationNumbers = 1,
    selector = !isTrain,
    smoothHeatMaps = 0,
    transformType = "scaleShear",
    noiseParameters = c(0, 0.0),
    sdSimulatedBiasField = 0.0,
    sdHistogramWarping = 0.0,
    sdAffine = 0.05, # limited
    numpynames = testfilename,
    numberOfSimulations = 12
    )
# visualize example augmented images
# layout( matrix(1:8,nrow=2))
# for ( k in 1:8 ) {
#  temp = as.antsImage( gg[["images"]][k,,,,1] )
#  antsImageWrite( temp, '/tmp/temp.nii.gz' )
#  temp = as.antsImage( gg[["segmentation"]][k,,,,1] )
#  antsImageWrite( temp, '/tmp/temps.nii.gz' )
#  plot( temp )
#  }
print("LOOP IT")
gc()
while( TRUE ) {
  for ( k in 1:nFiles ) {
    trnfilename = as.character(trainTestFileNames[k,grep("train",colnames(trainTestFileNames))])
    print( paste(k, trnfilename[1] ) )
    gg = generateDiskPointAndSegmentationData(
        inputImageList = ilistFull,
        pointsetList = plist,
        slistR,   # should match the correct side of the landmark
        cropping=c(trainTestFileNames$whichPoint[1],tardim), # just train one side first
        segmentationNumbers = 1,
        selector = isTrain,
        smoothHeatMaps = 0,
#        maskIndex = 2,
        transformType = "scaleShear",
        noiseParameters = c(0, 0.0),
        sdSimulatedBiasField = 0.0,
        sdHistogramWarping = 0.0,
        sdAffine = 0.15,
        numpynames = trnfilename,
        numberOfSimulations = 64
        )
    gc()
    }
  }
