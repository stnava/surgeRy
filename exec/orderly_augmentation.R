#!/usr/bin/env Rscript
# args<-commandArgs(TRUE)
Sys.setenv( "CUDA_VISIBLE_DEVICES"=-1)
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
prefix = "numpySegNbm"
smoothHeat = smthhm = 0.0
segnums = 0:3
ntestsim = 32
nFiles = 48 # number of aggregated training arrays
dir.create(prefix,showWarnings=FALSE)
dir.create('numpyinference',showWarnings=FALSE)
rawdatafn = paste0( prefix,'/',prefix,'trainTestRawData.csv')
resam=FALSE
rightside=TRUE
txtype = 'rigid'
if ( ! resam ) {
  factor=2
  rezzer='SR'
} else {
  factor=1
  rezzer='OR'
  }
ttfn = paste0( prefix, "/", prefix, "TrainTestArrays", rezzer, ".csv" )
types = c("images.npy", "pointset.npy", "mask.npy", "coordconv.npy")
npns = paste0("numpyinference/INFGEN",sample(1:100000,1),types)
lodim = c( 88, 128, 128 )
if ( ! file.exists( rawdatafn ) ) {
  sfnsL = Sys.glob( paste0( "manual/*leftNBM3PARCREGmanual.nii.gz") )
  sfnsL = c( sfnsL, Sys.glob( paste0( "evaluationResults13/*leftNBMX3PARCSRLJLF.nii.gz") ) )
  sfnsR = Sys.glob( paste0( "manual/*rightNBM3PARCREGmanual.nii.gz") )
  sfnsR = c( sfnsR, Sys.glob( paste0( "evaluationResults13/*rightNBMX3PARCSRLJLF.nii.gz") ) )
  ifns = gsub( "leftNBM3PARCREGmanual", "SRnocsf", sfnsL )
  ifns = gsub( "manual/", "images/", ifns )
  ifns = gsub( "leftNBMX3PARCSRLJLF", "SRnocsf", ifns )
  ifns = gsub( "evaluationResults13/", "images/", ifns )

  # define the image train / test data frame
  trainTestRawData = data.frame( images=ifns, leftSeg=sfnsL, rightSeg=sfnsR )
  usubs = substr( unique( basename( ifns )),0,10)
  usubsTest = sample( usubs, 12 )
  trainTestRawData$isTrain=TRUE
  for ( k in 1:length(usubsTest) )
    trainTestRawData$isTrain[ grep(usubsTest[k],trainTestRawData$images)]=FALSE

  print( table( trainTestRawData$isTrain ))
  write.csv( trainTestRawData, rawdatafn  )
  } else trainTestRawData = read.csv( rawdatafn )

if ( ! file.exists( ttfn ) ) {
  if ( ! exists( "uid" ) )
    uid = paste0("BF",rezzer,sample(1:1000000,nFiles),"Z")
  types = c("images.npy", "pointset.npy", "mask.npy", "coordconv.npy", "segmentation.npy", "heatmap.npy" )
  nmats = length( types ) # 5 arrays out
  trainTestFileNames = data.frame( matrix("",nrow=nFiles,ncol=nmats*2))
  colnamesTT = c(
    paste0("train",gsub(".npy","",types)),
    paste0("test",gsub(".npy","",types)) )
  colnames( trainTestFileNames ) = colnamesTT
  for ( k in 1:nFiles ) {
    twogroups = paste0(prefix,"/",uid[k],c("train","test"))
    npextsTr = paste0( twogroups[1], types )
    npextsTe = paste0( twogroups[2], types )
    trainTestFileNames[k,]=as.character( c(npextsTr,npextsTe) )
  }
  # record some of the parameters
  trainTestFileNames$side = 'both'
  trainTestFileNames$whichPoint = 1
  trainTestFileNames$lowX = lodim[1]
  trainTestFileNames$lowY = lodim[2]
  trainTestFileNames$lowZ = lodim[3]
  trainTestFileNames$patchX = 32 * factor
  trainTestFileNames$patchY = 32 * factor
  trainTestFileNames$patchZ = 16 * factor
  write.csv( trainTestFileNames, ttfn, row.names=FALSE)
} else trainTestFileNames = read.csv( ttfn )
################################################################################
# below we generate just one train and one test pair - paralleization is good
################################################################################
nlayers = 4   # for unet
nPoints = 8   # of points the net predicts
unet = createUnetModel3D(
       list( NULL, NULL, NULL, 1 ),
       numberOfOutputs = nPoints, # number of landmarks must be known
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
load_model_weights_hdf5( unet, 'lm8bf_weights.h5' )
print("done with LM load")

template = antsImageRead("~/.antspyt1w/CIT168_T1w_700um_pad_adni.nii.gz")
bfPriorL = antsImageRead( "~/.antspyt1w/CIT168_basal_forebrain_adni_prob_2_left.nii.gz" )
bfPriorR = antsImageRead( "~/.antspyt1w/CIT168_basal_forebrain_adni_prob_2_right.nii.gz" )

ilist = list()
ilistFull = list()
slist = list()
plistu = list()
sidelist = list()
print("COLLECT DATA")
ct = 1
subrows = sample( 1:nrow( trainTestRawData ), ntestsim, replace=TRUE )
ifns = trainTestRawData$images[subrows]
sfnsL = trainTestRawData$leftSeg[subrows]
sfnsR = trainTestRawData$rightSeg[subrows]
print( paste( "I", file.exists(ifns) ) )
print( paste( "L", file.exists(sfnsL) ) )
print( paste( "R", file.exists(sfnsR) ) )
for ( k in 1:length( sfnsL ) ) {
  print(k)
  imageF = iMath( antsImageRead( ifns[k] ), "Normalize" )
  if ( resam ) imageF = resampleImage( imageF, dim( imageF )/2, useVoxels=TRUE )
  image = antsImageRead( ifns[k] ) %>% resampleImage( lodim, useVoxels=TRUE )
  image = iMath( image, "Normalize" )
  reg = antsRegistration( image, template, 'SyN' )
  priorL = antsApplyTransforms( imageF, bfPriorL, reg$fwdtransforms ) %>% smoothImage( 3 ) %>% iMath("Normalize")
  priorR = antsApplyTransforms( imageF, bfPriorR, reg$fwdtransforms ) %>% smoothImage( 3 ) %>% iMath("Normalize")
  mask = thresholdImage( imageF, 0.01, 1.0 )
  ilistFull[[ct]] = list( imageF, priorL, mask )
  if ( rightside ) ilistFull[[ct+1]] = list( imageF, priorR, mask  )
  mask = thresholdImage( image, 0.01, 1.0 )
  ilist[[1]] = list( image, mask )
  seg = antsImageRead( sfnsL[k] )
  if ( resam ) seg = resampleImage( seg, dim( seg )/2, useVoxels=TRUE )
  slist[[ct]] = seg
  seg = antsImageRead( sfnsR[k] )
  if ( resam ) seg = resampleImage( seg, dim( seg )/2, useVoxels=TRUE )
  if ( rightside )  slist[[ct+1]] = seg
  sidelist[[ct]] = 'left'
  if ( rightside )  sidelist[[ct+1]]='right'
  ptmat = rbind( getCentroids( mask )[,1:3], getCentroids( mask )[,1:3] )
  plistu[[ct]] = ptmat
  nsim = 4
  gg = generateDiskPointAndSegmentationData(
      inputImageList = ilist,
      pointsetList = list( matrix(0,ncol=3,nrow=1)),
      maskIndex = 2,
      transformType = "scaleShear",
      noiseParameters = c(0, 0.001),
      sdSimulatedBiasField = 0.0,
      sdHistogramWarping = 0.0,
      sdAffine = 0.02, # limited augmentation for inference
      numpynames = npns,
      numberOfSimulations = nsim
      )
   unetp = predict( unet, list( gg[[1]], gg[[3]], gg[[4]] ) )
   mattoavg = matrix( 0, nrow=1, ncol=3 ) #
   for ( jj in 1:nsim ) mattoavg = mattoavg + colMeans(unetp[[2]][jj,3:5,])/nsim
   plistu[[ct]] = matrix( mattoavg, nrow=1 )
   # repeat the same thing for the right side
   mattoavg = matrix( 0, nrow=1, ncol=3 ) #
   for ( jj in 1:nsim ) mattoavg = mattoavg + colMeans(unetp[[2]][jj,6:8,])/nsim
   if ( rightside ) plistu[[ct+1]] = matrix( mattoavg, nrow=1 )
   if ( rightside ) off = 2 else off = 1
   ct = ct + off
}

sidelist = unlist(sidelist)
# identify the number of segmentation classes
rm( ilist )
gc()
k = sample( which( !file.exists( trainTestFileNames$trainimages ) ) , 1 )
mysdaff = 0.15
tardim = c(trainTestFileNames$patchX[k],trainTestFileNames$patchY[k],trainTestFileNames$patchZ[k])

if ( sum( !trainTestRawData[subrows,"isTrain"]  ) > 0 ) {
  print("TEST DATA")
  testfilename = as.character(trainTestFileNames[k,grep("test",colnames(trainTestFileNames))])
  gg = generateDiskPointAndSegmentationData(
    inputImageList = ilistFull,
    pointsetList = plistu,
    slist,   # should match the correct side of the landmark
    cropping=c(trainTestFileNames$whichPoint[k],tardim), # just train one side first
    segmentationNumbers = segnums, # both left and right CH13
    selector =  !trainTestRawData[subrows,"isTrain"] ,
    maskIndex = 3,
    transformType = txtype,
    noiseParameters = c(0, 0.01),
    sdSimulatedBiasField = 0.0,
    sdHistogramWarping = 0.0,
    sdAffine = mysdaff, # limited
    numpynames = testfilename,
    numberOfSimulations = ntestsim
    )
  }
if ( sum( trainTestRawData[subrows,"isTrain"]  ) > 0 ) {
  print("TRAIN DATA")
  sdbias = 0.05
  sdhist = 0.05
  trnfilename = as.character(trainTestFileNames[k,grep("train",colnames(trainTestFileNames))])
  print( paste(k, trnfilename[1] ) )
  gg = generateDiskPointAndSegmentationData(
        inputImageList = ilistFull,
        pointsetList = plistu,
        slist,   # should match the correct side of the landmark
        cropping = c(trainTestFileNames$whichPoint[k],tardim), # just train one side first
        segmentationNumbers = segnums,
        selector = trainTestRawData[subrows,"isTrain"],
        maskIndex = 3,
        transformType = txtype, # should match priors
        noiseParameters = c(0, 0.01),
        sdSimulatedBiasField = sample( c(0,sdbias))[1],
        sdHistogramWarping = sample( c(0,sdhist))[1],
        sdAffine = mysdaff, # limited
	      numpynames = trnfilename,
        numberOfSimulations = ntestsim
        )
      }
