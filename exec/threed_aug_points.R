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
# collect your images - a list of lists - multiple entries in each list are fine
# mydf = read.csv( "manual/ba_notes.csv" )
ifns = Sys.glob( paste0( "images/*nocsf.nii.gz" ) )
sfns = Sys.glob( paste0( "nbm3parcCH13/*SRnbm3CH13.nii.gz" ) )
ilist = list()
slist = list()
plist = list()
for ( k in 1:length( ifns ) ) {
  print(k)
  image = antsImageRead( ifns[k] ) %>% resampleImage( c( 88, 128, 128 ), useVoxels=TRUE )
  image = iMath( image, "Normalize" )
  mask = thresholdImage( image, 0.01, 1.0 )
  seg = antsImageRead( sfns[k] )
  ilist[[k]] = list( image, mask )
  ptmat = getCentroids( seg )[,1:3]
  plist[[k]] = ptmat
  if ( k == 1 ) {
    npoints = nrow( ptmat )
    print(paste("expect:",npoints))
  }
  stopifnot( nrow( ptmat ) == npoints )
}
# identify the number of segmentation classes
isTrain = c( rep(TRUE,length(ilist)-20), FALSE )

nFiles = 24
if ( ! exists( "uid" ) )
  uid = paste0("LM",sample(1:1000000,nFiles),"Z")
types = c("images.npy", "pointset.npy", "mask.npy", "coordconv.npy",
  "heatmap.npy" ) # "segmentation.npy" )
nmats = length(types) # 5 arrays out
trainTestFileNames = data.frame( matrix("",nrow=nFiles,ncol=nmats*2))
colnamesTT = c(
  paste0("train",gsub(".npy","",types)),
  paste0("test",gsub(".npy","",types)) )
colnames( trainTestFileNames ) = colnamesTT
for ( k in 1:nFiles ) {
  twogroups = paste0("numpyPoints8/",uid[k],c("train","test"))
  npextsTr = paste0( twogroups[1], types )
  npextsTe = paste0( twogroups[2], types )
  trainTestFileNames[k,]=as.character( c(npextsTr,npextsTe) )
}
write.csv( trainTestFileNames, "numpyPoints8/LMtrainttestfiles.csv", row.names=FALSE)

print("Begin test generation")
testfilename = as.character(trainTestFileNames[1,grep("test",colnames(trainTestFileNames))])
gg = generateDiskPointAndSegmentationData(
    inputImageList = ilist,
    pointsetList = plist,
    selector = !isTrain,
    smoothHeatMaps = smoothHeat,
    maskIndex = 2,
    transformType = "scaleShear",
    noiseParameters = c(0, 0.01),
    sdSimulatedBiasField = 0.01,
    sdHistogramWarping = 0.01,
    sdAffine = 0.1, # limited
    numpynames = testfilename,
    numberOfSimulations = 24
    )
# visualize example augmented images
# layout( matrix(1:8,nrow=2))
# for ( k in 1:8 ) {
#  temp = as.antsImage( gg[[1]][k,,,,1] )
#  plot( temp )
#  }
print("Begin train generation")

while( TRUE ) {
  for ( k in 1:nFiles ) {
    trnfilename = as.character(trainTestFileNames[k,grep("train",colnames(trainTestFileNames))])
    print( paste(k, trnfilename[1] ) )
    gg = generateDiskPointAndSegmentationData(
        inputImageList = ilist,
        pointsetList = plist,
        selector = isTrain,
        smoothHeatMaps = smoothHeat,
        maskIndex = 2,
        transformType = "scaleShear",
        noiseParameters = c(0, 0.01),
        sdSimulatedBiasField = 0.01,
        sdHistogramWarping = 0.01,
        sdAffine = 0.1,
        numpynames = trnfilename,
        numberOfSimulations = 32
        )
    }
  }
