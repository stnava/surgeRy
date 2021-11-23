set.seed( 000 )
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

## ----howdowereadintime,echo=TRUE,eval=FALSE-----------------------------------
trtefns = read.csv( "numpy/LMtrainttestfiles.csv" ) # critical - same file name
trnnames = colnames(trtefns)[grep("train", colnames(trtefns) )]
tstnames = colnames(trtefns)[grep("test", colnames(trtefns) )]
loadfirst = chooseTrainingFilesToRead( trtefns[,trnnames] )
Xtr = surgeRy::loadNPData( loadfirst )
mybs = dim( Xtr[[1]] )[1]
Xte = surgeRy::loadNPData( trtefns[1,tstnames] )


nlayers = 4   # for unet
# set up the network - all parameters below could be optimized for the application
unet = createUnetModel3D(
       list( NULL, NULL, NULL, 1 ),
       numberOfOutputs = dim(Xte[[2]])[2], # number of landmarks must be known
       numberOfLayers = nlayers, # should optimize this wrt criterion
       numberOfFiltersAtBaseLayer = 16, # should optimize this wrt criterion
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
unetLM = patchMatchR::deepLandmarkRegressionWithHeatmaps( unet,
  activation = 'none', theta=0 )
unetLM1 = patchMatchR::deepLandmarkRegressionWithHeatmaps( unet,
  activation = 'relu', theta=0 )



## ----seghistory,echo=TRUE,eval=TRUE-------------------------------------------
gpuid='MYVIGN'
gpuid = Sys.getenv(x = "CUDA_VISIBLE_DEVICES")
mydf = data.frame()
epoch = 1
wtfn=paste0('lm_weights_gpu', gpuid,'.h5')
csvfn = paste0('lm_weights_gpu', gpuid,'.csv')

# ----training,echo=TRUE,eval=FALSE--------------------------------------------
mydf = data.frame()
epoch = 1
for ( ptwt in c( 0.001, 0.005, 0.01 ) ) {
  if ( ptwt == 0.01 ) unetLM = unetLM1
  ptWeight = tf$cast( ptwt, mytype )
  num_epochs = 500
  if ( ptwt == 0.01 ) num_epochs = 1000
  optimizerE <- tf$keras$optimizers$Adam(1.e-6)
  batchsize = 2
  for (epoch in 1:num_epochs ) {
    if ( (epoch %% round(mybs/batchsize) ) == 0 & epoch > 1  ) {
        # refresh the data
        locfns = chooseTrainingFilesToRead( trtefns[,trnnames] )
        print( locfns[1] )
        Xtr = surgeRy::loadNPData( locfns )
      }
    ct = nrow( mydf ) + 1
    mysam = sample( 1:nrow(Xtr[[1]]), batchsize )
    datalist = list()
    # this is the point set
    datalist[[2]] = array( Xtr[[2]][mysam,,], dim=dim(Xtr[[2]][mysam,,]) ) %>% tf$cast( mytype )
    for ( jj in c(1,3:length(Xtr)) )
      datalist[[jj]] = array( Xtr[[jj]][mysam,,,,],
        dim=c(batchsize,tail(dim(Xtr[[jj]]),4)) ) %>% tf$cast( mytype )
    with(tf$GradientTape(persistent = FALSE) %as% tape, {
      preds = unetLM( datalist[c(1,3:4)] )
      lossht = tf$keras$losses$mse( datalist[[5]], preds[[1]] ) %>% tf$reduce_mean( )
      losspt = tf$keras$losses$mse( datalist[[2]], preds[[2]] ) %>% tf$reduce_mean( )
      loss = losspt * ptWeight + lossht
      })
    unet_gradients <- tape$gradient(loss, unetLM$trainable_variables)
    optimizerE$apply_gradients(purrr::transpose(list(
        unet_gradients, unetLM$trainable_variables )))
    mydf[ct,'train_loss'] = as.numeric( loss )
    mydf[ct,'train_ptloss'] = as.numeric( losspt )
    mydf[ct,'train_htloss'] = as.numeric( lossht )
    mydf[ct,'ptWeight'] = as.numeric( ptWeight )
    mydf[ct,'trainData'] = locfns[1]
    if( epoch > 3 & epoch %% 10 == 0 ) {
      with(tf$device("/cpu:0"), {
        preds = predict( unetLM, Xte[c(1,3:4)] )
        lossht = tf$keras$losses$mse( Xte[[5]], preds[[1]] ) %>% tf$reduce_mean( )
        losspt = tf$keras$losses$mse( Xte[[2]], preds[[2]] ) %>% tf$reduce_mean( )
        loss = tf$cast(losspt, mytype) * ptWeight + tf$cast(lossht, mytype)
      })
      # compute the same thing in test data
      mydf[ct,'test_loss'] = as.numeric( loss )
      mydf[ct,'test_ptloss'] = as.numeric( losspt )
      mydf[ct,'test_htloss'] = as.numeric( lossht )
      if ( mydf[ct,'test_ptloss'] <= min(mydf[1:epoch,'test_ptloss'],na.rm=TRUE) ) {
        print(paste("Saving",epoch))
        keras::save_model_weights_hdf5( unetLM, wtfn )
        gc()
      }
    }
  print( mydf[ct,] )
  write.csv( mydf, csvfn, row.names=FALSE )
  }
}

derkaderkaderkaderkaderkaderkaderkaderkaderkaderkaderkaderkaderkaderkaderkaderka

# ----traintestcurves,eval=FALSE,echo=TRUE-------------------------------------
mydf = read.csv( paste0( "lm_weights_gpu", gpuid, ".csv" ) )
mydfnona = mydf[ !is.na( mydf$test_loss),  ]
plot( ts( mydfnona ) )


# ----vizp,echo=TRUE,eval=FALSE------------------------------------------------
layout(matrix(1:2,nrow=1))
wsub = 1
testimg = as.antsImage( Xte[[1]][wsub,,,1] ) %>%
  antsCopyImageInfo2( rids( 1 ) )
preds = predict( unetLM, Xte[c(1,3:4)] )
# these are in physical space
predPoints = as.array( preds[[2]] )[wsub,,]
truPoints = Xte[[2]][wsub,,] # true points
print(paste("Error",norm(truPoints-predPoints,"F")))
print(paste("%Error",norm(truPoints-predPoints,"F")/norm(truPoints,"F")*100,"%"))
truIndex = round( antsTransformPhysicalPointToIndex( testimg, truPoints ) )
predIndex = round( antsTransformPhysicalPointToIndex( testimg, predPoints ) )
# make point images
truPointsImage = predPointsImage = testimg * 0
for ( j in 1:nrow( truPoints ) ) {
  truPointsImage[truIndex[j,1],truIndex[j,2]]=j
  predPointsImage[predIndex[j,1],predIndex[j,2]]=j
  }
plot( testimg, iMath( truPointsImage, "GD",2) )
plot( testimg, iMath( predPointsImage, "GD",2) )
