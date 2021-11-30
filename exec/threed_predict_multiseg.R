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

binary_dice <- function( y_true, y_pred )
{
  K <- tensorflow::tf$keras$backend
  smoothing_factor = tf$cast( 0.01, mytype )
  y_true_f = K$flatten( y_true )
  y_pred_f = K$flatten( y_pred )
  intersection = K$sum( y_true_f * y_pred_f )
  return( -1.0 * ( 2.0 * intersection + smoothing_factor )/
        ( K$sum( y_true_f ) + K$sum( y_pred_f ) + smoothing_factor ) )
}

weighted_mse <- function( y_true, y_pred, weights )
{
  K <- tensorflow::tf$keras$backend
  totalLoss = tf$cast( 0.0, mytype )
  for ( j in 1:tail(dim(y_true),1) ) {
    totalLoss = totalLoss + tf$keras$losses$mse(y_true[,,,,j], y_pred[,,,,j]) * weights[j]
    }
  return( totalLoss )
}

# set up a weighted dice loss
weighted_dice <- function( y_true, y_pred, weights=c(1.0,1.0) )
{
  K <- tensorflow::tf$keras$backend
  smoothing_factor = tf$cast( 0.01, mytype )
  totalLoss = tf$cast( 0.0, mytype )
  for ( j in 1:tail(dim(y_true),1) ) {
    y_true_f = K$flatten( y_true[,,,,j] )
    y_pred_f = K$flatten( y_pred[,,,,j] )
    intersection = K$sum( y_true_f * y_pred_f )
    locloss = ( -1.0 * weights[j] * ( 2.0 * intersection + smoothing_factor )/
        ( K$sum( y_true_f ) + K$sum( y_pred_f ) + smoothing_factor ) )
    totalLoss = totalLoss + locloss
    }
  return( totalLoss )
}


## ----howdowereadintime,echo=TRUE,eval=FALSE-----------------------------------
trtefns = read.csv( "numpySegM/multisegtrainttestfiles.csv" ) # critical - same file name
trnnames = colnames(trtefns)[grep("train", colnames(trtefns) )]
tstnames = colnames(trtefns)[grep("test", colnames(trtefns) )]
whichcolstouse
loadfirst = chooseTrainingFilesToRead( trtefns[,trnnames[whichcolstouse]] )
Xtr = surgeRy::loadNPData( loadfirst )
mybs = dim( Xtr[[1]] )[1]
Xte = surgeRy::loadNPData( trtefns[1,tstnames[whichcolstouse]] )
nclasstoseg = 0 # GET THIS FROM Xtr/Xte

nlayers = 4   # for unet
# set up the network - all parameters below could be optimized for the application
unet = createUnetModel3D(
       list( NULL, NULL, NULL, 1 ),
       numberOfOutputs = nclasstoseg, # number of landmarks must be known
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
# load_model_weights_hdf5( unet, 'nbm_left_weights_gpu1good.h5',
#  by_name = TRUE, skip_mismatch = TRUE, reshape = TRUE )
gpuid = Sys.getenv(x = "CUDA_VISIBLE_DEVICES")
mydf = data.frame()
epoch = 1
wtfn=paste0('nbm_left_weights_gpu', gpuid,'.h5')
csvfn = paste0('nbm_left_weights_gpu', gpuid,'.csv')

if ( file.exists( wtfn ) ) load_model_weights_hdf5( unet, wtfn )

# ----training,echo=TRUE,eval=FALSE--------------------------------------------
mydf = data.frame()
epoch = 1
num_epochs = 50000
optimizerE <- tf$keras$optimizers$Adam(1.e-5)
batchsize = 2
for (epoch in 1:num_epochs ) {
    if ( (epoch %% round(mybs/batchsize) ) == 0 & epoch > 1  ) {
      loadfirst = chooseTrainingFilesToRead( trtefns[,trnnames[whichcolstouse]], notFirst=TRUE )
      Xtr = surgeRy::loadNPData( loadfirst )
      }
    ct = nrow( mydf ) + 1
    mysam = sample( 1:nrow(Xtr[[1]]), batchsize )
    datalist = list()
    for ( jj in 1:2 )
      datalist[[jj]] = array( Xtr[[jj]][mysam,,,,],
        dim=c(batchsize,tail(dim(Xtr[[jj]]),4)) ) %>% tf$cast( mytype )
    with(tf$GradientTape(persistent = FALSE) %as% tape, {
      preds = unet( datalist[[1]] )
      loss = weighted_dice( datalist[[2]], preds )
      })
    unet_gradients <- tape$gradient(loss, unet$trainable_variables)
    optimizerE$apply_gradients(purrr::transpose(list(
        unet_gradients, unet$trainable_variables )))
    mydf[ct,'train_loss'] = as.numeric( loss )
    mydf[ct,'trainData'] = loadfirst[1]
    if( epoch > 3 & epoch %% 20 == 0 ) {
      with(tf$device("/cpu:0"), {
        preds = predict( unet, Xte[[1]] )
        loss = weighted_dice( tf$cast(Xte[[2]],mytype), tf$cast(preds, mytype ))
      })
      # compute the same thing in test data
      mydf[ct,'test_loss'] = as.numeric( loss )
      if ( mydf[ct,'test_loss'] <= min(mydf[1:epoch,'test_loss'],na.rm=TRUE) ) {
        print(paste("Saving",epoch))
        keras::save_model_weights_hdf5( unet, wtfn )
        gc()
      }
    }
  print( mydf[ct,] )
  write.csv( mydf, csvfn, row.names=FALSE )
  }
