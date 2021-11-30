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

reparameterize <- function(mean, logvar) {
  eps <- k_random_normal(shape = mean$shape, dtype = tf$float32)
  eps * k_exp(logvar * 0.5) + mean
}


# Loss and optimizer ------------------------------------------------------

normal_loglik <- function(sample, mean, logvar ) {
  loglik <- k_constant(0.5) *
    (k_log(2 * k_constant(pi)) +
       logvar +
       k_exp(-logvar) * (sample - mean) ^ 2)
  - k_sum(loglik )
}

compute_kernel <- function(x, y, sigma=tf$cast( 1e1, mytype ) ) {
  x_size <- k_shape(x)[1]
  y_size <- k_shape(y)[1]
  dim <- k_shape(x)[2]
  tiled_x <- k_tile(
    k_reshape(x, k_stack(list(x_size, tf$cast(1,"int32"), dim))),
    k_stack(list(tf$cast(1,"int32"), y_size, tf$cast(1,"int32")))
  )
  tiled_y <- k_tile(
    k_reshape(y, k_stack(list(tf$cast(1,"int32"), y_size, dim))),
    k_stack(list(x_size, tf$cast(1,"int32"), tf$cast(1,"int32")))
  )
  sigmaterm = tf$cast( 2.0, mytype ) * k_square( sigma )
  k_exp(-k_mean(k_square(tiled_x - tiled_y)/sigmaterm, axis = 3) /
          k_cast(dim, tf$float32))
}

compute_mmd <- function( x, y, sigma=tf$cast( 1e1, mytype ), takeMean = FALSE ) {
  x_kernel <- compute_kernel(x, x, sigma=sigma )
  y_kernel <- compute_kernel(y, y, sigma=sigma )
  xy_kernel <- compute_kernel(x, y, sigma=sigma )
  if ( takeMean ) {
    myout = k_mean(x_kernel) + k_mean(y_kernel) - 2 * k_mean(xy_kernel)
  } else {
    myout = (x_kernel) + (y_kernel) - 2 * (xy_kernel)
  }
  return( myout )
}


## ----howdowereadintime,echo=TRUE,eval=FALSE-----------------------------------
trtefns = read.csv( "numpyPoints8/LMtrainttestfiles.csv" ) # critical - same file name
noheatmap = grep("eatmap", names(trtefns) )
trtefns = trtefns[,-noheatmap]
trnnames = colnames(trtefns)[grep("train", colnames(trtefns) )]
tstnames = colnames(trtefns)[grep("test", colnames(trtefns) )]
locfns = chooseTrainingFilesToRead( trtefns[,trnnames] )
Xtr = surgeRy::loadNPData( locfns )
# Xtr[[1]] = abind::abind( Xtr[[1]], Xtr[[4]], along=5 )
mybs = dim( Xtr[[1]] )[1]
Xte = surgeRy::loadNPData( trtefns[1,tstnames] )
# Xte[[1]] = abind::abind( Xte[[1]], Xte[[4]], along=5 )


nlayers = 4   # for unet
# set up the network - all parameters below could be optimized for the application
unet = createUnetModel3D(
       list( NULL, NULL, NULL, 1 ), # image and coordconv = 4, otherwise 1
       numberOfOutputs = dim(Xte[[2]])[2], # number of landmarks must be known
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
unetLM = patchMatchR::deepLandmarkRegressionWithHeatmaps( unet,
  activation = 'none', theta=0 )
unetLM1 = patchMatchR::deepLandmarkRegressionWithHeatmaps( unet,
  activation = 'relu', theta=0 )
# these are pre-trained weights
load_model_weights_hdf5( unetLM, 'lm_weights_gpu2.h5',
  by_name = TRUE, skip_mismatch = TRUE, reshape = TRUE )
load_model_weights_hdf5( unetLM1, 'lm_weights_gpu2.h5',
  by_name = TRUE, skip_mismatch = TRUE, reshape = TRUE )



## ----seghistory,echo=TRUE,eval=TRUE-------------------------------------------
gpuid = Sys.getenv(x = "CUDA_VISIBLE_DEVICES")
mydf = data.frame()
epoch = 1
prefix = paste0( 'lm_nbm_weights_gpu', gpuid )
wtfn=paste0( prefix,'.h5')
csvfn = paste0( prefix, '.csv')
myptwts = c( 0.0001, 0.0005, 0.001, 0.005, 0.01 )
if ( file.exists( wtfn ) ) {
  unetLM = unetLM1
  load_model_weights_hdf5( unetLM, wtfn )
  load_model_weights_hdf5( unetLM1, wtfn )
  myptwts = c( 0.005, 0.01, 0.02 )
}
# ----training,echo=TRUE,eval=FALSE--------------------------------------------
mydf = data.frame()
epoch = 1
mmdWeight = tf$cast( 5.0, mytype )
myptwts = 0.01
for ( ptwt in myptwts ) {
  if ( ptwt >= myptwts[4] ) unetLM = unetLM1
  ptWeight = tf$cast( ptwt, mytype )
  ptWeight2 = tf$cast( 1.0 - ptwt, mytype )
  num_epochs = 200
  if ( ptwt >= myptwts[5] ) num_epochs = 20000
  optimizerE <- tf$keras$optimizers$Adam(2.e-5)
  batchsize = 2
  epoch = 1
  for (epoch in epoch:num_epochs ) {
    if ( (epoch %% round(mybs/batchsize) ) == 0 & epoch > 1  ) {
        # refresh the data
        locfns = chooseTrainingFilesToRead( trtefns[,trnnames], notFirst=TRUE )
        print( locfns[1] )
        Xtr = surgeRy::loadNPData( locfns )
#        Xtr[[1]] = abind::abind( Xtr[[1]], Xtr[[4]], along=5 )
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
      loss_mmd = tf$cast( 0.0, mytype )
      for ( k in 1:batchsize) {
        loss_mmd = loss_mmd + compute_mmd( datalist[[2]][k,,],
          preds[[2]][k,,], takeMean=TRUE ) * mmdWeight / tf$cast(batchsize,mytype)
        }
      lossht = tf$cast( 0.0, mytype )
#      lossht = tf$keras$losses$mse( datalist[[5]], preds[[1]] ) %>% tf$reduce_mean( )
      losspt = tf$keras$losses$mse( datalist[[2]], preds[[2]] ) %>% tf$reduce_mean( )
      loss = losspt * ptWeight + loss_mmd * mmdWeight + lossht * ptWeight2
      })
    unet_gradients <- tape$gradient(loss, unetLM$trainable_variables)
    optimizerE$apply_gradients(purrr::transpose(list(
        unet_gradients, unetLM$trainable_variables )))
    mydf[ct,'train_loss'] = as.numeric( loss )
    mydf[ct,'train_ptlossW'] = as.numeric( losspt * ptWeight)
    mydf[ct,'train_htlossW'] = as.numeric( lossht * ptWeight2 )
    mydf[ct,'train_ptloss'] = as.numeric( losspt )
    mydf[ct,'train_htloss'] = as.numeric( lossht )
    mydf[ct,'train_mmd'] = as.numeric( loss_mmd )
    mydf[ct,'ptWeight'] = as.numeric( ptWeight )
    mydf[ct,'ptWeight2'] = as.numeric( ptWeight2 )
    mydf[ct,'trainData'] = locfns[1]
    if( epoch > 3 & epoch %% 50 == 0 ) {
      with(tf$device("/cpu:0"), {
        preds = predict( unetLM, Xte[c(1,3:4)] )
        lossht = tf$cast( 0.0, mytype )
        local_batch_size = nrow( Xte[[1]] )
        loss_mmd = tf$cast( 0.0, mytype )
        for ( k in 1:local_batch_size) {
          loss_mmd = loss_mmd + compute_mmd( tf$cast(Xte[[2]][k,,],mytype),
            tf$cast(preds[[2]][k,,],mytype), takeMean=TRUE ) * mmdWeight / tf$cast(local_batch_size,mytype)
          }
#        lossht = tf$keras$losses$mse( Xte[[5]], preds[[1]] ) %>% tf$reduce_mean( )
        losspt = tf$keras$losses$mse( Xte[[2]], preds[[2]] ) %>% tf$reduce_mean( )
        loss = tf$cast(losspt, mytype) * ptWeight + tf$cast(lossht, mytype) * ptWeight2
      })
      # compute the same thing in test data
      mydf[ct,'test_loss'] = as.numeric( loss )
      mydf[ct,'test_ptloss'] = as.numeric( losspt )
      mydf[ct,'test_htloss'] = as.numeric( lossht )
      mydf[ct,'test_mmd'] = as.numeric( loss_mmd )
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
