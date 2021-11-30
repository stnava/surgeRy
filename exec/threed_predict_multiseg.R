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

add_background <-function( x ) {
  odim = newdim = dim( x )
  n = length( odim )
  newdim[ n ] = odim[ n ] + 1
  newx = array( 0, dim = newdim )
  newx[,,,,1] = 1
  for ( j in 2:newdim[n] ) {
    newx[,,,,j] = x[,,,,j-1]
    newx[,,,,1] = newx[,,,,1] - x[,,,,j-1]
  }
  return( newx )
}

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
    totalLoss = totalLoss +
      tf$reduce_mean( tf$keras$losses$mse(y_true[,,,,j], y_pred[,,,,j]) * weights[j] )
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
    locloss = binary_dice( y_true[,,,,j] , y_pred[,,,,j] )
    print( paste( j , as.numeric(locloss ) ) )
    totalLoss = totalLoss + locloss * tf$cast( weights[j], mytype )
    }
  return( totalLoss )
}


## ----howdowereadintime,echo=TRUE,eval=FALSE-----------------------------------
trtefns = read.csv( "numpySegM/multisegtrainttestfiles.csv" ) # critical - same file name
trnnames = colnames(trtefns)[grep("train", colnames(trtefns) )]
tstnames = colnames(trtefns)[grep("test", colnames(trtefns) )]
whichcolstouse = c( 1, 5 )
loadfirst = trtefns[1,trnnames[whichcolstouse]]
Xtr = surgeRy::loadNPData( loadfirst )
Xtr[[2]] = add_background( Xtr[[2]] )
mybs = dim( Xtr[[1]] )[1]
Xte = surgeRy::loadNPData( trtefns[1,tstnames[whichcolstouse]] )
Xte[[2]] = add_background( Xte[[2]] )
nclasstoseg = tail( dim( Xte[[2]] ), 1 ) # GET THIS FROM Xtr/Xte
#############################################################################$$$
nlayers = 4   # for unet
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
#       additionalOptions = c( "attentionGating" ),
       mode = c( "classification" )
     )
unet2 = clone_model( unet )
gpuid = Sys.getenv(x = "CUDA_VISIBLE_DEVICES")
mydf = data.frame()
epoch = 1
wtfn=paste0('mseg_weights_gpu', gpuid,'.h5')
csvfn = paste0('mseg_weights_gpu', gpuid,'.csv')

# if ( file.exists( wtfn ) ) load_model_weights_hdf5( unet, wtfn, T, T, T )

# ----training,echo=TRUE,eval=FALSE--------------------------------------------
mydf = data.frame()
epoch = 1
num_epochs = 50000
optimizerE <- tf$keras$optimizers$Adam(1.e-5)
batchsize = 2
for (epoch in 1:num_epochs ) {
    if ( (epoch %% round(mybs/batchsize) ) == 0 & epoch > 1 ) {
      fe = file.exists( trtefns[,trnnames[whichcolstouse][1]])
      loadfirst = chooseTrainingFilesToRead( trtefns[fe,trnnames[whichcolstouse]], random=TRUE, notFirst=TRUE)
      Xtr = surgeRy::loadNPData( loadfirst )
      Xtr[[2]] = add_background( Xtr[[2]] )
      }
    ct = nrow( mydf ) + 1
    mysam = sample( 1:nrow(Xtr[[1]]), batchsize )
    datalist = list()
    for ( jj in 1:2 )
      datalist[[jj]] = array( Xtr[[jj]][mysam,,,,],
        dim=c(batchsize,tail(dim(Xtr[[jj]]),4)) ) %>% tf$cast( mytype )
    with(tf$GradientTape(persistent = FALSE) %as% tape, {
      preds = ( unet( datalist[[1]] ) )
#      preds = tf$nn$softmax( unet( datalist[[1]] ) )
#      predsMX = tf$nn$softmax( preds )
#      diceBoth = binary_dice(
#        datalist[[2]][,,,,1] + datalist[[2]][,,,,2],
#        predsMX[,,,,1] + predsMX[,,,,2 ] )
#      dice1 = binary_dice( datalist[[2]][,,,,2], preds[,,,,2] )
#      dice2 = binary_dice( datalist[[2]][,,,,3], preds[,,,,3] )
#      mymse = weighted_mse( datalist[[2]], preds, weights = c( 0.0, 1.0, 1.0 ) )
#      print( paste( "DICE",as.numeric(dice1), as.numeric(dice2), as.numeric(mymse) ))
      mycce = tf$keras$losses$categorical_crossentropy( datalist[[2]], preds ) %>% tf$reduce_mean()
      # loss = dice1 + dice2 + mycce + mymse * 0.01
#      loss = tf$keras$losses$categorical_crossentropy( datalist[[2]], preds )
#      loss = multilabel_dice_coefficient()( datalist[[2]], predsMX ) * 20.0 +
#        tf$nn$sigmoid_cross_entropy_with_logits( datalist[[2]], preds ) %>% tf$reduce_mean()
      loss = multilabel_dice_coefficient()( datalist[[2]], preds ) + mycce
      })
    unet_gradients <- tape$gradient(loss, unet$trainable_variables)
    optimizerE$apply_gradients(purrr::transpose(list(
        unet_gradients, unet$trainable_variables )))
    mydf[ct,'train_loss'] = as.numeric( loss )
    mydf[ct,'trainData'] = loadfirst[1]
    if( epoch > 3 & epoch %% 20 == 0 ) {
      with(tf$device("/cpu:0"), {
        preds = predict( unet, Xte[[1]] )
        x1 = tf$cast(Xte[[2]],mytype)
        x2 = tf$cast(preds, mytype )
        loss = multilabel_dice_coefficient()( x1, x2 )
        loss2 = binary_dice(  x1[,,,,3], x2[,,,,3] )
        loss1 = binary_dice(  x1[,,,,2], x2[,,,,2] )
#        loss = weighted_dice( tf$cast(Xte[[2]],mytype), tf$cast(preds, mytype ))
#        loss = tf$keras$losses$categorical_crossentropy( tf$cast(Xte[[2]],mytype), tf$cast(preds, mytype ) ) %>% tf$reduce_mean()
#    loss = tf$nn$sigmoid_cross_entropy_with_logits( tf$cast(Xte[[2]],mytype), tf$cast(preds, mytype ) ) %>% tf$reduce_mean()
#      loss = tf$keras$losses$categorical_crossentropy( tf$cast(Xte[[2]],mytype), tf$cast(preds, mytype ) ) %>% tf$reduce_mean()
      })
      # compute the same thing in test data
      mydf[ct,'test_loss1'] = as.numeric( loss1 )
      mydf[ct,'test_loss2'] = as.numeric( loss2 )
      mydf[ct,'test_loss'] = as.numeric( loss )
      loe = epoch - 200 # best recent result
      if ( loe < 1 ) loe = 1
      if ( mydf[ct,'test_loss'] <= min(mydf[loe:epoch,'test_loss'],na.rm=TRUE) ) {
        print(paste("Saving",epoch))
        keras::save_model_weights_hdf5( unet, wtfn )
        gc()
      }
    }
  print( mydf[ct,] )
  write.csv( mydf, csvfn, row.names=FALSE )
  }
