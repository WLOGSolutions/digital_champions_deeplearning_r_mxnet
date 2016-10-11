library(mxnet)

model <- mx.model.load("mlp", 35)

#Devices
#Using CPU
#devs <- mx.cpu()

#For graphic devices
gpus_cnt <- 0
devs <- lapply(X = 0:gpus_cnt, FUN = function(i) mx.gpu(i))

#How many learning rounds?
num_round <- 100

#How fast follow the gradient?
learning_rate <- 0.1

#How to deal with distributed learning?
kv_store <- "local"

#Checkpoint model

checkpoint_mlp <- mx.callback.save.checkpoint(prefix = "mlp", 
                                              period = 5)


model2 <- mx.model.FeedForward.create(
  X                  = mnist_train,
  eval.data          = mnist_validate,
  ctx                = devs,
  arg.params         = model$arg.params,
  aux.params         = model$aux.params,
  symbol             = model$symbol,
  eval.metric        = mx.metric.accuracy,
  num.round          = num_round,
  learning.rate      = 0.001,
  momentum           = 0.9,
  wd                 = 0.00001,
  kvstore            = kv_store,
  array.batch.size   = batch_size,
  epoch.end.callback = checkpoint_mlp,
  batch.end.callback = mx.callback.log.train.metric(50))
