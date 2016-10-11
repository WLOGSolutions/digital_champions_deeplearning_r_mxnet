library(mxnet)

#Devices
#Using CPU
#devs <- mx.cpu()

#For graphic devices
#use nvidia-smi to check status
gpus_cnt <- 0
devs <- lapply(X = 0:gpus_cnt, FUN = function(i) mx.gpu(i))

#How many learning rounds?
num_round <- 50

#How fast follow the gradient?
learning_rate <- 0.1

#How to deal with distributed learning?
kv_store <- "local"

#Checkpoint model
?mx.callback.save.checkpoint
?mx.model.FeedForward.create



checkpoint_mlp <-  function(iteration, nbatch, env, verbose=TRUE) {
  print(paste("Iteration=",iteration,format(Sys.time(), "%H:%M:%S")))
  if (iteration %% 5 == 0) {
      mx.model.save(env$model, "mlp", iteration)
      cat(sprintf("Model checkpoint saved to %s-%04d.params\n", "mlp", iteration))
  }
  return(TRUE)
}




model <- mx.model.FeedForward.create(
  X                  = mnist_train,
  eval.data          = mnist_validate,
  ctx                = devs,
  symbol             = mlp,
  eval.metric        = mx.metric.accuracy,
  num.round          = num_round,
  learning.rate      = learning_rate,
  momentum           = 0.9,
  wd                 = 0.00001,
  kvstore            = kv_store,
  array.batch.size   = batch_size,
  epoch.end.callback = checkpoint_mlp,
  batch.end.callback = mx.callback.log.train.metric(150))
  


