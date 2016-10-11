library(mxnet)

data_dir <- "/home/ubuntu/R/mnist"

#Train data iterator

data_shape <- 28*28 #784
batch_size <- 128
flat <- TRUE

mnist_train <- mx.io.MNISTIter(
  image       = file.path(data_dir, "train-images-idx3-ubyte"),
  label       = file.path(data_dir, "train-labels-idx1-ubyte"),
  input_shape = data_shape,
  batch_size  = batch_size,
  shuffle     = TRUE,
  flat        = flat)

mnist_validate <- mx.io.MNISTIter(
  image       = file.path(data_dir, "t10k-images-idx3-ubyte"),
  label       = file.path(data_dir, "t10k-labels-idx1-ubyte"),
  input_shape = data_shape,
  batch_size  = batch_size,
  flat        = flat)


