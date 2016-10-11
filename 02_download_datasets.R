data_dir <- "mnist"

dir.create(data_dir, showWarnings = FALSE)

if ((!file.exists(file.path(data_dir, 'train-images-idx3-ubyte'))) ||
    (!file.exists(file.path(data_dir, 'train-labels-idx1-ubyte'))) ||
    (!file.exists(file.path(data_dir, 't10k-images-idx3-ubyte'))) ||
    (!file.exists(file.path(data_dir, 't10k-labels-idx1-ubyte')))) {
  download.file(url = 'http://data.dmlc.ml/mxnet/data/mnist.zip',
                destfile = 'mnist.zip', 
                method = 'internal')
  unzip("mnist.zip", exdir = data_dir)
  file.remove("mnist.zip")
}
