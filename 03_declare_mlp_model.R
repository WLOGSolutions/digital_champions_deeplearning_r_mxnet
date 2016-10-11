library(mxnet)

#Ten digits
num_classes <- 10

# Data layer
data <- mx.symbol.Variable('data')

# First layer
fc1 <- mx.symbol.FullyConnected(data = data, name = 'fc1', num_hidden = 128)
# relu - activation
act1 <- mx.symbol.Activation(data = fc1, name = 'relu1', act_type = "relu")

# Second layer
fc2 <- mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
# relu - activation
act2 <- mx.symbol.Activation(data = fc2, name = 'relu2', act_type = "relu")

# Third layer
fc3 <- mx.symbol.FullyConnected(data = act2, name = 'fc3', num_hidden = num_classes)

# Softmax output layer
mlp <- mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')

