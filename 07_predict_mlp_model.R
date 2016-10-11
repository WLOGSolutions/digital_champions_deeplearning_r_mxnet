library(mxnet)

model <- mx.model.load("mlp", 35)
preds <- predict(model, 
                 X = mnist_validate,
                 ctx = mx.cpu())


mnist_validate$reset()
mnist_validate$iter.next()
val <- mnist_validate$value()
val$data <- as.array(val$data)
val$label  <- as.array(val$label)

#plot digit

image(t(apply(matrix(val$data[, 1], nrow = 28, byrow = TRUE),2,rev)),  col = grey(seq(0,1,length.out = 256)))
title(main =sprintf("Record number=[%s]. Label = %s", 1, val$label[1]))


for (t in 100:120) {
  image(t(apply(matrix(val$data[, t], nrow = 28, byrow = TRUE),2,rev)),  col = grey(seq(0,1,length.out = 256)))
  title(main =sprintf("Record number=[%s]. Label = %s", t, val$label[t]))
}

