library(mxnet)

#model <- mx.model.load("mlp", 35)


mnist_validate$reset()
mnist_validate$iter.next()

val <- mnist_validate$value()
val$data <- as.array(val$data)
val$label  <- as.array(val$label)

#preds <- predict(model, 
#                 X = val$data,
#                 ctx = mx.cpu())
#ypred = max.col(t(as.array(preds)))
#head(ypred,20)

preds <- predict(model, 
                 X = mnist_validate,
                 ctx = mx.cpu())
ypred = max.col(t(as.array(preds)))-1
head(ypred,20)

head(val$label,20)


t=1
image(t(apply(matrix(val$data[, t], nrow = 28, byrow = TRUE),2,rev)),  col = grey(seq(0,1,length.out = 256)))
title(main =sprintf("Record number=[%s]. Label = %s. Pred = %s", t, val$label[t],ypred[t]))


for (t in 1:25) {
  image(t(apply(matrix(val$data[, t], nrow = 28, byrow = TRUE),2,rev)),  col = grey(seq(0,1,length.out = 256)))
  title(main =sprintf("Record number=[%s]. Label = %s. Pred = %s", t, val$label[t],ypred[t]))
}

