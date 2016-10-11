.libPaths("libs")

install.packages("checkpoint")
repo <- checkpoint:::getSnapshotUrl(snapshotDate = "2016-09-20")

install.packages("devtools", repos = repo)
install.packages("drat", repos = repo)

drat:::addRepo("dmlc")
install.packages("mxnet")
