rm(list=ls())
dyn.load("normal.so")

.C("normal")
