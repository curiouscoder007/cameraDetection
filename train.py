#####
#This file calls the functions required to train a model
#
#####

from trainModel import trainCNNModel

trainer = trainCNNModel('somefile') #use a filename with merge=False if using a single pickle file
trainer.loadData()
trainer.saveModel('cnn_model_v2.h5',epochs=1,weights=True)
