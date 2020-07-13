#####
#This file calls the functions required to test a model from the training data split
#
#####


from trainModel import trainCNNModel

trainer = trainCNNModel('somefile')
trainer.loadData()
score,report = trainer.testModel('cnn_model_v2.h5')
print(score)
print(report)