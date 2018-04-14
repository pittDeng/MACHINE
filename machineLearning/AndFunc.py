from Perceptron import Perceptron
def myfunc(x):
    return 1 if x>0 else 0
def get_training_dataset():
    dataset=[[1,1,1],[1,1,0],[1,0,1],[1,0,0],[0,1,1],[0,1,0],[0,0,1],[0,0,0]]
    labels=[1,0,0,0,0,0,0,0];
    return dataset,labels


if __name__=='__main__':
    percep=Perceptron(3,myfunc,0.1)
    dataset,labels=get_training_dataset()
    percep.train_and_iterate(20,dataset,labels)
