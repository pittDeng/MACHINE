from functools import reduce
class Perceptron(object):
    def __init__(self,input_num,activator,rate):
        self.actvator=activator
        self.weight=[0.0 for _ in range(input_num)]
        self.bias=0
        self.rate=rate
    def predict(self,input_vector):
        return\
            self.actvator(reduce(
                lambda a,b:a+b,map(
                    lambda x, w: x*w,
                    input_vector,self.weight),0.0)
                          +self.bias)
    def update_parameter(self,input_vector,delta):
        zipped=zip(self.weight,input_vector)
        i=0
        for(weight,input) in zipped:
            weight=self.rate*delta*input+weight
            self.weight[i]=weight
            i+=1
        # self.weight=map(lambda w,x:self.rate*delta*x+w,self.weight,input_vector)
        self.bias+=self.rate*delta
    def train(self,input_vector,label,actual_output):

        delta=label-actual_output
        self.update_parameter(input_vector,delta)
    def _one_iteration(self,dataset,labels):
        all_data=zip(dataset,labels)
        right=0.0
        for(input_vector,label) in all_data:
            actual_output = self.predict(input_vector)
            if(actual_output==label):
                right+=1
            self.train(input_vector,label,actual_output)
        return right/len(labels)
    def __str__(self):
        return 'weights\t%s\nbias\t%f'%(list(self.weight),self.bias)
    def train_and_iterate(self,num,dataset,labels):
        '''

        :param num: the number of iterating
        :param dataset: input dataset
        :param label: nominal result
        :return: none
        '''
        for i in range(num):
            rightratio=self._one_iteration(dataset,labels)
            print(self)
            print('%f\n'%rightratio)