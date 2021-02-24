from .layer import *

class Network(object):
    def __init__(self):
        self.fc1 = FullyConnected(28*28, 400)
        self.relu1 = ReLU()
        
        self.fc2 = FullyConnected(400, 60)
        self.relu2 = ReLU()
        

        self.classifier = FullyConnected(60, 47)

        self.smce = SoftmaxWithCE()

    def forward(self, input, target):
        ##example
        h1 = self.fc1.forward(input)
        a1 = self.relu1.forward(h1)
        h2 = self.fc2.forward(a1)
        a2 = self.relu2.forward(h2)
        out = self.classifier.forward(a2)
        pred, loss = self.smce.forward(out, target)

        return pred, loss

    def backward(self):
        input_gradient_smce = self.smce.backward()
        input_gradient_relu2 = self.classifier.backward(input_gradient_smce)
        input_gradient_fc2 = self.relu2.backward(input_gradient_relu2)
        input_gradient_relu1 = self.fc2.backward(input_gradient_fc2)
        input_gradient_fc1 = self.relu1.backward(input_gradient_relu1)
        input_gradient = self.fc1.backward(input_gradient_fc1)
        
    def update(self, lr):
        ##
        self.fc1.weight -= lr*self.fc1.weight_grad
        self.fc1.bias -= lr*self.fc1.bias_grad
        
        self.fc2.weight -= lr*self.fc2.weight_grad
        self.fc2.bias -= lr*self.fc2.bias_grad

        self.classifier.weight -= lr*self.classifier.weight_grad 
        self.classifier.bias -= lr*self.classifier.bias_grad
        
        
