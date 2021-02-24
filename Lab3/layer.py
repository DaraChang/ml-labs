import numpy as np

class _Layer(object):
    def __init__(self):
        pass

    def forward(self, *input):
        r"""Define the forward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def backward(self, *output_grad):
        r"""Define the backward propagation of this layer.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError

class FullyConnected(_Layer):
    def __init__(self, in_features, out_features):
        self.weight = np.random.randn(in_features, out_features) * 0.01
        self.bias = np.zeros((1, out_features))

        self.weight_grad = np.zeros((in_features, out_features))
        self.bias_grad = np.zeros((1, out_features))

    def forward(self, input):
    	###useful at backward
        self._save_for_backward = input
        output = np.dot(input, self.weight)+self.bias

        ######

        return output

    def backward(self, output_grad):
    	###########################
        
        #p = np.tile(output_grad.sum(axis=1, keepdims=True),(1,60))
        #self.weight_grad = self._save_for_backward * p###########
        self.weight_grad = np.matmul((self._save_for_backward).T, output_grad)
        self.bias_grad = output_grad.sum(axis=0, keepdims=True)
        output = np.matmul(output_grad, (self.weight).T)
        return output

class ReLU(_Layer):
    def __init__(self):
        pass

    def forward(self, input):
        self._save_for_backward = input

        output = np.maximum(input, 0)

        return output

    def backward(self, output_grad):
        input_grad = output_grad.copy()
        input_grad[self._save_for_backward < 0] = 0

        return input_grad

    
class SoftmaxWithCE(_Layer):
    def __init__(self):
        pass
    #input : train_data[it*Batch_size:(it+1)*Batch_size], target : train_label_onehot[it*Batch_size:(it+1)*Batch_size]
    def forward(self, input, target):
        self._save_for_backward_target = target
        '''Softmax'''
        e_input = np.exp(input - np.max(input))
        predict = [row / sum(row) for row in e_input]
        predict = np.array(predict)
        
        self._save_for_backward_predict = predict

        '''Average Cross Entropy'''
        loss = 0
        for i in range(predict.shape[0]):
            for j, k in zip(predict[i], target[i]):
                loss = loss - k * np.log(j)
        ce = loss / input.shape[0]
        
        return predict, ce

    def backward(self):
        #########
        input_grad = self._save_for_backward_predict - self._save_for_backward_target
        
        return input_grad    