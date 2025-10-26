import torch
import numpy as np
from collections import OrderedDict
class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()


    def ReluDerivative(x):
        
        x[x>=0] = 1
        x[x<0] = 0
        return x
    
    def SigmoidDerivative(x):
        sig = torch.sigmoid(x)
        derivativeSig = sig*(1-sig)
        return derivativeSig
    
    def IdentityDerivative(x):
        # shape = x.size()
        # iden_deriv=np.ones(x.shape, dtype=float)
        iden_deriv = torch.ones(x.shape)
        return iden_deriv

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # TODO: Implement the forward function
        batch_size = x.shape[0]

        W1 = self.parameters['W1']
        b1 = self.parameters['b1']
        # print(W1.shape)
        # print(b1.shape)
        # print(x.shape)
        # print(torch.transpose(x, 0, 1).shape)
        # print(torch.matmul(W1,torch.transpose(x, 0, 1)).shape)

        b1_reshape = b1.unsqueeze(1).repeat(1, batch_size)

        # print(b1)
        # print(b1i)
        z1 = torch.matmul(W1,torch.transpose(x, 0, 1)) + b1_reshape
        z1 = torch.transpose(z1, 0, 1)
        # print(z1.shape)
        f = self.f_function
        g = self.g_function


        if (f=='relu'):
            R = torch.nn.ReLU()
            z2 = R(z1)
        if (f=='sigmoid'):
            z2 = torch.sigmoid(z1)
        if (f=='identity'):
            I = torch.nn.Identity(z1)
            z2 = I(z1)

        
        
        W2 = self.parameters['W2']
        b2 = self.parameters['b2']

        # print(z2.shape)
        # print(torch.matmul(W2,z2).shape)
        # print(b2.shape)

        b2_reshape = b2.unsqueeze(1).repeat(1, batch_size)
        z3 = torch.matmul(W2,torch.transpose(z2, 0, 1)) + b2_reshape
        z3 = torch.transpose(z3, 0, 1)
        # self.cache['y_hat']=y_hat
        self.cache['z3']=z3
        self.cache['z2']=z2
        self.cache['z1']=z1
        # self.cache['W2']=W2
        self.cache['x']=x

        if (g=='relu'):
            R = torch.nn.ReLU()
            y_hat = R(z3)
        if (g=='sigmoid'):
            y_hat = torch.sigmoid(z3)
        if (g=='identity'):
            I=torch.nn.Identity()
            y_hat = I(z3)
        
        self.cache['y_hat']=y_hat
        # print(y_hat.shape)
        # y_hat_transpose = torch.transpose(y_hat, 0, 1)
        return y_hat


    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function
        if (self.g_function=='relu'):
            dy_hatdz3 = MLP.ReluDerivative(self.cache['z3'])
        if (self.g_function=='sigmoid'):
            dy_hatdz3 = MLP.SigmoidDerivative(self.cache['z3'])
        if (self.g_function=='identity'):
            dy_hatdz3 = MLP.IdentityDerivative(self.cache['z3'])

        # print(dJdy_hat.shape)
        # print(dy_hatdz3.shape)
        dJdz3 = dJdy_hat * dy_hatdz3
        # print(dJdy_hat)
        # print(dJdz3)
        # print(dJdz3.shape)
        dz3dW2 = self.cache['z2']
        dz3db2 = 1

        # print(dz3dW2.shape)
        dJdW2= torch.matmul(torch.transpose(dJdz3,0,1),dz3dW2)
        dJdb2 = torch.sum(dJdz3*dz3db2,0)
        # print(dJdb2.shape)

        dz3dz2=self.parameters['W2']
        # print(dz3dz2.shape)
        # print(dJdz3.shape)
        dJdz2 = torch.matmul(dJdz3,dz3dz2)
        # print(dJdz2.shape)

        if (self.f_function=='relu'):
            dz2dz1 = MLP.ReluDerivative(self.cache['z1'])
        if (self.f_function=='sigmoid'):
            dz2dz1 = MLP.SigmoidDerivative(self.cache['z1'])
        if (self.f_function=='identity'):
            dz2dz1 = MLP.IdentityDerivative(self.cache['z1'])

        # print(dz2dz1)
        dJdz1 = dJdz2*dz2dz1

        dz1dW1=self.cache['x']
        dz1db1=1

        # print(dJdz1.shape)
        # print(dz1dW1.shape)
        dJdW1 = torch.matmul(torch.transpose(dJdz1,0,1),dz1dW1)
        # print(dJdW1.shape)
        dJdb1 = dJdz1*dz1db1
        dJdb1=torch.sum(dJdb1,0)
        # print(dJdb1.shape)

        self.grads['dJdW1'] =  dJdW1
        self.grads['dJdb1'] = dJdb1
        self.grads['dJdW2'] = dJdW2
        self.grads['dJdb2'] = dJdb2




    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the mse loss
    # print(y.shape)
    # print(y_hat.shape)
    # k=  np.prod(y_hat.shape[:-1])
    

    k=y_hat.shape[0]*y_hat.shape[1]
    # print(y_hat.shape)
    l1 = torch.pow(y_hat - y, 2)
    J = torch.norm(l1)
    dJdy_hat = (2*(y_hat-y))/k
    # print(dJdy_hat.shape,"&&")

    return J,dJdy_hat
    
    

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    k=y_hat.shape[0]*y_hat.shape[1]
    eps=1e-9
    y_hat = torch.clamp(y_hat, eps, 1 - eps)
    
    # Binary cross entropy computation
    loss = - (y * torch.log(y_hat) + (1 - y) * torch.log(1 - y_hat))

    grad = ((y_hat - y) /(k* (y_hat  - y_hat*y_hat)))
    # print(grad.shape)
    
    loss = loss.mean()

    return loss,grad




    
