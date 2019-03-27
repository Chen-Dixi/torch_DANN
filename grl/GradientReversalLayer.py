from torch.autograd import Function as function
#import torch.autograd.Function as function 这样是不正确的，Function是一个类，python这样的语法是import Module，Function要和torch.autograd分开写

class GradientReversalLayer(function):

    @staticmethod
    def farward(ctx,x,lambda_p):
        ctx.lambda_p = lambda_p
        return x.view_as(x)

    @staticmethod
    def backward(ctx,grad_output):
        grad_output = grad_output.neg() * ctx.lambda_p
        return grad_output , None # return as many tnesors as input to forward

    def grad_reverse(x,lambda_p):
        return GradientReversalLayer.apply(x,lambda_p)