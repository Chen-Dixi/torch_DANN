import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import os
from dixitool.data import datasetsFactory
from dixitool.pytorch.optim import functional as optimizerF
from dixitool.pytorch.module import functional as moduleF
import torch.nn as nn
import torch.optim as optim
from models.models import Feature_Extractor,Class_classifier,Domain_classifier
import numpy as np
from utils import log
import dann_train_test 
# experiment parameter
mnistdata_path= "../data"
mnistmdata_path= "../data/MNIST_M"
gamma=10
batch_size = 512
theta = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

source_trainloader, source_testloader = datasetsFactory.create_data_loader("MNIST",mnistdata_path,batch_size)

target_trainloader, target_testloader = datasetsFactory.create_data_loader("MNIST_M",mnistmdata_path,batch_size)

# Network Variable
feature_extractor = Feature_Extractor()
class_classifier = Class_classifier()
domain_classifier = Domain_classifier()
#Remember to class fy
feature_extractor.to(device)
class_classifier.to(device)
domain_classifier.to(device)
#NLL_Loss 
domain_criterion = nn.BCELoss()
class_criterion = nn.NLLLoss()

trainlog_closure = log.get_machinelearning_logger(log.dann_train_logger)
testlog_closure = log.get_machinelearning_logger(log.dann_test_logger)

#init optimizer
#三个params 放在SGD父类Optimizer(object)的self.param_groups里面
optimizer = optim.SGD([{'params': feature_extractor.parameters()},
                        {'params': class_classifier.parameters()},
                        {'params': domain_classifier.parameters()}], lr= 0.01, momentum= 0.9)

dann_train_test.dann_train(feature_extractor, 
    class_classifier,
    domain_classifier,
    class_criterion,
    domain_criterion,
    source_trainloader,
    target_trainloader,
    optimizer,0,2,device=device,closure=trainlog_closure)

dann_train_test.dcnn_test(feature_extractor, class_classifier, domain_classifier, 
    class_criterion,domain_criterion ,source_testloader,target_testloader,
    0,2,closure=testlog_closure)

