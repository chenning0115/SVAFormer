import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from models import transformer as transformer
import utils
from utils import recorder
from evaluation import HSIEvaluation
import itertools
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from utils import device


class SKlearnTrainer(object):
    def __init__(self, params) -> None:
        self.params = params
        self.net_params = params['net']
        self.train_params = params['train']
        self.evalator = HSIEvaluation(param=params)


        self.model = None
        self.real_init()

    def real_init(self):
        pass
        

    def train(self, trainX, trainY):
        self.model.fit(trainX, trainY)
        print(self.model, "trian done.") 


    def final_eval(self, testX, testY):
        predictY = self.model.predict(testX)
        temp_res = self.evalator.eval(testY, predictY)
        print(temp_res['oa'], temp_res['aa'], temp_res['kappa'])
        return temp_res

    def test(self, testX):
        return self.model.predict(testX)

            
class SVMTrainer(SKlearnTrainer):
    def __init__(self, params) -> None:
        super(SVMTrainer, self).__init__(params)

    def real_init(self):
        kernel = self.net_params.get('kernel', 'rbf')
        gamma = self.net_params.get('gamma', 'scale')
        c = self.net_params.get('c', 1)
        self.model = svm.SVC(C=c, kernel=kernel, gamma=gamma)

class RandomForestTrainer(SKlearnTrainer):
    def __init__(self, params) -> None:
        super().__init__(params)

    def real_init(self):
        n_estimators = self.net_params.get('n_estimators', 200)
        self.model = RandomForestClassifier(n_estimators = n_estimators, max_features="auto", criterion="entropy")

class KNNTrainer(SKlearnTrainer):
    def __init__(self, params) -> None:
        super().__init__(params)

    def real_init(self):
        n = self.net_params.get('n', 10)
        self.model = KNeighborsClassifier(n_neighbors=n)

class BaseTrainer(object):
    def __init__(self, params) -> None:
        self.params = params
        self.net_params = params['net']
        self.train_params = params['train']
        self.device = device 
        self.evalator = HSIEvaluation(param=params)

        self.net = None
        self.criterion = None
        self.optimizer = None
        self.clip = 15
        self.unlabel_loader=None
        self.real_init()

    def real_init(self):
        pass

    def get_loss(self, outputs, target):
        return self.criterion(outputs, target)
       
    def train(self, train_loader, unlabel_loader=None, test_loader=None):
        epochs = self.params['train'].get('epochs', 100)
        total_loss = 0
        epoch_avg_loss = utils.AvgrageMeter()
        max_oa = 0
        for epoch in range(epochs):
            self.net.train()
            epoch_avg_loss.reset()
            for i, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.net(data)
                loss = self.get_loss(outputs, target)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
                self.optimizer.step()
                # batch stat
                total_loss += loss.item()
                epoch_avg_loss.update(loss.item(), data.shape[0])
            recorder.append_index_value("epoch_loss", epoch + 1, epoch_avg_loss.get_avg())
            print('[Epoch: %d]  [epoch_loss: %.5f]  [all_epoch_loss: %.5f] [current_batch_loss: %.5f] [batch_num: %s]' % (epoch + 1,
                                                                             epoch_avg_loss.get_avg(), 
                                                                             total_loss / (epoch + 1),
                                                                             loss.item(), epoch_avg_loss.get_num()))
            # 一定epoch下进行一次eval
            if test_loader and (epoch+1) % 60 == 0:
                y_pred_test, y_test = self.test(test_loader)
                temp_res = self.evalator.eval(y_test, y_pred_test)
                max_oa = max(max_oa, temp_res['oa'])
                recorder.append_index_value("train_oa", epoch+1, temp_res['oa'])
                recorder.append_index_value("train_aa", epoch+1, temp_res['aa'])
                recorder.append_index_value("train_kappa", epoch+1, temp_res['kappa'])
                recorder.append_index_value("max_oa", epoch+1, max_oa)
                print('[--TEST--] [Epoch: %d] [oa: %.5f] [aa: %.5f] [kappa: %.5f] [num: %s]' % (epoch+1, temp_res['oa'], temp_res['aa'], temp_res['kappa'], str(y_test.shape)))
        print('Finished Training')
        return True

    def final_eval(self, test_loader):
        y_pred_test, y_test = self.test(test_loader)
        temp_res = self.evalator.eval(y_test, y_pred_test)
        return temp_res

    def get_logits(self, output):
        if type(output) == tuple:
            return output[0]
        return output

    def test(self, test_loader):
        """
        provide test_loader, return test result(only net output)
        """
        count = 0
        self.net.eval()
        y_pred_test = 0
        y_test = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(self.device)
            logits = self.get_logits(self.net(inputs))
            if len(logits.shape) == 1:
                continue
            outputs = np.argmax(logits.detach().cpu().numpy(), axis=1)
            
            # ------
            # outputs shape [batch], logits: [batch, class_num]
            # p = torch.softmax(logits, dim=-1)
            # for k, temp_label in enumerate(labels):
            #     if temp_label != outputs[k]:
            #         pre = outputs[k]
            #         real = temp_label.detach().cpu().numpy()
            #         pp = p[k].detach().cpu().numpy()
            #         print("real=%s, %.3f, pred=%s, %.3f" % (real, pp[real], pre, pp[pre]))

            # ------

            if count == 0:
                y_pred_test = outputs
                y_test = labels
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))
                y_test = np.concatenate((y_test, labels))
        return y_pred_test, y_test



class TransformerTrainer(BaseTrainer):
    def __init__(self, params):
        super(TransformerTrainer, self).__init__(params)


    def real_init(self):
        # net
        self.net = transformer.TransFormerNet(self.params).to(self.device)
        # self.net = cross_without_all.HSINet(self.params).to(self.device)
        # self.net = cross_without_center.HSINet(self.params).to(self.device)
        # self.net = cross_without_rotate.HSINet(self.params).to(self.device)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def get_loss(self, outputs, target):
        '''
            A_vecs: [batch, dim]
            B_vecs: [batch, dim]
            logits: [batch, class_num]
        '''
        logits = outputs
        
        loss_main = nn.CrossEntropyLoss()(logits, target) 

        return loss_main   


def get_trainer(params):
    trainer_type = params['net']['trainer']
    if trainer_type == "transformer":
        return TransformerTrainer(params)

    assert Exception("Trainer not implemented!")

