import time
import os.path as osp
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer,
)
from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval,
)
from tensorboardX import SummaryWriter, summary, writer
from collections import deque
from tqdm import tqdm

class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model, self.para_model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)

    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        label = torch.arange(args.way, dtype=torch.int16).repeat(args.query)
        label_aux = torch.arange(args.way, dtype=torch.int8).repeat(args.shot + args.query)
        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()  
        return label, label_aux
    
    def string_label_name(filePath):
        filePath=osp.join(filePath + '.csv')
        df=pd.read_csv(filePath,header=1,index_col=0)
        for i, item in df.iteritems():
            human_readable_label_name=item.unique()
        return human_readable_label_name


    def train(self):
        args = self.args
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()
        
        # start FSL training for template matching
        label, label_aux = self.prepare_label()
        writer = SummaryWriter(logdir=args.save_path)
        for epoch in range(1, args.max_epoch + 1):
            self.train_epoch += 1
            self.model.train()
            if self.args.fix_BN:
                self.model.encoder.eval()
            
            tl1 = Averager()
            tl2 = Averager()
            ta = Averager()
            
            start_tm = time.time()
            for batch in self.train_loader:
                self.train_step += 1

                if torch.cuda.is_available():
                    #data, gt_label = [_.cuda() for _ in batch] # use this and delete next two lines of code when dataloader loads actual image and tensor both
                    #data, real_label= [_.cuda() for _ in batch]
                    data,gt_label,file_name= batch
                    data=data.cuda()
                else:
                    data, gt_label = batch[0], batch[1]
               
                data_tm = time.time()
                self.dt.add(data_tm - start_tm)

                # get saved centers
                logits, reg_logits = self.para_model(data)
                if reg_logits is not None:
                    loss = F.cross_entropy(logits, label)
                    total_loss = loss + args.balance * F.cross_entropy(reg_logits, label_aux)
                else:
                    loss = F.cross_entropy(logits, label)
                    total_loss = F.cross_entropy(logits, label)
                    
                tl2.add(loss)
                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)
                acc = count_acc(logits, label)

                tl1.add(total_loss.item())
                ta.add(acc)

                self.optimizer.zero_grad()
                total_loss.backward()
                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)

                self.optimizer.step()
                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)    

                # refresh start_tm
                start_tm = time.time()
            self.lr_scheduler.step()
            print('epoch {}, total loss={:.4f}, loss={:.4f} acc={:.4f}'
                  .format(epoch,tl1.item(), tl2.item(), ta.item()))
            self.try_evaluate(epoch)

            print('ETA:{}/{}'.format(
                    self.timer.measure(),
                    self.timer.measure(self.train_epoch / args.max_epoch))
            )

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')

    def evaluate(self, data_loader):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))       
        with torch.no_grad():
            for i, batch in enumerate(data_loader, 1):
                if torch.cuda.is_available():
                    #data, _ = [_.cuda() for _ in batch] #use this and delete next two lines of code when dataloader loads actual image and tensor both
                    #data, real_label= [_.cuda() for _ in batch]
                    #below lines added so that confusion matrix can be plotted 
                    data,_,_= batch
                    data=data.cuda()
                else:
                    data = batch[0]

                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
                
                
        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        
        # train mode
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        return vl, va, vap

    def evaluate_test(self):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
        self.model.eval()
        record = np.zeros((self.args.iterations, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        #confusion_matrix=torch.zeros(args.eval_way,args.eval_way) 
        confusion_matrix=torch.zeros(39,39) 
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.test_loader, 1)):

                if torch.cuda.is_available():
                    #data, real_label= [_.cuda() for _ in batch]
                    data,real_label,file_name= batch
                    data=data.cuda()
                else:
                    data = batch[0]
                real_label = real_label.type(torch.LongTensor)[args.eval_way:]
                if torch.cuda.is_available():
                    real_label = real_label.cuda()
                tested_file_name=file_name[args.eval_way:]
                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                pred = torch.argmax(logits, dim=1)
                actual_pred=torch.where(pred.eq(label),real_label,real_label[label[pred]])
                from model.dataloader.scanimage import test_path
                readable_label_list=FSLTrainer.string_label_name(test_path)
                majorCount=0
                testlength=0
                #below section is used to make model more explainable for inference of images and major and minor classes accuracy
                for imageName,realLabel,actualLabel,logitValue in zip(tested_file_name,real_label,actual_pred,logits):
                    testlength+=1
                    #print('\ntested file name is:',imageName)
                    #print('actual label is:',readable_label_list[realLabel.item()])
                    actualMajorClass=readable_label_list[realLabel.item()][:-8]
                    #print('predicted label is:',readable_label_list[actualLabel.item()])  
                    predictedMajorClass=readable_label_list[actualLabel.item()][:-8]
                    if (actualMajorClass==predictedMajorClass):
                        majorCount=majorCount+1
                    else:
                        pass
                print('major class correctness in percentage is:',majorCount/testlength*100)
                acc = count_acc(logits, label)
                #print('acc is:',acc)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
                real_label = batch[1]
                #to plot confusion matrix
                #for t,p in zip(real_label.view(-1), actual_pred.view(-1)):
                 #   confusion_matrix[t.long(),p.long()]+=1
        #cm=confusion_matrix.numpy()
        #cm_col=cm / cm.sum(axis=1)  
        #cm_row=cm / cm.sum(axis=0)
        
        #fig, ax = plt.subplots(figsize=(25,20))
        #heatmap=sns.heatmap(cm_col,annot=True,linewidths=.5, linecolor="0.3",
         #                  square=True,cbar=True, vmin=0, vmax=1,xticklabels=classes_allseen, yticklabels=classes_allseen)
        #plt.savefig("output_col_normalized.png")
        #fig, ax = plt.subplots(figsize=(25,20))
        #heatmap=sns.heatmap(cm_row,annot=True,linewidths=.1, linecolor="0.3",
        #                   square=True,cbar=True, vmin=0, vmax=1,xticklabels=classes_allseen, yticklabels=classes_allseen)
        #plt.savefig("output_row_normalized.png")

        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        
        self.trlog['test_acc'] = va
        self.trlog['test_acc_interval'] = vap
        self.trlog['test_loss'] = vl

        print('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        print('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))

        return vl, va, vap
    
    '''def evaluate_test_demonstration(self):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc.pth'))['params'])
        self.model.eval()
        record = np.zeros((self.args.iterations, 2)) # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.test_loader, 1)):

                if torch.cuda.is_available():
                    #data, real_label= [_.cuda() for _ in batch]
                    data,real_label,file_name= batch
                    data=data.cuda()
                else:
                    data = batch[0]
                real_label = real_label.type(torch.LongTensor)[args.eval_way:]
                if torch.cuda.is_available():
                    real_label = real_label.cuda()
                tested_file_name=file_name[args.eval_way:]
                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                pred = torch.argmax(logits, dim=1)
                actual_pred=torch.where(pred.eq(label),real_label,real_label[label[pred]])
                from model.dataloader.scanimage import test_path
                redable_label_list=FSLTrainer.string_label_name(test_path)
                for imageName,realLabel,actualLabel in zip(tested_file_name,real_label,actual_pred):
                    print('\ntested file name is:',imageName)
                    print('actual label is:',redable_label_list[realLabel.item()])
                    print('predicted label is:',redable_label_list[actualLabel.item()])  
                acc = count_acc(logits, label)
                print('acc is:',acc)
                record[i-1, 0] = loss.item()
                record[i-1, 1] = acc
                real_label = batch[1]

        assert(i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:,0])
        va, vap = compute_confidence_interval(record[:,1])
        
        self.trlog['test_acc'] = va
        self.trlog['test_acc_interval'] = vap
        self.trlog['test_loss'] = vl

        print('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
        print('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))

        return vl, va, vap
        '''

    def final_record(self):
        # save the best performance in a txt file
        
        with open(osp.join(self.args.save_path, '{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval'])), 'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))            
