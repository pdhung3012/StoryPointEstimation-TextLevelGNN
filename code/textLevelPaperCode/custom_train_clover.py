import torch
from model import Model
from custom_data_helper import DataHelper
import numpy as np
import tqdm
import sys, random
import argparse
import time, datetime
import os
from custom_pmi import cal_PMI
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import precision_score,accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
NUM_ITER_EVAL = 100
EARLY_STOP_EPOCH = 25

fopDataset='../../../../dataPapers/dataTextLevelPaper/'
fnSystem='clover'
fpLabel=fopDataset+fnSystem+"/test_label.txt"
fpPred=fopDataset+fnSystem+"/test_pred.txt"
fpResultSEEShort=fopDataset+"/resultSEE_"+fnSystem+"_short.txt"
fpResultSEEDetails=fopDataset+"/resultSEE_"+fnSystem+"_details.txt"
fpResultSEEAna=fopDataset+"/resultSEE_"+fnSystem+"_ana.txt"
fpTextLabel = fopDataset+fnSystem+'/label.txt'


def edges_mapping(vocab_len, content, ngram):
    count = 1
    mapping = np.zeros(shape=(vocab_len, vocab_len), dtype=np.int32)
    for doc in content:
        for i, src in enumerate(doc):
            for dst_id in range(max(0, i-ngram), min(len(doc), i+ngram+1)):
                dst = doc[dst_id]

                if mapping[src, dst] == 0:
                    mapping[src, dst] = count
                    count += 1

    for word in range(vocab_len):
        mapping[word, word] = count
        count += 1

    return count, mapping


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return datetime.timedelta(seconds=int(round(time_dif)))


def dev(model, dataset):
    data_helper = DataHelper(dataset, mode='dev')

    total_pred = 0
    correct = 0
    iter = 0
    for content, label, _ in data_helper.batch_iter(batch_size=64, num_epoch=1):
        iter += 1
        model.eval()

        logits = model(content)
        pred = torch.argmax(logits, dim=1)

        correct_pred = torch.sum(pred == label)

        correct += correct_pred
        total_pred += len(content)


    print('type {}\t{}'.format(type(total_pred),type(correct)))
    if not isinstance(total_pred,int):
        total_pred = int(total_pred)
    if not isinstance(correct,int):
        correct = correct.int()
    # print(torch.div(correct, total_pred))
    if total_pred ==0:
        return 0
    return torch.div(correct, total_pred)


def test(model_name, dataset,dictLabel):
    model = torch.load(os.path.join('.', model_name + '.pkl'))

    data_helper = DataHelper(dataset, mode='test')

    total_pred = 0
    correct = 0
    iter = 0
    list_pred=[]
    list_label = []
    for content, label, _ in data_helper.batch_iter(batch_size=64, num_epoch=1):
        iter += 1
        model.eval()

        logits = model(content)
        pred = torch.argmax(logits, dim=1)
        #print('label and pred {}\n{}'.format(label,pred))
        for i in range(0,len(label)):
            list_pred.append(int(dictLabel[int(pred[i])]))
            list_label.append(int(dictLabel[int(label[i])]))

        correct_pred = torch.sum(pred == label)

        correct += correct_pred
        total_pred += len(content)

    total_pred = float(total_pred)
    correct = correct.float()
    # print(torch.div(correct, total_pred))
    return torch.div(correct, total_pred).to('cpu'),list_label,list_pred


def train(ngram, name, bar, drop_out, dataset, is_cuda=False, edges=True):
    start_time = time.time()
    tupLst = []
    print('load data helper.')
    data_helper = DataHelper(dataset, mode='train')
    f1 = open(data_helper.current_set, 'r')
    arrContent = f1.read().strip().split('\n')
    tupLst.append(len(arrContent))
    f1.close()

    if os.path.exists(os.path.join('.', name+'.pkl')) and name != 'temp_model':
        print('load model from file.')
        model = torch.load(os.path.join('.', name+'.pkl'))
    else:
        print('new model.')
        if name == 'temp_model':
            name = 'temp_model_%s' % dataset
        # edges_num, edges_matrix = edges_mapping(len(data_helper.vocab), data_helper.content, ngram)
        edges_weights, edges_mappings, count = cal_PMI(dataset=dataset,window_size=20)
        print('count {} ds {}'.format(count,dataset))
        tupLst.append(len(data_helper.vocab))
        tupLst.append(count)
        
        model = Model(class_num=len(data_helper.labels_str), hidden_size_node=200,
                      vocab=data_helper.vocab, n_gram=ngram, drop_out=drop_out, edges_matrix=edges_mappings, edges_num=count,
                      trainable_edges=edges, pmi=edges_weights, cuda=is_cuda)

    print(model)
    if is_cuda:
        print('cuda')
        model.cuda()
    loss_func = torch.nn.CrossEntropyLoss()

    optim = torch.optim.Adam(model.parameters(), weight_decay=1e-6)

    iter = 0
    if bar:
        pbar = tqdm.tqdm(total=NUM_ITER_EVAL)
    best_acc = 0.0
    last_best_epoch = 0
    start_time = time.time()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for content, label, epoch in data_helper.batch_iter(batch_size=32, num_epoch=200):
        improved = ''
        model.train()

        logits = model(content)
        loss = loss_func(logits, label)

        pred = torch.argmax(logits, dim=1)

        correct = torch.sum(pred == label)

        total_correct += correct
        total += len(label)

        total_loss += loss.item()

        optim.zero_grad()
        loss.backward()
        optim.step()

        iter += 1
        if bar:
            pbar.update()
        if iter % NUM_ITER_EVAL == 0:
            if bar:
                pbar.close()

            val_acc = dev(model, dataset=dataset)
            if val_acc > best_acc:
                best_acc = val_acc
                last_best_epoch = epoch
                improved = '*'
                torch.save(model, name + '.pkl')
            elif best_acc==0:
                torch.save(model, name + '.pkl')

            if epoch - last_best_epoch >= EARLY_STOP_EPOCH:
                end_time = time.time()
                running_time = end_time - start_time
                tupLst.append(running_time)
                return name, tupLst
            msg = 'Epoch: {0:>6} Iter: {1:>6}, Train Loss: {5:>7.2}, Train Acc: {6:>7.2%}' \
                  + 'Val Acc: {2:>7.2%}, Time: {3}{4}' \
                  # + ' Time: {5} {6}'

            print(msg.format(epoch, iter, val_acc, get_time_dif(start_time), improved, total_loss/ NUM_ITER_EVAL,
                             float(total_correct) / float(total)))

            total_loss = 0.0
            total_correct = 0
            total = 0
            if bar:
                pbar = tqdm.tqdm(total=NUM_ITER_EVAL)

    end_time = time.time()
    running_time = end_time - start_time
    tupLst.append(running_time)
    return name, tupLst


def word_eval():
    print('load model from file.')
    data_helper = DataHelper('r8')
    edges_num, edges_matrix = edges_mapping(len(data_helper.vocab), data_helper.content, 1)
    model = torch.load(os.path.join('word_eval_1.pkl'))

    edges_weights = model.seq_edge_w.weight.to('cpu').detach().numpy()

    core_word = 'billion'
    core_index = data_helper.vocab.index(core_word)

    results = {}
    for i in range(len(data_helper.vocab)):
        word = data_helper.vocab[i]
        n_word = edges_matrix[i, core_index]
        # n_word = edges_matrix[i, i]
        if n_word != 0:
            results[word] = edges_weights[n_word][0]

    sort_results = sorted(results.items(), key=lambda d: d[1])

    print(sort_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngram', required=False, type=int, default=1, help='ngram number')
    parser.add_argument('--name', required=False, type=str, default='temp_model', help='project name')
    parser.add_argument('--bar', required=False, type=int, default=0, help='show bar')
    parser.add_argument('--dropout', required=False, type=float, default=0.5, help='dropout rate')
    parser.add_argument('--dataset', required=False, type=str,default=fnSystem, help='dataset')
    parser.add_argument('--edges', required=False, type=int, default=1, help='trainable edges')
    parser.add_argument('--rand', required=False, type=int, default=7, help='rand_seed')

    args = parser.parse_args()

    print('ngram: %d' % args.ngram)
    print('project_name: %s' % args.name)
    print('dataset: %s' % args.dataset)
    print('trainable_edges: %s' % args.edges)
    # #
    SEED = args.rand
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    if args.bar == 1:
        bar = True
    else:
        bar = False
    
    if args.edges == 1:
        edges = True
        print('trainable edges')
    else:
        edges = False

    dictLabel = {}
    fff = open(fpTextLabel, 'r')
    lUn = fff.read().split('\n')
    fff.close()

    for i in range(0, len(lUn)):
        dictLabel[i] = lUn[i].strip()

    o2 = open(fpResultSEEDetails, 'w')
    o2.write('')
    o2.close()
    lstResultOverProjects = []
    lstStrResultOverProjects = []

    model, tupLst = train(args.ngram, args.name, bar, args.dropout, dataset=args.dataset, is_cuda=True, edges=edges)
    #model='temp_model_'+fnSystem
    result,lLabel,lPred=test(model, args.dataset,dictLabel)
    print('test acc: ', result.numpy())
    # maeAccuracy = mean_absolute_error(lLabel, lPred)
    y_test=lLabel
    predicted=lPred
    classAccuracy = accuracy_score(y_test, predicted)

    print('{}\t{:.2f}'.format(fnSystem, classAccuracy))

    o2 = open(fpResultSEEDetails, 'a')
    o2.write(fnSystem + '\n')
    o2.write('Result for GCN TextLevel \n')
    # o2.write('MAE {}\nMQE {}\n\n\n'.format(maeAccuracy,mqeAccuracy))

    # o2.write(str(sum(cross_val) / float(len(cross_val))) + '\n')
    o2.write(str(confusion_matrix(y_test, predicted)) + '\n')
    o2.write(str(classification_report(y_test, predicted)) + '\n\n\n')
    o2.close()
    lstResultOverProjects.append(classAccuracy)
    lstStrResultOverProjects.append('{}\t{}'.format(fnSystem, classAccuracy))

    fff=open(fpLabel,'w')
    fff.write('\n'.join(map(str,lLabel)))
    fff.close()
    fff=open(fpPred,'w')
    fff.write('\n'.join(map(str,lPred)))
    fff.close()
    fff = open(fpResultSEEDetails, 'a')
    fff.write('{}\t{}\n'.format(fnSystem, classAccuracy))
    fff.close()
    fff = open(fpResultSEEAna, 'w')
    fff.write('{}\t{}\n'.format(fnSystem, '\t'.join(map(str, tupLst))))
    fff.close()

    averageAccuracy = np.average(lstResultOverProjects)
    lstStrResultOverProjects.append('{}\t{}'.format('Avg', averageAccuracy))
    f1 = open(fpResultSEEShort, 'w')
    f1.write('\n'.join(lstStrResultOverProjects))
    f1.close()

