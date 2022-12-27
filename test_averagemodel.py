import os
import pandas as pd
from torch.autograd import Variable
from sklearn import metrics
from SNRGCN_model import *

# Path
Dataset_Path = "./Dataset/"
Model_Path = "./Model/"
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

def evaluate(model1,model2,model3,model4,model5, data_loader):
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()

    epoch_loss = 0.0
    n = 0
    valid_pred1 = []
    valid_pred2 = []
    valid_pred3 = []
    valid_pred4 = []
    valid_pred5 = []
    valid_true = []

    for data in data_loader:
        with torch.no_grad():
            sequence_names, _, labels, node_features, graphs = data

            if torch.cuda.is_available():
                node_features = Variable(node_features.to(device))
                graphs = Variable(graphs.to(device))
                y_true = Variable(labels.to(device))
            else:
                node_features = Variable(node_features)
                graphs = Variable(graphs)
                y_true = Variable(labels)

            node_features = torch.squeeze(node_features)
            graphs = torch.squeeze(graphs)
            y_true = torch.squeeze(y_true)

            softmax = torch.nn.Softmax(dim=1)
            y_pred1 = model1(node_features, graphs)
            y_pred1 = softmax(y_pred1)
            y_pred1 = y_pred1.cpu().detach().numpy()

            y_pred2 = model2(node_features, graphs)
            y_pred2 = softmax(y_pred2)
            y_pred2 = y_pred2.cpu().detach().numpy()

            y_pred3 = model3(node_features, graphs)
            y_pred3 = softmax(y_pred3)
            y_pred3 = y_pred3.cpu().detach().numpy()

            y_pred4 = model4(node_features, graphs)
            y_pred4 = softmax(y_pred4)
            y_pred4 = y_pred4.cpu().detach().numpy()

            y_pred5 = model5(node_features, graphs)
            y_pred5 = softmax(y_pred5)
            y_pred5 = y_pred5.cpu().detach().numpy()

            valid_pred1 += [pred[1] for pred in y_pred1]
            valid_pred2 += [pred[1] for pred in y_pred2]
            valid_pred3 += [pred[1] for pred in y_pred3]
            valid_pred4 += [pred[1] for pred in y_pred4]
            valid_pred5 += [pred[1] for pred in y_pred5]

            y_true = y_true.cpu().detach().numpy()
            valid_true += list(y_true)



    return valid_true, valid_pred1, valid_pred2,valid_pred3,valid_pred4,valid_pred5


def analysis(y_true, y_pred, best_threshold = 0.32):
    if best_threshold == None:
        best_f1 = 0
        best_threshold = 0
        for threshold in range(0, 100):
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            f1 = metrics.f1_score(binary_true, binary_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
    binary_true = y_true

    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    AUC = metrics.roc_auc_score(binary_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)

    results = {
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC': AUC,
        'AUPRC': AUPRC,
        'mcc': mcc,
        'threshold': best_threshold
    }
    return results


def test(test_dataframe):
    test_loader = DataLoader(dataset=ProDataset(test_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


    model1 = SNRGCN(LAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT, LAMBDA, ALPHA, VARIANT)
    if torch.cuda.is_available():
        model1.to(device)
    model1.load_state_dict(torch.load(Model_Path + "Fold1_best_model.pkl", map_location='cuda:2'))

    model2 = SNRGCN(LAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT, LAMBDA, ALPHA, VARIANT)
    if torch.cuda.is_available():
        model2.to(device)
    model2.load_state_dict(torch.load(Model_Path + "Fold2_best_model.pkl", map_location='cuda:2'))

    model3 = SNRGCN(LAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT, LAMBDA, ALPHA, VARIANT)
    if torch.cuda.is_available():
        model3.to(device)
    model3.load_state_dict(torch.load(Model_Path + "Fold3_best_model.pkl", map_location='cuda:2'))

    model4 = SNRGCN(LAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT, LAMBDA, ALPHA, VARIANT)
    if torch.cuda.is_available():
        model4.to(device)
    model4.load_state_dict(torch.load(Model_Path + "Fold4_best_model.pkl", map_location='cuda:2'))

    model5 = SNRGCN(LAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT, LAMBDA, ALPHA, VARIANT)
    if torch.cuda.is_available():
        model5.to(device)
    model5.load_state_dict(torch.load(Model_Path + "Fold5_best_model.pkl", map_location='cuda:2'))

    test_true, test_pred1,test_pred2,test_pred3,test_pred4,test_pred5 = evaluate(model1,model2,model3,model4,model5, test_loader)
    test_pred=[]
    for index in range(len(test_pred1)):
        avg = test_pred1[index]+test_pred2[index]+test_pred3[index]+test_pred4[index]+test_pred5[index]
        avg=avg/5
        test_pred.append(avg)
    result_test = analysis(test_true, test_pred)


    print("========== Evaluate Test set ==========")
    print("Test binary acc: ", result_test['binary_acc'], result_test['precision'], result_test['recall'],
          result_test['f1'], result_test['AUC'], result_test['AUPRC'], result_test['mcc'])
    print("Threshold: ", result_test['threshold'])

    # fout1 = open("Graph_315_pred" + ".txt", "w")
    # for ele in test_pred:
    #     ele = float(ele)
    #     fout1.write(str(ele) + "\n")
    # fout1.close()
    #
    # fout2 = open("Graph_315_label"+".txt", "w")
    # for ele in test_true:
    #     ele=float(ele)
    #     fout2.write(str(ele) + "\n")
    # fout2.close()

def test_one_dataset(dataset):
    IDs, sequences, labels = [], [], []
    for ID in dataset:
        IDs.append(ID)
        item = dataset[ID]
        sequences.append(item[0])
        labels.append(item[1])
    test_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    test_dataframe = pd.DataFrame(test_dic)
    test(test_dataframe)


def main():
    with open(Dataset_Path + "Test_60.pkl", "rb") as f:
        Test_60 = pickle.load(f)

    with open(Dataset_Path + "Test_315.pkl", "rb") as f:
        Test_315 = pickle.load(f)

    print("Evaluate GraphPPIS on Test_60")
    test_one_dataset(Test_60)

    print("Evaluate GraphPPIS on Test_315")
    test_one_dataset(Test_315)

if __name__ == "__main__":
    main()
