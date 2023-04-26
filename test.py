import os
import pandas as pd
from torch.autograd import Variable
from sklearn import metrics
from SNRGCN_model import *

# Path
Dataset_Path = "./Dataset/"
Model_Path = "./Model/"
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

def evaluate(model3, data_loader):

    model3.eval()


    epoch_loss = 0.0
    n = 0

    valid_pred3 = []
    valid_true=[]

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


            y_pred3 = model3(node_features, graphs)
            y_pred3 = softmax(y_pred3)
            y_pred3 = y_pred3.cpu().detach().numpy()


            valid_pred3 += [pred[1] for pred in y_pred3]


            y_true = y_true.cpu().detach().numpy()
            valid_true += list(y_true)



    return valid_true, valid_pred3


def analysis(y_true, y_pred, best_threshold = None):
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
    # MAP_CUTOFF=14
    # LAYER = 8
    # DROPOUT = 0.1
    # LAMBDA = 1.2
    # ALPHA = 0.7
    # LEARNING_RATE = 1e-3
    test_loader = DataLoader(dataset=ProDataset(test_dataframe), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    model3 = GraphPPIS(LAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT, LAMBDA, ALPHA, VARIANT)
    if torch.cuda.is_available():
        model3.to(device)  # cuda()
    model3.load_state_dict(torch.load(Model_Path + "Fold1_best_model.pkl", map_location='cuda:7'))
    test_true,  test_pred3= evaluate( model3,  test_loader)

    # the result of model3
    result_test = analysis(test_true, test_pred3)
    print("========== Evaluate Test set3 ==========")
    print("Test binary acc: ", result_test['binary_acc'], result_test['precision'], result_test['recall'],
          result_test['f1'], result_test['AUC'], result_test['AUPRC'], result_test['mcc'])
    print("Threshold: ", result_test['threshold'])


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
    with open(Dataset_Path + "test_71.pkl", "rb") as f:
        Test_71 = pickle.load(f)

    with open(Dataset_Path + "Test_315.pkl", "rb") as f:
        Test_315 = pickle.load(f)

    print("Evaluate GraphPPIS on Test_71")
    test_one_dataset(Test_71)

    print("Evaluate GraphPPIS on Test_315")
    test_one_dataset(Test_315)

if __name__ == "__main__":
    main()
