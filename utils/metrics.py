import numpy as np


class Metric_Manager:
    def __init__(self, num_classes=1):
        self.num_classes = num_classes
        self.TP = np.zeros(self.num_classes)
        self.TN = np.zeros(self.num_classes)
        self.FP = np.zeros(self.num_classes)
        self.FN = np.zeros(self.num_classes)
        self.total_count = 0
        self.epsilon = 1e-6
    
    def compute_single_metrics(self, predictions, labels, normal = False):
        TP = TN = FP = FN = 0
        if not normal:
            for index in range(len(predictions)):
                pred = predictions[index]
                actual = labels[index]
                if pred == 1 and actual == 1:
                    TP += 1
                elif pred == 0 and actual == 0:
                    TN += 1
                elif pred == 1 and actual == 0:
                    FP += 1
                elif pred == 0 and actual == 1:
                    FN += 1
                
        else:
            for index in range(len(predictions)):
                if 1 in predictions[index]:
                    pred = 0
                else:
                    pred = 1
                actual = labels[index]
                
                if pred == 1 and actual == 1:
                    TP += 1
                elif pred == 0 and actual == 0:
                    TN += 1
                elif pred == 1 and actual == 0:
                    FP += 1
                elif pred == 0 and actual == 1:
                    FN += 1
                # print(pred, predictions[index])
            
        return TP, TN, FP, FN

    def update(self, predictions, labels):                
        TP = np.zeros(self.num_classes)
        TN = np.zeros(self.num_classes)
        FP = np.zeros(self.num_classes)
        FN = np.zeros(self.num_classes)

        for index in range(self.num_classes):
            TP[index], TN[index], FP[index], FN[index] = self.compute_single_metrics(predictions[:, index], labels[:, index])
        
        self.TP += TP
        self.TN += TN
        self.FP += FP
        self.FN += FN
        assert len(predictions) == len(labels), "predictions and labels must have the same length"
        self.total_count += len(predictions)
        # return TP, TN, FP, FN
    
    def reset(self):
        self.TP = np.zeros(self.num_classes)
        self.TN = np.zeros(self.num_classes)
        self.FP = np.zeros(self.num_classes)
        self.FN = np.zeros(self.num_classes)
        self.total_count = 0
        self.accuracy = 0.
        self.recall = 0.
        self.precision = 0.
        
    def get_metrics(self):
        self.accuracy = (self.TP + self.TN)/(self.TP + self.TN + self.FP + self.FN)
        self.recall = self.TP/(self.TP + self.FN + self.epsilon)
        self.precision = self.TP/(self.TP + self.FP + self.epsilon)
        self.specificity = self.TN/(self.TN + self.FP + self.epsilon)
        return self.accuracy, self.recall, self.precision, self.specificity
    
    def compute_score(self):
        total_TP = np.sum(self.TP[1:])
        total_TN = np.sum(self.TN[1:])
        total_FP = np.sum(self.FP[1:])
        total_FN = np.sum(self.FN[1:])
        average_accuracy = (total_TP + total_TN) / (total_TP + total_TN + total_FP + total_FN)
        average_recall = total_TP / (total_TP + total_FN)
        average_precision = total_TP / (total_TP + total_FP)
        average_specificity = total_TN / (total_TN + total_FP)
        
        # print(average_accuracy)
        total_score = (average_accuracy + average_recall + average_precision) / 3
        return [total_score, average_accuracy, average_recall, average_precision, average_specificity]
    
    def get_matrix(self):
        return self.TP, self.TN, self.FP, self.FN
    
    
class Metric_Manager_Normal(Metric_Manager):
    def __init__(self, num_classes=1):
        super(Metric_Manager_Normal, self).__init__()
        self.num_classes = num_classes
        self.TP = np.zeros(self.num_classes)
        self.TN = np.zeros(self.num_classes)
        self.FP = np.zeros(self.num_classes)
        self.FN = np.zeros(self.num_classes)
        self.total_count = 0
        self.epsilon = 1e-6
        
    def update(self, predictions, labels):                
        TP = np.zeros(self.num_classes)
        TN = np.zeros(self.num_classes)
        FP = np.zeros(self.num_classes)
        FN = np.zeros(self.num_classes)

        for index in range(1, self.num_classes):
            TP[index], TN[index], FP[index], FN[index] = self.compute_single_metrics(predictions[:, index-1], labels[:, index])
        
        TP[0], TN[0], FP[0], FN[0] = self.compute_single_metrics(predictions, labels[:, 0], normal=True)

        self.TP += TP
        self.TN += TN
        self.FP += FP
        self.FN += FN
        assert len(predictions) == len(labels), "predictions and labels must have the same length"
        self.total_count += len(predictions)
        # return TP, TN, FP, FN