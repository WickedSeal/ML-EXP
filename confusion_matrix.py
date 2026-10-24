import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#define true and predicted values as per give dataset
truevalue = [0,0,1,1,1,0,1,0,0,0,1,1,1,0,1]
predicted = [1,0,0,1,1,0,1,0,1,0,1,0,0,0,1]

#calculate confusion matrix
cf = confusion_matrix(truevalue, predicted)

#create a pandas series for true and predicted values
exp_series = pd.Series(truevalue)
pred_series = pd.Series(predicted)

#create a cross-tabulation (contingency) table
ct = pd.crosstab(exp_series, pred_series, rownames=['actual'], colnames=['predicted'], margins=True)

#display the contingency table
print(ct)

#create a plot for the confusion matrix
plt.imshow(cf, cmap=plt.cm.Purples, interpolation='nearest')
plt.colorbar()
plt.title('Confusion Matrix without Normalization')
plt.xlabel('Predicted')
plt.ylabel('Actual')

#set tick marks and class labels
unique_classes = sorted(set(truevalue))
tick_marks = np.arange(len(unique_classes))
class_labels = [str(cls) for cls in unique_classes]
plt.xticks(tick_marks, class_labels)
plt.yticks(tick_marks, class_labels)

#set threshold for text color
thresh = cf.max()/2
for i,j in itertools.product(range(cf.shape[0]), range(cf.shape[1])):
    plt.text(j, i, f'{cf[i,j]:d}', horizontalalignment='center', color='white' if cf[i, j] > thresh else 'black')

plt.show()