import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),'Chronic_Kidney_Disease'))
from arff2csv import arff2csv # https://github.com/Hutdris/arff2csv - Convert input_file.arff to out_file.csv
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import subprocess
import numpy as np
import re
import time
# import pydot
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


subst_dict = {'abnormal': 1,'notpresent': 1,'no': 1,'poor': 1,'notckd': 1, 'normal': 0,'present': 0,'yes': 0,'good': 0,'ckd': 0, '\?': np.NaN}
files_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Chronic_Kidney_Disease/chronic_kidney_disease.arff')
arff2csv(files_path)
files_path = files_path.replace(".arff", ".csv")
df = pd.read_csv(files_path, header=0, error_bad_lines=False)
features = [x.replace('\'','') for x in list(df.columns)]
df.columns = features


regex_str = '^\s*STR\s*$'  # this regex finds any exactly matching 'STR' with or without extra spaces(typos).
for key, value in subst_dict.items():
        df = df.replace(to_replace=regex_str.replace('STR', key), value=value, regex=True)

usefull_data={}
for index, row in df.iterrows():
    usefull_data[index] = {}
    usefull_data[index]['nbr_value_ok'] = []
    usefull_data[index]['nbr_no_value'] = []

    usefull_data[index]['nbr_value_ok'] = list(row.isnull()).count(False)  # count how many valid values each row has
    usefull_data[index]['nbr_no_value'] = list(row.isnull()).count(True)  # count how many nan each row has


#normalizing
float_array = df.values.astype(float)  # transform df in a float array (still contains nan)
min_max_scaler = preprocessing.MinMaxScaler()
scaled_array = min_max_scaler.fit_transform(float_array)
df = pd.DataFrame(scaled_array, columns=features)
# 1)
more_than_20_valid_indexes = [k for k, v in usefull_data.items() if v['nbr_value_ok'] >= 20]
more_than_20_valid = df.reindex(more_than_20_valid_indexes)
# 2)
equal_than_25_valid_indexes = [k for k, v in usefull_data.items() if v['nbr_value_ok'] == 25]
equal_than_25_valid = df.reindex(equal_than_25_valid_indexes)
X = equal_than_25_valid
# 3)
missing_1_indexes = [k for k, v in usefull_data.items() if v['nbr_value_ok'] == 24]
missing_1 = df.reindex(missing_1_indexes)


# 3)a)
# Identify which column is missing on a given row
# missing_1_recovered_normalized = []
# for index, row in missing_1.iterrows():
#     id = np.argwhere(np.isnan(row))[0][0]
#     mask = np.ones(len(row), dtype=bool)
#     mask[id] = False
#     data_test = row[mask]
#     data_train = X
#     data_train = data_train.drop(data_train.columns[id], axis=1) # {0 or ‘index’, 1 or ‘columns’}, default 0
#     y_train = X
#     y_train = y_train.reindex(columns=[y_train.columns[id]])
#     ridgereg = Ridge(alpha=10.0)
#     ridgereg.fit(data_train.values, y_train.values)
#     y_pred = ridgereg.predict(data_test.values.reshape(1, -1))[0][0]
#     # data_test.insert(, missing_1.columns[5], y_pred)
#     missing_1_recovered_normalized.append(list(np.insert(data_test.values, id, y_pred)))
#
# missing_1_recovered_normalized = df.from_records(missing_1_recovered_normalized, columns=features)
# missing_1_recovered_denormalized = min_max_scaler.inverse_transform(missing_1_recovered_normalized)
# missing_1_recovered_denormalized = df.from_records(missing_1_recovered_denormalized, columns=features)
# boolean_features = {'rbc': 0,'pc': 0,'pcc': 0,'ba': 0, 'sc': 0,'sod': 0,'pot': 0,'hemo': 1,'pcv': 0, 'htn': 0,'dm': 0,'cad': 0, 'appet': 0, 'pe': 0, 'ane': 0, 'class': 0}
# missing_1_recovered_denormalized_rounded = missing_1_recovered_denormalized.round(boolean_features)




dict = {}
for index, row in more_than_20_valid.iterrows():
    dict[index] = []
    ids = np.argwhere(np.isnan(row))
    ids = [ids[i][0] for i in range(len(ids))]
    nbr_missing_features = len(ids)
    predicted = []
    dict[index] = []
    new_row = list(row.values)
    for id in ids:
        mask = np.ones(len(row), dtype=bool)
        mask[ids] = False
        data_test = row[mask]
        data_train = X
        data_train = data_train.drop(data_train.columns[ids].values, axis=1)
        y_train = X
        y_train = y_train.reindex(columns=[y_train.columns[id]])
        ridgereg = Ridge(alpha=10.0)
        ridgereg.fit(data_train.values, y_train.values)
        y_pred = ridgereg.predict(data_test.values.reshape(1, -1))[0][0]
        new_row[id] = y_pred
    dict[index].extend(new_row)
recovered_data = pd.DataFrame.from_dict(dict, orient='index', columns=features)
recovered_data_denormalized = min_max_scaler.inverse_transform(recovered_data)
recovered_data_denormalized = df.from_records(recovered_data_denormalized, columns=features)
boolean_features = {'rbc': 0,'pc': 0,'pcc': 0,'ba': 0, 'sc': 0,'sod': 0,'pot': 0,'hemo': 1,'pcv': 0, 'htn': 0,
                    'dm': 0,'cad': 0, 'appet': 0, 'pe': 0, 'ane': 0, 'class': 0}
recovered_data_denormalized_rounded = recovered_data_denormalized.round(boolean_features)


# Defining data and the target
target = recovered_data_denormalized_rounded['class']
data = recovered_data_denormalized_rounded.drop('class', axis=1)

# Decison tree
clf = tree.DecisionTreeClassifier("entropy")
clf = clf.fit(data, target)

dot_data = tree.export_graphviz(
    clf,
    out_file='Tree.dot',
    feature_names= [features[i] for i in range(0, 24)],
    class_names=['ckd','notckd'],
    filled=True,
    rounded=True,
    special_characters=True)

# time.sleep(2)
# subprocess.check_call(['dot','-Tpng','Tree.dot','-o','OutputFile.png'])

# Plot predictions
predictions = clf.predict(data)

print("Healthy -> " + str(np.count_nonzero(predictions == 1)))
print("Unhealthy -> " + str(np.count_nonzero(predictions == 0)))

labels = 'Unhealthy', 'Healthy'
sizes = [np.count_nonzero(predictions == 0), np.count_nonzero(predictions == 1)]
colors = ['lightcoral', 'lightgreen']

plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()

# #Plot the feature importances
# feature_importances = clf.feature_importances_
# indexes = np.argsort(feature_importances)[::-1]
# plt.figure()
# plt.title("Importance of each feature")
# plt.bar(range(data.shape[1]), feature_importances[indexes],
#        color="r", align="center")
# plt.xticks(range(data.shape[1]), [features[i] for i in indexes], rotation='vertical')
# plt.ylabel('normalized importance [0-1(max)]')
# plt.xlim([-1, data.shape[1]])
# plt.show()
