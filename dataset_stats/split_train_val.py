from tqdm import tqdm
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

ogDfPath  = Path("../train.csv")
trainPath = Path("positive_patients_train.csv")
valPath   = Path("positive_patients_val.csv")

ogDf    = pd.read_csv(ogDfPath)
trainDf = pd.read_csv(trainPath)
valDf   = pd.read_csv(valPath)

oneMinusTarget = ogDf.copy().loc[:, ['patient_id', 'target']]
oneMinusTarget['target'] = ogDf['target'].apply(lambda x: 1-x)

groups = ogDf.loc[:, ['patient_id', 'target']].groupby('patient_id')
oneMinusGroups = oneMinusTarget.loc[:, ['patient_id', 'target']].groupby('patient_id')

trainSubsetDf = oneMinusGroups.sum().loc[trainDf['patient_id'], :]
trainSubsetDf.sort_values('target', ascending=False, inplace=True)

valSubsetDf = oneMinusGroups.sum().loc[valDf['patient_id'], :]
valSubsetDf.sort_values('target', ascending=False, inplace=True)

total = len(ogDf)
trainTotal = trainDf['target'].sum()+trainSubsetDf['target'].sum()
valTotal = valDf['target'].sum()+valSubsetDf['target'].sum()

print("train: {} | {:.1f}%".format(trainTotal, trainTotal/total*100))
print("\tpositives: {} | {:.1f}%".format(trainDf['target'].sum(), trainDf['target'].sum()/trainTotal*100))
print("\tnegatives: {} | {:.1f}%".format(trainSubsetDf['target'].sum(), trainSubsetDf['target'].sum()/trainTotal*100))

print("val: {} | {:.1f}%".format(valTotal, valTotal/total*100))
print("\tpositives: {} | {:.1f}%".format(valDf['target'].sum(), valDf['target'].sum()/valTotal*100))
print("\tnegatives: {} | {:.1f}%".format(valSubsetDf['target'].sum(), valSubsetDf['target'].sum()/valTotal*100))

# Save real csvs
trainIds = trainDf['patient_id']
trainSaveDf = ogDf.query('patient_id in @trainIds')
trainSaveDf.to_csv('train_set.csv')

valIds = valDf['patient_id']
valSaveDf = ogDf.query('patient_id in @valIds')
valSaveDf.to_csv('val_set.csv')

# trainGroup = groups.get_group(trainDf['patient_id'])
# print(trainGroup)
# valIndex   = []
# trainIndex = []
# for i in tqdm(oneMinusTarget.loc[:, 'patient_id']):
#     for trainIter in range(len(trainDf)):
#         if i == trainDf.loc[trainIter, 'patient_id'] and trainDf.loc[trainIter, 'target'] == 1:
#             trainIndex.append(i)
#     for valIter in range(len(valDf)):
#         if i == valDf.loc[valIter, 'patient_id'] and valDf.loc[valIter, 'target'] == 1:
#             valIndex.append(i)

# print("og index: ", len(ogDf))
# print("train index: ", len(trainDf))
# print("val index: ", len(valDf))
# print("new train negatives index: ", len(trainIndex))
# print("new val negatives index: ",   len(valIndex))
# print("check: ", (len(trainIndex)+len(valIndex)) == oneMinusTarget['target'].sum())
# print(oneMinusTarget)

# trainSubDf = ogDf.loc[trainDf['patient_id'], ['patient_id', 'target']]

# trainNegIndex = trainGroup.sum()

# print(trainGroup)

# pos 584
# neg 32542

# Train
# 468
# 26033

# Val
# 116
# 6509