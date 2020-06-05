import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

trainPath = Path("../train.csv")

trainDf = pd.read_csv(trainPath)

print(trainDf.shape)
print(trainDf.head())
print("Positive targets: ", trainDf['target'].sum())
# print(trainDf.describe())

groups = trainDf.groupby('patient_id')
patientCounts = groups.count()
print(patientCounts)

print("\nTotal patients:\t", trainDf['patient_id'].nunique())
print("Total images:\t",   trainDf['image_name'].nunique())
print(patientCounts.describe())

patientCounts.sort_values(by='target', inplace=True)
print(patientCounts['target'])

# patientCounts.iloc[0:100, :]['target'].plot(style='k.')
# plt.ylim(0)
# plt.show()

groups = trainDf.loc[:, ['patient_id', 'target']].groupby('patient_id')
patientPositives = groups.sum()
print("\n")
print(patientPositives)

patientPositives.sort_values(by='target', ascending=True, inplace=True)
print(patientPositives.query('target > 0'))
patientPositives.to_csv('positive_patients.csv')

# pos 584
# neg 32542

# Train
# 468
# 26033

# Val
# 116
# 6509