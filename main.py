import os
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
import seaborn as sns
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from utils import *

def testClassifier(classifier, X, y, split=0.7, ntrials=100):

    means = np.zeros(ntrials, )
    for trial in range(ntrials):
        xTr, yTr, xTe, yTe, trIdx, teIdx = trteSplitEven(X, y, split, trial)
        # Train
        scaler = RobustScaler()
        scaler.fit(xTr)
        xTr = scaler.transform(xTr)
        xTe = scaler.transform(xTe)
        trained_classifier = classifier.trainClassifier(xTr, yTr)
        # Predict
        yPr = trained_classifier.classify(xTe)
        # Compute classification error
        #if trial % 10 == 0:
        print("Trial:", trial, "Accuracy", "%.3g" % (100 * np.mean((yPr == yTe).astype(float))))

        means[trial] = 100 * np.mean((yPr == yTe).astype(float))

    print("Final mean classification accuracy ", "%.3g" % (np.mean(means)), "with standard deviation",
          "%.3g" % (np.std(means)))


def trteSplitEven(X, y, pcSplit, seed=None):
    labels = np.unique(y)
    xTr = np.zeros((0, X.shape[1]))
    xTe = np.zeros((0, X.shape[1]))
    yTe = np.zeros((0,), dtype=int)
    yTr = np.zeros((0,), dtype=int)
    trIdx = np.zeros((0,), dtype=int)
    teIdx = np.zeros((0,), dtype=int)
    np.random.seed(seed)
    for label in labels:
        classIdx = np.where(y == label)[0]
        NPerClass = len(classIdx)
        Ntr = int(np.rint(NPerClass * pcSplit))
        idx = np.random.permutation(NPerClass)
        trClIdx = classIdx[idx[:Ntr]]
        teClIdx = classIdx[idx[Ntr:]]
        trIdx = np.hstack((trIdx, trClIdx))
        teIdx = np.hstack((teIdx, teClIdx))
        # Split data
        xTr = np.vstack((xTr, X[trClIdx, :]))
        yTr = np.hstack((yTr, y[trClIdx]))
        xTe = np.vstack((xTe, X[teClIdx, :]))
        yTe = np.hstack((yTe, y[teClIdx]))

    return xTr, yTr, xTe, yTe, trIdx, teIdx


def label_race(row):
    if (df_room3_outliers.index == row['Time stamp']).any():
        return 1
    elif (absent_on.index == row['Time stamp']).any():
        return 2
    else:
        return 0


def fix_date(column):
    column = pd.to_datetime(column)
    change = True
    i = 0
    while i < column.shape[0]:
        for c in range(13):
            if i + c >= column.shape[0]:
                break
            if c != 0 and column[i + c].hour >= 12:
                change = not change
                break
            else:
                if change:
                    column[i + c] = (
                            """%s%s""" % (column[i + c], " AM"))
                else:
                    column[i + c] = (
                            """%s%s""" % (column[i + c], " PM"))
        i = i + c
    return column


a = os.getcwd()
os.chdir("Elena_202109 - per hour")
df_room3_lightning_toilet = pd.read_excel('Rum&Toilett Y/Belysning Lägenhet C - toilet.xlsx', engine='openpyxl')
df_room3_lightning_toilet['Time stamp'] = pd.to_datetime(df_room3_lightning_toilet['Time stamp'])
grouped_by_presence = df_room3_lightning_toilet.groupby('Lägenhet 3 - Närvaroindikering Badrum')[
    ['Time stamp', 'Ljusstyrka Badrum']].mean()
df_room3_temperature_toilet = pd.read_excel('Rum&Toilett Y/Temperatur 3 - Toilet.xlsx')
df_room3_temperature_toilet['Time stamp'] = pd.to_datetime(df_room3_temperature_toilet['Time stamp'])
df_room3_toilet_merged = df_room3_lightning_toilet.merge(df_room3_temperature_toilet, on='Time stamp')
df_room3_living_area = pd.read_excel('Rum&Toilett Y/Belysning Lägenhet C room.xlsx')
df_room3_living_area['Time stamp'] = pd.to_datetime(df_room3_living_area['Time stamp'])
df_room3 = df_room3_living_area.merge(df_room3_toilet_merged, on='Time stamp')
absent_bathroom_lighton = df_room3[
    (df_room3['Lägenhet 3 - Närvaroindikering Badrum'] == 0.0) & (df_room3['Ljusstyrka Badrum'] > 0.0)]
absent_bathroom_lighoff = df_room3[(df_room3['Lägenhet 3 - Närvaroindikering Badrum'] == 0.0) & (
            df_room3['Lägenhet 3 - Närvaroindikering rum'] == 0.0)]
absent_room_lighton = df_room3[
    (df_room3['Lägenhet 3 - Närvaroindikering rum'] == 0.0) & (df_room3['Ljusstyrka srum'] > 0.0)]

df_room3 = df_room3.assign(Rounded_TS=df_room3['Time stamp'].dt.round('H'))

df_room3_water2020 = pd.read_csv('Rum&Toilett Y/water labtrino/Lgh 3 2020.csv')
df_room3_water2021 = pd.read_csv('Rum&Toilett Y/water labtrino/Lgh 3 2021.csv')

# Water is in 12 hour format but AM PM is missing
df_room3_water2021['hour'] = fix_date(df_room3_water2021['hour'])
df_room3_water2020['hour'] = fix_date(df_room3_water2020['hour'])
df_room3_water = df_room3_water2020.append(df_room3_water2021)
df_room3 = df_room3.merge(df_room3_water, left_on="Rounded_TS", right_on="hour")
df_room3 = df_room3.drop(columns=['Rounded_TS', 'hour'])
df_room3[' Total Volume m3'].describe()
df_room3[' Total Volume m3'].hist(bins=100, log=True)
df_room3['Hour'] = df_room3['Time stamp'].dt.hour
df_room3['Weekday'] = df_room3['Time stamp'].dt.weekday
df_agg = df_room3.groupby([df_room3['Time stamp'].dt.date]).sum()
df_agg[' Total Volume m3'].describe()
corr = df_room3.drop(columns=['Time stamp', ' Total Volume m3']).corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
#plt.show()

df_room3_relevant = df_room3.drop(columns=['Time stamp', 'Temperatur Badrum', '  Cold Water m3', ' Hot Water m3'])
df_room3_electricity = pd.read_excel('Rum&Toilett Y/Electricity total.xlsx')
df_room3_electricity['Time stamp'] = pd.to_datetime(df_room3_electricity['Time stamp'])
df_room3 = df_room3.merge(df_room3_electricity, on='Time stamp')
df_agg = df_room3.groupby([df_room3['Time stamp'].dt.date]).sum()
df_agg.boxplot('Lägenhet 3 - Energiförbrukning föregående timme')
IN_BATHROOM = df_room3['Lägenhet 3 - Närvaroindikering Badrum'] == 1.0
IN_ROOM = df_room3['Lägenhet 3 - Närvaroindikering rum'] == 1.0
df_agg = df_room3.groupby([df_room3['Time stamp'].dt.date]).sum()
std_water = df_agg[' Total Volume m3'].std()
avg_water = df_agg[' Total Volume m3'].mean()
df_agg['EXCESSIVE_WATER'] = 0
#df_agg[df_agg[' Total Volume m3'] > avg_water + std_water]['EXCESSIVE_WATER'] = 1

X = df_room3.drop(columns=['Time stamp', ' Total Volume m3']).dropna().to_numpy()
gm = GaussianMixture(n_components=2, random_state=0).fit(X)
mean = gm.means_
transformer = FastICA(n_components=2, random_state=0)
X_transformed = transformer.fit_transform(X)
x = X_transformed[:, 0]
y = X_transformed[:, 1]
stds = (x.std(), y.std())
df_room3['Hour'] = df_room3['Time stamp'].dt.hour
df_room3['Weekday'] = df_room3['Time stamp'].dt.weekday

_ = sns.displot(x=X_transformed[:, 0], y=X_transformed[:, 1], kind="kde",
                hue=df_room3.drop(columns=['Time stamp', ' Total Volume m3']).dropna()['Weekday'])
#plt.show()
sns.scatterplot(x=X_transformed[:, 0], y=X_transformed[:, 1],
                hue=df_room3.drop(columns=['Time stamp', ' Total Volume m3']).dropna()['Weekday'])
#plt.show()
outliers = [True if abs(x[0] - x.mean()) > x.std() and abs(x[1] - y.mean()) > y.std() else False for x in X_transformed]
out = X_transformed[outliers]
df_room3_outliers = df_room3.dropna()[outliers]
sns.scatterplot(x=out[:, 0], y=out[:, 1],
                hue=df_room3_outliers.drop(columns=['Time stamp', ' Total Volume m3']).dropna()['Weekday'])
#plt.show()
absent_on = df_room3.loc[
    ((df_room3['Lägenhet 3 - Närvaroindikering Badrum'] == 0.0) & (df_room3['Ljusstyrka Badrum'] > 0.0)) | (
                (df_room3['Lägenhet 3 - Närvaroindikering rum'] == 0.0) & (df_room3['Ljusstyrka srum'] > 0.0))]

labels=[]
df_room3= df_room3.dropna()
for index, row in df_room3.iterrows():
    vals= absent_on['Time stamp'].values
    if row['Time stamp'] in absent_on['Time stamp'].values:
        labels.append(1)
    elif row['Time stamp'] in df_room3_outliers['Time stamp'].values:
        labels.append(2)
    else:
        labels.append(0)
df_room3['Label']= labels
#inputs= np.concatenate((X_transformed,np.reshape(np.array(labels),(-1,1))),axis=1)

labels= np.array(labels,dtype=int)
print("Random Forest: ")
testClassifier(RandForestClassifier(), X_transformed, labels, split=0.7)
print("SVM: ")
testClassifier(SVMClassifier(), X_transformed, labels, split=0.7)
print("Bayesian Classifier: ")
testClassifier(BayesClassifier(), X_transformed, labels, split=0.7)
#testClassifier(BoostClassifier(RandForestClassifier(), T=10), X_transformed, labels, split=0.7)
