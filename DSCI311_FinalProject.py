#                                               +JMJ+

#Imports
import pandas as pd
from scipy.stats import wasserstein_distance
import numpy as np
import ot
from sklearn.datasets import load_iris, load_digits
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import zscore


#Function to calculate Added Wasserstein Distances
def calc_awd(dist1,dist2):
    #Initialize Added Wasserstein Distance to 0
    awd = 0.0

    #Loop through all features and sum the Wasserstein Distances for each feature
    for col in dist1.columns:
        wd = wasserstein_distance(u_values=dist1[col],v_values=dist2[col])
        awd += wd

    #Return final Added Wasserstein Distance
    return awd

#Function to fill a matrix of a specified size with a specified list of values
def fill_matrix(values, size=3):
    mat = np.zeros((size, size))
    idx = 0
    for i in range(size):
        for j in range(i + 1, size):
            mat[i, j] = values[idx]
            mat[j, i] = values[idx]
            idx += 1
    return mat

        ### SECTION ONE: IRIS DATASET ###

###Section 1.1: Load and organize data

#Get dataset and make dataframe
iris = load_iris()
irisdf = load_iris(return_X_y=True,as_frame=True)

#Add column listing the flower species in irisdf
irisdf[0]['Flower'] = irisdf[1]
irisdf = irisdf[0]

#Separate data into the data, the label, and the label names
X = iris['data']
y = iris['target']
labels = iris['target_names']

#Make separate datasets by variety
setosa = 0
versicolor = 1
virginica = 2

#Get the feature vectors for each species
X1 = X[y == setosa]
X2 = X[y == versicolor]
X3 = X[y == virginica]


#Create dataframes for ease of running Added Wasserstein Distance
setosadf = irisdf[irisdf['Flower']==setosa].drop(columns='Flower')
versicolordf = irisdf[irisdf['Flower']==versicolor].drop(columns='Flower')
virginicadf = irisdf[irisdf['Flower']==virginica].drop(columns='Flower')

#Make uniform weight vectors
a = np.ones(len(X1))/len(X1)
b = np.ones(len(X2))/len(X2)
c = np.ones(len(X3))/len(X3)

###Section 1.2: Calculate EMD and AWD

#Compute cost matrices (squared distances between points)
Mset_ver = ot.dist(X1,X2, metric='euclidean')**2
Mset_vir = ot.dist(X1,X3, metric='euclidean')**2
Mver_vir = ot.dist(X2,X3, metric='euclidean')**2

#Calculate full Earth Mover's distances between distributions
EMDset_ver = ot.emd2(a,b,Mset_ver)
EMDset_vir = ot.emd2(a,c,Mset_vir)
EMDver_vir = ot.emd2(b,c,Mver_vir)
full_EMDs = [EMDset_ver, EMDset_vir, EMDver_vir]

#Calculate AWD's
AWDset_ver = calc_awd(setosadf, versicolordf)
AWDset_vir = calc_awd(setosadf, virginicadf)
AWDver_vir = calc_awd(versicolordf, virginicadf)
awds = [AWDset_ver,AWDset_vir,AWDver_vir]

###Section 1.3: Visualize results

#Obtain color mapping of wvu colors
wvu_colors = ["#EAAA00","#002855"]  # Blue to Gold
wvu_cmap = LinearSegmentedColormap.from_list(name="wvu",colors=wvu_colors)

#Visualization 1: Heatmaps of pairwise EMD/AWD values
#Generate matrices of pairwise EMD and AWD values
emd_matrix = fill_matrix(full_EMDs, size=3)
awd_matrix = fill_matrix(awds, size=3)

#Mask the lower half of the triangle because it is redundant
mask = np.tril(np.ones_like(emd_matrix, dtype=bool), k=-1)

#Create plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sns.heatmap(emd_matrix,
            mask=mask,
            annot=False,
            fmt=".2f",
            cmap=wvu_cmap,
            xticklabels=labels,
            yticklabels=labels,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=axes[0])
axes[0].set_title("Pairwise EMD Between Flower Types")

sns.heatmap(awd_matrix,
            mask=mask,
            annot=False,
            fmt=".2f",
            cmap=wvu_cmap,
            xticklabels=labels,
            yticklabels=labels,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=axes[1])
axes[1].set_title("Pairwise AWD Between Flower Types")

plt.tight_layout()

                                ### SECTION TWO: DIGITS DATASET ###

###Section 2.1: Load and organize data

#Load data, create dataframe, separate components as with Iris
digits = load_digits()
digitsdf = load_digits(return_X_y=True,as_frame=True)
digitsdf[0]['Digit'] = digitsdf[1]
digitsdf = digitsdf[0]

#Get a dataframe for each digit
digitDataFrames = [digitsdf[digitsdf['Digit']==digit].drop(columns='Digit') for digit in range(0,10)]

#Now: break data into components for EMD calculations, obtain feature and weight vectors
Xd = digits['data']
yd = digits['target']
feature_vecs = [Xd[yd == target] for target in range(0,10)]#Feature vectors for each class (digit)
weight_vecs = [np.ones(len(x))/len(x) for x in feature_vecs]#Weight vectors (uniform) for each class

###Section 2.2: Calculate pairwise Earth Mover's Distance and Added Wasserstein Distance

#Compute cost matrices and calculate full Earth Mover's Distance between all classes
costMats = []
emd_values = []
for digit1 in range(0,9):
    digit1X = feature_vecs[digit1]
    digit1Weights = weight_vecs[digit1]
    for digit2 in range(digit1+1,10):
        digit2X = feature_vecs[digit2]
        digit2Weights = weight_vecs[digit2]
        costMat = ot.dist(digit1X,digit2X,metric='euclidean')**2
        costMats.append(costMat)
        fullEMD = ot.emd2(digit1Weights,digit2Weights,costMat)
        emd_values.append(fullEMD)

#Calculate pairwise Added Wasserstein Distances
awd_values = []
for digit1 in range(0,9):
    digit1DF = digitDataFrames[digit1]
    for digit2 in range(digit1+1,10):
        digit2DF = digitDataFrames[digit2]
        awdVal = calc_awd(digit1DF,digit2DF)
        awd_values.append(awdVal)

###Section 2.3: Visualize results

#Visualization 1: Scatterplot of EMD (x) vs. AWD (y) values with regression fit

#Fit regression line of data to plot
m, b = np.polyfit(x=emd_values,y=awd_values,deg=1)
emd_array = np.array(emd_values)
linear_fit = m * emd_array + b
equation = f"y = {m:.2f}x + {b:.2f}"

#Create new figure for scatterplot
plt.figure()

#Plot data with regression line
plt.scatter(x=emd_values,y=awd_values,color=wvu_colors[1])
plt.plot(emd_values, linear_fit,color=wvu_colors[0],label=equation)
plt.legend()
plt.title("Scatterplot of Pairwise EMD and AWD values")
plt.ylabel("Added Wasserstein Distance")
plt.xlabel("Earth Mover's Distance")

#Visualization 2: Heatmaps of Pairwise EMD/AWD Calculations
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

#Create matrix of EMD values
emd_matrix = fill_matrix(emd_values,size=10)

#Create new mask of correct size
mask = np.tril(np.ones_like(emd_matrix, dtype=bool), k=-1)

#Plot heatmat in WVU colors
sns.heatmap(emd_matrix, mask=mask,
            annot=False,
            fmt=".2f",
            cmap=wvu_cmap,
            xticklabels=range(10),
            yticklabels=range(10),
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=axes[0])
axes[0].set_title("Pairwise EMD Between Digits", fontsize=14)
axes[0].set_xlabel("Digit")
axes[0].set_ylabel("Digit")


#Create matrix of AWD values
awd_matrix = fill_matrix(awd_values,size=10)

sns.heatmap(awd_matrix,
            mask=mask,
            annot=False,
            fmt=".2f",
            cmap=wvu_cmap,
            xticklabels=range(10),
            yticklabels=range(10),
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=axes[1])
axes[1].set_title("Pairwise AWD Between Digits", fontsize=14)
axes[1].set_xlabel("Digit")
axes[1].set_ylabel("Digit")

plt.tight_layout()

#Visualization 3 and 4: Kernel Density Plots (unstandardized and standardized) of EMD and AWDimport matplotlib.pyplot as plt

#Create KDE plot overlaying EMD and AWD distributions
plt.figure(figsize=(8, 5))

sns.kdeplot(emd_values, label="EMD Values", fill=True, color=wvu_colors[1], alpha=0.6, linewidth=2)
sns.kdeplot(awd_values, label="AWD Values", fill=True, color=wvu_colors[0], alpha=0.6, linewidth=2)

plt.title("KDE Plot of EMD and AWD Values")
plt.xlabel("Distance Value")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()

#Plot 2: Standardized KDE

plt.figure()
# Standardize EMD and AWD
emd_standardized = zscore(emd_values)
awd_standardized = zscore(awd_values)

sns.kdeplot(emd_standardized, label="Standardized EMD", fill=True, color=wvu_colors[1], alpha=0.6, linewidth=2)
sns.kdeplot(awd_standardized, label="Standardized AWD", fill=True, color=wvu_colors[0], alpha=0.6, linewidth=2)

plt.title("KDE Plot of Standardized EMD and AWD Values")
plt.xlabel("Z-Score")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()