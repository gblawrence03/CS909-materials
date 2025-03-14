import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv('https://warwick.ac.uk/fac/sci/dcs/teaching/material/cs909/protein_expression_data.csv')

df['specimen_id']=df.VisSpot.apply(lambda x: x.split('-')[2]) #create specimen id field
df['image_id']=df.VisSpot.apply(lambda x: x.split('-')[2])+'_'+df.id #create image id field
train_specimens = ['B1', 'C1', 'D1']
df = df.set_index('image_id').sort_index()
protein_names = ['SMAa', 'CD11b',
       'CD44', 'CD31', 'CDK4', 'YKL40', 'CD11c', 'HIF1a', 'CD24', 'TMEM119',
       'OLIG2', 'GFAP', 'VISTA', 'IBA1', 'CD206', 'PTEN', 'NESTIN', 'TCIRG1',
       'CD74', 'MET', 'P2RY12', 'CD163', 'S100B', 'cMYC', 'pERK', 'EGFR',
       'SOX2', 'HLADR', 'PDGFRa', 'MCT4', 'DNA1', 'DNA3', 'MHCI', 'CD68',
       'CD14', 'KI67', 'CD16', 'SOX10']

df_train = df.loc[df['specimen_id'].isin(train_specimens)]
df_test = df.loc[df['specimen_id'] == 'A1']

image_folder = '/content/patches_256/'
from skimage.feature import local_binary_pattern, hog
from skimage.color import rgb2gray, rgba2rgb, rgb2hed
from skimage.io import imread
from skimage.util import img_as_ubyte
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import os
import glob

training_image_paths = glob.glob('2025/A2/'+image_folder+'[BCD]*.png')

from tqdm import tqdm

training_images = {}
for path in tqdm(training_image_paths):
    
    image_id = os.path.splitext(os.path.basename(path))[0]
    # We don't care about the image if it doesn't appear in the CSV
    if image_id not in df_train.index:
        continue
    I = plt.imread(path)
    if I.shape[2] == 4: # It's an RGBA image
        I = rgba2rgb(I)
    training_images[path] = I

n_train = len(training_images)

average_hs = []
average_bs = []
average_es = []
cd11b_values = []
specimens = []

for path in tqdm(training_images.keys()):
    image_id = os.path.splitext(os.path.basename(path))[0]
    I = training_images[path]

    row = df_train.loc[image_id]
    cd11b_values.append(row['CD11b'])
    specimens.append(row['specimen_id'])

    I_b = I[:,:,2]
    I_hed = rgb2hed(I)
    I_h = I_hed[:,:,0];
    I_e = I_hed[:,:,1]
    average_hs.append(np.mean(I_h))
    average_es.append(np.mean(I_e))
    average_bs.append(np.mean(I_b))

average_hs = np.array(average_hs)
average_bs = np.array(average_bs)
average_es = np.array(average_es)
cd11b_values = np.array(cd11b_values)

print(f"Average H: {np.mean(average_hs)}")
print(f"Average Blue: {np.mean(average_bs)}")
print(f"Average E: {np.mean(average_es)}")

b_indices = [i for i, s in enumerate(specimens) if s == 'B1']
c_indices = [i for i, s in enumerate(specimens) if s == 'C1']
d_indices = [i for i, s in enumerate(specimens) if s == 'D1']

from sklearn.decomposition import PCA

width, height = list(training_images.values())[0].shape[:2]

X = np.array(list(training_images.values()))
X_HED = np.zeros_like(X)

features = {
    "mean_R": np.mean(X[..., 0], axis=(1, 2)),  # Mean of Red channel
    "var_R": np.var(X[..., 0], axis=(1, 2)),    # Variance of Red channel
    "mean_G": np.mean(X[..., 1], axis=(1, 2)),  # Green
    "var_G": np.var(X[..., 1], axis=(1, 2)),    
    "mean_B": np.mean(X[..., 2], axis=(1, 2)),  # Blue
    "var_B": np.var(X[..., 2], axis=(1, 2)),    
    "mean_H": np.mean(X_HED[..., 0], axis=(1, 2)),   # H
    "var_H": np.var(X_HED[..., 0], axis=(1, 2)),
    "mean_E": np.mean(X_HED[..., 1], axis=(1, 2)),   # E
    "var_E": np.var(X_HED[..., 1], axis=(1, 2)),
    "mean_D": np.mean(X_HED[..., 2], axis=(1, 2)),   # D
    "var_D": np.var(X_HED[..., 2], axis=(1, 2)),
}

# Convert to DataFrame
df_features = pd.DataFrame(features)

# Compute Pearson correlation with CD11b
pearson_corr = df_features.apply(lambda x: pearsonr(x, df_train["CD11b"]).correlation).to_frame(name="Pearson Correlation")
spearman_corr = df_features.apply(lambda x: spearmanr(x, df_train["CD11b"]).correlation).to_frame(name="Spearman Correlation")

# Combine both Pearson and Spearman correlation results
df_corr = pd.concat([pearson_corr, spearman_corr], axis=1)

# Reshape to table format (transpose for better readability)
df_corr_table = df_corr.T

# Plot heatmap
color_scale = 0.05  # Adjust for better visualization
plt.figure(figsize=(10, 2))
sns.heatmap(df_corr_table, annot=True, cmap="coolwarm", center=0, fmt=".3f", vmin=-color_scale, vmax=color_scale)
plt.title("Pearson & Spearman Correlation of Mean/Variance Features with CD11b Expression")
plt.show()