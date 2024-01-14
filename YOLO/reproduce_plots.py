# This file to reproduce some plots for the YOLO, data used from the training output

import pandas as pd
import matplotlib.pyplot as plt

# load the data from the CSV file
# results_path = r"C:\Users\marah\OneDrive\Skrivebord\exp135_cam3_m14\results.csv"
# results_path = r"C:\Users\marah\OneDrive\Skrivebord\exp127_cam4_m15\results.csv"
results_path = r"C:\Users\marah\OneDrive\Skrivebord\exp133_cam5_m15\results.csv"


results_df = pd.read_csv(results_path)
print(results_path)

# strip the column names of any leading/trailing whitespace
results_df.columns = results_df.columns.str.strip()
fontsize = 18



# Plot mAP over Epochs
plt.figure(figsize=(10, 6))
plt.plot(results_df['epoch'], results_df['metrics/mAP_0.5'], linestyle='-', color='blue')
plt.xlabel('Epoch', fontsize=fontsize)
plt.ylabel('mAP@0.5', fontsize=fontsize)
# plt.title('mAP@0.5 over Epochs', fontsize=16)
plt.grid(True)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.tight_layout()
plt.show()


# Plot training and validation loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(results_df['epoch'], results_df['train/box_loss'], label='train loss', color='blue')
plt.plot(results_df['epoch'], results_df['val/box_loss'], label='validation loss', color='orange')
plt.xlabel('Epoch', fontsize=fontsize)
plt.ylabel('Loss', fontsize=fontsize)
# plt.title('Training and Validation Loss over Epochs', fontsize=fontsize)
plt.legend(loc='upper right', fontsize=fontsize)
plt.grid(True)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.tight_layout()
plt.show()


# To print the F1-score results
p = results_df['metrics/precision']
r = results_df['metrics/recall']
# Calculate F1-score for each epoch
results_df['F1-score'] = 2 * (p * r) / (p + r)
results_df['F1-score'].fillna(0, inplace=True)
# print the results
print(results_df[['epoch', 'F1-score']])
