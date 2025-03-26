import os
import torch
import pathlib
import pickle
import numpy as np
import torch.nn as nn
import math
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import MSELoss
import tifffile
from torchsummary import summary
from tqdm import tqdm  # Import tqdm for a progress bar
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from volumentations import *
from torchvision import transforms
from matplotlib import pyplot as plt
from preprocess_3d import TiffDataGenerator
from resnet_3d_train_parallel import  resnet50

code_base = '/scistor/guest/mzu240/BAL/'
_DATA_SELECTOR = '*.tif'
classes_name = ['Neutrophils', 'Eosinophils', 'Lymphocytes', 'Macrophages']
batch_size = 1
lr = 0.0012
neuro_hidden =32
dropout =0.3
D=40
W=400
H=400

def run(dataset):
    # ------------------------------------- Configuration options -------------------------------------------
    results = {}
    torch.manual_seed(42)  # Set fixed random seed

    # Get all patient directories
    dirs = os.listdir(dataset)

    # Categorize samples by types
    bal_samples = [d for d in dirs if "BAL" in d and "BALI" not in d]
    bali_samples = [d for d in dirs if "BALI" in d and "Hep" not in d]
    hep_samples = [d for d in dirs if "Hep" in d]

    # Combine remaining samples
    all_samples = bal_samples + bali_samples

    # Create dataframe for data split
    samples = []
    for r in all_samples:
        if 'BALI' in r:
            samples.append('BALI')
        else:
            samples.append('BAL')

    df = pd.DataFrame()
    df['dirs'] = all_samples
    df['sample'] = samples

    # Prepare test data
    test_dirs = df.dirs
    test_folders = test_dirs
    test_images = [str(img) for dir in test_folders for img in pathlib.Path(os.path.join(dataset, dir)).rglob(_DATA_SELECTOR)]
    test_labels = [str(pathlib.Path(case).parents[0] / 'labels.txt') for case in test_images]

    print(f'test_folders:')
    for folder in test_folders:
            print(os.path.join(dataset_dir, folder))

    test_generator = TiffDataGenerator(test_images, test_labels, D, W, H, batch_size, augmentations=False, clahe=True)

    # Load model and criterion
    model = resnet50(D, W, H, neuro_hidden, dropout)
    criterion = torch.nn.L1Loss()
    model.load_state_dict(torch.load('/scistor/guest/mzu240/BAL/Results_3D_BAL_ResNet_LDS_1119/11-19 16:20/resnet3d50_BAL_lr0.0012_dropout0.3_SGD_neurons_hidden32/model.pth'))
    model.eval()

    # Create log directory
    result_dir = code_base + '/test-ResNet-BAL-11191620_1209'
    now_time = datetime.now()
    time_str = now_time.strftime('%m-%d %H:%M')
    log_dir = os.path.join(result_dir, time_str, f'ResNet3d_test_lr{lr}_dropout{dropout}_11191620')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    mae_scores = []
    patient_results = {}
    cytology_results = []
    deep_learning_results=[]
    # Test the model
    with torch.no_grad():
        index = 0
        for batch in test_generator:
            
            if batch is None:
                break

            inputs, labels = batch
            predictions = model(inputs)

            true_labels = labels.numpy()
            predicted_labels = predictions.numpy()

            cytology_results.append(true_labels)
            deep_learning_results.append(predicted_labels)
            
            # Iterate over classes to plot individual bars
            for i in range(true_labels.shape[1]):  
                true_labels_i = true_labels[:, i]  # Take mean if there are multiple samples
                predicted_labels_i = predicted_labels[:, i]  # Take mean if there are multiple samples
                mae_i = mean_absolute_error(true_labels[:, i], predicted_labels[:, i])
                mae_scores.append(mae_i)
                # Add bar plot for each class
                writer.add_scalars('True_vs_Predicted/Class_{}'.format(i + 1),
                        {'True_Label': true_labels_i, 'Predicted_Label': predicted_labels_i},
                            global_step=index)
                writer.add_scalars('MAE/Class_{}'.format(i + 1),
                        {'MAE':mae_i}, global_step=index)
            index += 1
        cytology_results = np.array(cytology_results)
        deep_learning_results = np.array(deep_learning_results)
        mae_scores_results=np.array(mae_scores)
        print("cytology_results_shape:",cytology_results.shape )
        print("deep_learning_results_shape:",deep_learning_results.shape )
        # Calculate average metrics across all samples
        avg_mae = np.mean(mae_scores)  
        print("Average MAE:", avg_mae)

    samples_per_patient = 5
    data = []
    # Average the predictions and true labels for each patient
    for p, patient in enumerate(all_samples):
        start_idx = p * samples_per_patient
        end_idx = start_idx + samples_per_patient
    
        # Slice the arrays to get data for the current patient
        patient_cytology = cytology_results[start_idx:end_idx, :,:]
        patient_deep_learning = deep_learning_results[start_idx:end_idx, :,:]

        avg_true_labels = np.mean(patient_cytology, axis=0)  # Average across samples
        avg_predicted_labels = np.mean( patient_deep_learning, axis=0)
        std_predicted_labels = np.std(patient_deep_learning, axis=0)  # Std for error bars
        mean_mae = np.mean(mae_scores_results)
        data.append({
        'Patient': patient,
        'Average True Labels': avg_true_labels.flatten(),  # Convert to 1D if necessary
        'Average Predicted Labels': avg_predicted_labels.flatten(),
        'Standard Deviation': std_predicted_labels.flatten(),
        'mean_MAE': mean_mae
        })


        # Plot the results for each patient
        plot_patient_results(patient, avg_true_labels, avg_predicted_labels, std_predicted_labels)
    
    df = pd.DataFrame(data)
    df.to_csv(full_path, index=False)   
    print(f'Results saved to {full_path}')
    
def plot_patient_results(patient, true_labels, predicted_labels, predicted_std):
    """
    Function to plot bar graphs comparing true labels vs predicted labels with error bars for predicted values.
    """
    plt.rc('font', family='Times New Roman')
    labels = ['Neutrophils', 'Eosinophils', 'Lymphocytes', 'Macrophages']  
    idx = np.arange(len(labels))  
    label_fontsize = 16
    tick_fontsize = 16
    legend_fontsize = 14
    figure_width = 6  
    figure_height = 4
   

    width = 0.2  # Width of the bars
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    true_labels=true_labels.flatten()
    predicted_labels =predicted_labels.flatten()
    print("Shape of true_labels:", np.shape(true_labels))


    # Plot true labels
    bars1 = ax.bar(idx, true_labels*100, width, label='Cytology results', color=['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd'])

    # Plot predicted labels with error bars
    bars2 = ax.bar(idx + width, predicted_labels*100, width, yerr=predicted_std*100, label='ResNet3d predict', capsize=5, color=['#c6dbef', '#fdd0a2', '#a1d99b', '#dadaeb'])

    # Labels and title
    ax.set_ylabel('Cell percentages (%)',fontsize=label_fontsize)
    ax.set_xticks(idx + width / 2)
    ax.set_xticklabels(labels, fontsize=tick_fontsize, rotation=10, ha='center')
    ax.legend()
    ax.set_yscale('log')
    ax.set_ylim(0.1, 120) 
    y_ticks = [0.1, 1, 2, 5, 10, 20, 50, 100] 
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)
    ax.tick_params(axis='y', labelsize=15)
    # Show grid only for the y-axis
    ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=1) 

    # Save the figure with the sample name
    svg_save_path = os.path.join(output_folder, f'{patient}_resnet-11191620.svg')
    tif_save_path = os.path.join(output_folder, f'{patient}_resnet-11191620.tif')
    plt.tight_layout()
    
    plt.savefig(svg_save_path, format='svg',dpi=600)
    print(f"Figure saved for {patient}: {svg_save_path}")

    plt.savefig(tif_save_path, format='tiff', dpi=600)
    print(f"TIFF Figure saved for {patient}: {tif_save_path}")

if __name__ == '__main__':

  dataset_dir = code_base + "dataset_test/"
  output_folder = os.path.join(code_base, 'predict_plot_2', 'BAL')
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
  csv_filename = 'patient_ResNet_BAL11191620.csv'
  full_path = os.path.join(output_folder, csv_filename)
  run(dataset_dir)