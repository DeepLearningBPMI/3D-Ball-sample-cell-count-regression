import os
import torch
import pathlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from preprocess_3d import TiffDataGenerator
from Vit_3d import ViT

code_base = r"E:\Model_3D_Cell_differenciation"
_DATA_SELECTOR = '*.tif'
classes = ['Neutrophils(%)', 'Eosinophils(%)', 'Lymphocytes(%)', 'Monocytes(%)']
batch_size = 1
lr = 1e-6
dropout=0.22
gamma = 0.7

D=32
W=256
H=256

def run(dataset):
    # ------------------------------------- Configuration options -------------------------------------------
   
    # Get all patient directories
    dirs = os.listdir(dataset)
    # Categorize samples by types
    bal_samples = [d for d in dirs if "BAL" in d and "BALI" not in d]
    bali_samples = [d for d in dirs if "BALI" in d and "Hep" not in d]
    hep_samples = [d for d in dirs if "Hep" in d]

    # Combine remaining samples
    all_samples = hep_samples

    # Prepare test data
    test_dirs = all_samples
    test_folders = test_dirs
    test_images = [str(img) for dir in test_folders for img in pathlib.Path(os.path.join(dataset, dir)).rglob(_DATA_SELECTOR)]

    print(f'test_folders:')
    for folder in test_folders:
            print(os.path.join(dataset_dir, folder))

    test_generator = TiffDataGenerator(test_images,  D, W,H, batch_size, augmentations=False, clahe = True)
    print("Total batches in test_generator:", len(test_generator))

     # ------------------------------------ step 5/5 : testing -------------------------------------------------- 
    # Print about testing
    print('Starting testing')   
    model = ViT(
            image_size = W,         # image size
            frames = D,               # number of frames
            image_patch_size = 16,     # image patch size
            frame_patch_size = 2,      # frame patch size
            num_classes = 4,
            dim = 1536,             #patch_size*patch_size*frame_patch_size*3
            depth = 6,
            heads = 8,
            mlp_dim = 1024,
            dropout = 0.22,
            emb_dropout = 0.22
    )

    model.load_state_dict(torch.load(r"E:\Model_3D_Cell_differenciation\Model\ViT_3D\model_vit_Blood.pth"))
    print("Model loaded for testing.")
    model.eval()

    deep_learning_results=[]
    # Test the model
    with torch.no_grad():
        for batch in test_generator:
            
            if batch is None:
                break

            inputs= batch
            predictions = model(inputs)
            predicted_labels = predictions.numpy()
            deep_learning_results.append(predicted_labels)
            
        deep_learning_results = np.array(deep_learning_results)
        print("deep_learning_results_shape:",deep_learning_results.shape )

    samples_per_patient = 5
    data = []
    file_exists = os.path.isfile(full_path)
    # Average the predictions and true labels for each patient
    for p, patient in enumerate(all_samples):
        start_idx = p * samples_per_patient
        end_idx = start_idx + samples_per_patient
    
        # Slice the arrays to get data for the current patient
        patient_deep_learning = deep_learning_results[start_idx:end_idx, :,:]

        avg_predicted_labels = np.round(np.mean( patient_deep_learning, axis=0).flatten()*100,3)
        std_predicted_labels = np.round(np.std(patient_deep_learning, axis=0).flatten()*100,3)  # Std for error bars
     
        
        # formatted_values = [f"{avg}±{std}" for avg, std in zip(avg_predicted_labels, std_predicted_labels)]
        row = {'Patient': patient}
        for i, class_name in enumerate(classes):
            row[class_name] = f"{avg_predicted_labels[i]:.3f}+-{std_predicted_labels[i]:.3f}"
        data.append(row) 
        
        # Plot the results for each patient
        plot_patient_results(patient, avg_predicted_labels, std_predicted_labels)
    
    df = pd.DataFrame(data)
    df.to_csv(full_path, mode='a', index=False,header=not file_exists, encoding='utf-8-sig')   
    print(f'Results saved to {full_path}')
     
def plot_patient_results(patient, predicted_labels, predicted_std):
    """
    Function to plot bar graphs comparing true labels vs predicted labels with error bars for predicted values.
    """
    plt.rc('font', family='Times New Roman')
    labels = ['Neutrophils', 'Eosinophils', 'Lymphocytes', 'Monocytes']  
    idx = np.arange(len(labels))  
    label_fontsize = 16
    tick_fontsize = 16
    figure_width = 6  
    figure_height = 4
   

    width = 0.3  # Width of the bars
    predicted_labels =predicted_labels.flatten()
    predicted_std = predicted_std.flatten()
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    
    # Plot predicted labels with error bars
    ax.bar(idx, predicted_labels, width, yerr=predicted_std, label='ViT3d predict', capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd'])

    # Labels and title
    ax.set_ylabel('Cell percentages (%)',fontsize=label_fontsize)
    ax.set_xticks(idx )
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
    svg_save_path = os.path.join(output_folder, f'{patient}__ViT.svg')
    tif_save_path = os.path.join(output_folder, f'{patient}__ViT.tif')
    plt.tight_layout()

    plt.savefig(svg_save_path, format='svg',dpi=600)
    print(f"Figure saved for {patient}: {svg_save_path}")

    plt.savefig(tif_save_path, format='tiff', dpi=600)
    print(f"TIFF Figure saved for {patient}: {tif_save_path}")


if __name__ == '__main__':

  dataset_dir = os.path.join(code_base, "data")
  output_folder = os.path.join(code_base, 'predict_plot', 'Blood')
  # Create the output folder if it doesn't exist
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
  csv_filename = 'patient_vit_blood.csv'
  full_path = os.path.join(output_folder, csv_filename)
 
  run(dataset_dir)