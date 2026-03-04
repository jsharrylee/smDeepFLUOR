smDeepFLUOR: Single-Molecule Deep Learning Fluorescence Classification

All experiments were executed using Python 3.9 and TensorFlow 2.15. The
complete computational environment is provided in environment.yml.

Environment setup

To reproduce the environment:

    conda env create -f environment.yml
    conda activate smdeepfluor

Processing pipeline

The smDeepFLUOR workflow consists of the following steps:
0. Subtract background (Fiji imagJ)
1. Single-particle tracking
2. Single-particle cropping
3. Training/test dataset split
4. Data loading
5. Data standardization
6. Model training
7. Model validation

------------------------------------------------------------------------

0. Subtract background (Fiji ImagJ)

1)  Open Fiji ImageJ
    Open tiff file with Fiji ImageJ → Process → Subtract background → Rolling ball radius (4.0) pixels → Save processed tiff

------------------------------------------------------------------------

1. Single particle cropping

Particle tracking with Fiji ImageJ

1)  Open Particle tracker
    Open tiff file with Fiji ImageJ → Plugins → Mosaic → Particle
    Tracker

2)  Particle tracker parameters

Radius: 3 – particle size
Cutoff: 0.001 – size variation
Per/Abs: absolute 150–450 – intensity variation

Particle Linking:
Link Range: 1 – maximum jump frame
Displacement: 1 – maximum jump distance

(Optional) Use Preview Detected to confirm detected particles.

3)  Run

4)  Extract Data
    Click All Trajectories to Table → data exported as CSV file

Output: Particle tracking information CSV file.

------------------------------------------------------------------------

Single particle cropping from TIRF images (Python)

Run the first cell “Single particle cropping from TIRF images” with
custom configuration.

    csv_file_path = r'Imagej_plugin_particle_tracking_output.csv'
    tiff_path     = r'TIRF_image.tif'
    output_dir    = r'output_folder'

Output: cropped .npz files with dimension (7, 7, frames).

------------------------------------------------------------------------

2. Training set & test set split

Run the second cell “Train set & test set split”.

    root_dir = r'Root directory where cropped npz files are'

Output: random split of training set and test set.

------------------------------------------------------------------------

3. Data loading (Class A & Class B)

Run the third and fourth cells:

-   Data loading (classA)
-   Data loading (classB)

    base_path = r'root_path where training and test folders are'

Output: datasets loaded into the environment.

------------------------------------------------------------------------

4. Data standardization

Run the fifth cell “Data standardization”.

Output: standardized dataset as described in the manuscript Methods
section.

------------------------------------------------------------------------

5. Model training

Run the sixth cell “Model training”.

    BATCH_SIZE = 32
    LEARNING_RATE = 1e-5
    EPOCHS = 30

Model saving:

    model_dir = 'directory to save a model'
    os.makedirs(model_dir, exist_ok=True)

    model_name = f'best_model_training_bs{BATCH_SIZE}_lr{LEARNING_RATE:.0e}.keras'
    model_path = os.path.join(model_dir, model_name)

Output: trained model saved to disk.

------------------------------------------------------------------------

6. Model loading

Run the seventh cell “Model loading”.

    model = load_model('saved directory/model.keras')

Output: trained model successfully loaded.

------------------------------------------------------------------------

7. Model validation

Run the eighth cell “Model validation”.

    # Define trained model before this script
    # model = load_model('saved directory/model.keras')

    root_dir = r'root_folder where validation data are'

Output: summary of predictions for the validation dataset.
