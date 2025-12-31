
# 🚀 To run the CER-HV framework, follow the following steps
## Note : the CER-HV code will be available after paper acceptance 

The CER-HV workflow consists of two main stages:

---


# **Stage 1 Automated Noise Detection**

## 1️⃣ **Train the CRNN Model**

Use `trainer.py` with the configuration file for your dataset (example: *Muharaf*):

```bash
python trainer.py config-muharaf.yaml \
    --saved_model_path "saved_models/muharaf_saved_models" \
    --wand_project "htr-bp-muharaf" \
    --decoded_samples_file "decoded_samples_muharaf.txt"
```
## 2️⃣ Evaluate the Model on the Validation/Test Splits

Run:

```bash
python evaluate.py config-muharaf.yaml
```

### The above command will generate  (muharaf_predictions.csv)



# Stage 2 — Human Verification and Error Categorization  
## 1️⃣ Prepare Ranked CER Files for Human Review
```bash
python prepare_csv_to_EA.py \
    --csv_path "results/muharaf_predictions.csv" \
    --output_path "ranked_muharaf.csv" \
    --gt_path "../data/muharaf_prepared/train/gt.txt"
```
 
 ## 2️⃣ Run the following to verify and label noisy samples 

 ```bash
python  error_classification.py
   ```
## The error classification/human review framework looks like:


<p align="center">
  <img src="../images/error-classification1.png" width="750" style="border: 2px solid #ccc; padding: 4px;">
</p>

## Select the error type (for this example, it is orientation error ):

<p align="center">
  <img src="../images/error-classification2.png" width="750" style="border: 2px solid #ccc; padding: 4px;" >
</p>




## Note: 📁 Dataset Format Requirements

All datasets used in this repository must follow a consistent directory structure and file format to ensure compatibility with the CER-HV pipeline.

Each dataset should be placed inside the `data/` directory and organized into three splits:
```text
data/
└── <dataset_name>/
├── train/
│     ├── images/
│     └── gt.txt
├── val/
│     ├── images/
│     └── gt.txt
└── test/
├── images/
└── gt.txt
```
### ✔ Ground-Truth File Format (`gt.txt`)

Each `gt.txt` file must contain one line per sample in the following format:

<image_id>  <ground_truth_transcription>