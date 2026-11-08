Oral Microbiome Machine Learning Analysis

About this project

This code analyses bacterial abundance in mouth samples. It compares oral lichen planus (OLP), oral potentially malignant disorders (OPMDs), and oral squamous cell carcinoma (OSCC) with healthy controls.

The analysis examines bacteria at genus and species level. It compares four machine-learning methods: elastic-net logistic regression, linear support vector machine, random forest, and XGBoost.

Main file

run_ml_analysis.py runs the analysis and produces results tables and figures. If your downloaded file is named run_ml_analysis(2).py, rename it to run_ml_analysis.py before running the command below.

Required data

The program requires four tab-separated text files:

OPMD_Genus_relative_abundance_OTU_table_with_meta_data(1).txt
OPMD_Species_relative_abundance_OTU_table_with_meta_data(1).txt
OSCC_Genus_relative_abundance_OTU_table_with_meta_data(1).txt
OSCC_Species_relative_abundance_OTU_table_with_meta_data(1).txt

Keep these filenames unchanged and place the files in a folder called data.

Each row represents one sample. Columns contain sample information, study name, disease status, and bacterial relative abundances. Abundance values should add up to approximately 1 per sample. For example, 20% should be entered as 0.20.

The data must be supplied separately from the code. Each main comparison requires at least three eligible studies, with both cases and healthy controls in each included study.

Install the required packages

With Python installed, run:

python -m pip install numpy pandas scipy scikit-learn xgboost matplotlib seaborn

Run the analysis

Open a terminal in the folder containing the script and run:

python run_ml_analysis.py --input-dir data --output-dir results

The program leaves one complete study out for testing and uses the other studies for training. It also checks clinical-data availability, summarizes bacterial features, and performs statistical checks. The full analysis may take substantial time.

Output files

results/tables/ contains analysis tables and model performance summaries.

results/predictions/ contains predictions for samples in the held-out studies.

results/figures/ contains figures in PNG and PDF format.

results/run_manifest.json contains run settings and selected software versions.

Research use

This code supports research on current disease status. The results require scientific interpretation and do not establish a clinical diagnostic test.
