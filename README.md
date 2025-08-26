# 🧠 CALAMITI  
_Image Harmonization Using Deep Learning Models_

---

## 📌 Overview  
**CALAMITI** (Cross-modAL AMI Transfer for Image harmonization) is a deep learning framework designed to address the variability in medical imaging datasets.  
Different MRI scanners, acquisition protocols, and reconstruction pipelines often produce heterogeneous data that hinders reliable downstream analysis.  

This project implements and extends **unsupervised harmonization** techniques to standardize medical images, enabling:  
- ✅ Better comparability across datasets  
- ✅ Improved robustness for ML/DL downstream tasks  
- ✅ Preservation of anatomical & diagnostic information  

---

## 🎯 Objectives  
- Develop a deep learning pipeline to harmonize **multi-site MRI data**.  
- Implement both **2D & 3D encoding/decoding architectures**.  
- Ensure **structural consistency** between input and harmonized outputs.  
- Benchmark harmonized outputs against ground truth using quantitative metrics.  

---

## 🏗️ Project Structure  

CALAMITI/
│── code/ # Core training & model scripts
│ ├── modules/ # Model building blocks
│ │ ├── dataset.py
│ │ ├── fusion.py
│ │ ├── model.py
│ │ ├── network.py
│ │ └── utils.py
│ │
│ ├── scripts/ # Helper shell scripts
│ │ ├── encode_2d_oas-01-t1.sh
│ │ ├── encode_2d_oas-04-t1.sh
│ │ ├── decode_2d_oas-01-t1-to-oas-04-t1.sh
│ │ ├── decode_2d_oas-04-t1-to-oas-01-t1.sh
│ │ └── train_harmonization_sample_code.sh
│ │
│ ├── combine_images.py # Merge encoded-decoded outputs
│ ├── decode_3d.py # Decoding for 3D MRI scans
│ ├── encode_3d.py # Encoding for 3D MRI scans
│ ├── train_fusion.py # Train fusion models
│ ├── train_harmonization.py# Train harmonization pipeline
│ └── requirements.txt # Dependencies
│
│── decode/ # Decoding outputs (MRI reconstruction)
│── encode/ # Encoded files for experiments
│── encoded/ # Pre-encoded sample MRI slices
│── CALAMITI_NNK_clean.ipynb # Main Colab Notebook (entry point)
│── requirements.txt # Root-level requirements
│── README.md # Project documentation
│── LICENSE # License file


---

## ⚙️ Installation  

Clone the repository:  
```bash
git clone https://github.com/nanda-81/CALAMITI.git
cd CALAMITI

Install dependencies:
pip install -r requirements.txt
For Colab users, simply open CALAMITI_NNK_clean.ipynb and run the cells in sequence.
🚀 Usage
🔹 Training

Train the harmonization model:

python code/train_harmonization.py


Train the fusion model:

python code/train_fusion.py

🔹 Encoding & Decoding
# Encode MRI slices
sh code/scripts/encode_2d_oas-01-t1.sh  

# Decode MRI slices
sh code/scripts/decode_2d_oas-01-t1-to-oas-04-t1.sh

🔹 Notebook Workflow

Upload dataset into encode/ or decode/

Run preprocessing & harmonization

Visualize outputs directly inside Colab

📊 Results

The harmonized images preserve anatomical fidelity while minimizing scanner-induced variability.
Evaluation metrics demonstrate improved consistency across domains.

Structural Similarity Index (SSIM): ↑

Peak Signal-to-Noise Ratio (PSNR): ↑

Visual Assessment: Sharper edges, reduced artifacts

📌 Figures (add these images under /assets/ or /results/ and reference them here):

assets/architecture.png → Model Architecture

assets/flowchart.png → Training Workflow

results/sample_before_after.png → Input vs Harmonized Output

results/metrics_plot.png → Quantitative Evaluation

🛠️ Tech Stack

Python 3.8+

PyTorch – Deep learning backbone

NumPy, SciPy, scikit-image – Pre/post-processing

NiBabel – Neuroimaging data handling

Matplotlib / Seaborn – Visualizations

Google Colab / Jupyter – Interactive experimentation

📂 Dataset

This project was tested on publicly available MRI datasets (OASIS / ADNI).
Due to licensing, raw datasets are not included here.

Download datasets into /data/ (local) or Google Drive (for Colab).

Update paths in training scripts or mount Drive inside Colab.

🤝 Contribution

We welcome contributions that improve reproducibility, extend experiments, or refine harmonization pipelines.

Steps to contribute:

Fork the repo

Create a new branch (feature-newidea)

Commit your changes

Submit a Pull Request 🚀

📜 License

This project is licensed under the MIT License – see LICENSE
 for details.

✨ Acknowledgements

Original methodology inspired by NeuroImage 2021: CALAMITI framework.

Special thanks to the open-source community and dataset providers.

🔮 Future Work

Scaling to multi-modal MRIs (T1, T2, FLAIR)

Integration with clinical pipelines

Lightweight inference models for real-time harmonization

👨‍💻 Authors

Nanda @nanda-81

Collaborators & Research Mentors
