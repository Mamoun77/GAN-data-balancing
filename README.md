Here is the comprehensive README.md content, ready for you to copy and paste.

# GAN-Based Data Balancing for Imbalanced MNIST Classification

## 📌 Project Overview
This project addresses the challenge of **class imbalance** in machine learning datasets. Using the **MNIST** dataset as a case study, we artificially created a severe imbalance by reducing "Class 0" to just ~300 samples (vs ~6,000 for other classes). 

The core objective is to evaluate the effectiveness of three different **Generative Adversarial Network (GAN)** architectures—Vanilla GAN, DCGAN, and CGAN—in generating high-quality synthetic minority class samples to re-balance the dataset and improve classification performance.

## 🚀 Key Features
*   **Imbalance Simulation**: Artificially reduced MNIST Class 0 to simulate a real-world minority class scenario (1:20 ratio).
*   **Three GAN Architectures**:
    *   **Vanilla GAN**: Standard fully connected generator/discriminator.
    *   **DCGAN**: Deep Convolutional GAN for better image stability.
    *   **CGAN**: Conditional GAN trained on the full dataset to leverage shared features.
*   **Data Augmentation**: Generated 5,700 synthetic images per model to restore balance.
*   **Performance Evaluation**: Trained CNN classifiers on four datasets (Original vs. 3 Augmented) and compared metrics like Accuracy, F1-Score, and Recall.

---
A sample of the generated images for each GAN variation:

<img width="870" height="302" alt="image" src="https://github.com/user-attachments/assets/31f179e1-dfe9-497c-bfeb-47a480ffd48e" />



## 📂 Repository Structure
```text
.
├── classifiers_results/        # Classifier models, training logs, confusion matrices, and plots
├── data/                       # Directory for MNIST dataset
├── gan_saved_models/           # Checkpoints for trained GAN models (Vanilla, DCGAN, CGAN)
├── generated_images_samples/   # Progress images generated during GAN training
├── synthetic_data/             # Final synthetic datasets generated for augmentation
├── classifier.ipynb            # Notebook: Data loading, augmentation, classifier training & eval
├── gan_train.ipynb             # Notebook: GAN model definitions and training loops
├── class_0_reduced.pt          # The reduced minority class subset used for training
└── README.md                   # Project documentation
🛠️ Installation & Requirements
Ensure you have Python 3.x installed. Install the necessary dependencies:

pip install torch torchvision matplotlib numpy scikit-learn tqdm
Note: A GPU (CUDA) is highly recommended for training the GANs and Classifiers efficiently.

💻 Usage
1. Train GANs & Generate Data
Open gan_train.ipynb. This notebook covers:

Loading the reduced minority class data.
Defining Vanilla GAN, DCGAN, and CGAN architectures.
Training the models (Vanilla/DCGAN on minority only, CGAN on full data).
Evaluating generation quality using FID Score.
Generating 5,700 synthetic images into the synthetic_data/ directory.
2. Train Classifiers & Evaluate
Open classifier.ipynb. This notebook covers:

Loading the original imbalanced MNIST data.
Loading the synthetic data generated in the previous step.
Creating four training sets (Original, +Vanilla, +DCGAN, +CGAN).
Training a CNN classifier for each scenario.
Comparing performance using Confusion Matrices and Classification Reports.
📊 Results Summary
We found that data augmentation using GANs consistently improved classifier performance on the minority class.

Scenario	Test Accuracy	Class 0 Recall	Class 0 F1-Score
Original (Baseline)	98.59%	97.04%	0.9835
Vanilla GAN	98.90%	97.45%	0.9866
DCGAN	98.77%	97.45%	0.9861
CGAN	98.71%	97.24%	0.9860

Conclusion: The Vanilla GAN augmentation provided the best overall improvement for this specific dataset, demonstrating that even simple generative models can effectively mitigate class imbalance bias.
