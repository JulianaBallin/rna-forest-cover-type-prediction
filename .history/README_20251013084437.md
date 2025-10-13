# 🌳 Forest Cover Type Prediction: Comparing Neural Networks and Discriminant Analysis

## 📋 About the Project

This repository contains the development and comparison of machine learning models designed to predict forest cover types using cartographic variables.

The activity is based on the classic study: **"Comparative accuracies of artificial neural networks and discriminant analysis in predicting forest cover types from cartographic variables"** (Blackard and Dean, 1999).

### 🎯 Activity Objective

The main goal of this activity is to replicate and expand upon the findings of the 1999 study, which compared the accuracy of two distinct methods for classifying tree types in a given area:

1.  **Artificial Neural Networks (ANN):** A non-linear machine learning model.
2.  **Discriminant Analysis (DA):** A traditional statistical model (linear and/or quadratic).

We will use the **Forest Cover Type Dataset**, which contains cartographic and environmental data (such as elevation, slope, hillshade, and distances to hydrology and roadways) to classify each observation into one of seven forest cover types.

### 💡 Rationale and Scope

The original study by Blackard and Dean concluded that the Artificial Neural Network model outperformed Discriminant Analysis (Gaussian Discriminant Analysis) in terms of prediction accuracy.

In this project, we aim to:

* Implement and evaluate **Artificial Neural Network** and **Discriminant Analysis (LDA/QDA)** models.
* Analyze and compare the performance metrics of each technique (such as accuracy, precision, recall, and F1-score).
* Conduct exploratory data analysis and necessary data preprocessing.
* (Optional): Explore modern Machine Learning models (like Random Forest or XGBoost) to contextualize the improvement in accuracy over the years.

---

## 💻 Technologies Used

* **Python**
* **Jupyter Notebook/Google Colab**
* **Pandas & NumPy** (Data Manipulation)
* **Scikit-learn** (Implementation of Discriminant Analysis and other classifications)
* **TensorFlow/Keras or PyTorch** (Implementation of the Artificial Neural Network)
* **Matplotlib & Seaborn** (Data Visualization)

---

## ⚙️ Repository Structure

* `notebooks/`: Contains the Jupyter notebooks with the analysis, modeling, and comparison code.
* `requirements.txt`: Lists all Python dependencies for environment setup (e.g., pandas, scikit-learn, tensorflow, matplotlib).
* `README.md`: This file.

---

## 👥 Team Members

- **Juliana Ballin Lima**  
    Registration: 2315310011   
    [GitHub Profile](https://github.com/JulianaBallin)

- **Marcelo Heitor de Almeida Lira**  
    Registration: 2315310043  
    [GitHub Profile](https://github.com/Marcelo-Heitor-de-Almeida-Lira)

- **Lucas Maciel Gomes**  
    Registration: 2315310014  
    [GitHub Profile](https://github.com/lucassmaciel)

- **Ryan da Silva Marinho**  
    Registration: 2315310047  
    [GitHub Profile](https://github.com/RyanDaSilvaMarinho)

- **Vitória Gabrielle Kinshasa Silva de Almeida**  
    Registration: 2415280044  
    [GitHub Profile](httos://github.com/VitoriaKinshasa)

---


## 🔗 Useful Links

- [NumPy Documentation](https://numpy.org/doc/stable/)  
- [Matplotlib Scatter Plots](https://matplotlib.org/3.3.0/api/_as_gen/matplotlib.pyplot.scatter.html)  
- [Python Data Science Handbook – Scatter Plots](https://jakevdp.github.io/PythonDataScienceHandbook/04.02-simple-scatter-plots.html)  
- [Google Colab](http://colab.research.google.com/)  

## Installation and Setup

This project uses "uv" for dependency management. You need to install it on your computer:

### UV Installation

**Windows:**
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
``` 

**MacOS | Linux:**
```
curl -LsSf https://astral.sh/uv/install.sh | sh
``` 

### Environment Setup

After installing UV, run the command below in the project root:
```
uv sync
``` 

This command will:
- Create the virtual environment (.venv)
- Install all necessary dependencies

If the required Python version is not installed with 'uv sync', use:
```
uv python install
``` 
This command will:
- Install the correct Python version for the project
---


## 📚 Primary Reference

* Blackard, J. A., & Dean, D. J. (1999). **Comparative accuracies of artificial neural networks and discriminant analysis in predicting forest cover types from cartographic variables.** *Computers and Electronics in Agriculture*, 24(3), 131–151.<p>
* **Dataset:** UCI Machine Learning Repository - Forest Cover Type Data.