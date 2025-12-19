# Model Decision Visualization with LIME and SHAP

<img width="608" height="210" alt="image" src="https://github.com/user-attachments/assets/1eb69742-d326-48db-a14f-db4829455d6f" />

This repository provides a Python implementation for visualizing and interpreting machine learning model predictions using **LIME** (Local Interpretable Model-agnostic Explanations) and **SHAP** (SHapley Additive exPlanations). The project demonstrates how to generate local explanations for individual predictions and compare the results from both explainability methods.

## Features

- Train and evaluate two classification models: **Logistic Regression** and **Random Forest**
- Generate local explanations for individual predictions using **LIME** and **SHAP**
- Visualize feature importance via bar plots and waterfall charts
- Compare explainer outputs using:
  - Spearman rank correlation of absolute importances
  - Top-k feature overlap
- Interactive web interface built with **Gradio** for real-time explanation visualization
- Works with the **Iris dataset** and can be extended to other datasets like Titanic

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/your-username/visualizing-model-decisions.git
cd visualizing-model-decisions
pip install -r requirements.txt
```
Or install dependencies manually:
```bash
pip install pandas scikit-learn lime shap matplotlib seaborn gradio
```
## Usage

## Jupyter Notebook / Script Execution

Run the main script to train models, generate explanations, and compare results:
```bash
python visualizing_model_decisions_with_lime_or_shap.py
```
This will:
1. Load and preprocess the Iris dataset
2. Train Logistic Regression and Random Forest classifiers
3. Generate LIME and SHAP explanations for a test instance
4. Display visual comparisons and correlation metrics

## Interactive Web Interface

Launch the Gradio web app to interactively explore explanations:
```bash
python app.py
```
Then open your browser to the local URL (usually http://127.0.0.1:7860) to:

- Select test instances via a slider
- View SHAP and LIME explanations side by side
- Compare how different models justify their predictions

## Explanation Methods

## LIME (Local Interpretable Model-agnostic Explanations)

- Creates local surrogate models to explain individual predictions
- Provides interpretable feature importance scores
- Works with any black-box model

## SHAP (SHapley Additive exPlanations)

- Based on cooperative game theory
- Provides consistent and theoretically grounded feature attributions
- Supports model-specific explainers (TreeExplainer, KernelExplainer)

## Results Comparison

The script computes two comparison metrics between LIME and SHAP explanations:

1. **Spearman Rank Correlation:** Measures the correlation between feature importance rankings
2. **Top-k Overlap:** Counts how many of the top k important features are identified by both methods

## Example Output
- SHAP absolute importance bar plots for both models
- LIME explanation lists with feature weights
- Side-by-side comparison visualizations
- Correlation and overlap metrics printed in the console

## Extending to Other Datasets
To use a different dataset (e.g., Titanic):

- Load and preprocess your dataset
- Update feature names and target variable
- Adjust model training as needed
- The explanation pipeline remains the same

## License
MIT License

## Contributing
- Contributions are welcome! Please feel free to submit a Pull Request.
- Fork the repository
- Create your feature branch (git checkout -b feature/AmazingFeature)
- Commit your changes (git commit -m 'Add some AmazingFeature')
- Push to the branch (git push origin feature/AmazingFeature)
- Open a Pull Request

## Acknowledgments
- LIME
- SHAP
- scikit-learn
- Gradio
