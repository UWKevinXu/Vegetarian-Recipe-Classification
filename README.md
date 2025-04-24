# Vegetarian Recipe Classification

This repository presents a machine learning project focused on classifying recipes as vegetarian or non-vegetarian based on ingredient lists and textual descriptions. The project leverages both traditional NLP techniques and transformer-based models to achieve high classification performance.

## Dataset

- The dataset includes a collection of online recipes.
- Each recipe contains:
  - Name
  - List of ingredients
  - Description text
  - Binary label indicating whether the recipe is vegetarian
- The dataset was cleaned and preprocessed prior to modeling.

*Note: The dataset is not included due to licensing restrictions. Please refer to the code for guidance on preprocessing your own recipe data.*

## Context

This project was created to explore the feasibility of using natural language processing for dietary classification. Understanding whether a recipe is vegetarian based solely on its ingredients and description can be useful for:
- Personalized meal recommendation systems
- Recipe filtering for dietary needs
- Health and nutrition-related applications

Key elements of the project:
- TF-IDF for feature extraction in traditional ML pipelines
- Pre-trained transformers (DistilBERT, ELECTRA) for deep contextual understanding
- Evaluation of model performance using classification metrics

## Findings

1. **Model Performance**  
   - DistilBERT achieved the highest accuracy of **87%**, outperforming both ELECTRA and traditional models.

2. **Preprocessing Impact**  
   - Proper cleaning and tokenization significantly improved results, especially when paired with transformer models.

3. **Transformer Advantage**  
   - Pre-trained models captured deeper semantic meaning in recipe descriptions, leading to more reliable predictions.

## Recommendations

1. **Deploy DistilBERT**  
   - Use DistilBERT in production environments for its strong balance between performance and speed.

2. **Data Expansion**  
   - Include recipes from diverse cuisines to improve generalizability and reduce bias.

3. **API Integration**  
   - Wrap the classification model in a REST API using Flask or FastAPI to enable real-time classification in applications.

## Model Comparison Table

| Model               | Feature Extraction | Accuracy |
|---------------------|--------------------|----------|
| Logistic Regression | TF-IDF             | 76%      |
| ELECTRA             | Token Embedding    | 83%      |
| DistilBERT          | Token Embedding    | **87%**  |

## Files

- `preprocessing.ipynb` – Data cleaning and TF-IDF vectorization
- `modeling_distilbert.ipynb` – Fine-tuning and evaluating DistilBERT
- `modeling_electra.ipynb` – Training ELECTRA model
- `train_model.py` – Modular script for model training and evaluation
- `results/metrics_summary.md` – Summary of model performance metrics

## Appendix

Additional materials such as confusion matrices, error analysis, and hyperparameter tuning logs can be found in the `results/` directory.

---

Thank you for exploring this project! Feel free to fork, open issues, or submit pull requests. For questions or collaborations, please contact the repository owner.
