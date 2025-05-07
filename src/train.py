import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from preprocessing import preprocess_dataframe

# Load and preprocess training data
print("Loading and preprocessing training data...")
df_train = pd.read_csv('development.csv')
df_preprocessed, ohe_day = preprocess_dataframe(df_train)

df_preprocessed = df_preprocessed.drop_duplicates(subset=['ids'], keep='first')

X = df_preprocessed[['filtered_text', 'contains_links', 'contains_mentions', 'hour'] + 
                    [col for col in df_preprocessed.columns if col.startswith('day_')]]
y = df_preprocessed['sentiment']

# Define the model pipeline
pipeline = Pipeline([
    ('features', ColumnTransformer([
        ('text', TfidfVectorizer(), 'filtered_text'),
        ('additional', 'passthrough', ['contains_links', 'contains_mentions', 'hour']),
        ('day', 'passthrough', [col for col in X.columns if col.startswith('day_')])
    ])),
    ('classifier', LinearSVC(class_weight='balanced', max_iter=10000, dual=False))
])

# Define hyperparameter search space
param_distributions = {
    'features__text__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'features__text__max_df': [0.1, 0.2, 0.5],
    'features__text__min_df': [2, 5, 10],
    'features__text__max_features': [5000, 10000, 15000],
    'classifier__C': np.linspace(0.1, 1.0, 20),
    'classifier__penalty' : ['l2']
}

# Set up cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Hyperparameter tuning
print("Running hyperparameter tuning...")
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=50,
    scoring='f1_macro',
    cv=cv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# Train the model
random_search.fit(X_train, y_train)

# Save the best model
print("Best parameters:", random_search.best_params_)
y_pred_train = random_search.best_estimator_.predict(X_train)
train_f1 = f1_score(y_train, y_pred_train, average='macro')
print(f"F1 Score on training set: {train_f1:.4f}")

y_pred_test = random_search.best_estimator_.predict(X_test)
test_f1 = f1_score(y_test, y_pred_test, average='macro')
print(f"F1 Score on test set: {test_f1:.4f}")
print(f"Difference between training and test F1 Score: {train_f1 - test_f1:.4f}")
