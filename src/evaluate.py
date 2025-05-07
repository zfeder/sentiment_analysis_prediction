import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from preprocessing import preprocess_dataframe
from train import random_search
import matplotlib.pyplot as plt

# Load evaluation data
print("Loading and preprocessing evaluation data...")
df_eval = pd.read_csv('evaluation.csv')
df_eval_preprocessed, _ = preprocess_dataframe(df_eval, ohe_day=None)

# Prepare evaluation data in the same format as training data
X_eval = df_eval_preprocessed[['filtered_text', 'contains_links', 'contains_mentions', 'hour'] + 
                              [col for col in df_eval_preprocessed.columns if col.startswith('day_')]]

y_pred_eval = random_search.best_estimator_.predict(X_eval)

# Prepare the submission file
submission = pd.DataFrame({
    'Id': df_eval['ids'],
    'Predicted': y_pred_eval
})

submission.to_csv('submission.csv', index=False)

print("Submission file created: submission.csv")

# Plot confusion matrix (optional)
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

# Uncomment this line if you have the true labels for the evaluation set
# plot_confusion_matrix(y_eval, y_pred_eval, classes=np.unique(y_eval))
