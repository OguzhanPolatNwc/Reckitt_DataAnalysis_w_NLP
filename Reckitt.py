import re
import numpy as np
import pandas as pd

# loading data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data.csv')

# Check values of language column
df['language'].nunique()
df['language'].value_counts()
df['proj_id'].nunique()
df['file_id'].nunique()

# Delete id files
for col in ["proj_id", "file_id"]:
    del df[col]

df.head()

# Drop na values
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

df.isna().sum()

X, y = df.file_body, df.language
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model params
token_pattern = r"""(\b[A-Za-z_]\w*\b|[!\#\$%\&\*\+:\-\./<=>\?@\\\^_\|\~]+|[ \t\(\),;\{\}\[\]`"'])"""


def preprocess(x):
    return pd.Series(x).replace(r'\b([A-Za-z])\1+\b', '', regex=True)\
        .replace(r'\b[A-Za-z]\b', '', regex=True)

# Pipe steps
transformer = FunctionTransformer(preprocess)
vectorizer = TfidfVectorizer(token_pattern=token_pattern, max_features=3000)
clf = RandomForestClassifier(n_jobs=4)

pipe_RF = Pipeline([
    ('preprocessing', transformer),
    ('vectorizer', vectorizer),
    ('clf', clf)]
)

# Setting best params
best_params = {
    'clf__criterion': 'gini',
    'clf__max_features': 'sqrt',
    'clf__min_samples_split': 3,
    'clf__n_estimators': 300
}

pipe_RF.set_params(**best_params)

# Fitting
pipe_RF.fit(X_train, y_train)

a = pipe_RF.predict(X_test)
a.shape

# Evaluation
print(f'Accuracy: {pipe_RF.score(X_test, y_test)}')

plt.figure(figsize=(20,20))
sns.heatmap(confusion_matrix(y_test,a), annot=True)
plt.show()
