#dataset downloaded from kaggle and uplaoded in colab for analysis 
#basic information about dataset
import pandas as pd
df=pd.read_csv('data_amazon.csv')
print(df.head())
print(df.describe())

#1- data preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df['Review']=df['Review'].astype(str)
if 'Review' in df.columns:
    print("The 'Review' column exists.")
else:
    print("The 'Review' column is missing.")

def clean_text(text):
    text=text.lower()  
    text=re.sub(r'\d+', '', text)  
    text=re.sub(r'[^\w\s]', '', text)  
    text=re.sub(r'\s+', ' ', text)  
    return text

sample_text=df['Review'].iloc[0]
print("Original Sample Text:",sample_text)
print("Cleaned Sample Text:",clean_text(sample_text))
df['Cleaned_Review'] = df['Review'].apply(clean_text)
print("Columns in DataFrame after cleaning:", df.columns)
print(df[['Review','Cleaned_Review']].head())
vectorizer=CountVectorizer(stop_words='english')
X=vectorizer.fit_transform(df['Cleaned_Review'])
df['Sentiment']=df['Cons_rating'].apply(lambda x: 1 if x >= 4 else 0)
y=df['Sentiment']
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=42)

#2- Visualization
plt.figure(figsize=(12, 8))
sns.countplot(data=subset_data,x='Cons_rating',palette='viridis',order=sorted(subset_data['Cons_rating'].unique()))
plt.title('Distribution of Cons_rating', fontsize=16)
plt.xlabel('Cons_rating', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

plt.figure(figsize=(12, 8))
sns.countplot(data=subset_data, x='Sentiment', palette='viridis', order=[0, 1])
plt.title('Distribution of Sentiments', fontsize=16)
plt.xlabel('Sentiment', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

#3- Model building
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

cm=confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
print('Classification Report:\n', report)

