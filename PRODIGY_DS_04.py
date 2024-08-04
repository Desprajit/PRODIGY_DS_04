import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk


nltk.download('vader_lexicon')


df = pd.read_csv(r"C:\Users\Despr\Downloads\social_media_data.csv")


sia = SentimentIntensityAnalyzer()


def analyze_sentiment(text):
    sentiment = sia.polarity_scores(text)
    return sentiment


df['sentiment'] = df['text'].apply(analyze_sentiment)


df['compound'] = df['sentiment'].apply(lambda x: x['compound'])


df['sentiment_class'] = df['compound'].apply(
    lambda x: 'positive' if x >= 0.05 else 'negative' if x <= -0.05 else 'neutral'
)


plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='date', y='compound', marker='o')
plt.title('Sentiment Analysis Over Time')
plt.xlabel('Date')
plt.ylabel('Compound Sentiment Score')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment_class', data=df)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Class')
plt.ylabel('Count')
plt.show()


