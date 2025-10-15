import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
from textblob import TextBlob
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

# Make sure required packages are downloaded
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Load CSV file
def load_data(file_path):
    df = pd.read_csv(file_path)
    
    data_clean_comments = df.loc[ df['Comment'].str.lower().str.contains("see attached") == False,:]
    data_clean_comments = data_clean_comments.dropna(axis=1, how="all")

    if 'Comment' not in df.columns:
        raise ValueError("CSV must contain 'Comment' column.")
    return data_clean_comments
def remove_stop_words(df):
    specific_stops = ['proposed',
                  'proposal',
                  'epar10ow20170369',
                  'epa',
                  'pebble',
                  'mine',
                  'docket',
                  'decision',
                  'bristol',
                  'bay',
                  'would']
    stop_words = set(stopwords.words('english') + specific_stops)
    # stop_words = set(stopwords.words('english'))
    all_words = []

    for text in df['Comment'].dropna():
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and word not in stop_words]
    df['Comment'] = df['Comment'].apply(lambda x: " ".join([word for word in word_tokenize(x.lower()) if word.isalpha() and word not in stop_words]))
    print(df['Comment'])
    return df
# Basic information
def basic_info(df):
    print("\n=== Basic Info ===")
    print(df.info())
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    print("\n=== Sample Rows ===")
    print(df.head())

def temporal_analysis(df):
    # Report based on daily trends
    freq="D"
    # Ensure datetime type
    df.loc[:,'Received Date'] = pd.to_datetime(df['Received Date'])
    df = df.dropna(subset=['Received Date'])  # Drop invalid dates

    # Count posts per time period
    time_counts = pd.to_datetime(df['Received Date']).dt.to_period(freq).value_counts().sort_index()
    time_counts = time_counts.to_timestamp()

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(time_counts.index, time_counts.values, marker='o', linestyle='-')
    plt.title(f'Post Frequency Over Time ({freq} resolution)')
    plt.xlabel('Date')
    plt.ylabel('Number of Posts')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Text length analysis
def text_length_analysis(df):
    df.loc[:,'Text Length'] = df['Comment'].astype(str).apply(len)
    print("\n=== Text Length Stats ===")
    print(df['Text Length'].describe())

    plt.figure(figsize=(10, 4))
    sns.histplot(df['Text Length'], bins=30, kde=True)
    plt.title("Distribution of Text Lengths")
    plt.xlabel("Number of Characters")
    plt.ylabel("Document Count")
    plt.tight_layout()
    plt.show()
    
# Conduct Topic Modeling using LDA
def topic_modeling(df):
    
    number_topics = 10 
    texts = df['Comment'].str.split().tolist()
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below = 2)

    # generate corpus as BoW
    corpus = [dictionary.doc2bow(text) for text in texts]

    # train LDA model
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, random_state=4583, chunksize=20, num_topics=number_topics, passes=50, iterations=50)

    print("LDA topics")
    for topic in lda_model.print_topics(num_topics=number_topics, num_words=10):
        print(topic)
    print("Document-Topic Table")
    all_doc_topics = [lda_model.get_document_topics(doc) for doc in corpus]
    df.loc[:,'Topics'] = pd.Series(all_doc_topics)
    
    
# Word frequency analysis
def word_frequency(df, top_n=20):
    
    # all_words = [i for sublist in df['Comment'].tolist() for i in sublist]
    all_words = [i for sublist in df['Comment'].str.split() for i in sublist]
    word_counts = Counter(all_words)
    common_words = word_counts.most_common(top_n)

    print(f"\n=== Top {top_n} Most Common Words ===")
    for word, count in common_words:
        print(f"{word}: {count}")

    # Plotting
    words, counts = zip(*common_words)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(words), y=list(counts))
    plt.title(f"Top {top_n} Most Common Words")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Word cloud generation
def generate_wordcloud(df):
    text_combined = ' '.join(df['Comment'])
    wordcloud = WordCloud(stopwords=stopwords.words('english'),
                          background_color='white',
                          width=800, height=400).generate(text_combined)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# Sentiment analysis using TextBlob
def sentiment_analysis(df):
    def get_sentiment(text):
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return None

    df['Sentiment'] = df['Comment'].astype(str).apply(get_sentiment)

    print("\n=== Sentiment Stats ===")
    print(df['Sentiment'].describe())

    plt.figure(figsize=(10, 4))
    sns.histplot(df['Sentiment'].dropna(), bins=30, kde=True)
    plt.title("Sentiment Polarity Distribution")
    plt.xlabel("Polarity (-1 = Negative, 1 = Positive)")
    plt.ylabel("Document Count")
    plt.tight_layout()
    plt.show()

# Sentiment Analysis using VADER
# (Valence Aware Dictionary and sEntiment Reasoner)
def sentiment_analysis_vader(df):
    analyzer = SentimentIntensityAnalyzer()
    df.loc[:,'Sentiment_Vader'] = df['Comment'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    
    print("\n=== Sentiment Stats ===")
    print(df['Sentiment_Vader'].describe())

    plt.figure(figsize=(10, 4))
    sns.histplot(df['Sentiment_Vader'].dropna(), bins=30, kde=True)
    plt.title("VADER Sentiment Polarity Distribution")
    plt.xlabel("Polarity (-1 = Negative, 1 = Positive)")
    plt.ylabel("Document Count")
    plt.tight_layout()
    plt.show()

def jobs_analysis(df):
    df_subset = df[df['Comment'].str.lower().str.contains("job")]
    df_subset.loc[:,'job_count'] = df_subset['Comment'].str.lower().str.count("job")
    print()
    print("##############")
    print("'Job' Comments")
    print("##############")
    for text in df_subset.loc[:,['Comment', 'job_count']].sort_values(by='job_count', ascending=False).head(10)['Comment'].tolist():
        print(text)
    print("######################")
    print("End of 'Job' comments")
    print("######################")
    

def foreign_analysis(df):
    df_subset = df[df['Comment'].str.lower().str.contains("foreign")]    
    df_subset.loc[:,'foreign_count'] = df_subset['Comment'].str.lower().str.count("foreign")
    print()
    print("##################")
    print("'Foreign' Comments")
    print("##################")
    for text in df_subset.loc[:,['Comment', 'foreign_count']].sort_values(by='foreign_count', ascending=False).head(10)['Comment'].tolist():
        print(text)
    print("#########################")
    print("End of 'Foreign' Comments")
    print("#########################")
    
def administrator_analysis(df):
    df_subset = df[df['Comment'].str.lower().str.contains("administrator")]
    df_subset.loc[:,'admin_count'] = df_subset['Comment'].str.lower().str.count("administrator")
    print()
    print("########################")
    print("'Administrator' Comments")
    print("########################")
    for text in df_subset.loc[:,['Comment', 'admin_count']].sort_values(by='admin_count', ascending=False).head(10)['Comment'].tolist():
        print(text)
    print("###############################")
    print("End of 'Administrator' Comments")
    print("###############################")
    
def verify_extreme_sentiments(df):
    
    # Most positive TextBlob comments
    print("Most Positive TextBlob")
    top_tb = df[['Comment','Sentiment']].sort_values('Sentiment', ascending=False).head(20)
    print(top_tb)
    top_tb.to_csv("Positive_TextBlob.csv")
    # Most negative TextBlob comments
    bottom_tb = df[['Comment','Sentiment']].sort_values('Sentiment', ascending=True).head(20)
    print("Most Negative TextBlob")
    print(bottom_tb)
    bottom_tb.to_csv("Negative_TextBlob.csv")
    
    
    # Most positive VADER comments
    print("Most Positive VADER")
    top_vader = df[['Comment','Sentiment_Vader']].sort_values('Sentiment_Vader', ascending=False).head(20)
    print(top_vader)
    top_vader.to_csv("Positive_VADER.csv")
    # Most negative VADER comments
    print("Most Negative VADER")
    bottom_vader = df[['Comment','Sentiment_Vader']].sort_values('Sentiment_Vader', ascending=True).head(20)
    bottom_vader.to_csv("Negative_VADER.csv")
    print(bottom_vader)
    
# Run all analyses
def run_eda(file_path):
    df = load_data(file_path)
    basic_info(df)
    temporal_analysis(df)
    text_length_analysis(df)
    generate_wordcloud(df)
    sentiment_analysis(df)
    sentiment_analysis_vader(df)
    verify_extreme_sentiments(df)
    jobs_analysis(df)
    foreign_analysis(df)
    administrator_analysis(df)
    df_clean = remove_stop_words(df)
    topic_modeling(df_clean)
    word_frequency(df_clean)
    generate_wordcloud(df_clean)
run_eda("bulk-data-download.csv")

