import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from transformers import AutoTokenizer, AutoModel
import torch
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
import nltk
import requests
from io import StringIO
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# URLs
XANEW_URL = 'https://raw.githubusercontent.com/JULIELab/XANEW/master/Ratings_Warriner_et_al.csv'
EMOBANK_URL = 'https://raw.githubusercontent.com/PramodaKAPS/SongsEmotions/main/emobank.csv'
SPOTIFY_URL = 'https://raw.githubusercontent.com/PramodaKAPS/SongsEmotions/main/spotify_songs.csv'

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained('distilbert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to GPU/CPU

# Download datasets
def download_csv(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download {url}")
    return pd.read_csv(StringIO(response.text))

# Load datasets
xanew_df = download_csv(XANEW_URL)
xanew_df = xanew_df[['Word', 'V.Mean.Sum', 'A.Mean.Sum']].rename(columns={'Word': 'word', 'V.Mean.Sum': 'valence', 'A.Mean.Sum': 'arousal'})
xanew_scaler = MinMaxScaler()
xanew_df[['valence', 'arousal']] = xanew_scaler.fit_transform(xanew_df[['valence', 'arousal']])

sentence_df = download_csv(EMOBANK_URL)
song_df = download_csv(SPOTIFY_URL)

# Normalize EmoBank arousal/valence
emo_scaler = MinMaxScaler()
sentence_df[['V', 'A']] = emo_scaler.fit_transform(sentence_df[['V', 'A']])
sentence_df = sentence_df[sentence_df['split'] == 'train']

# Audio features (Taylor & Francis 2020)
audio_features = ['tempo', 'loudness', 'energy', 'speechiness', 'danceability', 'mode']

# Handle missing values
song_df[audio_features] = song_df[audio_features].fillna(song_df[audio_features].mean())

# Normalize audio features
scaler_audio = MinMaxScaler()
audio_scaled = scaler_audio.fit_transform(song_df[audio_features])
audio_scaled_df = pd.DataFrame(audio_scaled, columns=audio_features)

# Linear regression coefficients (Taylor & Francis 2020, fixed weights)
def calculate_linear_arousal(audio_df):
    weights = {
        'tempo': 0.4,
        'loudness': 0.3,
        'energy': 0.2,
        'speechiness': 0.05,
        'danceability': 0.05
    }
    arousal = sum(weights[f] * audio_df[f] for f in weights)
    arousal = (arousal - arousal.min()) / (arousal.max() - arousal.min() + 1e-10)
    return arousal

def calculate_linear_valence(audio_df):
    weights = {
        'energy': 0.35,
        'mode': 0.25,
        'tempo': 0.15,
        'energy_extra': 0.15,  # Combined RMS/spectral centroid
        'danceability': 0.1
    }
    valence = (weights['energy'] * audio_df['energy'] +
               weights['mode'] * audio_df['mode'] +
               weights['tempo'] * audio_df['tempo'] +
               weights['energy_extra'] * audio_df['energy'] +
               weights['danceability'] * audio_df['danceability'])
    valence = (valence - valence.min()) / (valence.max() - valence.min() + 1e-10)
    return valence

arousal_audio = calculate_linear_arousal(audio_scaled_df)
valence_audio = calculate_linear_valence(audio_scaled_df)

# Text preprocessing
def preprocess_text(text):
    if not isinstance(text, str):
        return [], ''
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [WordNetLemmatizer().lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return tokens, ' '.join(tokens).strip()

# X-ANEW features
def get_xanew_features(tokens, is_lyric=False):
    arousal_scores = []
    valence_scores = []
    weights = []
    for token in tokens:
        if token in xanew_df['word'].values:
            row = xanew_df[xanew_df['word'] == token]
            arousal_scores.append(row['arousal'].values[0])
            valence_scores.append(row['valence'].values[0])
            weight = 2.0 if is_lyric and tokens.count(token) > 1 else 1.0
            weights.append(weight)
    total_weight = sum(weights) if weights else 1.0
    arousal = sum(a * w for a, w in zip(arousal_scores, weights)) / total_weight if arousal_scores else 0.5
    valence = sum(v * w for v, w in zip(valence_scores, weights)) / total_weight if valence_scores else 0.5
    return arousal, valence

# POS tagging
def apply_pos_context(tokens, arousal, valence):
    tagged = pos_tag(tokens)
    for _, (word, tag) in enumerate(tagged):
        if tag.startswith('JJ'):
            arousal *= 1.2
            valence *= 1.2
        elif tag in ['RB', 'RBR', 'RBS']:
            arousal *= 1.1
            valence *= 1.1
        elif tag.startswith('VB') and word in ['kill', 'destroy']:
            valence *= 0.8
        elif word in ['not', 'never', 'dont']:
            valence = 1.0 - valence
    return arousal, valence

# Preprocess texts
sentence_df['tokens'], sentence_df['cleaned_text'] = zip(*sentence_df['text'].apply(preprocess_text))
song_df['tokens'], song_df['cleaned_lyrics'] = zip(*song_df['lyrics'].apply(preprocess_text))

# Get X-ANEW features
sentence_df['xanew_arousal'], sentence_df['xanew_valence'] = zip(*sentence_df['tokens'].apply(lambda x: get_xanew_features(x)))
song_df['xanew_arousal'], song_df['xanew_valence'] = zip(*song_df['tokens'].apply(lambda x: get_xanew_features(x, is_lyric=True)))

# Apply POS context
sentence_df['xanew_arousal'], sentence_df['xanew_valence'] = zip(*sentence_df.apply(
    lambda row: apply_pos_context(row['tokens'], row['xanew_arousal'], row['xanew_valence']), axis=1))
song_df['xanew_arousal'], song_df['xanew_valence'] = zip(*song_df.apply(
    lambda row: apply_pos_context(row['tokens'], row['xanew_arousal'], row['xanew_valence']), axis=1))

# DistilBERT embeddings
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained('distilbert-base-uncased')

def get_bert_embeddings(texts, max_length=512):
    embeddings = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define device inside function (optional)
    for text in texts:
        sentences = sent_tokenize(text) if len(text) > max_length else [text]
        sentence_embeddings = []
        for sentence in sentences:
            inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to device
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # Move back to CPU for numpy
            sentence_embeddings.append(embedding)
        embeddings.append(np.mean(sentence_embeddings, axis=0) if sentence_embeddings else np.zeros(768))
    return np.array(embeddings)

# Extract embeddings
sentence_embeddings = get_bert_embeddings(sentence_df['cleaned_text'])
lyrics_embeddings = get_bert_embeddings(song_df['cleaned_lyrics'])

# Lyrics-based kNN
from sklearn.neighbors import KNeighborsRegressor
X_text_sentence = np.hstack([sentence_embeddings, sentence_df[['xanew_arousal', 'xanew_valence']].values])
y_arousal = sentence_df['A'].values
y_valence = sentence_df['V'].values
knn_text_arousal = KNeighborsRegressor(n_neighbors=30, weights='distance')
knn_text_valence = KNeighborsRegressor(n_neighbors=30, weights='distance')
knn_text_arousal.fit(X_text_sentence, y_arousal)
knn_text_valence.fit(X_text_sentence, y_valence)

# Predict lyrics-based arousal/valence
X_text_lyrics = np.hstack([lyrics_embeddings, song_df[['xanew_arousal', 'xanew_valence']].values])
arousal_text = knn_text_arousal.predict(X_text_lyrics)
valence_text = knn_text_valence.predict(X_text_lyrics)

# Combine predictions
w_text = 0.6
w_audio = 0.4
arousal_final = w_text * arousal_text + w_audio * arousal_audio
valence_final = w_text * valence_text + w_audio * valence_audio

# Save predictions
predictions_df = pd.DataFrame({
    'track_id': song_df['track_id'],
    'track_name': song_df['track_name'],
    'track_artist': song_df['track_artist'],
    'arousal_text': arousal_text,
    'valence_text': valence_text,
    'arousal_audio': arousal_audio,
    'valence_audio': valence_audio,
    'arousal_final': arousal_final,
    'valence_final': valence_final
})
predictions_df.to_csv('song_emotion_predictions_taylor_francis.csv', index=False)
print("Predictions saved to 'song_emotion_predictions_taylor_francis.csv'")

# Validate valence
mse_valence = mean_squared_error(song_df['valence'], valence_audio)
print(f"Valence MSE (linear audio vs. Spotify valence): {mse_valence:.4f}")

# Validate arousal (correlation)
corr, _ = pearsonr(arousal_audio, arousal_text)
print(f"Arousal correlation (audio vs. text): {corr:.4f}")

# Thayer's Plot
def create_thayer_plot(predictions_df, output_file='thayer_plot_taylor_francis.png'):
    sns.set_style("darkgrid")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=predictions_df, x='valence_final', y='arousal_final',
                    hue='valence_final', size='arousal_final',
                    palette='viridis', alpha=0.6, sizes=(20, 200))
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.text(0.25, 0.75, 'Angry/Stressed', fontsize=10, ha='center')
    plt.text(0.25, 0.25, 'Sad/Depressed', fontsize=10, ha='center')
    plt.text(0.75, 0.75, 'Happy/Excited', fontsize=10, ha='center')
    plt.text(0.75, 0.25, 'Calm/Peaceful', fontsize=10, ha='center')
    top_songs = predictions_df.nlargest(5, 'arousal_final')
    for _, row in top_songs.iterrows():
        plt.text(row['valence_final'], row['arousal_final'], row['track_name'],
                 fontsize=8, ha='right', va='bottom')
    plt.xlabel('Valence (Negative to Positive)', fontsize=12)
    plt.ylabel('Arousal (Calm to Excited)', fontsize=12)
    plt.title('Thayer\'s Emotion Plane for Spotify Songs (Taylor & Francis 2020)', fontsize=14)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Thayer's plot saved to '{output_file}'")

create_thayer_plot(predictions_df, 'thayer_plot_taylor_francis.png')

# Quadrant labels
def assign_quadrant(arousal, valence):
    if arousal >= 0.5 and valence >= 0.5:
        return 'Happy/Excited'
    elif arousal >= 0.5 and valence < 0.5:
        return 'Angry/Stressed'
    elif arousal < 0.5 and valence >= 0.5:
        return 'Calm/Peaceful'
    else:
        return 'Sad/Depressed'

predictions_df['quadrant'] = predictions_df.apply(
    lambda row: assign_quadrant(row['arousal_final'], row['valence_final']), axis=1)
predictions_df.to_csv('song_emotion_predictions_with_quadrant_taylor_francis.csv', index=False)
print("Predictions with quadrant labels saved to 'song_emotion_predictions_with_quadrant_taylor_francis.csv'")
