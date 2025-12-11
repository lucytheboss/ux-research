import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Apple Music Analysis Dashboard",
    page_icon="🎧",
    layout="wide"
)

# ===============================
# Load Dataset
# ===============================

@st.cache_data
def load_data():
    df = pd.read_csv("data/apple_music_10000.csv")  # ← UPDATE filename if different

    # Re-run essential cleaning steps (same as your notebook)
    df['releaseDate'] = pd.to_datetime(df['releaseDate'], errors='coerce')
    df['primaryGenreName'] = df['primaryGenreName'].str.lower().str.strip()

    df['contentAdvisoryRating'] = df['contentAdvisoryRating'].fillna('clean')
    df['collectionPrice'] = df['collectionPrice'].fillna(df['collectionPrice'].median())
    df['trackPrice'] = df['trackPrice'].fillna(df['trackPrice'].median())
    df['isStreamable'] = df['isStreamable'].fillna("true")
    df['isStreamable'] = df['isStreamable'].map({'true': 1, 'false': 0})

    df['trackTimeSec'] = df['trackTimeMillis'] / 1000
    df['trackTimeMin'] = df['trackTimeSec'] / 60
    df['releaseYear'] = df['releaseDate'].dt.year
    df['releaseMonth'] = df['releaseDate'].dt.month

    df['isExplicit'] = df['trackExplicitness'].apply(lambda x: 1 if x == "explicit" else 0)

    return df

df = load_data()

# ===============================
# Sidebar Navigation
# ===============================

st.sidebar.title("🎧 Apple Music Dashboard")
page = st.sidebar.radio(
    "Go to:",
    ["Overview", "Genres", "Artists", "Trends", "Explicit Content", "Albums"]
)

# ===============================
# PAGE 1 — OVERVIEW
# ===============================

if page == "Overview":
    st.title("📀 Apple Music 10,000 Track Dataset — Overview")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe(include='all'))

    st.subheader("Missing Values")
    st.write(df.isna().sum())

# ===============================
# PAGE 2 — GENRES
# ===============================

elif page == "Genres":
    st.title("🎼 Genre Analysis")

    # Genre selection filter
    genres = df['primaryGenreName'].unique()
    selected_genre = st.sidebar.selectbox("Filter by Genre", sorted(genres))

    filtered = df[df['primaryGenreName'] == selected_genre]

    st.write(f"### Tracks in Genre: **{selected_genre}** ({len(filtered)} tracks)")
    st.dataframe(filtered[['trackName', 'artistName', 'releaseYear', 'trackTimeMin']])

    # Plot — Genre Distribution
    st.subheader("Top Genres Distribution")

    fig, ax = plt.subplots(figsize=(10, 4))
    df['primaryGenreName'].value_counts().head(15).plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title("Top 15 Genres")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Plot — Duration by Genre
    st.subheader("Track Duration by Genre")

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x='primaryGenreName', y='trackTimeMin', ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
    ax2.set_title("Duration (minutes) by Genre")
    st.pyplot(fig2)

# ===============================
# PAGE 3 — ARTISTS
# ===============================

elif page == "Artists":
    st.title("🎤 Artist Analysis")

    # Artist selector
    artists = df['artistName'].value_counts().head(100).index
    selected_artist = st.sidebar.selectbox("Choose an Artist", artists)

    artist_df = df[df['artistName'] == selected_artist]

    st.write(f"### Tracks by {selected_artist}")
    st.dataframe(artist_df[['trackName', 'collectionName', 'releaseYear', 'trackTimeMin']])

    # Top artists plot
    st.subheader("Top 20 Most Frequent Artists")

    fig, ax = plt.subplots(figsize=(10, 4))
    df['artistName'].value_counts().head(20).plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title("Top Artists")
    st.pyplot(fig)

# ===============================
# PAGE 4 — TRENDS
# ===============================

elif page == "Trends":
    st.title("📈 Time & Seasonal Trends")

    # Release Year Trend
    st.subheader("Tracks Released Per Year")

    fig, ax = plt.subplots(figsize=(10, 4))
    df['releaseYear'].value_counts().sort_index().plot(kind='line', ax=ax)
    ax.set_title("Release Year Trend")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Release Month Trend
    st.subheader("Tracks Released Per Month")

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    df['releaseMonth'].value_counts().sort_index().plot(kind='bar', ax=ax2, color='lightgreen')
    ax2.set_title("Release Month Distribution")
    st.pyplot(fig2)

    # Heatmap: Genre x Month
    st.subheader("Genre Release Frequency by Month (Heatmap)")
    genre_month = pd.crosstab(df['primaryGenreName'], df['releaseMonth'])

    fig3, ax3 = plt.subplots(figsize=(12, 10))
    sns.heatmap(genre_month, cmap='YlGnBu', ax=ax3)
    ax3.set_title("Heatmap: Genre vs Release Month")
    st.pyplot(fig3)

# ===============================
# PAGE 5 — EXPLICIT CONTENT
# ===============================

elif page == "Explicit Content":
    st.title("🔞 Explicit Content Analysis")

    # Explicit ratio by genre
    st.subheader("Explicit Ratio by Genre")

    explicit_by_genre = df.groupby('primaryGenreName')['isExplicit'].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    explicit_by_genre.head(20).plot(kind='bar', color='tomato', ax=ax)
    ax.set_title("Explicit Content Ratio (Top 20 Genres)")
    ax.set_ylabel("Percentage Explicit")
    st.pyplot(fig)

    # Pie chart
    st.subheader("Overall Explicit vs Non-Explicit")

    explicit_counts = df['isExplicit'].value_counts()
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.pie(explicit_counts, labels=["Non-Explicit", "Explicit"],
            autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    ax2.set_title("Explicit Content Breakdown")
    st.pyplot(fig2)

# ===============================
# PAGE 6 — ALBUMS
# ===============================

elif page == "Albums":
    st.title("💽 Album Structure Analysis")

    # Tracks per album
    st.subheader("Distribution of Tracks Per Album")

    album_track_counts = df.groupby('collectionId')['trackCount'].max()

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(album_track_counts, bins=30, kde=True, ax=ax)
    ax.set_title("Tracks Per Album Distribution")
    st.pyplot(fig)

    # Show albums of top artist
    st.subheader("Explore Albums by Artist")

    artists = df['artistName'].value_counts().index
    selected_artist = st.sidebar.selectbox("Choose Artist for Album View", artists)

    adf = df[df['artistName'] == selected_artist][['collectionName', 'trackName', 'trackNumber']]
    st.dataframe(adf.sort_values(['collectionName', 'trackNumber']))
