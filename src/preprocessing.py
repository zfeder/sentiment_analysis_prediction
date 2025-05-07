import re
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from nltk.tokenize import TweetTokenizer


def preprocess_dataframe(df, ohe_day=None):
    # Remove column "flag" from the dataset
    df_preprocessed = df.drop(columns=['flag'])
    
    # Dictionary for replacing special characters
    substitutions = {
        '&quot;': '"',
        '&amp;': '&',
        '&gt;': '>',
        '&lt;': '<',
    }
    
    # Emoticons to be replaced with text descriptions
    EMOTICONS_EMO = {
        ':‑)': 'Happy face or smiley',
        ':)': 'Happy face or smiley',
        ':))': 'Very Happy face or smiley',
        ':)))': 'Very very Happy face or smiley',
        ':D': 'Laughing, big grin or laugh with glasses',
        ':(': 'Frown, sad, andry or pouting',
        ":'‑(": 'Crying',
        ":'(": 'Crying',
        ':O': 'Surprise',
        ';)': 'Wink or smirk',
        ':P': 'Tongue sticking out, cheeky, playful or blowing a raspberry',
    }
    
    # Apply substitutions
    for abbr, full_form in substitutions.items():
        df_preprocessed['text'] = df_preprocessed['text'].str.replace(abbr, full_form, regex=False)

    # Apply emoticon substitutions
    for emoticon, description in EMOTICONS_EMO.items():
        df_preprocessed['text'] = df_preprocessed['text'].str.replace(emoticon, description, regex=False)
    
    # Removing links
    link_regex = r'(?:www\.|https?://)\S+'
    df_preprocessed['text'] = df_preprocessed['text'].str.replace(link_regex, '', regex=True)
    
    # Removing hashtags
    df_preprocessed['text'] = df_preprocessed['text'].str.replace(r'#\w+', '', regex=True)
    
    # Removing mentions
    df_preprocessed['text'] = df_preprocessed['text'].str.replace(r'@\w+', '', regex=True)
    
    # Tokenization
    tokenizer = TweetTokenizer()
    df_preprocessed['filtered_text'] = df_preprocessed['text'].apply(lambda tweet: " ".join([
        re.sub(r'(.)\1{2,}', r'\1\1', word.strip().lower())
        for word in tokenizer.tokenize(tweet)
        if (word.isalnum() or "'" in word) and len(word) > 1
    ]))

    # Add user information
    df_preprocessed['filtered_text'] += " " + df['user']
    
    # Parsing day and hour
    df_preprocessed['day'] = df_preprocessed['date'].str.split().str[0]
    df_preprocessed['hour'] = df_preprocessed['date'].str.split().str[3].str.split(':').str[0].astype(int)
    
    # Encoding days with OneHotEncoder
    if ohe_day is None:
        ohe_day = OneHotEncoder(sparse_output=False, dtype=int)
        day_encoded = ohe_day.fit_transform(df_preprocessed[['day']])
    else:
        day_encoded = ohe_day.transform(df_preprocessed[['day']])
    
    day_columns = [f'day_{i}' for i in range(day_encoded.shape[1])]
    
    df_preprocessed = df_preprocessed.drop(columns=['date', 'day'])
    df_preprocessed = pd.concat([df_preprocessed, pd.DataFrame(day_encoded, columns=day_columns)], axis=1)
    
    return df_preprocessed, ohe_day
