import pandas as pd
import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("Loading Pokemon entries...")
df = pd.read_csv('data/pokemon_entries.csv')

# --- NEW LOADING SECTION ---
print("Loading YOUR custom labeled entries...")
# This points to the file you just spent all that time making!
df = pd.read_csv('data/labeled_pokemon_entries.csv')

# We only use 'entry' and the 'nature' YOU assigned
X = df['entry']
y = df['nature']
# ---------------------------

NATURE_DESC = {
    'Brave':   'Strong and fearless — charges into battle without hesitation.',
    'Calm':    'Composed and steady — adapts fluidly to any situation.',
    'Gentle':  'Kind and nurturing — cares deeply for its environment.',
    'Hasty':   'Lightning quick — always moving faster than others can follow.',
    'Modest':  'Highly intelligent — relies on mental power over physical strength.',
    'Quiet':   'Cool and reserved — prefers stillness and careful observation.',
    'Adamant': 'Fierce and dominant — raw power defines its every action.',
    'Lonely':  'Solitary and secretive — operates best in the shadows alone.',
    'Timid':   'Gentle but enchanting — wins battles through charm and wit.',
    'Hardy':   'Balanced and dependable — a reliable all-rounder in any role.',
    'Jolly':   'Energetic and spirited — loves the thrill of competition.',
    'Naive':   'Free and adventurous — soars above challenges with ease.',
    'Bold':    'Resilient and daring — takes hits without flinching.',
    'Relaxed': 'Grounded and patient — endures anything with steady calm.',
    'Impish':  'Tough and stubborn — built to withstand whatever comes.',
    'Careful': 'Precise and methodical — thinks before every move.',
    'Sassy':   'Mysterious and cunning — unpredictable and hard to read.',
    'Serious': 'Disciplined and structured — follows a strict internal code.',
    'Bashful': 'Shy and reserved — avoids attention and keeps emotions hidden.',
    'Docile': 'Mild-mannered and obedient — goes with the flow without resistance.',
    'Lax': 'Carefree and relaxed — doesnt worry much about consequences.',
    'Mild': 'Gentle but capable — balances kindness with quiet strength.',
    'Naughty': 'Mischievous and rebellious — enjoys causing trouble or bending rules.',
    'Quirky': 'Eccentric and unpredictable — behaves in unusual and unexpected ways.',
    'Rash': 'Impulsive and reckless acts on instinct before considering the dangers.'
}

# Augment entries with type keywords to improve prediction per type
TYPE_KEYWORDS = {
    'fire':     'burns flame heat scorching fire blaze inferno temperature hot',
    'water':    'swims ocean river lake water rain sea fluid gentle flow',
    'grass':    'forest jungle leaves seeds plants trees nature grove growing',
    'electric': 'electricity speed lightning bolt fast charge current voltage',
    'psychic':  'psychic mind intelligence power brain telepathy mental mysterious',
    'ice':      'ice cold frozen snow blizzard freeze temperature cool glacier',
    'dragon':   'powerful ancient strong legendary fierce dragon wing dominate',
    'dark':     'dark shadow night lurk evil sinister cunning deceive alone',
    'fairy':    'cute charm beautiful enchant glitter magical fairy dance gentle',
    'normal':   'ordinary common balanced gentle normal friendly basic everyday',
    'fighting': 'fight punch kick strong battle martial combat power train fist',
    'flying':   'fly soar wing sky air free swift glide breeze aerial',
    'poison':   'poison toxic venom sludge chemical gas fume spray acid liquid',
    'ground':   'ground earth sand desert burrow dig cave underground stable',
    'rock':     'rock stone hard sturdy mineral cave mountain solid defensive',
    'bug':      'insect hive swarm spin web antenna metamorphosis careful tiny',
    'ghost':    'ghost spirit soul disappear invisible eerie haunt mysterious dark',
    'steel':    'steel iron metal hard armor shield construct mechanical strong',
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words='english',
        sublinear_tf=True
    )),
    ('clf', LogisticRegression(
        max_iter=2000,
        random_state=42,
        C=2.0,
        class_weight='balanced'
    ))
])

print("Training model...")
pipeline.fit(X_train, y_train)

print("\nModel evaluation on test set:")
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

os.makedirs('model', exist_ok=True)
with open('model/nature_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

with open('model/nature_desc.pkl', 'wb') as f:
    pickle.dump(NATURE_DESC, f)

print("Model saved to model/nature_model.pkl")
print("Done!")
