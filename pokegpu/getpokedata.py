import requests
import pandas as pd
import time
import os

print("Fetching Pokemon entries from PokeAPI (Gen 1-3)...")

VERSIONS = ['emerald', 'ruby', 'sapphire', 'firered', 'leafgreen',
            'heartgold', 'soulsilver', 'diamond', 'pearl', 'platinum']

def get_best_entry(entries):
    for version in VERSIONS:
        for e in entries:
            if e['language']['name'] == 'en' and e['version']['name'] == version:
                text = e['flavor_text']
                text = text.replace('\f', ' ').replace('\r', ' ').replace('\n', ' ')
                text = ' '.join(text.split())
                return text
    for e in entries:
        if e['language']['name'] == 'en':
            text = e['flavor_text']
            text = text.replace('\f', ' ').replace('\r', ' ').replace('\n', ' ')
            text = ' '.join(text.split())
            return text
    return None

results = []

for pokemon_id in range(1, 387):
    try:
        res = requests.get(f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}", timeout=10)
        data = res.json()

        name = data['name']
        types = [t['type']['name'] for t in data['types']]
        primary_type = types[0]

        spec_res = requests.get(data['species']['url'], timeout=10)
        spec = spec_res.json()

        entry = get_best_entry(spec['flavor_text_entries'])
        if not entry:
            continue

        results.append({
            'id': pokemon_id,
            'name': name,
            'type': primary_type,
            'entry': entry
        })

        print(f"  #{pokemon_id} {name.capitalize()} ({primary_type}) - OK")
        time.sleep(0.3)

    except Exception as ex:
        print(f"  #{pokemon_id} - ERROR: {ex}")
        time.sleep(1)

os.makedirs('data', exist_ok=True)
df = pd.DataFrame(results)
df.to_csv('data/pokemon_entries.csv', index=False)
print(f"\nDone! Saved {len(results)} entries to data/pokemon_entries.csv")
