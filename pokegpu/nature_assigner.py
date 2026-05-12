import pandas as pd
import os

# This dictionary helps the computer "guess" the nature so you don't have to!
KEYWORD_HINTS = {
    'Brave': ['brave', 'fearless', 'strong', 'bold', 'courage'],
    'Hasty': ['fast', 'quick', 'speed', 'swift', 'velocity'],
    'Calm': ['calm', 'steady', 'peaceful', 'composed', 'patient'],
    'Quiet': ['quiet', 'silent', 'reserved', 'still'],
    'Timid': ['timid', 'shy', 'scared', 'wary', 'flee'],
    'Jolly': ['happy', 'energetic', 'playful', 'cheerful'],
    'Modest': ['intelligent', 'smart', 'wise', 'mind']
}

def suggest_nature(entry):
    for nature, keywords in KEYWORD_HINTS.items():
        if any(word in entry.lower() for word in keywords):
            return nature
    return None

def run_semi_labeler():
    input_file = 'data/pokemon_entries.csv'
    output_file = 'data/labeled_pokemon_entries.csv'
    
    if not os.path.exists(input_file):
        print("Error: Run fetch_entries.py first!")
        return
        
    df = pd.read_csv(input_file)
    
    # Load existing progress if it exists
    if os.path.exists(output_file):
        labeled_df = pd.read_csv(output_file)
        labeled_ids = labeled_df['id'].tolist()
        to_label_df = df[~df['id'].isin(labeled_ids)]
    else:
        labeled_df = pd.DataFrame(columns=['id', 'name', 'type', 'entry', 'nature'])
        to_label_df = df

    new_labels = []
    
    print(f"\nRemaining to label: {len(to_label_df)}")
    print("Commands: [Enter] to accept suggestion | Type a new Nature | 's' to save\n")

    for _, row in to_label_df.iterrows():
        suggestion = suggest_nature(row['entry'])
        
        print("-" * 50)
        print(f"POKEMON: {row['name'].upper()} | TYPE: {row['type']}")
        print(f"ENTRY: {row['entry']}")
        
        if suggestion:
            print(f"SUGGESTION: {suggestion} (Press Enter to accept)")
        else:
            print("SUGGESTION: None (Please type one)")

        user_input = input("Assign Nature: ").strip().capitalize()
        
        if user_input.lower() == 's':
            break
        
        # If user just hits Enter, use the suggestion
        final_nature = user_input if user_input else suggestion
        
        if final_nature:
            row_data = row.to_dict()
            row_data['nature'] = final_nature
            new_labels.append(row_data)
            print(f"Saved as: {final_nature}")
        else:
            print("Skipped (No nature assigned)")

    if new_labels:
        current_session_df = pd.DataFrame(new_labels)
        final_df = pd.concat([labeled_df, current_session_df], ignore_index=True)
        final_df.to_csv(output_file, index=False)
        print(f"\nProgress saved! Total labeled: {len(final_df)}")

if __name__ == "__main__":
    run_semi_labeler()