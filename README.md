# PokéGPU: Computer System Architecture Mapping

This project builds a machine learning pipeline that classifies Pokémon Natures and maps them to physical GPU architecture components. It was developed for IT1910 - Computer System Architecture to demonstrate the relationship between software-driven logic and hardware-level execution.

## Project Structure

```
/
├── pokegpu_app.py              # Flask Backend & API Server
├── get_poke_data.py            # Data Extraction (PokéAPI)
├── nature_assigner.py          # Semi-supervised Data Labeling
├── train_engine.py             # ML Model Training Logic
├── pokedex_ui.html             # Retro Pokédex Frontend
├── data/
│   └── labeled_pokemon_entries.csv 
└── model/
    ├── nature_model.pkl
    └── nature_desc.pkl
```

## Workflow Overview

1. **Data Acquisition** — Run `get_poke_data.py` to mine raw descriptions and types for the first 386 Pokémon from the PokéAPI.

2. **Interactive Labeling** — Use `nature_assigner.py` to curate a custom dataset. This tool suggests a Nature based on behavioral keywords which the user confirms to build a labeled CSV.

3. **Model Training** — Run `train_engine.py` to process text via TF-IDF Vectorization and train a Logistic Regression classifier.

4. **Hardware Mapping & Inference** — Launch `pokegpu_app.py` to run the web interface. The system predicts a Pokémon's Nature and triggers a CSA Mapping Modal based on the Pokémon's Type.

## Key Features

- **Machine Learning Inference** — Uses Scikit-learn to predict personality traits from text descriptions.
- **Hardware Analogies** — Provides a conceptual bridge between Pokémon types and GPU components (e.g., Fire = Execution Units).
- **Serialized Artifacts** — Saves trained models as `.pkl` files for instant loading without retraining.
- **Clean UI** — A retro-styled Pokédex with a "Tech Spec" pop-up system to reduce visual clutter.

## How Each Script Works

### `get_poke_data.py`
- Connects to the PokéAPI to fetch name, type, and flavor text.
- Cleans and normalizes strings to ensure the ML model receives high-quality input.

### `nature_assigner.py`
- Loads raw entries and uses a keyword-matching dictionary to suggest "Brave," "Calm," "Timid," etc.
- Allows the user to manually override suggestions, creating a high-accuracy labeled dataset.

### `train_engine.py`
- **Preprocessing** — Implements a pipeline that converts text into a numerical matrix using TF-IDF.
- **Classification** — Trains a Logistic Regression model with a balanced class weight to handle varied Nature distributions.
- **Persistence** — Serializes the trained model and descriptions into the `model/` directory.

### `pokegpu_app.py`
- Acts as a Flask server providing a `/predict` endpoint.
- Hosts the retro Pokédex UI for interactive predictions.

## Installation

```bash
pip install flask flask-cors pandas scikit-learn
```

## Usage

```bash
# 1. Start the server
python pokegpu_app.py

# 2. Access the UI
# Open pokedex_ui.html in your browser
```

---

**Developed by:** Javier  
**Course:** IT1910 - Computer System Architecture
