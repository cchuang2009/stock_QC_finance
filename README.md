# Quantum + ML Alpha Engine

This project combines:
- Quantum feature extraction
- Machine learning classification
- Price prediction
- Multi-stock scanning
- Streamlit visualization

## Architecture

Market Data
    ↓
Feature Engineering
    ↓
Quantum Feature Layer
    ↓
ML Classification + Regression
    ↓
Trading Signal Engine

## Key Concepts

### Quantum Features
Quantum circuits encode market features into a Hilbert-space representation.

Features extracted:
- Probability entropy
- Nonlinear interaction structure
- Entanglement-inspired transformations

### Classification Model
Predicts:
Probability that price rises within next 5 bars.

### Price Predictor
Predicts:
Future expected price.

### Signal Engine
Generates:
- STRONG BUY
- BUY
- HOLD
- SELL
- STRONG SELL

Based on:
- probability
- volume shock
- trend
- acceleration

## Run

Install:

    pip install -r requirements.txt

Start app:

    streamlit run app.py

Run scanner:

    python scanner.py
    
    
# QC Finance v2

Enhanced Quantum + ML hybrid trading system.

New Features:
- Predict future RETURNS instead of raw prices
- Convert returns back into predicted prices
- Ensemble prediction:
  - LightGBM
  - Ridge Regression
- Quantum feature extraction
- Confidence score
- Signal engine
- Multi-stock scanner

## stable ML pipeline
- feature-consistent architecture
- scalable scanner
- ensemble predictor
- quantum-enhanced signal engine

## Notes

This is a research framework and not financial advice.
