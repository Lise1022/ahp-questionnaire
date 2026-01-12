# 🔩 AHP Questionnaire - Shear Connector Selection

An interactive web application for evaluating and ranking shear connector selection criteria using the **Analytic Hierarchy Process (AHP)** methodology.

## 📋 Description

This questionnaire is part of doctoral research at EPFL focused on **reuse-oriented composite structures**. It allows experts to compare and weight 11 criteria across 4 macro-categories:

- **M1: Structural Safety & Performance** (C1-C5)
- **M2: Serviceability & Compatibility** (C6-C7)
- **M3: Constructability & Practicality** (C8-C9)
- **M4: Durability & Circularity** (C10-C11)

## 🚀 Features

- Two input modes: Interactive sliders or Expert matrix input
- Real-time weight calculations
- Consistency Ratio (CR) verification
- Visual ranking chart
- Export results as JSON, CSV, or TXT
- Direct submission to database

## 🛠️ Installation (Local)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ahp-questionnaire.git
cd ahp-questionnaire

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 🌐 Online Access

Access the questionnaire online at: [YOUR_STREAMLIT_URL]

## 📊 Methodology

The questionnaire uses Saaty's AHP methodology:
- Pairwise comparisons on a 1-9 scale
- Geometric mean method for weight calculation
- Consistency Ratio verification (CR < 0.10)

## 📧 Contact

**Lise Bachmann**  
Doctoral Researcher - EPFL  
Email: lise.bachmann@epfl.ch

## 📄 License

This project is for academic research purposes.

---
*EPFL - RESSLAB - Doctoral Research on Shear Connectors for Reuse-Oriented Composite Structures*
