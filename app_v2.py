import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import io
import datetime
import json

st.set_page_config(page_title="AHP Questionnaire - Shear Connectors", page_icon="🔩", layout="wide")
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'serif'

# Custom CSS for better matrix styling
st.markdown("""
<style>
    .matrix-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .matrix-header {
        font-weight: bold;
        text-align: center;
        background-color: #e9ecef;
        padding: 8px;
        border-radius: 5px;
        margin-bottom: 5px;
    }
    .matrix-cell {
        text-align: center;
        padding: 5px;
    }
    .matrix-diagonal {
        background-color: #dee2e6;
        font-weight: bold;
    }
    .stNumberInput > div > div > input {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FONCTIONS AHP
# ============================================================================

def ahp_weights(matrix):
    n = matrix.shape[0]
    geo_mean = np.prod(matrix, axis=1) ** (1 / n)
    return geo_mean / np.sum(geo_mean)

def calculate_cr(matrix, ri_values):
    n = matrix.shape[0]
    if n <= 2:
        return 0.0
    eigvals, _ = np.linalg.eig(matrix)
    lambda_max = max(eigvals.real)
    CI = (lambda_max - n) / (n - 1)
    RI = ri_values.get(n, 0)
    return CI / RI if RI > 0 else 0

RI_VALUES = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

# ============================================================================
# INITIALISATION SESSION STATE
# ============================================================================

if 'macro_judgments' not in st.session_state:
    st.session_state.macro_judgments = {(0,1): 1.0, (0,2): 1.0, (0,3): 1.0, (1,2): 1.0, (1,3): 1.0, (2,3): 1.0}

if 'structural_judgments' not in st.session_state:
    st.session_state.structural_judgments = {
        (0,1): 1.0, (0,2): 1.0, (0,3): 1.0, (0,4): 1.0,
        (1,2): 1.0, (1,3): 1.0, (1,4): 1.0,
        (2,3): 1.0, (2,4): 1.0, (3,4): 1.0
    }

if 'serviceability_judgment' not in st.session_state:
    st.session_state.serviceability_judgment = 1.0

if 'constructability_judgment' not in st.session_state:
    st.session_state.constructability_judgment = 1.0

if 'durability_judgment' not in st.session_state:
    st.session_state.durability_judgment = 1.0

# ============================================================================
# DÉFINITIONS DES CRITÈRES - INDICES COHÉRENTS PAR FAMILLE
# ============================================================================

# Macro-criteria
macro_criteria = [
    "M1: Structural performance",
    "M2: Compatibility",
    "M3: Constructability",
    "M4: Durability"
]

macro_short = ["M1", "M2", "M3", "M4"]

macro_descriptions = {
    "M1": "Criteria related to structural behavior, code compliance, and mechanical performance",
    "M2": "Criteria related to compatibility with reused materials and geometric constraints",
    "M3": "Criteria related to ease of construction, assembly, and component simplicity",
    "M4": "Criteria related to maintenance, demountability, and circular economy principles"
}

# Critères regroupés par famille avec indices cohérents
# Structural: C1-C5, Serviceability: C6-C7, Constructability: C8-C9, Durability: C10-C11

all_criteria = {
    # Structural performance (C1-C5)
    "C1": ("Failure mode of the composite section", "M1: Structural performance",
           "This criterion evaluates which component fails first (concrete slab, steel beam, or connector) at the level of the composite cross-section. To guarantee the future reusability of the elements, failure in the connector is preferred as it allows the structural components (beam and slab) to remain undamaged and suitable for reuse."),
    "C2": ("Compliance with EC4 ductility requirements", "M1: Structural performance",
           "Eurocode 4 requires shear connectors to achieve a characteristic slip capacity (slip > 6 mm) before failure to be classified as ductile. This threshold ensures adequate deformation capacity and allows for moment redistribution in the composite beam."),
    "C3": ("Initial shear stiffness", "M1: Structural performance",
           "The connector's elastic stiffness governs deflections and vibrations under service loads. A gradual stiffness degradation (without sudden drops in resistance) is preferred for this study, as maintaining higher stiffness throughout the loading history reduces deformations and improves dynamic behavior, which is critical for serviceability performance."),
    "C4": ("Delay or avoidance of first slip", "M1: Structural performance",
           "First slip represents the initiation of non-linear behavior at the steel-concrete interface. Delaying or avoiding first slip improves serviceability performance by reducing long-term deformations and maintaining structural stiffness."),
    "C5": ("Shear resistance", "M1: Structural performance",
           "The ultimate shear capacity of the connector is fundamental for structural safety under extreme loads. This is a mandatory requirement for ultimate limit state (ULS) design verification."),
    
    # Compatibility (C6-C7)
    "C6": ("Compatibility with reused concrete slabs", "M2: Compatibility",
           "Since reused slabs are already fabricated and cannot be modified during casting, the connector must be compatible with post-installation techniques such as drilling and grouting. In other words, the primary objective is to place the tolerance in the concrete rather than requiring precise pre-cast provisions."),
    "C7": ("Size for double-slab configuration", "M2: Compatibility",
           "Using reused slabs requires placing two separate slabs on either side of the steel beam's top flange. The connector must be sufficiently compact to fit within standard beam flange widths without requiring custom-fabricated wide-flange sections. It should be noted that this study considers steel beams with relatively compact cross-sections, as the project's application scope targets office and industrial buildings."),
    
    # Constructability (C8-C9)
    "C8": ("Constructability and ease of assembly on-site", "M3: Constructability",
           "The connector must be practical for construction workers to install under typical site constraints. Simple, fast installation reduces labor costs, construction time, and risk of errors."),
    "C9": ("Simplicity of connector components", "M3: Constructability",
           "Connectors using standard catalog components (bolts, washers, threaded rods) are more economical and have shorter lead times than those requiring custom-made parts. Design simplicity also reduces fabrication costs and supply chain dependencies."),
    
    # Durability (C10-C11)
    "C10": ("Maintenance requirements during service life", "M4: Durability",
            "Connectors requiring periodic inspections, corrosion protection, or prestress monitoring increase life-cycle costs and operational disruptions. Low-maintenance or maintenance-free designs are preferred."),
    "C11": ("Demountability and reuse potential", "M4: Durability",
            "This criterion evaluates from a practical perspective whether the beam-slab assembly can be easily deconstructed and whether the connector can be readily replaced if necessary. Unlike criterion C1, which focuses on failure at the cross-section level, C11 assesses the ease of disassembly to enable component reuse and align with circular economy principles.")
}

# Codes par famille
structural_codes = ["C1", "C2", "C3", "C4", "C5"]
serviceability_codes = ["C6", "C7"]
constructability_codes = ["C8", "C9"]
durability_codes = ["C10", "C11"]

# ============================================================================
# FONCTIONS POUR CONSTRUIRE LES MATRICES ET CALCULER LES POIDS
# ============================================================================

def build_macro_matrix():
    A = np.ones((4, 4))
    for (i, j), v in st.session_state.macro_judgments.items():
        A[i, j] = v
        A[j, i] = 1 / v
    return A

def build_structural_matrix():
    A = np.ones((5, 5))
    for (i, j), v in st.session_state.structural_judgments.items():
        A[i, j] = v
        A[j, i] = 1 / v
    return A

def build_2x2_matrix(value):
    return np.array([[1.0, value], [1.0 / value, 1.0]])

def compute_all_weights():
    A_macro = build_macro_matrix()
    w_macro = ahp_weights(A_macro)
    cr_macro = calculate_cr(A_macro, RI_VALUES)
    
    A_structural = build_structural_matrix()
    w_structural = ahp_weights(A_structural)
    cr_structural = calculate_cr(A_structural, RI_VALUES)
    
    A_service = build_2x2_matrix(st.session_state.serviceability_judgment)
    w_service = ahp_weights(A_service)
    
    A_construct = build_2x2_matrix(st.session_state.constructability_judgment)
    w_construct = ahp_weights(A_construct)
    
    A_durability = build_2x2_matrix(st.session_state.durability_judgment)
    w_durability = ahp_weights(A_durability)
    
    global_weights = {}
    for i, code in enumerate(structural_codes):
        global_weights[code] = w_macro[0] * w_structural[i]
    for i, code in enumerate(serviceability_codes):
        global_weights[code] = w_macro[1] * w_service[i]
    for i, code in enumerate(constructability_codes):
        global_weights[code] = w_macro[2] * w_construct[i]
    for i, code in enumerate(durability_codes):
        global_weights[code] = w_macro[3] * w_durability[i]
    
    return {
        'macro': {'matrix': A_macro, 'weights': w_macro, 'CR': cr_macro},
        'structural': {'matrix': A_structural, 'weights': w_structural, 'CR': cr_structural},
        'serviceability': {'matrix': A_service, 'weights': w_service},
        'constructability': {'matrix': A_construct, 'weights': w_construct},
        'durability': {'matrix': A_durability, 'weights': w_durability},
        'global': global_weights
    }

# ============================================================================
# INTERFACE
# ============================================================================

st.title("🔩 AHP Questionnaire: Shear Connector Selection")
st.markdown("### Selection and Evaluation of Shear Connectors for Reuse-Oriented Composite Structures")
st.markdown("---")

# Participant info
st.header("📋 Participant Information")
col1, col2 = st.columns(2)
with col1:
    participant_name = st.text_input("Name", "")
    participant_role = st.text_input("Role/Position", "")
with col2:
    participant_organization = st.text_input("Organization", "")
    participant_email = st.text_input("Email", "")

st.markdown("---")

# Mode selection
st.header("⚙️ Input Mode")
input_mode = st.radio("Choose your preferred input method:", 
                      ["Interactive sliders (visual)", "Expert mode (direct matrix input)"], horizontal=True)
use_sliders = (input_mode == "Interactive sliders (visual)")

st.markdown("---")

# Instructions
with st.expander("📖 Complete Guide — How to Fill in This Questionnaire", expanded=True):

    st.markdown("""
    ## What is this questionnaire about?

    This questionnaire uses the **AHP method (Analytic Hierarchy Process)** to determine which criteria
    matter the most when selecting shear connectors for reuse-oriented composite structures.

    **Your task is simple:** compare criteria **two by two** and indicate which one you consider more
    important, and by how much. There are no right or wrong answers — this reflects **your expert judgment**.

    ---

    ## The two input modes

    This questionnaire offers two ways to enter your judgments:

    ### Interactive sliders (recommended for most users)

    - A **slider** is displayed for each pair of criteria.
    - The **left** criterion name appears on the left, the **right** criterion name appears on the right.
    - Slide **towards the left** to indicate the left criterion is more important.
    - Slide **towards the right** to indicate the right criterion is more important.
    - Leave the slider at **1** if both criteria are **equally important**.
    - A text summary below the slider confirms your choice in plain language.

    ### Expert mode (direct matrix input)

    - You fill in a **pairwise comparison matrix** directly.
    - You only fill in the **upper triangle** (above the diagonal). The lower triangle is computed automatically.
    - The diagonal is always **1** (a criterion compared to itself is always equal).
    - Enter a value **greater than 1** if the **row** criterion is more important than the **column** criterion.
    - Enter a value **less than 1** (e.g. 0.333) if the **column** criterion is more important than the **row** criterion.

    ---

    ## The Saaty scale: how to assign scores

    You compare criteria using **Saaty's 1–9 scale**. Scores must be **whole numbers** (integers).
    The intermediate values (2, 4, 6, 8) can be used if you hesitate between two levels.

    | Score | Meaning | How to think about it |
    |:-----:|---------|----------------------|
    | **1** | Equally important | Both criteria matter the same to me |
    | **2** | Equally to slightly more important | I have a very slight preference |
    | **3** | Slightly more important | I have a mild preference for one |
    | **4** | Slightly to strongly more important | My preference is moderate |
    | **5** | Strongly more important | I clearly prefer one over the other |
    | **6** | Strongly to very strongly more important | I have a strong preference |
    | **7** | Very strongly more important | One dominates the other clearly |
    | **8** | Very strongly to extremely more important | The dominance is very clear |
    | **9** | Extremely more important | The maximum possible difference in importance |

    ### What do the fractions (1/3, 1/5, 1/7…) mean?

    The fractions are **reciprocals**. They express the **opposite** preference:

    - If comparing **A vs B**, a score of **5** means "A is strongly more important than B".
    - The reciprocal is then **1/5** for B vs A, meaning "B is strongly less important than A".
    - In the **slider mode**, sliding to the right side automatically uses the reciprocal for you.
    - In the **expert mode**, you can directly enter 0.200 (= 1/5), 0.333 (= 1/3), etc.

    | You enter | It means | Equivalent fraction |
    |:---------:|----------|:-------------------:|
    | 1/3 = 0.333 | The other criterion is slightly more important | 1/3 |
    | 1/5 = 0.200 | The other criterion is strongly more important | 1/5 |
    | 1/7 = 0.143 | The other criterion is very strongly more important | 1/7 |
    | 1/9 = 0.111 | The other criterion is extremely more important | 1/9 |

    **Key rule:** always use **integer** scores on the Saaty scale (1, 2, 3, 4, 5, 6, 7, 8, 9).
    The fractions are automatically derived from those integers.

    ---

    ## Worked example: comparing fruit preferences

    To help you understand the process, here is a simple example with three statements:

    - **(A)** 🍎 I like eating **apples**
    - **(B)** 🍒 I like eating **cherries**
    - **(C)** 🍍 I like eating **pineapples**

    Suppose your preferences are: you clearly prefer apples over cherries, you mildly prefer apples
    over pineapples, and you slightly prefer pineapples over cherries.

    ### Step 1: Make all pairwise comparisons

    With 3 items, you need **3 comparisons** (🍎 vs 🍒, 🍎 vs 🍍, 🍒 vs 🍍):

    | Comparison | Your judgment | Score |
    |:----------:|--------------|:-----:|
    | 🍎 vs 🍒 | I strongly prefer apples over cherries | **5** |
    | 🍎 vs 🍍 | I slightly prefer apples over pineapples | **3** |
    | 🍒 vs 🍍 | I slightly prefer pineapples over cherries | **1/3** (= cherries is less important) |

    ### Step 2: Build the pairwise comparison matrix

    | | 🍎 A (Apples) | 🍒 B (Cherries) | 🍍 C (Pineapples) |
    |:-:|:----------:|:------------:|:---------------:|
    | **🍎 A** | 1 | 5 | 3 |
    | **🍒 B** | 1/5 | 1 | 1/3 |
    | **🍍 C** | 1/3 | 3 | 1 |

    - The diagonal is always **1** (🍎 apples vs 🍎 apples = equal).
    - If 🍎 vs 🍒 = 5, then 🍒 vs 🍎 = **1/5** (the reciprocal fills automatically).

    ### Step 3: Compute the weights

    The AHP method calculates a **priority weight** for each item from this matrix.
    In this example, the approximate result would be:

    | Item | Weight | Interpretation |
    |:----:|:------:|:--------------:|
    | 🍎 A (Apples) | **63%** | Most preferred |
    | 🍍 C (Pineapples) | **26%** | Second |
    | 🍒 B (Cherries) | **11%** | Least preferred |

    ### Step 4: Check the consistency (see below)

    Your comparisons say: 🍎 > 🍍 > 🍒. Let's verify consistency:
    - 🍎 is **5×** more important than 🍒
    - 🍎 is **3×** more important than 🍍
    - 🍍 should therefore be about **5/3 ≈ 1.67×** more important than 🍒
    - You entered **3** for 🍍 vs 🍒, which is close but not perfectly consistent

    This slight inconsistency is **normal and acceptable** — the Consistency Ratio (CR) will be
    low (below 0.10), confirming your judgments are coherent enough. Perfect mathematical consistency
    is rarely achieved by humans, and small deviations are expected.

    ---

    ## The Consistency Ratio (CR) — What is it and why does it matter?

    When you make multiple pairwise comparisons, the AHP method checks whether your answers are
    **logically consistent** with each other. This check is called the **Consistency Ratio (CR)**.

    ### The intuition — back to our fruits 🍎🍒🍍

    In the consistent example above, your judgments made sense together:
    you said 🍎 > 🍍 > 🍒, and the numerical ratios roughly matched.

    Now imagine a **contradictory** set of judgments:

    | Comparison | Your judgment | Score |
    |:----------:|--------------|:-----:|
    | 🍎 vs 🍒 | I extremely prefer apples over cherries | **9** |
    | 🍎 vs 🍍 | I slightly prefer apples over pineapples | **2** |
    | 🍒 vs 🍍 | I very strongly prefer cherries over pineapples | **7** |

    Let's check the logic:
    - You say 🍎 is **9×** more important than 🍒
    - You say 🍎 is only **2×** more important than 🍍
    - So 🍍 should be about **9/2 = 4.5×** more important than 🍒
    - **But you said the opposite!** You entered 🍒 vs 🍍 = **7**, meaning 🍒 is **7×** more important than 🍍

    This is a clear **contradiction**: according to your first two answers, 🍍 should dominate 🍒,
    but your third answer says 🍒 dominates 🍍. The CR will be **very high** (well above 0.10),
    warning you that something is off.

    ### How is the CR actually calculated?

    The CR is computed in three steps:

    **Step 1 — Find λ_max (the largest eigenvalue of your matrix)**

    The comparison matrix is multiplied by the weight vector. In a perfectly consistent matrix,
    every row would give exactly *n × w_i* (where *n* is the number of criteria). In practice,
    due to small inconsistencies, each row gives a slightly different value. The **largest eigenvalue**
    (λ_max) captures how far, on average, the matrix is from being perfectly consistent.
    For a perfectly consistent matrix, λ_max = *n* exactly.

    **Step 2 — Compute the Consistency Index (CI)**

    The Consistency Index measures the deviation from perfect consistency:

    > **CI = (λ_max − n) / (n − 1)**

    Where *n* is the number of criteria being compared. A perfectly consistent matrix has CI = 0.

    **Step 3 — Compute the Consistency Ratio (CR)**

    The CI alone doesn't tell you much because larger matrices are naturally more prone to
    inconsistency. So we normalize CI by a **Random Index (RI)**, which represents the average CI
    of thousands of randomly filled matrices of the same size:

    > **CR = CI / RI**

    The RI values used are:

    | Matrix size (n) | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
    |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
    | **RI** | 0.00 | 0.58 | 0.90 | 1.12 | 1.24 | 1.32 | 1.41 | 1.45 | 1.49 |

    ### Numerical example with the inconsistent fruits 🍎🍒🍍

    Using the contradictory matrix above:

    | | 🍎 | 🍒 | 🍍 |
    |:-:|:---:|:---:|:---:|
    | **🍎** | 1 | 9 | 2 |
    | **🍒** | 1/9 | 1 | 7 |
    | **🍍** | 1/2 | 1/7 | 1 |

    1. The largest eigenvalue is approximately **λ_max ≈ 4.47**
    2. CI = (4.47 − 3) / (3 − 1) = **0.74**
    3. For n = 3, RI = 0.58, so CR = 0.74 / 0.58 = **1.27**

    **CR = 1.27 ≫ 0.10** → ⚠️ The judgments are **highly inconsistent** and should be revised.

    Compare this with the consistent example (🍎 vs 🍒 = 5, 🍎 vs 🍍 = 3, 🍒 vs 🍍 = 1/3) which gives **CR ≈ 0.03** → ✅ acceptable.

    ### How to read the CR

    | CR value | Meaning | What to do |
    |:--------:|---------|------------|
    | **CR < 0.10** | ✅ Your comparisons are **consistent** enough | Nothing — your judgments are fine |
    | **CR ≥ 0.10** | ⚠️ Your comparisons may contain **contradictions** | Review your answers and adjust the ones that seem off |

    ### What if my CR is too high?

    Don't worry — it just means some of your comparisons are contradictory. Typical fixes:

    1. Look for a comparison where you gave a very high score (e.g. 9) or very low score (e.g. 1/9)
    2. Ask yourself: "Is this really **that** extreme compared to my other answers?"
    3. Check the transitivity: if A > B and B > C, make sure your A vs C score reflects A > C too
    4. Adjust the most extreme or contradictory comparisons and the CR usually drops below 0.10

    **Note:** For groups of only 2 criteria (e.g. C6 vs C7), there is only one comparison
    so the CR is always 0 — consistency is guaranteed with a single comparison.

    ---

    ## Summary of the questionnaire structure

    This questionnaire has **two levels**:

    1. **Level 1 — Macro-criteria (4 families):** You compare the 4 families of criteria against each other
       (6 comparisons).
    2. **Level 2 — Internal criteria:** Within each family, you compare the criteria against each other:
       - M1: Structural performance → 5 criteria → 10 comparisons
       - M2: Compatibility → 2 criteria → 1 comparison
       - M3: Constructability → 2 criteria → 1 comparison
       - M4: Durability → 2 criteria → 1 comparison

    In total, you will make **6 + 10 + 1 + 1 + 1 = 19 pairwise comparisons**.

    The final global weight of each criterion is computed as:
    **Global weight = Macro-criterion weight × Local weight within its family**.
    """)

# Criteria definitions
with st.expander("📚 Criteria Definitions", expanded=False):
    st.markdown("### Criteria grouped by family")
    
    st.markdown("#### 🔴 M1: Structural performance")
    for code in structural_codes:
        name, _, desc = all_criteria[code]
        st.markdown(f"**{code} - {name}**")
        st.caption(desc)
    
    st.markdown("#### 🔵 M2: Compatibility")
    for code in serviceability_codes:
        name, _, desc = all_criteria[code]
        st.markdown(f"**{code} - {name}**")
        st.caption(desc)
    
    st.markdown("#### 🟡 M3: Constructability")
    for code in constructability_codes:
        name, _, desc = all_criteria[code]
        st.markdown(f"**{code} - {name}**")
        st.caption(desc)
    
    st.markdown("#### 🟢 M4: Durability")
    for code in durability_codes:
        name, _, desc = all_criteria[code]
        st.markdown(f"**{code} - {name}**")
        st.caption(desc)

st.markdown("---")

# Helper functions
def get_options():
    return [9, 8, 7, 6, 5, 4, 3, 2, 1, 1/2, 1/3, 1/4, 1/5, 1/6, 1/7, 1/8, 1/9]

def fmt(x):
    if x >= 1:
        return f"{int(x)}" if x == int(x) else f"{x:.2f}"
    return f"1/{int(1/x)}" if 1/x == int(1/x) else f"{x:.3f}"

def show_result(val, n1, n2):
    if val == 1:
        st.caption(f"↔️ {n1} and {n2} are equally important")
    elif val > 1:
        st.caption(f"➡️ **{n1}** is more important (factor: {fmt(val)})")
    else:
        st.caption(f"⬅️ **{n2}** is more important (factor: {fmt(1/val)})")

def get_comparison_tooltip(code1, code2):
    """Génère un tooltip pour une comparaison"""
    name1, _, desc1 = all_criteria.get(code1, (code1, "", ""))
    name2, _, desc2 = all_criteria.get(code2, (code2, "", ""))
    return f"{code1}: {name1}\n{code2}: {name2}"

def get_macro_tooltip(m1, m2):
    """Génère un tooltip pour une comparaison macro"""
    desc1 = macro_descriptions.get(m1, "")
    desc2 = macro_descriptions.get(m2, "")
    return f"{m1}: {desc1}\n\n{m2}: {desc2}"

# ============================================================================
# LEVEL 1: MACRO-CRITERIA
# ============================================================================

st.header("1️⃣ Level 1: Macro-Criteria Comparisons")
st.markdown("Compare the importance of the four macro-criteria families.")

# Ordre des comparaisons: M1 vs M2 en premier (plus logique)
macro_comps = [
    ((0,1), "M1", "M2", "Structural safety", "Serviceability"),
    ((0,2), "M1", "M3", "Structural safety", "Constructability"),
    ((0,3), "M1", "M4", "Structural safety", "Durability"),
    ((1,2), "M2", "M3", "Serviceability", "Constructability"),
    ((1,3), "M2", "M4", "Serviceability", "Durability"),
    ((2,3), "M3", "M4", "Constructability", "Durability")
]

if use_sliders:
    for idx, ((i, j), m1, m2, name1, name2) in enumerate(macro_comps):
        tooltip = get_macro_tooltip(m1, m2)
        st.markdown(f"**Comparison {idx+1}/6:** {m1} vs {m2}")
        
        c1, c2, c3 = st.columns([1, 4, 1])
        with c1:
            st.markdown(f"**{m1}**")
            st.caption(name1)
        with c2:
            opts = get_options()
            cur = st.session_state.macro_judgments[(i, j)]
            idx_closest = min(range(len(opts)), key=lambda k: abs(opts[k] - cur))
            new_val = st.select_slider(f"m_{i}_{j}", options=opts, value=opts[idx_closest], 
                                       format_func=fmt, key=f"sl_m_{i}_{j}", label_visibility="collapsed",
                                       help=tooltip)
            st.session_state.macro_judgments[(i, j)] = new_val
            show_result(new_val, m1, m2)
        with c3:
            st.markdown(f"**{m2}**")
            st.caption(name2)
        st.markdown("---")
else:
    # Mode expert: matrice avec style amélioré
    st.markdown("#### Pairwise Comparison Matrix")
    st.caption("Enter values in the upper triangle. Values > 1 mean row is more important than column.")
    
    # Container avec style
    st.markdown('<div class="matrix-container">', unsafe_allow_html=True)
    
    # Header row
    header_cols = st.columns([1.5] + [1]*4)
    header_cols[0].markdown("**↓ vs →**")
    for idx, m in enumerate(macro_short):
        header_cols[idx+1].markdown(f"**{m}**", help=macro_descriptions[m])
    
    # Matrix rows
    for i in range(4):
        cols = st.columns([1.5] + [1]*4)
        cols[0].markdown(f"**{macro_short[i]}**", help=macro_descriptions[macro_short[i]])
        for j in range(4):
            with cols[j+1]:
                if i == j:
                    st.markdown("**1.000**")
                elif i < j:
                    cur = st.session_state.macro_judgments[(i, j)]
                    tooltip = get_macro_tooltip(macro_short[i], macro_short[j])
                    new_val = st.number_input(f"{macro_short[i]}vs{macro_short[j]}", 
                                              min_value=0.111, max_value=9.0, value=float(cur),
                                              step=0.5, format="%.3f", key=f"mx_m_{i}_{j}", 
                                              label_visibility="collapsed", help=tooltip)
                    st.session_state.macro_judgments[(i, j)] = new_val
                else:
                    mirror_val = 1/st.session_state.macro_judgments[(j,i)]
                    st.markdown(f"*{mirror_val:.3f}*")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Display macro results
data = compute_all_weights()
w_macro = data['macro']['weights']
CR_macro = data['macro']['CR']

st.subheader("📊 Macro-Criteria Results")
c1, c2 = st.columns(2)
with c1:
    df = pd.DataFrame({
        'Macro-Criterion': [f"{macro_short[i]}: {macro_criteria[i].split(': ')[1]}" for i in range(4)],
        'Weight': w_macro,
        'Percentage': [f"{w*100:.2f}%" for w in w_macro]
    })
    st.dataframe(df, hide_index=True, use_container_width=True)
with c2:
    if CR_macro < 0.10:
        st.success(f"✅ Consistency Ratio: {CR_macro:.4f} (acceptable < 0.10)")
    else:
        st.warning(f"⚠️ Consistency Ratio: {CR_macro:.4f} (should be < 0.10)")

st.markdown("---")

# ============================================================================
# LEVEL 2: INTERNAL CRITERIA
# ============================================================================

st.header("2️⃣ Level 2: Internal Criteria Comparisons")

# STRUCTURAL (C1-C5)
with st.expander("🔴 M1: Structural Safety & Performance (C1-C5)", expanded=True):
    data = compute_all_weights()
    w_macro = data['macro']['weights']
    st.markdown(f"**Macro-criterion weight: {w_macro[0]*100:.2f}%**")
    
    struct_comps = [(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
    
    if use_sliders:
        for idx, (i, j) in enumerate(struct_comps):
            code_i, code_j = structural_codes[i], structural_codes[j]
            name_i = all_criteria[code_i][0][:35] + "..."
            name_j = all_criteria[code_j][0][:35] + "..."
            tooltip = f"{code_i}: {all_criteria[code_i][0]}\n{all_criteria[code_i][2]}\n\n{code_j}: {all_criteria[code_j][0]}\n{all_criteria[code_j][2]}"
            
            st.markdown(f"**{idx+1}/10:** {code_i} vs {code_j}")
            c1, c2, c3 = st.columns([1, 4, 1])
            with c1:
                st.markdown(f"**{code_i}**")
                st.caption(name_i)
            with c2:
                opts = get_options()
                cur = st.session_state.structural_judgments[(i, j)]
                idx_closest = min(range(len(opts)), key=lambda k: abs(opts[k] - cur))
                new_val = st.select_slider(f"s_{i}_{j}", options=opts, value=opts[idx_closest],
                                           format_func=fmt, key=f"sl_s_{i}_{j}", 
                                           label_visibility="collapsed", help=tooltip)
                st.session_state.structural_judgments[(i, j)] = new_val
                show_result(new_val, code_i, code_j)
            with c3:
                st.markdown(f"**{code_j}**")
                st.caption(name_j)
    else:
        st.markdown("#### Pairwise Comparison Matrix")
        st.markdown('<div class="matrix-container">', unsafe_allow_html=True)
        
        header_cols = st.columns([1.5] + [1]*5)
        header_cols[0].markdown("**↓ vs →**")
        for idx, code in enumerate(structural_codes):
            header_cols[idx+1].markdown(f"**{code}**", help=f"{all_criteria[code][0]}\n\n{all_criteria[code][2]}")
        
        for i in range(5):
            cols = st.columns([1.5] + [1]*5)
            cols[0].markdown(f"**{structural_codes[i]}**", help=f"{all_criteria[structural_codes[i]][0]}")
            for j in range(5):
                with cols[j+1]:
                    if i == j:
                        st.markdown("**1.000**")
                    elif i < j:
                        cur = st.session_state.structural_judgments[(i, j)]
                        ci, cj = structural_codes[i], structural_codes[j]
                        tooltip = f"{ci}: {all_criteria[ci][0]}\n\n{cj}: {all_criteria[cj][0]}"
                        new_val = st.number_input(f"{ci}vs{cj}", min_value=0.111, max_value=9.0,
                                                  value=float(cur), step=0.5, format="%.3f",
                                                  key=f"mx_s_{i}_{j}", label_visibility="collapsed",
                                                  help=tooltip)
                        st.session_state.structural_judgments[(i, j)] = new_val
                    else:
                        st.markdown(f"*{1/st.session_state.structural_judgments[(j,i)]:.3f}*")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    data = compute_all_weights()
    w_struct = data['structural']['weights']
    w_macro = data['macro']['weights']
    CR_struct = data['structural']['CR']
    
    st.markdown("---")
    st.markdown("#### Results")
    c1, c2 = st.columns([2, 1])
    with c1:
        df = pd.DataFrame({
            'Code': structural_codes,
            'Criterion': [all_criteria[c][0][:40] + "..." for c in structural_codes],
            'Local %': [f"{w*100:.2f}%" for w in w_struct],
            'Global %': [f"{w_macro[0] * w * 100:.2f}%" for w in w_struct]
        })
        st.dataframe(df, hide_index=True, use_container_width=True)
    with c2:
        if CR_struct < 0.10:
            st.success(f"✅ CR: {CR_struct:.4f}")
        else:
            st.warning(f"⚠️ CR: {CR_struct:.4f}")

# SERVICEABILITY (C6-C7)
with st.expander("🔵 M2: Serviceability & Compatibility (C6-C7)", expanded=True):
    data = compute_all_weights()
    w_macro = data['macro']['weights']
    st.markdown(f"**Macro-criterion weight: {w_macro[1]*100:.2f}%**")
    
    code_i, code_j = "C6", "C7"
    tooltip = f"{code_i}: {all_criteria[code_i][0]}\n{all_criteria[code_i][2]}\n\n{code_j}: {all_criteria[code_j][0]}\n{all_criteria[code_j][2]}"
    
    st.markdown(f"**{code_i} vs {code_j}**")
    
    if use_sliders:
        c1, c2, c3 = st.columns([1, 4, 1])
        with c1:
            st.markdown(f"**{code_i}**")
            st.caption(all_criteria[code_i][0][:30] + "...")
        with c2:
            opts = get_options()
            cur = st.session_state.serviceability_judgment
            idx_closest = min(range(len(opts)), key=lambda k: abs(opts[k] - cur))
            new_val = st.select_slider("serv", options=opts, value=opts[idx_closest],
                                       format_func=fmt, key="sl_serv", label_visibility="collapsed",
                                       help=tooltip)
            st.session_state.serviceability_judgment = new_val
            show_result(new_val, code_i, code_j)
        with c3:
            st.markdown(f"**{code_j}**")
            st.caption(all_criteria[code_j][0][:30] + "...")
    else:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            st.markdown(f"**{code_i}** vs **{code_j}**")
        with c2:
            cur = st.session_state.serviceability_judgment
            new_val = st.number_input(f"{code_i}vs{code_j}", min_value=0.111, max_value=9.0,
                                      value=float(cur), step=0.5, format="%.3f",
                                      key="mx_serv", label_visibility="collapsed", help=tooltip)
            st.session_state.serviceability_judgment = new_val
            show_result(new_val, code_i, code_j)
    
    data = compute_all_weights()
    w_serv = data['serviceability']['weights']
    w_macro = data['macro']['weights']
    
    st.markdown("---")
    df = pd.DataFrame({
        'Code': serviceability_codes,
        'Criterion': [all_criteria[c][0] for c in serviceability_codes],
        'Local %': [f"{w*100:.2f}%" for w in w_serv],
        'Global %': [f"{w_macro[1] * w * 100:.2f}%" for w in w_serv]
    })
    st.dataframe(df, hide_index=True, use_container_width=True)

# CONSTRUCTABILITY (C8-C9)
with st.expander("🟡 M3: Constructability (C8-C9)", expanded=True):
    data = compute_all_weights()
    w_macro = data['macro']['weights']
    st.markdown(f"**Macro-criterion weight: {w_macro[2]*100:.2f}%**")
    
    code_i, code_j = "C8", "C9"
    tooltip = f"{code_i}: {all_criteria[code_i][0]}\n{all_criteria[code_i][2]}\n\n{code_j}: {all_criteria[code_j][0]}\n{all_criteria[code_j][2]}"
    
    st.markdown(f"**{code_i} vs {code_j}**")
    
    if use_sliders:
        c1, c2, c3 = st.columns([1, 4, 1])
        with c1:
            st.markdown(f"**{code_i}**")
            st.caption(all_criteria[code_i][0][:30] + "...")
        with c2:
            opts = get_options()
            cur = st.session_state.constructability_judgment
            idx_closest = min(range(len(opts)), key=lambda k: abs(opts[k] - cur))
            new_val = st.select_slider("constr", options=opts, value=opts[idx_closest],
                                       format_func=fmt, key="sl_constr", label_visibility="collapsed",
                                       help=tooltip)
            st.session_state.constructability_judgment = new_val
            show_result(new_val, code_i, code_j)
        with c3:
            st.markdown(f"**{code_j}**")
            st.caption(all_criteria[code_j][0][:30] + "...")
    else:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            st.markdown(f"**{code_i}** vs **{code_j}**")
        with c2:
            cur = st.session_state.constructability_judgment
            new_val = st.number_input(f"{code_i}vs{code_j}", min_value=0.111, max_value=9.0,
                                      value=float(cur), step=0.5, format="%.3f",
                                      key="mx_constr", label_visibility="collapsed", help=tooltip)
            st.session_state.constructability_judgment = new_val
            show_result(new_val, code_i, code_j)
    
    data = compute_all_weights()
    w_constr = data['constructability']['weights']
    w_macro = data['macro']['weights']
    
    st.markdown("---")
    df = pd.DataFrame({
        'Code': constructability_codes,
        'Criterion': [all_criteria[c][0] for c in constructability_codes],
        'Local %': [f"{w*100:.2f}%" for w in w_constr],
        'Global %': [f"{w_macro[2] * w * 100:.2f}%" for w in w_constr]
    })
    st.dataframe(df, hide_index=True, use_container_width=True)

# DURABILITY (C10-C11)
with st.expander("🟢 M4: Durability & Circularity (C10-C11)", expanded=True):
    data = compute_all_weights()
    w_macro = data['macro']['weights']
    st.markdown(f"**Macro-criterion weight: {w_macro[3]*100:.2f}%**")
    
    code_i, code_j = "C10", "C11"
    tooltip = f"{code_i}: {all_criteria[code_i][0]}\n{all_criteria[code_i][2]}\n\n{code_j}: {all_criteria[code_j][0]}\n{all_criteria[code_j][2]}"
    
    st.markdown(f"**{code_i} vs {code_j}**")
    
    if use_sliders:
        c1, c2, c3 = st.columns([1, 4, 1])
        with c1:
            st.markdown(f"**{code_i}**")
            st.caption(all_criteria[code_i][0][:30] + "...")
        with c2:
            opts = get_options()
            cur = st.session_state.durability_judgment
            idx_closest = min(range(len(opts)), key=lambda k: abs(opts[k] - cur))
            new_val = st.select_slider("durab", options=opts, value=opts[idx_closest],
                                       format_func=fmt, key="sl_durab", label_visibility="collapsed",
                                       help=tooltip)
            st.session_state.durability_judgment = new_val
            show_result(new_val, code_i, code_j)
        with c3:
            st.markdown(f"**{code_j}**")
            st.caption(all_criteria[code_j][0][:30] + "...")
    else:
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            st.markdown(f"**{code_i}** vs **{code_j}**")
        with c2:
            cur = st.session_state.durability_judgment
            new_val = st.number_input(f"{code_i}vs{code_j}", min_value=0.111, max_value=9.0,
                                      value=float(cur), step=0.5, format="%.3f",
                                      key="mx_durab", label_visibility="collapsed", help=tooltip)
            st.session_state.durability_judgment = new_val
            show_result(new_val, code_i, code_j)
    
    data = compute_all_weights()
    w_durab = data['durability']['weights']
    w_macro = data['macro']['weights']
    
    st.markdown("---")
    df = pd.DataFrame({
        'Code': durability_codes,
        'Criterion': [all_criteria[c][0] for c in durability_codes],
        'Local %': [f"{w*100:.2f}%" for w in w_durab],
        'Global %': [f"{w_macro[3] * w * 100:.2f}%" for w in w_durab]
    })
    st.dataframe(df, hide_index=True, use_container_width=True)

st.markdown("---")

# ============================================================================
# FINAL RESULTS
# ============================================================================

st.header("3️⃣ Final Global Weights and Ranking")

final_data = compute_all_weights()
global_weights = final_data['global']
w_macro_final = final_data['macro']['weights']
CR_macro_final = final_data['macro']['CR']
w_struct_final = final_data['structural']['weights']
CR_struct_final = final_data['structural']['CR']
w_serv_final = final_data['serviceability']['weights']
w_constr_final = final_data['constructability']['weights']
w_durab_final = final_data['durability']['weights']
A_macro_final = final_data['macro']['matrix']
A_struct_final = final_data['structural']['matrix']
A_serv_final = final_data['serviceability']['matrix']
A_constr_final = final_data['constructability']['matrix']
A_durab_final = final_data['durability']['matrix']

sorted_criteria = sorted(global_weights.items(), key=lambda x: x[1], reverse=True)

final_df = pd.DataFrame({
    'Rank': range(1, 12),
    'Code': [c for c, _ in sorted_criteria],
    'Criterion': [all_criteria[c][0] for c, _ in sorted_criteria],
    'Macro-criterion': [all_criteria[c][1].split(': ')[0] for c, _ in sorted_criteria],
    'Global Weight': [w for _, w in sorted_criteria],
    'Percentage': [f"{w*100:.2f}%" for _, w in sorted_criteria]
})

st.dataframe(final_df, hide_index=True, use_container_width=True)

total = sum(global_weights.values())
if abs(total - 1.0) < 0.001:
    st.success(f"✅ Total weight: {total:.6f} (correct)")
else:
    st.warning(f"⚠️ Total weight: {total:.6f} (should be 1.0)")

with st.expander("🔍 Calculation Details", expanded=False):
    st.markdown("### Global Weight = Macro Weight × Local Weight")
    st.markdown("---")
    st.markdown("#### Macro-criteria Weights")
    for i in range(4):
        st.markdown(f"- **{macro_short[i]}** ({macro_criteria[i].split(': ')[1]}): {w_macro_final[i]:.4f} ({w_macro_final[i]*100:.2f}%)")
    
    st.markdown("---")
    st.markdown("#### M1: Structural Safety (× local weights)")
    for i, code in enumerate(structural_codes):
        gw = global_weights[code]
        st.markdown(f"- **{code}**: {w_macro_final[0]:.4f} × {w_struct_final[i]:.4f} = {gw:.4f} ({gw*100:.2f}%)")
    
    st.markdown("---")
    st.markdown("#### M2: Serviceability (× local weights)")
    for i, code in enumerate(serviceability_codes):
        gw = global_weights[code]
        st.markdown(f"- **{code}**: {w_macro_final[1]:.4f} × {w_serv_final[i]:.4f} = {gw:.4f} ({gw*100:.2f}%)")
    
    st.markdown("---")
    st.markdown("#### M3: Constructability (× local weights)")
    for i, code in enumerate(constructability_codes):
        gw = global_weights[code]
        st.markdown(f"- **{code}**: {w_macro_final[2]:.4f} × {w_constr_final[i]:.4f} = {gw:.4f} ({gw*100:.2f}%)")
    
    st.markdown("---")
    st.markdown("#### M4: Durability (× local weights)")
    for i, code in enumerate(durability_codes):
        gw = global_weights[code]
        st.markdown(f"- **{code}**: {w_macro_final[3]:.4f} × {w_durab_final[i]:.4f} = {gw:.4f} ({gw*100:.2f}%)")

st.markdown("---")

# ============================================================================
# VISUALIZATION
# ============================================================================

st.header("📊 Visualization")

codes_plot = [c for c, _ in reversed(sorted_criteria)]
weights_plot = [w * 100 for _, w in reversed(sorted_criteria)]

# Nuances de gris professionnelles avec hachures distinctives
color_map = {
    "M1": ('#4a4a4a', '\\\\'),   # Gris foncé - Structural
    "M2": ('#8a8a8a', ''),        # Gris moyen - Serviceability
    "M3": ('#b0b0b0', 'xxx'),     # Gris clair - Constructability
    "M4": ('#6a6a6a', '///')      # Gris moyen-foncé - Durability
}

colors = [color_map[all_criteria[c][1].split(': ')[0]][0] for c in codes_plot]
hatches = [color_map[all_criteria[c][1].split(': ')[0]][1] for c in codes_plot]

# Figure compacte et haute résolution
fig, ax = plt.subplots(figsize=(8, 5), dpi=450)

bars = ax.barh(range(len(codes_plot)), weights_plot, color=colors, edgecolor='black', linewidth=0.8, height=0.7)

for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

for i, (bar, w) in enumerate(zip(bars, weights_plot)):
    ax.text(w + 0.5, i, f'{w:.1f}%', va='center', fontsize=9, fontweight='bold')

ax.set_yticks(range(len(codes_plot)))
ax.set_yticklabels(codes_plot, fontsize=10, fontweight='bold')
ax.set_xlabel('Global Weight (%)', fontsize=11, fontweight='bold')
ax.set_title('AHP Ranking of Shear Connector Selection Criteria\n(Reuse-Oriented Composite Structures)',
             fontsize=12, fontweight='bold', pad=15)
ax.set_xlim(0, max(weights_plot) * 1.12)
ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)

# Légende avec nuances de gris
legend_elements = [
    Patch(facecolor='#4a4a4a', edgecolor='black', hatch='\\\\', label='M1: Structural safety'),
    Patch(facecolor='#8a8a8a', edgecolor='black', label='M2: Serviceability'),
    Patch(facecolor='#b0b0b0', edgecolor='black', hatch='xxx', label='M3: Constructability'),
    Patch(facecolor='#6a6a6a', edgecolor='black', hatch='///', label='M4: Durability')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.95)

plt.tight_layout()

# Utiliser des colonnes pour limiter la largeur et centrer
col_left, col_center, col_right = st.columns([1, 2, 1])
with col_center:
    st.pyplot(fig, use_container_width=False)

# Boutons de téléchargement avec haute résolution
c1, c2 = st.columns(2)
with c1:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    buf.seek(0)
    st.download_button("📥 Download PNG (300 DPI)", buf, f"ahp_{participant_name.replace(' ','_')}.png", "image/png")
with c2:
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', bbox_inches='tight')
    buf.seek(0)
    st.download_button("📥 Download PDF (Vector)", buf, f"ahp_{participant_name.replace(' ','_')}.pdf", "application/pdf")

st.markdown("---")

# ============================================================================
# EXPORT
# ============================================================================

st.header("💾 Export and Save Results")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename_base = f"ahp_{participant_name.replace(' ', '_')}_{timestamp}"

results_summary = {
    "participant": {"name": participant_name, "role": participant_role,
                   "organization": participant_organization, "email": participant_email},
    "timestamp": datetime.datetime.now().isoformat(),
    "macro_criteria": {"weights": {macro_short[i]: float(w_macro_final[i]) for i in range(4)},
                      "CR": float(CR_macro_final), "matrix": A_macro_final.tolist()},
    "internal_criteria": {
        "structural": {"weights": {c: float(w_struct_final[i]) for i, c in enumerate(structural_codes)},
                      "CR": float(CR_struct_final), "matrix": A_struct_final.tolist()},
        "serviceability": {"weights": {c: float(w_serv_final[i]) for i, c in enumerate(serviceability_codes)},
                          "CR": 0.0, "matrix": A_serv_final.tolist()},
        "constructability": {"weights": {c: float(w_constr_final[i]) for i, c in enumerate(constructability_codes)},
                            "CR": 0.0, "matrix": A_constr_final.tolist()},
        "durability": {"weights": {c: float(w_durab_final[i]) for i, c in enumerate(durability_codes)},
                      "CR": 0.0, "matrix": A_durab_final.tolist()}
    },
    "global_weights": {c: float(w) for c, w in global_weights.items()},
    "ranking": [(c, float(w)) for c, w in sorted_criteria]
}

results_json = json.dumps(results_summary, indent=2)
csv_ranking = final_df.to_csv(index=False)

all_matrices = f"""PARTICIPANT: {participant_name}
ORGANIZATION: {participant_organization}
EMAIL: {participant_email}
TIMESTAMP: {timestamp}

MACRO-CRITERIA MATRIX (M1-M4)
{pd.DataFrame(A_macro_final, columns=macro_short, index=macro_short).to_csv()}

STRUCTURAL SAFETY MATRIX (C1-C5)
{pd.DataFrame(A_struct_final, columns=structural_codes, index=structural_codes).to_csv()}

SERVICEABILITY MATRIX (C6-C7)
{pd.DataFrame(A_serv_final, columns=serviceability_codes, index=serviceability_codes).to_csv()}

CONSTRUCTABILITY MATRIX (C8-C9)
{pd.DataFrame(A_constr_final, columns=constructability_codes, index=constructability_codes).to_csv()}

DURABILITY MATRIX (C10-C11)
{pd.DataFrame(A_durab_final, columns=durability_codes, index=durability_codes).to_csv()}

FINAL RANKING
{csv_ranking}
"""

st.subheader("📤 Submit Results")

GOOGLE_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSdax9QauAZsgQW6SCu5_nei4hjLC4wKWRD9HW3WaDFahJHRcQ/formResponse"
FORM_FIELDS = {'name': 'entry.1477330328', 'role': 'entry.1105829463', 'organization': 'entry.1434459689',
               'email': 'entry.1024721860', 'timestamp': 'entry.1649855533', 'results_json': 'entry.611170301',
               'ranking_csv': 'entry.2053258693', 'matrices_text': 'entry.1582487824'}

def submit():
    import requests
    data = {FORM_FIELDS['name']: participant_name, FORM_FIELDS['role']: participant_role,
            FORM_FIELDS['organization']: participant_organization, FORM_FIELDS['email']: participant_email,
            FORM_FIELDS['timestamp']: datetime.datetime.now().isoformat(),
            FORM_FIELDS['results_json']: results_json, FORM_FIELDS['ranking_csv']: csv_ranking,
            FORM_FIELDS['matrices_text']: all_matrices}
    try:
        requests.post(GOOGLE_FORM_URL, data=data, headers={'Content-Type': 'application/x-www-form-urlencoded'})
        return True, "Results submitted successfully!"
    except Exception as e:
        return False, str(e)

c1, c2 = st.columns([2, 1])
with c1:
    if st.button("📤 Submit to Database", type="primary", use_container_width=True):
        if not participant_name or not participant_email or "@" not in participant_email:
            st.error("⚠️ Please fill in your name and a valid email address")
        else:
            ok, msg = submit()
            if ok:
                st.success(f"✅ {msg}")
                st.balloons()
            else:
                st.error(f"❌ Error: {msg}")
with c2:
    if participant_name and participant_email and "@" in participant_email:
        st.success("✅ Ready to submit")
    else:
        st.warning("⚠️ Fill required fields")

st.markdown("---")
st.subheader("💾 Download Backup Copies")
c1, c2, c3 = st.columns(3)
with c1:
    st.download_button("📄 Complete Results (JSON)", results_json, f"{filename_base}.json", 
                      "application/json", use_container_width=True)
with c2:
    st.download_button("📊 Ranking Table (CSV)", csv_ranking, f"{filename_base}.csv", 
                      "text/csv", use_container_width=True)
with c3:
    st.download_button("📐 All Matrices (TXT)", all_matrices, f"{filename_base}.txt", 
                      "text/plain", use_container_width=True)


st.success("✅ Questionnaire completed! Thank you for your participation.")
st.markdown("---")
st.markdown("*AHP Questionnaire - Shear Connector Selection for Reuse-Oriented Composite Structures*")
st.markdown("*EPFL - Doctoral Research by Bachmann Lise*")
