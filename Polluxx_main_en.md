# OMNY Time Series Foundation Modeling
*EHR Foundation Model Engineering Architecture*

---

## Positioning & Scale
- **Type**: EHR Longitudinal Time-Series Prediction Foundation Model
- **Goal**: Model long-term health trajectories ➔ Predict future clinical events
- **Scale**: 1.38B Total Params | 322M Active Params (Forward Pass)
  - *Activation Rate 23.3% ➔ 4.3× Parameter Efficiency*
- **Data**: 63M Pre-tokenized Medical Records (Max Len: 4096)

---

## Core Architecture: [ Mamba × Attention × MoE ]
**Topology**: Hybrid Decoder-only | 16 Layers Total | **[M, M, M, A] × 4** Loop

### Mamba State Space (Backbone 75%)
- **Function**: Capture cross-month/year longitudinal **long-term temporal dependencies**

### Attention Mechanism (Local 25%)
- **Function**: Handle complex intra-visit event interactions

### Sparse MoE (Expert Network)
- **Mechanism**: 8 Experts Architecture | Top-2 Dynamic Routing (25% activation)
- **Function**: Automatically activate specialized knowledge domains (Cardiovascular/Neurology, etc.) based on patient features | 4× capacity expansion with zero compute overhead

---

## Minimalist Data Flow & Topology

### 1. Data Pipeline (Tokenization & Binning)
```text
[ Raw EHR Records ]
       │
       ▼
  Preprocessing & Mapping
 ├─ Static Features (Age/Gender) ───▶ [ Static Tokens (5) ]
 ├─ Continuous Time Feature (Days) ─▶ [ ContinuousTimeEncoder (1280-d vector) ]
 └─ Diagnoses/Vitals/Labs/Meds ─────▶ [ Clinical Tokens (Strict pre-sorting) ]
       │
       ▼
  Sequence Packing & Feature Fusion
 [ Patient A: Static + <start> + [Enc]* + <end> ] + [ Patient B: ... ]
 (Note: Continuous time features are directly added to Clinical Embedding, occupying no separate sequence slots)
```

### 2. Model Processing Pipeline
```text
[ Input: Static + <start> + [Encounter]* + <end> ]
                            │
                            ▼
 [ Token Embeddings ]  ⊕  [ Continuous Time Encoder ] (1280-d Feature Fusion)
                            │
  ┌─────────────────────────┼─────────────────────────┐
  │ 🔄 Loop 4 Times         ▼                         │
  │     ╔═════════════════════════════════════╗       │
  │     ║ 🟢 Mamba Block 1 (Linear extraction)║       │
  │     ║ 🟢 Mamba Block 2 (State evolution)  ║       │
  │     ║ 🟢 Mamba Block 3 (Longitudinal mem) ║       │
  │     ╚═══════════════════╤═════════════════╝       │
  │                         ▼                         │
  │     ╔═════════════════════════════════════╗       │
  │     ║ 🟠 Attention + Sparse MoE Block     ║       │
  │     ║  ├─ Multi-Head Attn (Local context) ║       │
  │     ║  └─ MoE Router (Select 2 of 8)      ║       │
  │     ╚═════════════════════════════════════╝       │
  └─────────────────────────┼─────────────────────────┘
                            ▼
                  [ Final RMS Norm ]
                            │
             [ LM Head (Tied with Embedding) ]
                            ▼
                   [ Predict Next Token ]
```

---

## Hardware & Engineering Optimization
- **Tensor Alignment (Tensor Core)**: Vocabulary limits padded to `26,880` (multiple of 64/128) ➔ GPU throughput ↑10-15%
- **VRAM Squeezing**: Enabled Gradient Checkpointing ➔ VRAM usage ↓73% (~12GB/GPU)
- **Numerical Stability**: RMSNorm (Pre-Norm) + bfloat16 ➔ Eliminated mean calculation, preventing gradient overflow
- **Tied Weights**: LM Head shares Embedding weights ➔ Natively unifies input/output spaces, saves 34M parameters
