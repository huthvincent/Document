# Time Series Foundation Modeling Data Preprocessing

In processing 87 million (including over 40 million real patient) massive medical records, OMNY-Twin employs a minimalist, high-density feature preprocessing and packing pipeline, completely eliminating sequence waste and state poisoning.

---

## 1. Global Vocabulary & Basic Tokens
The entire model is constrained to a streamlined vocabulary of `26,880` dimensions (aligned to multiples of 64/128 for perfect GPU Tensor Core optimization).

- **Static Dimension (Static Tokens)**: 5 tokens permanently allocated at the sequence head for static demographics like age and gender.
- **Dynamic Dimension (Clinical Tokens)**: Comprises nearly 26,868 entity medical codes, fully covering Diagnoses (Dx), Vitals & Signs (Vs), Laboratory Tests (Lb), and Prescriptions (Rx).
- **Layout Tokens (Special Tokens)**:
  - Patient Level: `<patient_start>`, `<patient_end>`
  - Encounter Level: `<enc_start>`, `<enc_end>`

---

## 2. Continuous Time Positional Encoding
We have **completely discarded** the traditional coarse-grained "discrete time placeholders" (e.g., using `<gap_31-90>` to represent an interval of 31 to 90 days), upgrading fully to **continuous numerical time feature fusion**.

1. **Extraction**: Retrieve the actual floating-point day difference between two adjacent encounters (e.g., `time_delta = 42.5` days).
2. **Encoding**: Utilize a `ContinuousTimeEncoder` (based on MLP or Sine/Cosine functions) to project `42.5` directly into a 1280-dimensional temporal feature vector.
3. **Additive Fusion**: Occupies no additional sequence position! This 1280-dimensional time feature is **directly added (Add)** to the Token Embeddings of all entity tokens within the current encounter.

> **Gain**: Massively expands the total encounters accommodated within the 4096 context window, while granting every diagnosis infinitely precise historical time-distance perception.

---

## 3. Canonical Ordering within Encounters
Facing massive un-ordered events concurrently prescribed in a single visit (Time = 0) (e.g., 3 medications and 2 lab tests on the same day), random arrangement confuses the Teacher Forcing mechanism, causing gradient collapse.

**Our Rule: Forced Structured Dimensionality Reduction**
All `Intra-Encounter Tokens` must establish rules via internal ordering during preprocessing:
1. **Major Categories Strictly Locked by Medical Logic**: Diagnoses (dx) ➔ Vitals (vs) ➔ Labs (lb) ➔ Medications (rx)
2. **Same Categories Strictly Ordered Lexicographically**: For example, when two medications are prescribed, Aspirin will always precede a medication starting with a later letter.

> **Gain**: Pacifies auto-regressive model penalties caused by "same meaning, different order," prompting the model to automatically reconstruct an extremely fluent, logical medical record writing pattern during inference.

---

## 4. Numerical Feature Tokenization
For continuous numerical features like Laboratory Tests (Labs), the model incorporates clinical Reference Ranges (High/Low Limits) for discrete mapping, rather than directly feeding floating-point values.

**Binning Logic**:
For each specific biochemical lab test, the system automatically segments its detection value into independent Tokens with semantic states:
- **Abnormally High** ➔ `<lb_{id}_high>` (e.g., `<lb_322_high>`, exceeding upper normal limit)
- **Normal Range** ➔ `<lb_{id}_norm>` (e.g., `<lb_322_norm>`, within medical safety interval)
- **Abnormally Low** ➔ `<lb_{id}_low>` (e.g., `<lb_322_low>`, dropping below lower normal limit)

> **Gain**: This forced introduction of medical common sense boundaries saves the LLM the compute waste of "guessing the meaning of floating-point numbers from scratch," directly endowing the data with ready-made "clinical judgment attributes" for immediate effective reasoning.

---

## 5. Final Data Topology Pipeline

```text
[A. Extract & Compute Continuous Days] ─────┐ 
   (e.g., 25.4 days, 112 days)              │
                                            ▼
[B. Sequence Pre-tokenization & Ordering]  [Continuous Time Encoder] (Output 1280-d feature)
           │                                │
           ▼                                ▼
   [ Token Embedding Layer ]       ⊕       (Injects time concept via additive fusion)
           │
           ▼
[C. Snake-like Sequence Packing to Fill 4096 Window]
 
 ┌─ Patient(ID: #204) ───────────────────────────────────────────────────────┐
 │ [Static_1..5]                                                             │
 │ <patient_start>                                                           │
 │   <enc_start> [dx_1, dx_2] ➔ [vs_1] ➔ [rx_1, rx_2] <enc_end>              │
 │   <enc_start> [dx_1] ➔ [lb_1, lb_2] <enc_end>  (Fused with 25.4 day feat) │
 │ <patient_end>                                                             │
 └───────────────────────────────────────────────────────────────────────────┘
                                     ▼
                                (Seamless concatenation)
                                     ▼
 ┌─ Patient(ID: #491) ───────────────────────────────────────────────────────┐
 │ [Static_1..5]                                                             │
 │ <patient_start> (Triggers Mamba state forced reset & diagonal mask isolate│
 │   <enc_start> [dx_3] ➔ [rx_3] <enc_end>                                   │
 │ ...                                                                       │
 └───────────────────────────────────────────────────────────────────────────┘
```
