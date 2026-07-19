# Turkish POS Tagger using Hidden Markov Models (HMM)

An elegant, dependency-free Part-of-Speech (POS) tagger for the Turkish language implemented completely **from scratch**. This project models natural language sequences using a generative **Hidden Markov Model (HMM)** and decodes optimal state sequences via the dynamic programming-based **Viterbi Algorithm**.

---

## Key Highlights & Features
* **Built From Scratch:** Pure Python/NumPy implementation of probability estimation, smoothing matrices, and trellis decoding without relying on sequence-labeling libraries (e.g., NLTK, spaCy).
* **Linguistic Focus:** Specifically designed and trained on morphologically complex, agglutinative Turkish corpora.
* **Underflow Prevention:** All mathematical operations are calculated in log-space to mitigate floating-point numeric underflow.
* **Granularity Trade-off Analysis:** Features benchmarking between a highly granular 14-tag system and a simplified 5-tag core syntactic category system.

---

## Dataset & Setup
The models are trained and evaluated on the **Google Turkish Treebank** (merging the `web.conllu` and `wiki.conllu` corpora)[cite: 1]:
* **Total Dataset Size:** 4,851 sentences[cite: 1]
* **Train / Test Split:** 80% Train (3,880 sentences) / 20% Test (971 sentences) using a robust reproducible seed (`random_state=42`)[cite: 1].
* **Boundary Markers:** Automatic ingestion of `<START>` and `<END>` tokens to correctly model initial state distributions and sequence terminations[cite: 1].

---

## Pipeline Architecture

### 1. Mathematical Probability Estimation (MLE)
The system trains parameters directly from the text corpus using **Maximum Likelihood Estimation (MLE)**:

* **Transition Probabilities:** Calculates the likelihood of moving from one hidden grammatical state to another:
  $$P(\text{tag}_i \mid \text{tag}_{i-1}) = \frac{\text{Count}(\text{tag}_{i-1}, \text{tag}_i)}{\text{Count}(\text{tag}_{i-1})}$$

* **Emission Probabilities:** Calculates the likelihood of a specific POS tag generating a given word:
  $$P(\text{word}_j \mid \text{tag}_i) = \frac{\text{Count}(\text{tag}_i, \text{word}_j)}{\text{Count}(\text{tag}_i)}$$

### 2. Numerical Stability & Laplace Smoothing
* **Add-k Smoothing:** To seamlessly handle Out-of-Vocabulary (OOV) words and unseen sequence transitions during testing, **Laplace smoothing** ($\alpha = 0.001$) is injected into calculations[cite: 1].
* **Log-Space Trellis:** Instead of vulnerable fractional multiplications ($A \times B \times C$), the execution paths are completely calculated using addition operations:
  $$\log(A \times B) = \log(A) + \log(B)$$

### 3. Viterbi Decoding
The structural search for the optimal hidden state sequence is solved using a dynamic programming trellis of size $(N + 2) \times (T + 2)$, where $N$ is the number of unique tags and $T$ is the sentence sequence duration[cite: 1].

---

## Experimental Framework & Results

We benchmarked the architecture across two primary structural layouts:
1. **Model A (Granular):** Employs all 14 baseline treebank tags (`ADJ`, `ADP`, `ADV`, `NOUN`, `VERB`, etc.)[cite: 1].
2. **Model B (Simplified):** Restricts tracking to 5 core syntactic categories (`ADJ`, `ADV`, `NOUN`, `VERB`, `PUNCT`)[cite: 1].

### Performance Summary
| Layout Configuration | Unique States | Accuracy | Weighted $F_1$-Score |
| :--- | :---: | :---: | :---: |
| **Model A (14 Tags)** | 14 | **83.40%** | **85.29%** |
| **Model B (5 Tags)** | 5 | **86.46%** | **87.47%** |

### Key Evaluation Takeaways
* **The Granularity Trade-off:** Dropping minor morphological variants (Model B) naturally reduces state-transition complexity, leading to an **~3.06% increase in tagging accuracy**.
* **Noun/Verb Confusion:** Error metrics show that standard misclassifications typically cluster around ambiguous Turkish root words that fluctuate between nominal and verbal characteristics depending on active suffixes.

---

## Architectural Limitations
* **Markovian Constraint:** Being a Bigram HMM, the state depends strictly on the immediate predecessor $P(\text{tag}_i \mid \text{tag}_{i-1})$, which fails to encapsulate complex long-range sentence framing typical of Turkish.
* **Surface-Level OOV Logic:** Unknown terms are generic targets smoothed globally; the system does not look at sub-word suffix formations (e.g., *-lar*, *-mak*) to make a structural deduction.

---

## Project Roadmap
Future updates intended to bypass sequence boundaries and scale performance:
- [ ] **Trigram Transition Matrix:** Transitioning to a historical sequence framework: $P(\text{tag}_i \mid \text{tag}_{i-1}, \text{tag}_{i-2})$.
- [ ] - [ ] **Kneser-Ney Smoothing:** Integrating advanced back-off smoothing for less frequent n-grams.
- [ ] **Morphological Token Pre-Parsing:** Incorporating morphological analyzers (e.g., *Zemberek*) to map agglutinative boundaries before training.
- [ ] **Discriminative Transition (CRF):** Elevating the project to a Conditional Random Field (CRF) layout to support rich arbitrary feature mappings.

---

## 📂 File Layout
```text
├── web.conllu                 # Web corpus from Google Turkish Treebank
├── wiki.conllu                # Wikipedia corpus from Google Turkish Treebank
├── turkish_hmm_pos_tagger.ipynb # Core notebook featuring code and evaluation
└── LICENSE                    # MIT License
