# Research Roadmap for Cuneiform NLP

## Overview
This document outlines the research tasks needed to establish baselines, find datasets, and identify comparison points for cuneiform NLP using PaleoCode encoding.

---

## 1. Cuneiform Datasets

### 1.1 Major Cuneiform Corpora

**Digital Libraries & Archives:**
- [ ] **CDLI (Cuneiform Digital Library Initiative)** - https://cdli.mpiwg-berlin.mpg.de/
  - 340,000+ cuneiform artifacts
  - Transliterations and some Unicode text
  - Download via API or bulk data

- [ ] **ORACC (Open Richly Annotated Cuneiform Corpus)** - http://oracc.museum.upenn.edu/
  - Multiple sub-corpora (Akkadian, Sumerian)
  - Lemmatized and annotated texts
  - ATF format with transliterations

- [ ] **eBL (electronic Babylonian Library)** - https://www.ebl.lmu.de/
  - Curated Babylonian texts
  - High-quality annotations
  - API access available

- [ ] **Hethitologie Portal Mainz** - https://www.hethport.uni-wuerzburg.de/
  - Hittite cuneiform texts
  - Specialized corpus

**Unicode Cuneiform Texts:**
- [ ] Check if CDLI provides Unicode cuneiform (not just transliterations)
- [ ] Look for digitized tablet images with Unicode transcriptions
- [ ] Contact cuneiform scholars for Unicode datasets

### 1.2 Dataset Preparation Tasks

- [ ] Convert transliterated texts to Unicode cuneiform (if necessary)
- [ ] Clean and normalize texts (remove damaged/uncertain signs)
- [ ] Create train/val/test splits by period, genre, or random
- [ ] Document dataset statistics:
  - Total signs/tokens
  - Vocabulary size
  - Text lengths
  - Genre distribution
  - Time period distribution

---

## 2. Baseline Models

### 2.1 Character-Level Baselines

- [ ] **N-gram Language Models**
  - Implement trigram/5-gram models at sign level
  - Calculate perplexity on test set
  - Use as simple baseline

- [ ] **Character-Level LSTM**
  - Train vanilla LSTM on sign sequences
  - Compare with stroke-level approach

### 2.2 Transliteration-Based Baselines

- [ ] Train models on transliterated text (Latin characters)
- [ ] Compare performance: Unicode cuneiform vs. transliteration
- [ ] Document information loss in transliteration

### 2.3 Standard NLP Baselines

- [ ] **Byte-Pair Encoding (BPE)**
  - Apply standard BPE to Unicode cuneiform
  - Compare with PaleoCode tokenization

- [ ] **Pre-trained Models**
  - Test multilingual models (mBERT, XLM-R) on cuneiform
  - Likely poor performance, but useful baseline

---

## 3. Relevant Papers & Prior Work

### 3.1 Cuneiform NLP Papers

**Search for:**
- [ ] "cuneiform natural language processing"
- [ ] "akkadian language model"
- [ ] "sumerian computational linguistics"
- [ ] "ancient language NLP"
- [ ] "low-resource language modeling"

**Venues to check:**
- ACL Anthology (ACL, EMNLP, NAACL)
- LREC (Language Resources and Evaluation)
- CHR (Computational Humanities Research)
- DH (Digital Humanities conferences)
- Journal of Cuneiform Studies

**Specific papers to find:**
- [ ] Any work on computational analysis of cuneiform
- [ ] Sign prediction or language modeling for cuneiform
- [ ] Machine translation for Akkadian/Sumerian
- [ ] Optical character recognition for cuneiform tablets

### 3.2 Related Work in Ancient Languages

- [ ] Egyptian hieroglyphics NLP
- [ ] Ancient Greek/Latin language modeling
- [ ] Dead language revival with NLP
- [ ] Low-resource language techniques applicable to cuneiform

### 3.3 Stroke/Component-Based Approaches

- [ ] Chinese character decomposition and stroke modeling
- [ ] Hierarchical tokenization for logographic scripts
- [ ] Component-based representation learning
- [ ] Structure-aware language models

### 3.4 PaleoCode System

- [ ] Find original PaleoCode publication/documentation
- [ ] Understand stroke encoding methodology
- [ ] Cite properly in papers

---

## 4. Evaluation Metrics & Tasks

### 4.1 Define Tasks

**Language Modeling:**
- [ ] Next-sign prediction (perplexity)
- [ ] Masked sign prediction (BERT-style)
- [ ] Text generation quality (if possible with expert evaluation)

**Classification:**
- [ ] Genre classification (contracts, letters, omens, etc.)
- [ ] Time period classification (Old Babylonian, Neo-Assyrian, etc.)
- [ ] Language identification (Akkadian vs. Sumerian)

**Sign-Level Tasks:**
- [ ] Sign frequency prediction
- [ ] Damaged sign restoration
- [ ] Sign variant identification

### 4.2 Metrics

**Quantitative:**
- [ ] Perplexity
- [ ] Accuracy (for classification)
- [ ] F1 score (for multi-class tasks)
- [ ] BLEU score (if doing generation/restoration)

**Qualitative:**
- [ ] Expert evaluation of generated text
- [ ] Analysis of learned sign representations
- [ ] Visualization of stroke embeddings

---

## 5. Experimental Design

### 5.1 Ablation Studies

- [ ] Sign-level vs. stroke-level vs. hybrid tokenization
- [ ] Effect of PaleoCode vs. simple Unicode encoding
- [ ] Model architecture comparison (Transformer vs. RNN)
- [ ] Impact of pre-training on larger corpora

### 5.2 Analysis

- [ ] What patterns do models learn about sign composition?
- [ ] Do stroke embeddings capture linguistic structure?
- [ ] Can models predict damaged or missing signs?
- [ ] How does performance vary by time period/genre?

---

## 6. Collaboration & Resources

### 6.1 Connect with Experts

- [ ] Contact cuneiformists/Assyriologists for:
  - Dataset recommendations
  - Evaluation of model outputs
  - Domain knowledge

- [ ] Join relevant mailing lists:
  - Cuneiform Digital Library Initiative
  - Ancient language NLP groups
  - Digital Humanities forums

### 6.2 Computational Resources

- [ ] Estimate computational requirements
- [ ] Identify GPU/cluster access for training
- [ ] Plan experiment timeline based on resources

---

## 7. Publication Plan

### 7.1 Contributions to Highlight

- [ ] First work on PaleoCode-based NLP (likely)
- [ ] Comparison of sign-level vs. stroke-level modeling
- [ ] New benchmark for cuneiform language modeling
- [ ] Analysis of compositional structure learning

### 7.2 Target Venues

**Primary:**
- ACL workshops (e.g., Ancient Language Processing)
- LREC (Language Resources and Evaluation)
- Digital Humanities conferences

**Secondary:**
- Journal of Cuneiform Studies (domain-specific)
- Computational Linguistics journal

---

## 8. Immediate Next Steps

### This Week:
1. [ ] Search CDLI for Unicode cuneiform datasets
2. [ ] Download sample corpus (even if small)
3. [ ] Run preprocessing script on sample data
4. [ ] Train simple baseline (n-gram or small LSTM)
5. [ ] Search Google Scholar for "cuneiform NLP" papers

### This Month:
1. [ ] Secure main dataset(s)
2. [ ] Implement all baseline models
3. [ ] Run first experiments with PaleoCode tokenization
4. [ ] Document results in experiment logs
5. [ ] Create comparison table: baselines vs. PaleoCode models

---

## Notes & Resources

### Useful Links
- CDLI API: https://cdli.mpiwg-berlin.mpg.de/wiki/doku.php?id=cdli_api
- ORACC JSON: http://oracc.museum.upenn.edu/doc/opendata/
- Unicode Cuneiform Block: U+12000 to U+123FF, U+12400 to U+1247F

### Key Questions
1. How much Unicode cuneiform text is available (vs. transliterations)?
2. What is the state-of-the-art for cuneiform NLP (if any)?
3. Can we create a standard benchmark for future work?
4. How do we evaluate text generation quality without expert knowledge?

---

## Progress Tracking

Update this section as tasks are completed:

**Datasets Found:**
-

**Baselines Implemented:**
-

**Papers Reviewed:**
-

**Key Insights:**
-
