# Cuneiform Dataset Sources & Research References

## Quick Links to Major Datasets

### 1. CDLI (Cuneiform Digital Library Initiative)
- **Website**: https://cdli.mpiwg-berlin.mpg.de/
- **Search**: https://cdli.mpiwg-berlin.mpg.de/search
- **API Documentation**: https://cdli.mpiwg-berlin.mpg.de/wiki/doku.php?id=cdli_api
- **Bulk Download**: Check their data download section
- **What to look for**: Unicode transcriptions (not just ATF transliterations)

### 2. ORACC (Open Richly Annotated Cuneiform Corpus)
- **Website**: http://oracc.museum.upenn.edu/
- **Projects List**: http://oracc.museum.upenn.edu/projectlist.html
- **Open Data**: http://oracc.museum.upenn.edu/doc/opendata/
- **Notable sub-corpora**:
  - SAAO (State Archives of Assyria Online)
  - RIAO (Royal Inscriptions of Assyria Online)
  - RINAP (Royal Inscriptions of the Neo-Assyrian Period)

### 3. Electronic Babylonian Library (eBL)
- **Website**: https://www.ebl.lmu.de/
- **About**: Curated Babylonian literature corpus
- **API**: May require registration/permission

### 4. Hethitologie Portal Mainz
- **Website**: https://www.hethport.uni-wuerzburg.de/
- **Focus**: Hittite cuneiform texts

---

## Paper Search Queries

### Google Scholar Searches
```
"cuneiform" AND "natural language processing"
"cuneiform" AND "language model"
"Akkadian" AND "computational linguistics"
"Sumerian" AND "NLP"
"ancient language" AND "neural"
"cuneiform" AND "machine learning"
"cuneiform sign" AND "prediction"
"PaleoCode" AND "cuneiform"
```

### Semantic Scholar
- https://www.semanticscholar.org/
- Search: "cuneiform computational analysis"
- Filter by: Computer Science, Linguistics

### ACL Anthology
- https://aclanthology.org/
- Search: "ancient language", "low-resource", "cuneiform"
- Check workshops: LowResourceNLP, Ancient Language Processing

---

## Relevant Research Groups

### Academic Institutions
- **LMU Munich** - Electronic Babylonian Library
- **UCLA** - Cuneiform Digital Library Initiative
- **University of Pennsylvania** - ORACC project
- **Yale University** - Babylonian Collection

### Researchers to Follow
- Look for authors publishing on:
  - Cuneiform digital humanities
  - Ancient language computational analysis
  - Low-resource NLP for historical texts

---

## Related Work to Review

### Ancient Language NLP
- **PapyGreek**: Greek papyri NLP
- **Egyptian hieroglyphics**: Any computational work
- **Latin/Ancient Greek**: Language models (Perseus Digital Library)

### Component-Based Modeling
- **Chinese NLP**: Stroke-based or radical-based models
- **Japanese Kanji**: Component decomposition
- Search: "Chinese character decomposition" + "neural"

### Low-Resource Language Modeling
- Techniques applicable to cuneiform:
  - Transfer learning
  - Multilingual models
  - Few-shot learning
  - Data augmentation

---

## Unicode Cuneiform Resources

### Unicode Standard
- **Cuneiform Block**: U+12000–U+123FF
- **Cuneiform Numbers Block**: U+12400–U+1247F
- **Document**: https://www.unicode.org/charts/PDF/U12000.pdf

### Fonts
- **Noto Sans Cuneiform**: https://fonts.google.com/noto/specimen/Noto+Sans+Cuneiform
- Free and open-source

---

## Baseline Implementations

### N-gram Models
- **KenLM**: https://github.com/kpu/kenlm
- Fast n-gram language modeling

### Pre-trained Models to Test
- **mBERT**: Multilingual BERT
- **XLM-R**: Cross-lingual RoBERTa
- (Expect poor performance, but useful baseline)

---

## Evaluation Resources

### Perplexity Calculation
- Standard metric for language modeling
- Lower is better
- Compare across tokenization schemes

### Inter-annotator Agreement
- If doing human evaluation
- Kappa statistics for agreement

---

## Contact Points

### CDLI
- General inquiries: Contact form on website
- Data access: May need to request bulk download

### ORACC
- Email maintainers for specific corpora
- Check individual project pages for contacts

### Academic Community
- Cuneiform mailing lists
- Digital Humanities forums
- Twitter: #Assyriology, #DigitalHumanities

---

## Tools & Libraries

### Text Processing
- **Pyoracc**: Python library for ORACC data
  - https://github.com/oracc/pyoracc

### Transliteration Tools
- May need to convert ATF to Unicode
- Look for existing converters or create custom

---

## Experiment Tracking Template

```markdown
## Experiment: [Name]
**Date**: YYYY-MM-DD
**Dataset**: [Name, size, source]
**Model**: [Architecture, hyperparameters]
**Tokenization**: [Sign/Stroke/Hybrid]
**Results**:
- Perplexity: X.XX
- Accuracy: XX%
**Notes**:
- [Key observations]
- [Issues encountered]
```

---

## Progress Checklist

### Datasets
- [ ] Downloaded CDLI sample
- [ ] Downloaded ORACC corpus
- [ ] Converted to Unicode (if needed)
- [ ] Created train/val/test splits
- [ ] Documented statistics

### Baselines
- [ ] N-gram model trained
- [ ] Character LSTM trained
- [ ] Transliteration baseline
- [ ] mBERT zero-shot test

### Literature Review
- [ ] Found 10+ relevant papers
- [ ] Created comparison table
- [ ] Identified research gaps

### Experiments
- [ ] Stroke tokenizer tested
- [ ] Sign tokenizer tested
- [ ] Hybrid tokenizer tested
- [ ] Results compared with baselines

---

## Next Actions

1. **Start here**: Visit CDLI and explore their search interface
2. **Search papers**: Google Scholar with queries above
3. **Join communities**: Find cuneiform/DH mailing lists
4. **Test with sample**: Use small dataset to verify pipeline works
5. **Document everything**: Keep experiment logs from day one
