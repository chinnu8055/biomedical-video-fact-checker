# CER comparison and local-ML milestone plan

## What is retained from CER

The reference [CER repository](https://github.com/PRAISELab-PicusLab/CER-Fact-Checking)
demonstrates the useful evidence→reasoning→classification pattern. Its public demo combines a
stored abstract corpus, SentenceTransformer embeddings, FAISS retrieval, Whisper, Llama reasoning,
and a transformer classifier.

The checked-in CER video path extracts audio and transcribes speech, but does not sample frames or
run OCR. Its Llama call is made through NVIDIA's hosted OpenAI-compatible endpoint, even though the
model itself is Meta Llama rather than an OpenAI model. Therefore, reproducing that exact call would
not satisfy this project's no-API requirement.

CER's experiments searched a downloaded PubMed abstract snapshot using BM25 and dense retrieval.
This project keeps CER's retrieve→rerank→verify structure but uses live NCBI E-utilities to avoid a
multi-gigabyte corpus. Inference remains local, and the main research contribution is detecting
claims visible only as text in video frames.

## Local adaptation

| Stage | Reference CER demo | This project |
|---|---|---|
| Input | Text, URL, or video | MP4 video only |
| Video speech | Whisper | Local Whisper with segment timestamps |
| Visible text | Not in video path | OpenCV plus local PaddleOCR/EasyOCR |
| Claim extraction | Hosted Llama prompt | Local DeBERTa zero-shot classification |
| Candidate evidence | Downloaded PubMed snapshot | Live NCBI PubMed ESearch and EFetch |
| Evidence ranking | BM25 or SBERT/FAISS | PubMed relevance plus local SBERT reranking |
| Verification | Llama justification plus DeBERTa | BiomedBERT fine-tuned locally on HealthFC |
| Explanation | Generated prose | Extractive top-evidence explanation |
| Runtime network | Hosted model call | NCBI PubMed only; no hosted model |
| Public surface | Streamlit multi-input demo | Streamlit video-only dashboard |

## HealthFC integration

The uploaded dataset contains the official 750 English/German health fact-check examples. The local
training pipeline uses:

- `en_claim`: claim/question input;
- `en_top_sentences`: evidence input; and
- `label`: 0 supported, 1 NEI, 2 refuted.

It deliberately excludes `en_explanation` from model input because it summarizes the expert verdict
and would leak the target label. The split is stratified 70% train, 15% validation, and 15% test.
Class-weighted loss addresses the strong NEI majority. The best epoch is chosen only by validation
macro-F1, and the test set is evaluated once after model selection.

## Stable architecture

```text
Streamlit
   ↓
BiomedicalFactCheckPipeline
   ├── FFmpegAudioExtractor → WhisperTranscriber
   ├── OpenCVFrameExtractor → PaddleOCRReader/EasyOCRReader
   ├── TimeAlignedTranscriptMerger
   ├── LocalBiomedicalClaimExtractor
   ├── PubMedAPIRetriever → local SentenceTransformer reranker
   └── HealthFCClassifierReasoner
           ↓
      AnalysisResult JSON
```

All stage dependencies are protocols in `ports.py`, so tests replace large local models with small
fakes. Training, indexing, inference, and presentation are separate responsibilities.

## Milestones

### M0 — scope and typed contracts (complete)

- Video-only application interface.
- TRUE/FALSE/NEI domain model.
- Separate speech, OCR, combined transcript, claim, evidence, reasoning, confidence, and timing.

### M1 — dual-modality ingestion (complete in code)

- FFmpeg mono audio extraction.
- Timestamped local Whisper speech recognition.
- Configurable frame sampling.
- PaddleOCR/EasyOCR adapters and persistent-caption de-duplication.

Acceptance still required: run speech-only, OCR-only, and combined real videos on the target machine.

### M2 — local ML pipeline with live scientific retrieval (complete in code)

- Removed OpenAI, Gemini, and hosted reasoning dependencies.
- Local zero-shot claim detector.
- NCBI ESearch/EFetch retriever with rate limiting, retry handling, and XML parsing.
- Local Sentence Transformer reranking of PubMed candidates.
- Local HealthFC-trained verdict classifier.
- Streamlit readiness indicators for the classifier and PubMed configuration.

Acceptance still required: download/cache models, train the classifier, and run live PubMed
retrieval on the target machine.

### M3 — HealthFC experiment (training command ready)

- Validate the supplied dataset schema and label distribution.
- Use reproducible stratified splits and class-weighted loss.
- Save the best model, split row IDs, learning history, confusion matrix, and test macro-F1.

Acceptance: run at least three seeds, report mean/standard deviation, and compare with a majority
baseline and an unfine-tuned zero-shot baseline.

### M4 — retrieval and end-to-end evaluation (next)

- Version and store every generated PubMed query plus returned PMIDs for reproducibility.
- Measure PubMed candidate recall@20 and locally reranked recall@3.
- Annotate a video set containing speech-only, OCR-only, duplicate, complementary, and conflicting
  claims.
- Report WER, OCR CER, claim precision/recall, verdict macro-F1, calibration, latency, and memory.
- Run speech-only/OCR-only/combined ablations.

### M5 — final demonstration hardening (after evaluation)

- Cache every model before the demonstration.
- Pin tested CPU/GPU builds and record hardware.
- Add representative test videos and expected outputs.
- Document temporary-file deletion and video privacy limitations.
- Create a presentation-ready error analysis with retrieval, OCR, and classifier failures separated.

## Research risks

- **Domain shift:** HealthFC evidence passages are not identical to PubMed abstracts.
- **Question wording:** HealthFC inputs are often yes/no research questions, while detected video
  claims are declarative. Include this distinction in error analysis.
- **Retrieval coverage:** PubMed lexical search may miss evidence for vague consumer language.
- **Reproducibility:** live search rankings can change as PubMed is updated.
- **Availability:** analysis requires internet access to NCBI E-utilities.
- **OCR errors:** decimal points and medicine names are especially consequential.
- **Confidence interpretation:** softmax confidence is not clinical certainty.
- **Explanation fidelity:** the local explanation is extractive and must not be presented as an
  expert-generated clinical justification.
