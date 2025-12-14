# Investigation: Content Loss in Long Documents

## Problem Description
When processing long documents (e.g., 13-page PDFs), only partial content is retained in the final output. The 13-page solar energy paper produced only ~67 lines of output, missing the title, abstract, introduction, and early methodology sections.

## Root Cause Analysis

### Pipeline Architecture
The processing pipeline has 4 stages:

1. **Stage 1: Core Text Transformation** (src/pdf_to_audio/core.py:115-177)
   - ✅ Processes in CHUNKS (pages → sub-chunks → API calls)
   - ✅ Accumulates all responses
   - ✅ No content loss

2. **Stage 2: Math Expression Handling** (src/pdf_to_audio/core.py:179-192)
   - ❌ Sends ENTIRE document in ONE API call
   - ❌ Limited by `max_tokens=4000` output limit
   - ❌ LLM truncates content when output exceeds limit

3. **Stage 3: Citations Optimization** (src/pdf_to_audio/core.py:193-206)
   - ❌ Sends ENTIRE document in ONE API call
   - ❌ Limited by `max_tokens=4000` output limit
   - ❌ Further truncation of already-truncated content

4. **Stage 4: Language/Style Refinement** (src/pdf_to_audio/core.py:207-220)
   - ❌ Sends ENTIRE document in ONE API call
   - ❌ Limited by `max_tokens=4000` output limit
   - ❌ Final truncation

### The Critical Issue

**Token Limit Constraint:**
- All LLM providers are initialized with `max_tokens=4000` (core.py:88, 96, 104, 112)
- This is defined in constants.py:4 as `MAX_TOKENS = 4000`

**For a 13-page document:**
- Estimated OCR output: ~20,000-30,000 characters = ~5,000-7,500 tokens
- Stage 1 output (after core transform): Similar size
- Stage 2 input: 5,000-7,500 tokens
- Stage 2 max output: **4,000 tokens** ← TRUNCATION OCCURS HERE
- Stages 3 & 4: Further potential truncation

**Why Stage 1 works fine:**
```python
# Stage 1 splits into chunks and processes each separately
for i in range(0, len(pages), pages_per_chunk):
    chunk_pages = pages[i:i + pages_per_chunk]
    sub_chunks = split_chunk(chunk_content)
    for sub_chunk in sub_chunks:
        response = make_api_call(...)  # Each call returns full content
        transformed_text += response   # Accumulate all responses
```

**Why Stages 2-4 fail:**
```python
# Stages 2-4 send everything at once
messages = [
    {"role": "system", "content": MATH_PROMPT},
    {"role": "user", "content": core_transformed_text},  # ENTIRE DOCUMENT
]
math_processed_text = make_api_call(...)  # Output limited to 4000 tokens
```

## Evidence

1. **File sizes:**
   - examples/solar-02-00026-v4.pdf: 13 pages, 2,337,750 bytes
   - examples/solar-02-0026-v4.txt: 67 lines (only sections 2.4 onwards)

2. **Code locations:**
   - src/pdf_to_audio/core.py:184 - Math stage sends full document
   - src/pdf_to_audio/core.py:198 - Citations stage sends full document
   - src/pdf_to_audio/core.py:212 - Language stage sends full document
   - src/pdf_to_audio/constants.py:4 - MAX_TOKENS = 4000

## Solution Design

### Option 1: Increase max_tokens (NOT RECOMMENDED)
- Pros: Simple one-line change
- Cons:
  - Still hits limits on very long documents
  - Higher API costs
  - Slower response times
  - Doesn't scale

### Option 2: Implement Chunking for Stages 2-4 (RECOMMENDED)
- Pros:
  - Scales to documents of any length
  - Consistent with Stage 1 architecture
  - Maintains quality across all content
  - No additional API costs per document
- Cons:
  - More complex implementation
  - Need to handle chunk boundaries carefully

## Recommended Fix

Implement intelligent chunking for stages 2-4:

1. **For Math Processing (Stage 2):**
   - Split document into chunks of ~3000 tokens each
   - Process each chunk independently
   - Concatenate results

2. **For Citations Optimization (Stage 3):**
   - Use similar chunking strategy
   - Ensure citation references aren't split across chunks

3. **For Language/Style Refinement (Stage 4):**
   - Use paragraph-aware chunking
   - Avoid splitting sentences across chunks

4. **Implementation approach:**
   - Create a reusable `process_in_chunks()` function
   - Use the existing `split_chunk()` utility from utils.py
   - Add overlap between chunks to preserve context

## Test Plan

1. Run test_each_stage.py to confirm the issue
2. Implement chunking for stages 2-4
3. Test with the solar-02-00026-v4.pdf document
4. Verify 100% content preservation
5. Test with varying document lengths (1 page, 5 pages, 20+ pages)
6. Ensure mathematical content preservation remains at 100%

## Files to Modify

- src/pdf_to_audio/core.py (main implementation)
- src/pdf_to_audio/constants.py (potentially adjust MAX_TOKENS)
- tests/ (add regression test for long documents)
