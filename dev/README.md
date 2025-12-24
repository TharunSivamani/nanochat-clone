# NanoChat Tokenizer Training & Evaluation

This document describes a full end-to-end test run of the NanoChat tokenizer pipeline: dataset preparation, tokenizer training, and tokenizer evaluation against GPT-2 and GPT-4.

The tokenizer is trained using a **custom Rust BPE implementation (`rustbpe`)** and wrapped with **tiktoken-compatible encoding** for downstream model use.

---

## Overview

The tokenizer pipeline consists of three main stages:

1. **Dataset preparation**
   Download and iterate over pretraining parquet shards.

2. **Tokenizer training**
   Train a Byte Pair Encoding (BPE) tokenizer on up to 1B characters using a Rust backend.

3. **Tokenizer evaluation**
   Compare token efficiency against GPT-2 and GPT-4 across multiple text domains.

---

## Environment Setup

```bash
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"

source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

* Uses a Rust extension module built via **maturin + pyo3**
* Installs the tokenizer backend in editable mode
* All tokenizer artifacts are stored under `$NANOCHAT_BASE_DIR`

---

## Dataset Preparation

```bash
python -m nanochat.dataset -n 4
```

* Downloads 4 parquet shards into:

  ```
  ~/.cache/nanochat/base_data/
  ```
* Skips already-downloaded shards
* Used as input for tokenizer training

---

## Tokenizer Training

### Command

```bash
python -m scripts.tok_train --max_chars=1000000000
```

### Configuration

* **Maximum characters:** 1,000,000,000
* **Document cap:** 10,000
* **Vocabulary size:** 65,536
* **Training backend:** Rust (incremental BPE)
* **Special tokens:** registered at encoding creation time

### Training Output (verbatim)

```
max_chars: 1,000,000,000
doc_cap: 10,000
vocab_size: 65,536
2025-12-24 07:45:06,083 - rustbpe - INFO - Processing sequences from iterator (buffer_size: 8192)
2025-12-24 07:45:27,016 - rustbpe - INFO - Processed 159744 sequences total, 1052765 unique
2025-12-24 07:45:27,082 - rustbpe - INFO - Starting BPE training: 65271 merges to compute
2025-12-24 07:45:27,082 - rustbpe - INFO - Computing initial pair counts from 1052765 unique sequences
2025-12-24 07:45:29,219 - rustbpe - INFO - Building heap with 15656 unique pairs
2025-12-24 07:45:29,221 - rustbpe - INFO - Starting merge loop
2025-12-24 07:45:30,838 - rustbpe - INFO - Progress: 1% (653/65271 merges) - Last merge: (585, 268) -> 908 (frequency: 76699)
2025-12-24 07:45:31,072 - rustbpe - INFO - Progress: 2% (1306/65271 merges) - Last merge: (304, 317) -> 1561 (frequency: 33101)
2025-12-24 07:45:31,211 - rustbpe - INFO - Progress: 3% (1959/65271 merges) - Last merge: (964, 1445) -> 2214 (frequency: 20182)
2025-12-24 07:45:31,334 - rustbpe - INFO - Progress: 4% (2611/65271 merges) - Last merge: (423, 97) -> 2866 (frequency: 13952)
2025-12-24 07:45:31,435 - rustbpe - INFO - Progress: 5% (3264/65271 merges) - Last merge: (277, 1322) -> 3519 (frequency: 10477)
2025-12-24 07:45:31,513 - rustbpe - INFO - Progress: 6% (3917/65271 merges) - Last merge: (2427, 280) -> 4172 (frequency: 8205)
2025-12-24 07:45:31,577 - rustbpe - INFO - Progress: 7% (4569/65271 merges) - Last merge: (3905, 493) -> 4824 (frequency: 6646)
2025-12-24 07:45:31,656 - rustbpe - INFO - Progress: 8% (5222/65271 merges) - Last merge: (725, 101) -> 5477 (frequency: 5544)
2025-12-24 07:45:31,710 - rustbpe - INFO - Progress: 9% (5875/65271 merges) - Last merge: (1165, 910) -> 6130 (frequency: 4681)
2025-12-24 07:45:31,769 - rustbpe - INFO - Progress: 10% (6528/65271 merges) - Last merge: (5910, 1686) -> 6783 (frequency: 4015)
2025-12-24 07:45:31,816 - rustbpe - INFO - Progress: 11% (7180/65271 merges) - Last merge: (5127, 274) -> 7435 (frequency: 3493)
2025-12-24 07:45:31,858 - rustbpe - INFO - Progress: 12% (7833/65271 merges) - Last merge: (4637, 390) -> 8088 (frequency: 3054)
2025-12-24 07:45:31,898 - rustbpe - INFO - Progress: 13% (8486/65271 merges) - Last merge: (2389, 1780) -> 8741 (frequency: 2698)
2025-12-24 07:45:31,966 - rustbpe - INFO - Progress: 14% (9138/65271 merges) - Last merge: (334, 522) -> 9393 (frequency: 2406)
2025-12-24 07:45:32,007 - rustbpe - INFO - Progress: 15% (9791/65271 merges) - Last merge: (2767, 1333) -> 10046 (frequency: 2164)
2025-12-24 07:45:32,044 - rustbpe - INFO - Progress: 16% (10444/65271 merges) - Last merge: (67, 78) -> 10699 (frequency: 1957)
2025-12-24 07:45:32,080 - rustbpe - INFO - Progress: 17% (11097/65271 merges) - Last merge: (2293, 3676) -> 11352 (frequency: 1774)
2025-12-24 07:45:32,130 - rustbpe - INFO - Progress: 18% (11749/65271 merges) - Last merge: (776, 1360) -> 12004 (frequency: 1619)
2025-12-24 07:45:32,158 - rustbpe - INFO - Progress: 19% (12402/65271 merges) - Last merge: (1308, 2271) -> 12657 (frequency: 1492)
2025-12-24 07:45:32,187 - rustbpe - INFO - Progress: 20% (13055/65271 merges) - Last merge: (528, 973) -> 13310 (frequency: 1378)
2025-12-24 07:45:32,221 - rustbpe - INFO - Progress: 21% (13707/65271 merges) - Last merge: (410, 2857) -> 13962 (frequency: 1285)
2025-12-24 07:45:32,259 - rustbpe - INFO - Progress: 22% (14360/65271 merges) - Last merge: (1112, 480) -> 14615 (frequency: 1191)
2025-12-24 07:45:32,282 - rustbpe - INFO - Progress: 23% (15013/65271 merges) - Last merge: (3445, 2083) -> 15268 (frequency: 1113)
2025-12-24 07:45:32,314 - rustbpe - INFO - Progress: 24% (15666/65271 merges) - Last merge: (1476, 926) -> 15921 (frequency: 1040)
2025-12-24 07:45:32,349 - rustbpe - INFO - Progress: 25% (16318/65271 merges) - Last merge: (2290, 2290) -> 16573 (frequency: 972)
2025-12-24 07:45:32,383 - rustbpe - INFO - Progress: 26% (16971/65271 merges) - Last merge: (10185, 368) -> 17226 (frequency: 911)
2025-12-24 07:45:32,404 - rustbpe - INFO - Progress: 27% (17624/65271 merges) - Last merge: (3107, 16700) -> 17879 (frequency: 860)
2025-12-24 07:45:32,429 - rustbpe - INFO - Progress: 28% (18276/65271 merges) - Last merge: (7518, 112) -> 18531 (frequency: 811)
2025-12-24 07:45:32,451 - rustbpe - INFO - Progress: 29% (18929/65271 merges) - Last merge: (13105, 85) -> 19184 (frequency: 765)
2025-12-24 07:45:32,476 - rustbpe - INFO - Progress: 30% (19582/65271 merges) - Last merge: (753, 341) -> 19837 (frequency: 723)
2025-12-24 07:45:32,495 - rustbpe - INFO - Progress: 31% (20235/65271 merges) - Last merge: (1771, 889) -> 20490 (frequency: 685)
2025-12-24 07:45:32,512 - rustbpe - INFO - Progress: 32% (20887/65271 merges) - Last merge: (1518, 282) -> 21142 (frequency: 648)
2025-12-24 07:45:32,533 - rustbpe - INFO - Progress: 33% (21540/65271 merges) - Last merge: (5720, 97) -> 21795 (frequency: 615)
2025-12-24 07:45:32,552 - rustbpe - INFO - Progress: 34% (22193/65271 merges) - Last merge: (1057, 794) -> 22448 (frequency: 583)
2025-12-24 07:45:32,571 - rustbpe - INFO - Progress: 35% (22845/65271 merges) - Last merge: (12147, 21803) -> 23100 (frequency: 556)
2025-12-24 07:45:32,593 - rustbpe - INFO - Progress: 36% (23498/65271 merges) - Last merge: (441, 513) -> 23753 (frequency: 529)
2025-12-24 07:45:32,613 - rustbpe - INFO - Progress: 37% (24151/65271 merges) - Last merge: (10015, 115) -> 24406 (frequency: 505)
2025-12-24 07:45:32,629 - rustbpe - INFO - Progress: 38% (24803/65271 merges) - Last merge: (8950, 105) -> 25058 (frequency: 481)
2025-12-24 07:45:32,648 - rustbpe - INFO - Progress: 39% (25456/65271 merges) - Last merge: (1036, 21160) -> 25711 (frequency: 460)
2025-12-24 07:45:32,666 - rustbpe - INFO - Progress: 40% (26109/65271 merges) - Last merge: (2364, 426) -> 26364 (frequency: 441)
2025-12-24 07:45:32,682 - rustbpe - INFO - Progress: 41% (26762/65271 merges) - Last merge: (441, 21354) -> 27017 (frequency: 422)
2025-12-24 07:45:32,699 - rustbpe - INFO - Progress: 42% (27414/65271 merges) - Last merge: (273, 709) -> 27669 (frequency: 404)
2025-12-24 07:45:32,731 - rustbpe - INFO - Progress: 43% (28067/65271 merges) - Last merge: (6037, 282) -> 28322 (frequency: 388)
2025-12-24 07:45:32,748 - rustbpe - INFO - Progress: 44% (28720/65271 merges) - Last merge: (419, 28734) -> 28975 (frequency: 372)
2025-12-24 07:45:32,760 - rustbpe - INFO - Progress: 45% (29372/65271 merges) - Last merge: (73, 75) -> 29627 (frequency: 358)
2025-12-24 07:45:32,775 - rustbpe - INFO - Progress: 46% (30025/65271 merges) - Last merge: (440, 378) -> 30280 (frequency: 345)
2025-12-24 07:45:32,787 - rustbpe - INFO - Progress: 47% (30678/65271 merges) - Last merge: (336, 600) -> 30933 (frequency: 332)
2025-12-24 07:45:32,803 - rustbpe - INFO - Progress: 48% (31331/65271 merges) - Last merge: (10680, 2822) -> 31586 (frequency: 320)
2025-12-24 07:45:32,818 - rustbpe - INFO - Progress: 49% (31983/65271 merges) - Last merge: (408, 5504) -> 32238 (frequency: 308)
2025-12-24 07:45:32,832 - rustbpe - INFO - Progress: 50% (32636/65271 merges) - Last merge: (301, 471) -> 32891 (frequency: 297)
2025-12-24 07:45:32,849 - rustbpe - INFO - Progress: 51% (33289/65271 merges) - Last merge: (408, 1266) -> 33544 (frequency: 287)
2025-12-24 07:45:32,863 - rustbpe - INFO - Progress: 52% (33941/65271 merges) - Last merge: (2824, 1391) -> 34196 (frequency: 278)
2025-12-24 07:45:32,880 - rustbpe - INFO - Progress: 53% (34594/65271 merges) - Last merge: (351, 21141) -> 34849 (frequency: 269)
2025-12-24 07:45:32,891 - rustbpe - INFO - Progress: 54% (35247/65271 merges) - Last merge: (2565, 760) -> 35502 (frequency: 261)
2025-12-24 07:45:32,902 - rustbpe - INFO - Progress: 55% (35900/65271 merges) - Last merge: (497, 31625) -> 36155 (frequency: 252)
2025-12-24 07:45:32,925 - rustbpe - INFO - Progress: 56% (36552/65271 merges) - Last merge: (377, 13462) -> 36807 (frequency: 244)
2025-12-24 07:45:32,935 - rustbpe - INFO - Progress: 57% (37205/65271 merges) - Last merge: (70, 369) -> 37460 (frequency: 236)
2025-12-24 07:45:32,947 - rustbpe - INFO - Progress: 58% (37858/65271 merges) - Last merge: (697, 22905) -> 38113 (frequency: 229)
2025-12-24 07:45:32,962 - rustbpe - INFO - Progress: 59% (38510/65271 merges) - Last merge: (1307, 756) -> 38765 (frequency: 222)
2025-12-24 07:45:32,973 - rustbpe - INFO - Progress: 60% (39163/65271 merges) - Last merge: (295, 1092) -> 39418 (frequency: 215)
2025-12-24 07:45:32,983 - rustbpe - INFO - Progress: 61% (39816/65271 merges) - Last merge: (109, 1071) -> 40071 (frequency: 209)
2025-12-24 07:45:32,991 - rustbpe - INFO - Progress: 62% (40469/65271 merges) - Last merge: (36236, 6977) -> 40724 (frequency: 204)
2025-12-24 07:45:33,004 - rustbpe - INFO - Progress: 63% (41121/65271 merges) - Last merge: (265, 3604) -> 41376 (frequency: 197)
2025-12-24 07:45:33,014 - rustbpe - INFO - Progress: 64% (41774/65271 merges) - Last merge: (1079, 121) -> 42029 (frequency: 192)
2025-12-24 07:45:33,024 - rustbpe - INFO - Progress: 65% (42427/65271 merges) - Last merge: (74, 3677) -> 42682 (frequency: 186)
2025-12-24 07:45:33,033 - rustbpe - INFO - Progress: 66% (43079/65271 merges) - Last merge: (116, 519) -> 43334 (frequency: 181)
2025-12-24 07:45:33,045 - rustbpe - INFO - Progress: 67% (43732/65271 merges) - Last merge: (230, 151) -> 43987 (frequency: 176)
2025-12-24 07:45:33,055 - rustbpe - INFO - Progress: 68% (44385/65271 merges) - Last merge: (19089, 36962) -> 44640 (frequency: 172)
2025-12-24 07:45:33,065 - rustbpe - INFO - Progress: 69% (45037/65271 merges) - Last merge: (71, 4462) -> 45292 (frequency: 167)
2025-12-24 07:45:33,077 - rustbpe - INFO - Progress: 70% (45690/65271 merges) - Last merge: (440, 263) -> 45945 (frequency: 163)
2025-12-24 07:45:33,089 - rustbpe - INFO - Progress: 71% (46343/65271 merges) - Last merge: (1192, 6834) -> 46598 (frequency: 159)
2025-12-24 07:45:33,097 - rustbpe - INFO - Progress: 72% (46996/65271 merges) - Last merge: (351, 25626) -> 47251 (frequency: 155)
2025-12-24 07:45:33,106 - rustbpe - INFO - Progress: 73% (47648/65271 merges) - Last merge: (105, 2742) -> 47903 (frequency: 151)
2025-12-24 07:45:33,114 - rustbpe - INFO - Progress: 74% (48301/65271 merges) - Last merge: (4522, 595) -> 48556 (frequency: 148)
2025-12-24 07:45:33,120 - rustbpe - INFO - Progress: 75% (48954/65271 merges) - Last merge: (84, 2127) -> 49209 (frequency: 144)
2025-12-24 07:45:33,130 - rustbpe - INFO - Progress: 76% (49606/65271 merges) - Last merge: (3904, 265) -> 49861 (frequency: 141)
2025-12-24 07:45:33,137 - rustbpe - INFO - Progress: 77% (50259/65271 merges) - Last merge: (21122, 465) -> 50514 (frequency: 138)
2025-12-24 07:45:33,144 - rustbpe - INFO - Progress: 78% (50912/65271 merges) - Last merge: (63, 93) -> 51167 (frequency: 134)
2025-12-24 07:45:33,153 - rustbpe - INFO - Progress: 79% (51565/65271 merges) - Last merge: (68, 14318) -> 51820 (frequency: 131)
2025-12-24 07:45:33,160 - rustbpe - INFO - Progress: 80% (52217/65271 merges) - Last merge: (9321, 30574) -> 52472 (frequency: 129)
2025-12-24 07:45:33,170 - rustbpe - INFO - Progress: 81% (52870/65271 merges) - Last merge: (2295, 1385) -> 53125 (frequency: 126)
2025-12-24 07:45:33,177 - rustbpe - INFO - Progress: 82% (53523/65271 merges) - Last merge: (283, 2833) -> 53778 (frequency: 123)
2025-12-24 07:45:33,185 - rustbpe - INFO - Progress: 83% (54175/65271 merges) - Last merge: (28288, 117) -> 54430 (frequency: 121)
2025-12-24 07:45:33,193 - rustbpe - INFO - Progress: 84% (54828/65271 merges) - Last merge: (8035, 1514) -> 55083 (frequency: 118)
2025-12-24 07:45:33,202 - rustbpe - INFO - Progress: 85% (55481/65271 merges) - Last merge: (401, 885) -> 55736 (frequency: 115)
2025-12-24 07:45:33,210 - rustbpe - INFO - Progress: 86% (56134/65271 merges) - Last merge: (6513, 276) -> 56389 (frequency: 113)
2025-12-24 07:45:33,219 - rustbpe - INFO - Progress: 87% (56786/65271 merges) - Last merge: (309, 399) -> 57041 (frequency: 110)
2025-12-24 07:45:33,229 - rustbpe - INFO - Progress: 88% (57439/65271 merges) - Last merge: (1996, 7066) -> 57694 (frequency: 108)
2025-12-24 07:45:33,236 - rustbpe - INFO - Progress: 89% (58092/65271 merges) - Last merge: (6539, 1271) -> 58347 (frequency: 106)
2025-12-24 07:45:33,244 - rustbpe - INFO - Progress: 90% (58744/65271 merges) - Last merge: (14320, 8196) -> 58999 (frequency: 104)
2025-12-24 07:45:33,253 - rustbpe - INFO - Progress: 91% (59397/65271 merges) - Last merge: (43234, 115) -> 59652 (frequency: 102)
2025-12-24 07:45:33,261 - rustbpe - INFO - Progress: 92% (60050/65271 merges) - Last merge: (95, 112) -> 60305 (frequency: 99)
2025-12-24 07:45:33,269 - rustbpe - INFO - Progress: 93% (60703/65271 merges) - Last merge: (34591, 8319) -> 60958 (frequency: 98)
2025-12-24 07:45:33,276 - rustbpe - INFO - Progress: 94% (61355/65271 merges) - Last merge: (24689, 687) -> 61610 (frequency: 96)
2025-12-24 07:45:33,286 - rustbpe - INFO - Progress: 95% (62008/65271 merges) - Last merge: (6562, 12626) -> 62263 (frequency: 94)
2025-12-24 07:45:33,295 - rustbpe - INFO - Progress: 96% (62661/65271 merges) - Last merge: (483, 8391) -> 62916 (frequency: 92)
2025-12-24 07:45:33,301 - rustbpe - INFO - Progress: 97% (63313/65271 merges) - Last merge: (40503, 2715) -> 63568 (frequency: 91)
2025-12-24 07:45:33,308 - rustbpe - INFO - Progress: 98% (63966/65271 merges) - Last merge: (3946, 294) -> 64221 (frequency: 89)
2025-12-24 07:45:33,318 - rustbpe - INFO - Progress: 99% (64619/65271 merges) - Last merge: (354, 399) -> 64874 (frequency: 87)
2025-12-24 07:45:33,324 - rustbpe - INFO - Progress: 100% (65271/65271 merges) - Last merge: (25707, 2583) -> 65526 (frequency: 86)
2025-12-24 07:45:33,324 - rustbpe - INFO - Finished training: 65271 merges completed
Training time: 27.94s
Saved tokenizer encoding to /root/.cache/nanochat/tokenizer/tokenizer.pkl
Saved token_bytes to /root/.cache/nanochat/tokenizer/token_bytes.pt
```

### Artifacts Produced

| File             | Description                                   |
| ---------------- | --------------------------------------------- |
| `tokenizer.pkl`  | Serialized tiktoken-compatible encoding       |
| `token_bytes.pt` | Token → byte mapping for inspection/debugging |

Both are saved under:

```
~/.cache/nanochat/tokenizer/
```

---

## Tokenizer Evaluation

### Command

```bash
python -m scripts.tok_eval
```

### Vocabulary Sizes

```
GPT-2: 50257
GPT-4: 100277
Ours: 65536
```

---

## Comparison with GPT-2

```
===============================================================================================
Text Type  Bytes    GPT-2           Ours            Relative     Better    
                    Tokens  Ratio   Tokens  Ratio   Diff %      
-----------------------------------------------------------------------------------------------
news       1819     404     4.50    375     4.85       +7.2%     Ours      
korean     893      745     1.20    731     1.22       +1.9%     Ours      
code       1259     576     2.19    494     2.55      +14.2%     Ours      
math       1834     936     1.96    971     1.89       -3.7%     GPT-2     
science    1112     260     4.28    227     4.90      +12.7%     Ours      
fwe-train  4208518  900364  4.67    856776  4.91       +4.8%     Ours      
fwe-val    4400098  954009  4.61    908625  4.84       +4.8%     Ours      
```

**Summary vs GPT-2**

* Strong improvements on **code**, **science**, and **news**
* Slight regression on **math**
* Overall token efficiency improves on both training and validation corpora

---

## Comparison with GPT-4

```
===============================================================================================
Text Type  Bytes    GPT-4           Ours            Relative     Better    
                    Tokens  Ratio   Tokens  Ratio   Diff %      
-----------------------------------------------------------------------------------------------
news       1819     387     4.70    375     4.85       +3.1%     Ours      
korean     893      364     2.45    731     1.22     -100.8%     GPT-4     
code       1259     309     4.07    494     2.55      -59.9%     GPT-4     
math       1834     832     2.20    971     1.89      -16.7%     GPT-4     
science    1112     249     4.47    227     4.90       +8.8%     Ours      
fwe-train  4208518  874799  4.81    856776  4.91       +2.1%     Ours      
fwe-val    4400098  927977  4.74    908625  4.84       +2.1%     Ours 
```

**Summary vs GPT-4**

* GPT-4 remains stronger on **code**, **math**, and **Korean**
* NanoChat tokenizer is **more efficient on scientific text**
* Slight gains on large-scale pretraining corpora

---

## Key Takeaways

* The tokenizer trains **~65k merges in under 30 seconds** on ~1B characters
* Achieves **better compression than GPT-2** on most domains
* Competitive with GPT-4 on large-scale English corpora
* Designed for **efficient pretraining**, not multilingual specialization
