import pandas as pd
import os
import glob


import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize


def chunk_by_word_count(text, n_chunks):

    words = text.split()
    total_words = len(words)
    chunk_size = max(1, total_words // n_chunks)
    
    chunks = []
    for i in range(0, total_words, chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    
    return chunks


def sentence_based_chunk(text, n_chunks):
    """
    Splits `text` into `n_chunks` roughly equal in word count,
    but always at sentence boundaries.
    """
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    # Count total words
    total_words = sum(len(s.split()) for s in sentences)
    target_words_per_chunk = total_words // n_chunks
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sent in sentences:
        sent_word_count = len(sent.split())
        # If adding this sentence exceeds target, start a new chunk
        if current_word_count + sent_word_count > target_words_per_chunk and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0
        current_chunk.append(sent)
        current_word_count += sent_word_count
    
    # Add any remaining sentences
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # If we have fewer chunks than desired, merge last chunks
    while len(chunks) > n_chunks:
        # Merge the last two chunks
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks.pop()
    
    return chunks



def paragraph_based_chunk(text, n_chunks):
    """
    Splits text into at most n_chunks, covering the entire document.
    Prioritizes paragraph boundaries (\n\n), then splits by sentences if needed,
    and finally merges if too many chunks were created.
    """
    # Split into paragraphs
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    total_paragraphs = len(paragraphs)

    # Start with paragraph-based chunks
    if total_paragraphs <= n_chunks:
        chunks = paragraphs[:]
    else:
        paragraphs_per_chunk = max(1, total_paragraphs // n_chunks)
        chunks, current_chunk = [], []
        for para in paragraphs:
            current_chunk.append(para)
            if len(current_chunk) >= paragraphs_per_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

    # If still fewer than n_chunks, split largest chunks by sentences
    while len(chunks) < n_chunks:
        idx = max(range(len(chunks)), key=lambda i: len(chunks[i].split()))
        sentences = sent_tokenize(chunks[idx])
        if len(sentences) <= 1:
            break
        mid = len(sentences) // 2
        left = " ".join(sentences[:mid])
        right = " ".join(sentences[mid:])
        chunks = chunks[:idx] + [left, right] + chunks[idx+1:]

    # If we overshot (too many chunks), merge smallest ones until n_chunks
    while len(chunks) > n_chunks:
        # Merge the two smallest chunks (by word count)
        idx = min(range(len(chunks)-1), key=lambda i: len(chunks[i].split()) + len(chunks[i+1].split()))
        merged = chunks[idx] + "\n\n" + chunks[idx+1]
        chunks = chunks[:idx] + [merged] + chunks[idx+2:]

    return chunks






# --- Step 1: Read CSV ---
df = pd.read_csv("master_clauses.csv")

# --- Step 2: Collect word counts from txt files ---
folder_path = "selected_contracts"
txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

word_counts = []
for file_path in txt_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        num_words = len(text.split())
        word_counts.append((os.path.basename(file_path), num_words))

# Sort by number of words
word_counts.sort(key=lambda x: x[1])

# WRONG FILENAMES IN DATASET:
# MACY'S,INC_05_11_2020-EX-99.4-JOINT FILING AGREEMENT.PDF
# MACY_S,INC_05_11_2020-EX-99.4-JOINT FILING AGREEMENT.txt
# NETGEAR,INC_04_21_2003-EX-10.16-AMENDMENT TO THE DISTRIBUTOR AGREEMENT BETWEEN INGRAM MICRO AND NETGEAR-.pdf
# NETGEAR,INC_04_21_2003-EX-10.16-AMENDMENT TO THE DISTRIBUTOR AGREEMENT BETWEEN INGRAM MICRO AND NETGEAR.txt

selected_files = [fname for fname, count in word_counts if count <= 2000]

filtered_df = df[df["Filename"].str.replace(".pdf", ".txt").str.replace(".PDF", ".txt").isin(selected_files)]


# --- Step 4: Keep only required columns ---
cols_to_keep = [
    "Exclusivity-Answer",
    "Anti-Assignment-Answer",
    "Revenue/Profit Sharing-Answer",
    "Minimum Commitment-Answer",
    "License Grant-Answer",
    # "Audit Rights-Answer",
    # "Uncapped Liability-Answer",
    # "Cap On Liability-Answer",
]

desc = [
    "Is there an exclusive dealing  commitment with the counterparty? This includes a commitment to procure all “requirements” from one party of certain technology, goods, or services or a prohibition on licensing or selling technology, goods or services to third parties, or a prohibition on  collaborating or working with other parties), whether during the contract or  after the contract ends (or both).",
    "Is consent or notice required of a party if the contract is assigned to a third party?",
    "Is one party required to share revenue or profit with the counterparty for any technology, goods, or services?",
    "Is there a minimum order size or minimum amount or units per-time period that one party must buy from the counterparty under the contract?",
    "Does the contract contain a license granted by one party to its counterparty?",   
]


final_df = filtered_df[cols_to_keep]



import pandas as pd

filtered_df = filtered_df.copy()
filtered_df.insert(
    0,
    "filename",
    filtered_df["Filename"]
)

long_df = filtered_df.melt(
    id_vars="filename",
    value_vars=cols_to_keep,
    var_name="clause_type",
    value_name="answer",
)

long_df["filename"] = long_df["filename"].str.replace(".PDF", ".txt").str.replace(".pdf", ".txt")

yes_df = long_df[long_df["answer"] == "Yes"]
no_df = long_df[long_df["answer"] == "No"]

# --- Sample 50 Yes and 50 No ---
yes_sample = yes_df.sample(n=50, random_state=42)
no_sample = no_df.sample(n=50, random_state=42)

sample_df = pd.concat([yes_sample, no_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
clause_to_desc = dict(zip(cols_to_keep, desc))
sample_df["description"] = sample_df["clause_type"].map(clause_to_desc)


for i in range(len(sample_df)):
    
    sample = sample_df.iloc[i]
    filename = sample['filename']
    filepath = os.path.join(folder_path, filename)

    file = open(filepath, "r")
    content = file.read()
    file.close()
    print(content)

    print("=========")

    x = 3
    
    # chunks = chunk_by_word_count(content, n_chunks=x)
    # chunks = sentence_based_chunk(content, n_chunks=x)
    chunks = paragraph_based_chunk(content, n_chunks=x)

    [print(f"{chunk}\n----------------------------------") for chunk in chunks]



    print("=================================================")
    print(len(chunks))

    input("=================================================")