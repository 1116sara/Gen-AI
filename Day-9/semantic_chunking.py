from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
sample_hr_policy = """  HUMAN RESOURCES POLICY

1. Employee Conduct and Workplace Ethics

All employees are expected to maintain professional behavior, treat colleagues with respect, and uphold the company's values at all times. Harassment, discrimination, bullying, or any form of unethical conduct will not be tolerated. Employees must comply with all company policies and maintain confidentiality of company information.

2. Attendance, Working Hours, and Leave

Employees are required to adhere to their assigned work schedules and notify their reporting manager in advance of any planned absence whenever possible. Attendance records will be maintained, and repeated unexplained absences may result in disciplinary action. Leave requests must be submitted through the approved process and are subject to managerial approval based on business requirements.

3. Performance Management and Professional Development

Employee performance will be reviewed periodically based on individual goals, job responsibilities, and overall contribution to the organization. Constructive feedback, coaching, and development opportunities will be provided to support professional growth. Consistent underperformance may lead to a performance improvement plan and further corrective actions if necessary.
"""

import nltk
def chunk_by_semantics(text, threshold = 0.4):
    sentences = nltk.sent_tokenize(text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    # print(f"Total sentences after filtering: {len(sentences)}")
    embeddings = model.encode(sentences)
    # print(f"Embeddings shape: {embeddings.shape}")
    # print(f"Running similarity for {len(sentences)-1} pairs...")
    chunks = []
    current_chunk = [sentences[0]]
    for sentence in range(len(sentences)-1):
        vec1 = embeddings[sentence]
        vec2 = embeddings[sentence + 1]
        dot_product = np.dot(vec1, vec2)
        magnitude = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        similarity = dot_product/magnitude
        if similarity < threshold:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_chunk.append(sentences[sentence+1])
        else:
            current_chunk.append(sentences[sentence+1])
        print(f"Pair ({sentence}, {sentence+1}): {similarity:.4f}")

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        current_chunk = []
    
    return(chunks)

testing_chunks = chunk_by_semantics(sample_hr_policy,threshold=0.25)

print(f"\nTotal Chunks: {len(testing_chunks)}")
print("="*50)
for i, chunk in enumerate(testing_chunks):
    print(f"\nChunk {i+1}:")
    print(chunk)
    print("-"*50)


