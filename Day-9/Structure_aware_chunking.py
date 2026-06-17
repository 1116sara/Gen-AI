import re

sample_hr_policy = """  HUMAN RESOURCES POLICY

1. Employee Conduct and Workplace Ethics

All employees are expected to maintain professional behavior, treat colleagues with respect, and uphold the company's values at all times. Harassment, discrimination, bullying, or any form of unethical conduct will not be tolerated. Employees must comply with all company policies and maintain confidentiality of company information.

2. Attendance, Working Hours, and Leave

Employees are required to adhere to their assigned work schedules and notify their reporting manager in advance of any planned absence whenever possible. Attendance records will be maintained, and repeated unexplained absences may result in disciplinary action. Leave requests must be submitted through the approved process and are subject to managerial approval based on business requirements.

3. Performance Management and Professional Development

Employee performance will be reviewed periodically based on individual goals, job responsibilities, and overall contribution to the organization. Constructive feedback, coaching, and development opportunities will be provided to support professional growth. Consistent underperformance may lead to a performance improvement plan and further corrective actions if necessary.
"""


def chunk_by_structure(text):
    pattern = r"^\d+\. "
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    for paragraph in paragraphs:
        if re.match(pattern,paragraph):
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            print("This is a section header.")
            current_chunk = []
            current_chunk.append(paragraph)
        else:
            print("This is a body text")
            current_chunk.append(paragraph)
        
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        current_chunk = []

    return(chunks)

chunks = chunk_by_structure(sample_hr_policy)
chunks = [chunk for chunk in chunks if len(chunk) > 50]
print(f"\nTotal chunks: {len(chunks)}")
print("="*50)
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}: ")
    print(chunk)
    print("-"*50)