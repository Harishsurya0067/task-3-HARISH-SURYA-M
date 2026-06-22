# ============================================
#   Project 3: Tech Stack Recommender
#   AI Recommendation Logic - DecodeLabs 2026
#   Method: TF-IDF + Cosine Similarity
# ============================================

import csv
import math

def load_dataset(filename):
    dataset = []
    with open(filename, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            job_role = row['job_role']
            skills = row['skills'].split()
            dataset.append({'role': job_role, 'skills': skills})
    return dataset

def build_vocabulary(dataset):
    vocab = set()
    for item in dataset:
        for skill in item['skills']:
            vocab.add(skill.lower())
    return sorted(list(vocab))

def compute_tf(skills):
    tf = {}
    total = len(skills)
    for skill in skills:
        skill = skill.lower()
        tf[skill] = tf.get(skill, 0) + 1
    for skill in tf:
        tf[skill] = tf[skill] / total
    return tf

def compute_idf(dataset, vocab):
    total_docs = len(dataset)
    idf = {}
    for term in vocab:
        count = sum(1 for item in dataset if term in [s.lower() for s in item['skills']])
        if count == 0:
            idf[term] = 0
        else:
            idf[term] = math.log(total_docs / count)
    return idf

def compute_tfidf_vector(skills, vocab, idf):
    tf = compute_tf(skills)
    vector = []
    for term in vocab:
        tf_val = tf.get(term, 0)
        idf_val = idf.get(term, 0)
        vector.append(tf_val * idf_val)
    return vector

def cosine_similarity(vec_a, vec_b):
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a ** 2 for a in vec_a))
    magnitude_b = math.sqrt(sum(b ** 2 for b in vec_b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)

def get_top_recommendations(user_skills, dataset, vocab, idf, top_n=3):
    user_vector = compute_tfidf_vector(user_skills, vocab, idf)
    scored_roles = []
    for item in dataset:
        role_vector = compute_tfidf_vector(item['skills'], vocab, idf)
        score = cosine_similarity(user_vector, role_vector)
        scored_roles.append((item['role'], score, item['skills']))
    scored_roles.sort(key=lambda x: x[1], reverse=True)
    return scored_roles[:top_n]

def main():
    print("=" * 55)
    print("   AI TECH STACK RECOMMENDER — DecodeLabs P3")
    print("=" * 55)

    dataset = load_dataset('raw_skills.csv')
    print(f"Loaded {len(dataset)} job roles.\n")

    vocab = build_vocabulary(dataset)
    idf = compute_idf(dataset, vocab)

    print("Enter your skills (minimum 3), separated by commas:")
    print("Example: Python, Machine_Learning, SQL")
    raw_input_str = input("Your skills: ")

    user_skills = [s.strip().lower() for s in raw_input_str.split(',') if s.strip()]

    if len(user_skills) < 3:
        print("Please enter at least 3 skills!")
        return

    results = get_top_recommendations(user_skills, dataset, vocab, idf, top_n=3)

    print("\n" + "=" * 55)
    print("        TOP 3 RECOMMENDED JOB ROLES")
    print("=" * 55)

    for rank, (role, score, role_skills) in enumerate(results, 1):
        match_pct = round(score * 100, 1)
        print(f"\n#{rank}  {role}")
        print(f"    Match Score : {match_pct}%")
        print(f"    Role Skills : {', '.join(role_skills[:5])}...")

    print("\n" + "=" * 55)
    print("Recommendation complete!")

if __name__ == "__main__":
    main()