# Project 3 Report: AI Recommendation System
**Name:** Harish Surya M
**Batch:** 2026
**Organization:** DecodeLabs
**Project:** AI Recommendation Logic — Tech Stack Recommender

---

## Objective

To build a content-based AI recommendation system that maps a user's
skills to the most relevant job roles using TF-IDF weighting and
Cosine Similarity scoring.

---

## Tools & Technologies Used

- Language: Python 3
- Libraries: csv (built-in), math (built-in)
- Dataset: raw_skills.csv (custom job role dataset)
- Algorithm: TF-IDF + Cosine Similarity
- Editor: VS Code

---

## How the System Works

The system follows a 4-step ranking pipeline:

1. INGESTION
   The user enters a minimum of 3 skills as input.
   These are parsed and cleaned into a list.

2. SCORING (TF-IDF + Cosine Similarity)
   Each skill input is converted into a weighted TF-IDF vector.
   The user vector is compared against every job role vector
   in the dataset using Cosine Similarity formula:
   cos(θ) = (A · B) / (||A|| × ||B||)

3. SORTING
   All job roles are sorted in descending order by their
   similarity score.

4. FILTERING
   Only the Top 3 highest-scoring job roles are displayed
   to prevent choice overload.

---

## Sample Output

Input Skills: Python, SQL, Machine_Learning

#1  Data Scientist        — Match Score: 41.1%
#2  Machine Learning Engineer — Match Score: 34.5%
#3  Backend Developer     — Match Score: 13.6%

---

## Key Concepts Demonstrated

- Content-Based Filtering (not Collaborative Filtering)
- Vector Mapping: converting text skills into numerical arrays
- TF-IDF Weighting: rewarding specific skills, penalizing generic ones
- Cosine Similarity: measuring angular alignment between vectors
- Cold Start Handling: zero-vector check to avoid division errors
- Top-N Filtering: returning only the best matches

---

## Conclusion

This project successfully demonstrates the core logic behind
modern AI recommendation engines. By implementing TF-IDF and
Cosine Similarity from scratch using only Python's built-in
libraries, I was able to build a functional Tech Stack Recommender
that maps raw user skills to relevant career paths with ranked
mathematical precision.

**Submitted by:** Harish Surya M
**Project:** DecodeLabs AI Track — Project 3