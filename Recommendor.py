#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

def load_libsvm_file(file_path):
    """
    Load a custom libsvm file with two columns (isbn, ratings) as a sparse matrix.
    Each line represents a new user, with sequential user IDs assigned automatically.
    """
    data, rows, cols = [], [], []
    user_ids = []  # Store original user IDs
    
    with open(file_path, 'r') as file:
        for idx, line in enumerate(file):
            features = line.strip().split()
            for feature in features:
                if ":" in feature:
                    index, value = map(float, feature.split(":"))
                    rows.append(idx)
                    cols.append(int(index))
                    data.append(value)
    
    return csr_matrix((data, (rows, cols)), shape=(max(rows) + 1, max(cols) + 1))

def load_user_isbn_mapping(ratings_csv):
    """
    Load user ID and ISBN mapping from ratings CSV
    """
    ratings_df = pd.read_csv(ratings_csv, sep=';')
    user_mapping = {idx: user_id for idx, user_id in enumerate(ratings_df['User-ID'].unique())}
    isbn_mapping = {idx: isbn for idx, isbn in enumerate(ratings_df['ISBN'].unique())}
    return user_mapping, isbn_mapping

def cosine_similarity_manual(target_vector, matrix):
    """
    Compute cosine similarity between a target vector and all rows in a matrix.
    """
    target_norm = np.sqrt(target_vector.multiply(target_vector).sum())
    matrix_norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    dot_products = matrix.dot(target_vector.T).toarray().flatten()

    matrix_norms[matrix_norms == 0] = 1
    if target_norm == 0:
        target_norm = 1

    return dot_products / (matrix_norms * target_norm)

def get_user_recommendations(user_idx, rating_matrix, books_df, user_mapping, isbn_mapping, k=10, top_n=5):
    """
    Generate recommendations for a specific user.
    """
    rating_matrix_csr = rating_matrix.tocsr()
    target_user_ratings = rating_matrix_csr[user_idx]
    
    similarities = cosine_similarity_manual(target_user_ratings, rating_matrix_csr)
    similar_users_indices = np.argsort(similarities)[-k-1:-1][::-1]
    similar_users_similarities = similarities[similar_users_indices]
    
    similar_users_ratings = rating_matrix_csr[similar_users_indices]
    weighted_ratings = similar_users_ratings.T.multiply(similar_users_similarities).sum(axis=1).A1
    similarity_sums = np.array(similar_users_ratings.astype(bool).sum(axis=0)).flatten()
    
    similarity_sums[similarity_sums == 0] = 1
    weighted_average = weighted_ratings / similarity_sums
    
    user_rated_books = target_user_ratings.toarray().flatten()
    weighted_average[user_rated_books > 0] = 0
    
    max_score = weighted_average.max()
    if max_score > 0:
        scaled_scores = np.floor(1 + (weighted_average / max_score * 8)).astype(int)
    else:
        scaled_scores = weighted_average.astype(int)
    
    top_indices = np.argsort(scaled_scores)[-top_n:][::-1]
    
    recommendations = []
    original_user_id = user_mapping.get(user_idx)
    
    for idx in top_indices:
        if scaled_scores[idx] > 0:
            isbn = isbn_mapping.get(idx)
            book_title = books_df[books_df['ISBN'] == isbn]['Title'].iloc[0] if isbn in books_df['ISBN'].values else f"Book_{idx}"
            score = int(scaled_scores[idx])
            recommendations.append({
                'User_ID': original_user_id,
                'Book_ID': isbn,
                'Title': book_title,
                'Recommendation_Score': score
            })
    
    return recommendations

def generate_all_user_recommendations(rating_matrix, books_df, user_mapping, isbn_mapping, output_file, k=10, top_n=5):
    """
    Generate recommendations for all users and save to CSV.
    """
    all_recommendations = []
    num_users = rating_matrix.shape[0]
    
    for user_idx in range(num_users):
        user_recs = get_user_recommendations(user_idx, rating_matrix, books_df, user_mapping, isbn_mapping, k, top_n)
        all_recommendations.extend(user_recs)
    
    recommendations_df = pd.DataFrame(all_recommendations)
    recommendations_df['Recommendation_Score'] = recommendations_df['Recommendation_Score'].astype(int)
    recommendations_df.to_csv(output_file, index=False)
    return recommendations_df

# File paths upload
libsvm_file = '/Users/vidya/Downloads/Ratings-2.libsvm'
ratings_csv = '/Users/vidya/Downloads/archive (11)/Ratings.csv'
books_csv = '/Users/vidya/Downloads/archive (11)/Books.csv'
output_file = 'user_recommendations_book_list.csv'

# Loading the data
rating_matrix = load_libsvm_file(libsvm_file)
books_df = pd.read_csv(books_csv, sep=';')
user_mapping, isbn_mapping = load_user_isbn_mapping(ratings_csv)

# Generating recommendations for all the users
recommendations_df = generate_all_user_recommendations(
    rating_matrix,
    books_df,
    user_mapping,
    isbn_mapping,
    output_file,
    k=10,
    top_n=5
)

# Example: Geting recommendations for a specific user
user_idx = 0
user_recommendations = get_user_recommendations(user_idx, rating_matrix, books_df, user_mapping, isbn_mapping)
original_user_id = user_mapping.get(user_idx)
print(f"\nRecommendations for User {original_user_id}:")
for i, rec in enumerate(user_recommendations, 1):
    print(f"{i}. {rec['Title']} (Score: {rec['Recommendation_Score']})")


# In[1]:


#import statements
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

def load_libsvm_file(file_path):
    """
    Load a custom libsvm file with two columns (isbn, ratings) as a sparse matrix.
    """
    data, rows, cols = [], [], []
    user_ids = []
    
    with open(file_path, 'r') as file:
        for idx, line in enumerate(file):
            features = line.strip().split()
            for feature in features:
                if ":" in feature:
                    index, value = map(float, feature.split(":"))
                    rows.append(idx)
                    cols.append(int(index))
                    data.append(value)
    
    return csr_matrix((data, (rows, cols)), shape=(max(rows) + 1, max(cols) + 1))

def load_user_isbn_mapping(ratings_csv):
    """
    Load user ID and ISBN mapping from ratings CSV
    """
    ratings_df = pd.read_csv(ratings_csv, sep=';')
    user_mapping = {idx: user_id for idx, user_id in enumerate(ratings_df['User-ID'].unique())}
    isbn_mapping = {idx: isbn for idx, isbn in enumerate(ratings_df['ISBN'].unique())}
    return user_mapping, isbn_mapping

#Calculating using cosine similarity

def cosine_similarity_manual(target_vector, matrix):
    """
    Compute cosine similarity between a target vector and all rows in a matrix.
    """
    target_norm = np.sqrt(target_vector.multiply(target_vector).sum())
    matrix_norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    dot_products = matrix.dot(target_vector.T).toarray().flatten()

    matrix_norms[matrix_norms == 0] = 1
    if target_norm == 0:
        target_norm = 1

    return dot_products / (matrix_norms * target_norm)

def get_user_recommendations(user_idx, rating_matrix, books_df, user_mapping, isbn_mapping, k=10, top_n=5):
    """
    Generate recommendations for a specific user.
    """
    rating_matrix_csr = rating_matrix.tocsr()
    target_user_ratings = rating_matrix_csr[user_idx]
    
    similarities = cosine_similarity_manual(target_user_ratings, rating_matrix_csr)
    similar_users_indices = np.argsort(similarities)[-k-1:-1][::-1]
    similar_users_similarities = similarities[similar_users_indices]
    
    similar_users_ratings = rating_matrix_csr[similar_users_indices]
    weighted_ratings = similar_users_ratings.T.multiply(similar_users_similarities).sum(axis=1).A1
    similarity_sums = np.array(similar_users_ratings.astype(bool).sum(axis=0)).flatten()
    
    similarity_sums[similarity_sums == 0] = 1
    weighted_average = weighted_ratings / similarity_sums
    
    user_rated_books = target_user_ratings.toarray().flatten()
    weighted_average[user_rated_books > 0] = 0
    
    max_score = weighted_average.max()
    if max_score > 0:
        scaled_scores = np.floor(1 + (weighted_average / max_score * 8)).astype(int)
    else:
        scaled_scores = weighted_average.astype(int)
    
    top_indices = np.argsort(scaled_scores)[-top_n:][::-1]
    
    recommendations = []
    original_user_id = user_mapping.get(user_idx)
    
    for idx in top_indices:
        isbn = isbn_mapping.get(idx)
        if isbn is not None:
            # Try to get title, use ISBN if not found
            title = books_df[books_df['ISBN'] == isbn]['Title'].iloc[0] if isbn in books_df['ISBN'].values else isbn
            score = int(scaled_scores[idx])
            recommendations.append({
                'User_ID': original_user_id,
                'Book_ID': isbn,
                'Title': title,
                'Recommendation_Score': score
            })
    
#5 recommendations per user
    while len(recommendations) < top_n:
        recommendations.append({
            'User_ID': original_user_id,
            'Book_ID': 'N/A',
            'Title': 'N/A',
            'Recommendation_Score': 0
        })
    
    return recommendations[:top_n]

def generate_all_user_recommendations(rating_matrix, books_df, user_mapping, isbn_mapping, output_file, k=10, top_n=5):
    """
    Generate recommendations for all users and save to CSV.
    """
    all_recommendations = []
    num_users = rating_matrix.shape[0]
    
    for user_idx in range(num_users):
        user_recs = get_user_recommendations(user_idx, rating_matrix, books_df, user_mapping, isbn_mapping, k, top_n)
        all_recommendations.extend(user_recs)
    
    recommendations_df = pd.DataFrame(all_recommendations)
    recommendations_df['Recommendation_Score'] = recommendations_df['Recommendation_Score'].astype(int)
    recommendations_df.to_csv(output_file, index=False)
    return recommendations_df

# Enter all the file paths
libsvm_file = '/Users/vidya/Downloads/Ratings-2.libsvm'
ratings_csv = '/Users/vidya/Downloads/archive (11)/Ratings.csv'
books_csv = '/Users/vidya/Downloads/archive (11)/Books.csv'
output_file = 'user_recommendations_book_list.csv'

# Loading the data
rating_matrix = load_libsvm_file(libsvm_file)
books_df = pd.read_csv(books_csv, sep=';')
user_mapping, isbn_mapping = load_user_isbn_mapping(ratings_csv)

# Generatimg the recommendations
recommendations_df = generate_all_user_recommendations(
    rating_matrix,
    books_df,
    user_mapping,
    isbn_mapping,
    output_file,
    k=10,
    top_n=5
)

user_idx = 0
user_recommendations = get_user_recommendations(user_idx, rating_matrix, books_df, user_mapping, isbn_mapping)
original_user_id = user_mapping.get(user_idx)
print(f"\nRecommendations for User {original_user_id}:")
for i, rec in enumerate(user_recommendations, 1):
    print(f"{i}. {rec['Title']} (Score: {rec['Recommendation_Score']})")


# In[ ]:


import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

def load_libsvm_file(file_path, user_ids):
    """
    Load a custom libsvm file with two columns (isbn, ratings) as a sparse matrix.
    Maps each row to the corresponding user_id from ratings.csv
    """
    data, rows, cols = [], [], []
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    
    with open(file_path, 'r') as file:
        for idx, line in enumerate(file):
            if idx < len(user_ids):  # Only process rows that have matching user_ids
                features = line.strip().split()
                for feature in features:
                    if ":" in feature:
                        index, value = map(float, feature.split(":"))
                        rows.append(idx)
                        cols.append(int(index))
                        data.append(value)
    
    return csr_matrix((data, (rows, cols)), shape=(len(user_ids), max(cols) + 1))

def load_user_isbn_mapping(ratings_csv):
    """
    Load user ID and ISBN mapping from ratings CSV
    """
    ratings_df = pd.read_csv(ratings_csv, sep=';')
    user_ids = ratings_df['User-ID'].unique()
    isbn_mapping = {idx: isbn for idx, isbn in enumerate(ratings_df['ISBN'].unique())}
    return user_ids, isbn_mapping

def get_user_recommendations(user_idx, rating_matrix, books_df, user_ids, isbn_mapping, k=10, top_n=5):
    """
    Generate recommendations for a specific user.
    """
    rating_matrix_csr = rating_matrix.tocsr()
    target_user_ratings = rating_matrix_csr[user_idx]
    
    similarities = cosine_similarity_manual(target_user_ratings, rating_matrix_csr)
    similar_users_indices = np.argsort(similarities)[-k-1:-1][::-1]
    similar_users_similarities = similarities[similar_users_indices]
    
    similar_users_ratings = rating_matrix_csr[similar_users_indices]
    weighted_ratings = similar_users_ratings.T.multiply(similar_users_similarities).sum(axis=1).A1
    similarity_sums = np.array(similar_users_ratings.astype(bool).sum(axis=0)).flatten()
    
    similarity_sums[similarity_sums == 0] = 1
    weighted_average = weighted_ratings / similarity_sums
    
    user_rated_books = target_user_ratings.toarray().flatten()
    weighted_average[user_rated_books > 0] = 0
    
    max_score = weighted_average.max()
    if max_score > 0:
        scaled_scores = np.floor(1 + (weighted_average / max_score * 8)).astype(int)
    else:
        scaled_scores = weighted_average.astype(int)
    
    top_indices = np.argsort(scaled_scores)[-top_n:][::-1]
    
    recommendations = []
    original_user_id = user_ids[user_idx]
    
    for idx in top_indices:
        isbn = isbn_mapping.get(idx)
        if isbn is not None:
            title = books_df[books_df['ISBN'] == isbn]['Title'].iloc[0] if isbn in books_df['ISBN'].values else isbn
            score = int(scaled_scores[idx])
            recommendations.append({
                'User_ID': original_user_id,
                'Book_ID': isbn,
                'Title': title,
                'Recommendation_Score': score
            })
    
    while len(recommendations) < top_n:
        recommendations.append({
            'User_ID': original_user_id,
            'Book_ID': 'N/A',
            'Title': 'N/A',
            'Recommendation_Score': 0
        })
    
    return recommendations[:top_n]

def generate_all_user_recommendations(rating_matrix, books_df, user_ids, isbn_mapping, output_file, k=10, top_n=5):
    """
    Generate recommendations for all users and save to CSV.
    """
    all_recommendations = []
    
    for user_idx in range(len(user_ids)):
        user_recs = get_user_recommendations(user_idx, rating_matrix, books_df, user_ids, isbn_mapping, k, top_n)
        all_recommendations.extend(user_recs)
    
    recommendations_df = pd.DataFrame(all_recommendations)
    recommendations_df['Recommendation_Score'] = recommendations_df['Recommendation_Score'].astype(int)
    recommendations_df.to_csv(output_file, index=False)
    return recommendations_df

# File paths
libsvm_file = '/Users/vidya/Downloads/Ratings-2.libsvm'
ratings_csv = '/Users/vidya/Downloads/archive (11)/Ratings.csv'
books_csv = '/Users/vidya/Downloads/archive (11)/Books.csv'
output_file = 'user_recommendations_book_list.csv'

# Load data
books_df = pd.read_csv(books_csv, sep=';')
user_ids, isbn_mapping = load_user_isbn_mapping(ratings_csv)
rating_matrix = load_libsvm_file(libsvm_file, user_ids)

# Generate recommendations
recommendations_df = generate_all_user_recommendations(
    rating_matrix,
    books_df,
    user_ids,
    isbn_mapping,
    output_file,
    k=10,
    top_n=5
)

# Example for a specific user
user_idx = 0
user_recommendations = get_user_recommendations(user_idx, rating_matrix, books_df, user_ids, isbn_mapping)
original_user_id = user_ids[user_idx]
print(f"\nRecommendations for User {original_user_id}:")
for i, rec in enumerate(user_recommendations, 1):
    print(f"{i}. {rec['Title']} (Score: {rec['Recommendation_Score']})")


# In[ ]:




