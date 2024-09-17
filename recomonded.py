import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Expanded sample data with more users
ratings = pd.DataFrame({
    'User ID': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10],
    'Book Title': ['The Catcher in the Rye', 'To Kill a Mockingbird', '1984',
                   'The Catcher in the Rye', 'Pride and Prejudice', '1984',
                   '1984', 'The Great Gatsby', 'Moby Dick',
                   'Moby Dick', 'To Kill a Mockingbird', 'The Great Gatsby',
                   'Pride and Prejudice', 'The Great Gatsby', 'The Catcher in the Rye',
                   '1984', 'Moby Dick', 'Pride and Prejudice',
                   'The Catcher in the Rye', '1984', 'Pride and Prejudice',
                   'To Kill a Mockingbird', 'Moby Dick', 'The Great Gatsby',
                   'Pride and Prejudice', 'The Catcher in the Rye', '1984',
                   'To Kill a Mockingbird', 'Moby Dick', 'The Great Gatsby'],
    'Rating': [5, 4, 5, 4, 5, 3, 4, 5, 3, 3, 4, 5, 4, 5, 4, 5, 4, 3, 5, 4, 4, 3, 5, 4, 2, 3, 5, 4, 4, 5]
})

# Pivot the ratings data
ratings_pivot = ratings.pivot(index='User ID', columns='Book Title', values='Rating').fillna(0)

# Calculate similarity matrix
similarity_matrix = cosine_similarity(ratings_pivot)
similarity_df = pd.DataFrame(similarity_matrix, index=ratings_pivot.index, columns=ratings_pivot.index)

def recommend_books(user_id):
    if user_id not in ratings_pivot.index:
        return "User ID not found."
    
    # Find similar users
    user_similarities = similarity_df[user_id]
    similar_users = user_similarities.sort_values(ascending=False).index
    
    # Get ratings of similar users
    similar_users_ratings = ratings_pivot.loc[similar_users]
    
    # Average ratings of similar users, weighted by their similarity
    weighted_ratings = similar_users_ratings.T.dot(user_similarities) / user_similarities.sum()
    
    # Recommend books not already rated by the user
    user_rated_books = ratings_pivot.loc[user_id]
    recommendations = weighted_ratings[~user_rated_books.index.isin(user_rated_books[user_rated_books > 0].index)]
    
    return recommendations.sort_values(ascending=False).index.tolist()

# Get user ID from the terminal
try:
    user_id = int(input("Enter your user ID: "))
    recommendations = recommend_books(user_id)
    
    if isinstance(recommendations, str):  # Check if the response is an error message
        print(recommendations)
    else:
        print(f"Recommended books for user {user_id}:")
        print(recommendations)
except ValueError:
    print("Please enter a valid integer for the user ID.")

