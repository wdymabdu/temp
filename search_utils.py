from sentence_transformers import SentenceTransformer
import numpy as np

# Load model once globally (so we don't reload it every time)
model = SentenceTransformer('all-MiniLM-L6-v2')


def keyword_search(collection, query, limit=10):
    """
    Performs keyword/text search on flight data
    Uses MongoDB's text search on origin, destination, and airline
    """
    results = collection.find(
        {"$text": {"$search": query}},
        {"score": {"$meta": "textScore"}}
    ).sort([("score", {"$meta": "textScore"})]).limit(limit)
    
    # Convert to list and normalize scores
    results_list = list(results)
    
    if results_list:
        # Normalize scores to 0-1 range
        max_score = max([r.get('score', 0) for r in results_list])
        if max_score > 0:
            for r in results_list:
                r['keyword_score'] = r.get('score', 0) / max_score
    
    return results_list


def semantic_search(collection, query, limit=10):
    """
    Performs semantic/vector search on flight data
    Finds flights similar in meaning to the query
    """
    # Generate embedding for the search query
    query_embedding = model.encode(query).tolist()
    
    # Get all documents with embeddings
    all_docs = list(collection.find({}))
    
    # Calculate cosine similarity with each document
    for doc in all_docs:
        if 'embedding' in doc:
            # Calculate similarity between query and document
            doc_embedding = doc['embedding']
            similarity = cosine_similarity(query_embedding, doc_embedding)
            doc['semantic_score'] = similarity
        else:
            doc['semantic_score'] = 0
    
    # Sort by similarity and return top results
    all_docs.sort(key=lambda x: x.get('semantic_score', 0), reverse=True)
    
    return all_docs[:limit]


def cosine_similarity(vec1, vec2):
    """
    Calculates how similar two vectors are (0 to 1)
    1 = identical, 0 = completely different
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return float(dot_product / (norm1 * norm2))


def hybrid_search(collection, query, alpha=0.5, limit=10):
    """
    Combines keyword and semantic search
    
    alpha: weight parameter (0 to 1)
    - alpha = 0: only semantic search
    - alpha = 1: only keyword search
    - alpha = 0.5: equal weight to both
    """
    # Get results from both search methods
    keyword_results = keyword_search(collection, query, limit=limit*2)
    semantic_results = semantic_search(collection, query, limit=limit*2)
    
    # Create a dictionary to combine scores
    combined_scores = {}
    
    # Add keyword scores
    for doc in keyword_results:
        doc_id = str(doc['_id'])
        combined_scores[doc_id] = {
            'doc': doc,
            'keyword_score': doc.get('keyword_score', 0),
            'semantic_score': 0
        }
    
    # Add semantic scores
    for doc in semantic_results:
        doc_id = str(doc['_id'])
        if doc_id in combined_scores:
            combined_scores[doc_id]['semantic_score'] = doc.get('semantic_score', 0)
        else:
            combined_scores[doc_id] = {
                'doc': doc,
                'keyword_score': 0,
                'semantic_score': doc.get('semantic_score', 0)
            }
    
    # Calculate hybrid score for each document
    for doc_id in combined_scores:
        k_score = combined_scores[doc_id]['keyword_score']
        s_score = combined_scores[doc_id]['semantic_score']
        
        # Hybrid formula: alpha * keyword + (1-alpha) * semantic
        hybrid_score = alpha * k_score + (1 - alpha) * s_score
        combined_scores[doc_id]['hybrid_score'] = hybrid_score
    
    # Sort by hybrid score
    sorted_results = sorted(
        combined_scores.values(),
        key=lambda x: x['hybrid_score'],
        reverse=True
    )
    
    # Return top results with scores
    final_results = []
    for item in sorted_results[:limit]:
        doc = item['doc']
        doc['keyword_score'] = round(item['keyword_score'], 3)
        doc['semantic_score'] = round(item['semantic_score'], 3)
        doc['hybrid_score'] = round(item['hybrid_score'], 3)
        final_results.append(doc)
    
    return final_results


def optimize_alpha(collection, query, relevance_feedback):
    """
    BONUS: Optimizes alpha value based on user feedback
    
    relevance_feedback: list of doc_ids that user found relevant
    Returns: optimized alpha value
    """
    # Try different alpha values
    alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_alpha = 0.5
    best_score = 0
    
    for alpha in alpha_values:
        results = hybrid_search(collection, query, alpha=alpha, limit=10)
        
        # Calculate how many relevant docs are in top results
        relevant_count = 0
        for i, doc in enumerate(results):
            doc_id = str(doc['_id'])
            if doc_id in relevance_feedback:
                # Give higher weight to results that appear earlier
                position_weight = 1 / (i + 1)
                relevant_count += position_weight
        
        if relevant_count > best_score:
            best_score = relevant_count
            best_alpha = alpha
    
    return best_alpha