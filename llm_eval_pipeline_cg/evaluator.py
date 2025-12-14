from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

def evaluate_relevance_and_completeness(query, answer, llm_call, metrics):
    q_emb = model.encode([query])
    a_emb = model.encode([answer])
    relevance = float(cosine_similarity(q_emb, a_emb)[0][0])

    prompt = f"""
    Question: {query}
    Answer: {answer}

    Is the answer complete?
    Score from 0 to 1.
    Return only the number.
    """

    response = llm_call(prompt)
    metrics.record_llm_call(prompt, response)

    completeness = float(response.strip())
    return relevance, completeness


def evaluate_hallucination(answer, contexts, llm_call, metrics):
    context_text = "\n".join(c.get("text", "") for c in contexts)

    prompt = f"""
    Context:
    {context_text}

    Answer:
    {answer}

    Does the answer contain unsupported facts?
    Score from 0 to 1.
    Return only the number.
    """

    response = llm_call(prompt)
    metrics.record_llm_call(prompt, response)

    return float(response.strip())
