import pymupdf
import chromadb

from sentence_transformers import SentenceTransformer


# Load the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="django_chatbot_knowledge",
    metadata={"hnsw:space": "cosine"}
)


# Extract text and create chunks from PDF
def extract_text_from_pdf(pdf_path):

    pdf = pymupdf.open(pdf_path)

    chunks = []

    chunk_size = 500
    overlap = 100
    step = chunk_size - overlap

    for page_number, page in enumerate(pdf, start=1):

        page_text = page.get_text()

        paragraphs = [
            paragraph.strip()
            for paragraph in page_text.split("\n\n")
            if paragraph.strip()
        ]

        for paragraph in paragraphs:

            start = 0

            while start < len(paragraph):

                chunk = paragraph[start:start + chunk_size]

                chunks.append({
                    "text": chunk,
                    "page": page_number
                })

                start += step

    pdf.close()

    return chunks


# Generate embeddings and store PDF chunks
def store_pdf_chunks(pdf_path):

    chunks = extract_text_from_pdf(pdf_path)

    if not chunks:
        return 0

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(texts)

    filename = pdf_path.name

    ids = [
        f"{filename}_page_{chunk['page']}_chunk_{i}"
        for i, chunk in enumerate(chunks)
    ]

    metadatas = [
        {
            "filename": filename,
            "page": chunk["page"]
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    return len(chunks)


# Retrieve relevant chunks from ChromaDB
def retrieve_relevant_chunks(question):

    if collection.count() == 0:
        print("ChromaDB is empty.")
        return []

    question_embedding = embedding_model.encode(question)

    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=min(3, collection.count())
    )

    retrieved_chunks = []

    threshold = 0.3

    if results["documents"] and results["documents"][0]:

        for document, distance, metadata in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0]
        ):

            similarity_score = 1 - distance

            print("\nSimilarity score:", round(similarity_score, 2))

            if similarity_score >= threshold:

                print("Retrieved chunk:", document)
                print("Source PDF:", metadata["filename"])
                print("Page number:", metadata["page"])

                retrieved_chunks.append({
                    "text": document,
                    "filename": metadata["filename"],
                    "page": metadata["page"]
                })

            else:

                print("Chunk rejected because similarity is too low.")

    return retrieved_chunks


# Generate answer using Groq and retrieved PDF information
def generate_rag_answer(question, groq_client):

    retrieved_chunks = retrieve_relevant_chunks(question)

    if not retrieved_chunks:

        return "I couldn't find relevant information in the uploaded PDFs."

    context = "\n\n".join(
        chunk["text"] for chunk in retrieved_chunks
    )

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the provided PDF context. "
                    "If the answer is not available in the context, "
                    "say you could not find it in the PDFs. "
                    "Do not invent information."
                )
            },
            {
                "role": "user",
                "content": (
                    f"PDF context:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ]
    )

    return response.choices[0].message.content