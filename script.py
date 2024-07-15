# Import necessary libraries
import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity
import torch.optim as optim
import torch.nn.functional as F
from llama_cpp import Llama
import nltk

# Download NLTK data
nltk.download('punkt')

# Define the text
text = '''Quantization of Large Language Models (LLMs) using llama optimizes model
            efficiency by reducing precision from 32-bit floats to 8-bit integers.
            llama.cpp streamlines this process, enhancing deployment speed and
            memory usage without sacrificing model accuracy. By converting weights
            and activations to lower bit-widths, llama enables faster inference while
            maintaining LLM capabilities crucial for real-world applications in NLP,
            ensuring scalable and efficient model deployment across diverse computing environments.'''

# Tokenize the text using NLTK
tokens = nltk.word_tokenize(text)

# Load the BAAI/bge-large-en-v1.5 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")

# Tokenize the text and generate embeddings for the parent model
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    embeddings_parent = outputs.last_hidden_state.mean(dim=1)  # Mean pooling of token embeddings

# Print the embeddings for the parent model
print("Embeddings (Parent Model):")
print(embeddings_parent)

# Initialize Llama for embedding generation
llama_model = Llama("./bge-large-en-1.5.gguf", embedding=True)  # Adjust path as per your actual setup

# Generate embeddings for the student model
embeddings_student = torch.tensor(llama_model.embed(text))

# Normalize embeddings
embeddings_parent_normalized = torch.nn.functional.normalize(embeddings_parent, p=2, dim=-1)
embeddings_student_normalized = torch.nn.functional.normalize(embeddings_student, p=2, dim=-1)

# Calculate cosine similarity
cosine_sim = torch.matmul(embeddings_parent_normalized, embeddings_student_normalized.T)
mean_similarity = torch.mean(cosine_sim).item()
print("Mean Cosine Similarity:", mean_similarity)

# Define target similarity
target_similarity = torch.tensor([1.0])  # Example target similarity

# Calculate Mean Squared Error (MSE) loss
mse_loss = F.mse_loss(torch.tensor([mean_similarity]), target_similarity)
print("MSE Loss:", mse_loss.item())

# Example target embeddings for your downstream task
target_embeddings = torch.randn_like(embeddings_student)  # Replace with your actual targets

# Example optimizer setup
initial_embeddings = torch.tensor(llama_model.embed(text), requires_grad=True)  # Ensure requires_grad=True
optimizer = optim.Adam([initial_embeddings], lr=0.001)  # Optimizing input text for Llama

# Example usage for training
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Generate embeddings with current text input
    embeddings = torch.tensor(llama_model.embed(text), requires_grad=True)  # Ensure requires_grad=True

    # Calculate loss (example MSE loss)
    loss = F.mse_loss(embeddings, target_embeddings)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Print loss
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# After optimization, use the final text for inference
final_text = text
final_embeddings = torch.tensor(llama_model.embed(final_text), requires_grad=True)

# Print the final embeddings
print("Final Embeddings:")
print(final_embeddings)
