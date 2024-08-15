# LLM Quantization Project

This project focuses on the quantization of Large Language Models (LLMs) using `llama.cpp` and explores various techniques to optimize model efficiency, deployment speed, and memory usage without sacrificing model accuracy. The project includes embedding generation, cosine similarity calculation, and optimization using advanced deep learning techniques.

## Features

- **Embedding Generation**: Generate embeddings using the `BAAI/bge-large-en-v1.5` model.
- **Quantization**: Optimize LLMs by reducing precision from 32-bit floats to 8-bit integers using `llama.cpp`.
- **Similarity Calculation**: Calculate cosine similarity between different embeddings.
- **Optimization**: Optimize embeddings using Mean Squared Error (MSE) loss and backpropagation.

## Technologies Used

- Python
- PyTorch
- Transformers (`transformers` library)
- `llama-cpp-python`
- NLTK (Natural Language Toolkit)

## Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (for CUDA support)

### Steps

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/LLM-Quantization.git
   cd LLM-Quantization
   ```

2. **Install Dependencies**:

   ```bash
   pip install torch transformers nltk
   apt-get update
   apt-get install -y build-essential cmake
   CMAKE_ARGS="-DGGML_CUDA=ON" pip install llama-cpp-python --no-cache-dir
   pip install -r llama.cpp/requirements.txt
   ```

3. **Verify CUDA Installation**:

   ```bash
   nvcc --version
   nvidia-smi
   ```

4. **Download NLTK Data**:
   ```python
   import nltk
   nltk.download('punkt')
   ```

## Usage

### Embedding Generation

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")
model = AutoModel.from_pretrained("BAAI/bge-large-en-v1.5")

# Example text
text = '''Quantization of Large Language Models (LLMs) using llama optimizes model
            efficiency by reducing precision from 32-bit floats to 8-bit integers.
            llama.cpp streamlines this process, enhancing deployment speed and
            memory usage without sacrificing model accuracy. By converting weights
            and activations to lower bit-widths, llama enables faster inference while
            maintaining LLM capabilities crucial for real-world applications in NLP,
            ensuring scalable and efficient model deployment across diverse computing environments.'''

# Tokenize and generate embeddings
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    embeddings_parent = outputs.last_hidden_state.mean(dim=1)

print("Embeddings:")
print(embeddings_parent)
```

### Cosine Similarity Calculation

```python
import torch
from torch.nn.functional import cosine_similarity

# Normalize embeddings and calculate cosine similarity
embeddings_tensor1 = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
embeddings_tensor2 = torch.nn.functional.normalize(embeddings_parent, p=2, dim=-1)
cosine_similarity = torch.matmul(embeddings_tensor1, embeddings_tensor2.T)

mean_similarity = torch.mean(cosine_similarity).item()
print("Mean Cosine Similarity:", mean_similarity)
```

### Optimization using MSE Loss

```python
import torch
import torch.optim as optim
import torch.nn.functional as F
from llama_cpp import Llama

# Initialize Llama and optimizer
llama_model = Llama("./bge-large-en-1.5.gguf", embedding=True)
target_embeddings = torch.randn_like(torch.tensor(llama_model.embed(text)))
initial_embeddings = torch.tensor(llama_model.embed(text), requires_grad=True)
optimizer = optim.Adam([initial_embeddings], lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    embeddings = torch.tensor(llama_model.embed(text), requires_grad=True)
    loss = F.mse_loss(embeddings, target_embeddings)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

final_text = text
final_embeddings = torch.tensor(llama_model.embed(final_text), requires_grad=True)
```

## Contribution

Feel free to fork this repository, create a feature branch, and submit a pull request. Contributions are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
