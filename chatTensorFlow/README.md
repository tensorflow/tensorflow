# ChatTensorFlow

ChatTensorFlow is an intelligent assistant that helps users with TensorFlow related queries by utilizing the power of LangGraph based Agents, vector embeddings, and a conversational interface powered by various LLM providers.

![ChatTensorFlow](https://via.placeholder.com/800x400?text=ChatTensorFlow+Assistant)

## Features

- **Multi-provider support**: Works with multiple LLM providers (Google Gemini, OpenAI, Anthropic, Cohere) 
- **Content processing**: Splits and embeds TensorFlow documentation for semantic search
- **Query routing**: Intelligently classifies user questions and determines next steps
- **Assisted research**: Follows a research plan to answer complex TensorFlow queries
- **Context-aware responses**: Provides answers with citations from official TensorFlow documentation

## Architecture

ChatTensorFlow is built with a modular architecture:

1. **Document Crawler**: Efficiently crawls TensorFlow documentation using sitemap.xml
2. **Data Ingestion**: Processes web content into chunks for embedding
3. **Vector Store**: Stores embeddings for semantic search using Chroma
4. **Retriever**: Retrieves relevant content when answering questions
5. **Researcher Graph**: Follows a structured plan to research answers
6. **Router Graph**: Determines how to handle each user query
7. **Response Generator**: Creates coherent, accurate responses based on retrieved information

## Getting Started

### Prerequisites

- Python 3.9+
- API keys for at least one supported LLM provider

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/chatTensorFlow.git
   cd chatTensorFlow
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

### Data Collection

To crawl and process the Scikit-learn documentation:

1. Run the web crawler:
   ```bash
   python -m processor.crawler
   ```

2. Process and ingest the data:
   ```bash
   python -m processor.data_ingestion
   ```

This will create a Chroma vector database with the embedded documentation chunks.

## Usage

Run the application:

```bash
python app/main.py
```

The assistant can answer questions about:
- TensorFlow API usage
- Deep learning concepts
- Model building and training
- TensorFlow data pipelines
- Common errors and troubleshooting

Example queries:
- "How do I compile and train a Keras model?"
- "How do I use tf.data.Dataset to load and preprocess large datasets?"
- "What are the parameters for a Conv2D layer?"
- "How do I create and use a custom loss function?"

## Configuration

The application can be configured by setting the following environment variables:

```
# LLM Provider (gemini, openai, anthropic, cohere)
LLM_PROVIDER=gemini

# Embedding Provider (gemini, openai, cohere)
EMBEDDING_PROVIDER=gemini

# API Keys
GOOGLE_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
COHERE_API_KEY=your_cohere_api_key
export LANGSMITH_API_KEY="your_langsmith_api_key"
export LANGSMITH_TRACING=true

# Model selections (optional, defaults provided)
GEMINI_LLM_MODEL=gemini-2.0-flash
OPENAI_LLM_MODEL=gpt-4o
ANTHROPIC_LLM_MODEL=claude-3-sonnet-20240229
COHERE_LLM_MODEL=command
```

## Project Structure

```
chatTensorFlow/
├── app/
│   ├── core/         # Core utilities and model interfaces
│   ├── graphs/       # LangGraph state management and flow
│   ├── retriever/    # Document retrieval components
│   ├── static/       # Static files (CSS, JS, images)
│   ├── templates/    # HTML templates
│   └── main.py       # Application entry point
├── config/           # Application configuration
├── DATA/             # Vector store data
├── logs/             # Application logs
├── processor/        # Web crawling and data ingestion
├── tensorflow_docs/  # Crawled TensorFlow documentation
└── tests/            # Unit and integration tests
```

## How It Works

1. **User Query Analysis**: The system classifies the query as TensorFlow related, general, or requiring more information
2. **Research Planning**: For TensorFlow queries, a research plan is created
3. **Document Retrieval**: Relevant documentation is retrieved using semantic search
4. **Response Generation**: A detailed, accurate response is generated citing relevant documentation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph)
- Uses [ChromaDB](https://github.com/chroma-core/chroma) for vector storage
- Powered by various LLM providers (Gemini, OpenAI, Anthropic, Cohere)
- Documentation content from [TensorFlow](https://www.tensorflow.org/)
- Developed by [Lalan Kumar](https://github.com/kumar8074)