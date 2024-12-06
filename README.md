# AI Real Estate Assistant

A Telegram bot powered by AI that helps users find real estate properties using natural language queries. The bot uses vector similarity search and large language models to provide relevant property recommendations.

## Features

- Natural language property search
- Support for multiple LLM backends (GPT, Claude, Llama)
- Vector similarity search using Chroma or FAISS
- Persistent vector storage
- Property matching based on multiple criteria
- Detailed property information display

## Requirements

- Python 3.9 or higher
- 8GB RAM minimum (16GB recommended for Llama models)
- GPU optional but recommended for Llama models
- Docker (optional, for development)

## Installation

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.template .env
```

## Configuration

Edit `.env` file with your settings:

### LLM Configuration

Choose one of the following LLM configurations:

1. OpenAI (GPT):
```env
LLM_TYPE=gpt
OPENAI_API_KEY=your_key_here
```

2. Anthropic (Claude):
```env
LLM_TYPE=claude
ANTHROPIC_API_KEY=your_key_here
```

3. Llama:
```env
LLM_TYPE=llama
LLAMA_MODEL_PATH=/path/to/model.gguf
```

### Vector Store Configuration

Choose your vector store:

1. ChromaDB (recommended for most cases):
```env
VECTOR_STORE_TYPE=chroma
CHROMA_PERSIST_DIR=./chroma_db
```

2. FAISS:
```env
VECTOR_STORE_TYPE=faiss
CHROMA_PERSIST_DIR=./faiss_db
```

### Telegram Bot Setup

1. Create a new bot with @BotFather on Telegram
2. Add the bot token to your .env:
```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
```

## Running the Bot

### Normal Start
```bash
python app.py
```

### Force Vector Database Reload
```bash
python app.py --reload-vectors
```

## Usage

1. Start conversation with bot using `/start`
2. Use natural language to describe properties you're looking for
3. Use `/help` to see example queries and tips

Example queries:
- "Show me 2-bedroom apartments in Guardamar under 150000 euros"
- "I need a villa with a pool near the beach"
- "Find properties with 3 bedrooms and 2 bathrooms in Torrevieja"

## Development

### Project Structure
```
real_estate_bot/
├── requirements.txt
├── .env
├── data/
│   └── properties.xml
├── models/
│   ├── __init__.py
│   └── property.py
├── vectorstore/
│   ├── __init__.py
│   └── base.py
├── llm/
│   ├── __init__.py
│   └── base.py
├── utils/
│   ├── __init__.py
│   └── xml_loader.py
├── config.py
└── app.py
```

## Troubleshooting

### Common Issues

1. Memory errors with Llama:
- Reduce context size in llama_handler.py
- Use a smaller model
- Increase system RAM

2. Slow responses:
- Check internet connection for API-based models
- Consider switching to FAISS for faster searches
- Reduce the number of properties in the database

3. Vector store errors:
- Delete the persist directory and restart
- Check disk space
- Verify file permissions

### Performance Optimization

1. For faster responses:
- Use FAISS instead of Chroma
- Reduce `top_k` in search operations
- Use local models when possible

2. For better accuracy:
- Use GPT-4 or Claude instead of local models
- Increase context size if using Llama
- Fine-tune embeddings model if needed
