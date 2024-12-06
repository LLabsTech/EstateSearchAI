# Real Estate Bot Setup Instructions

## Prerequisites
- Python 3.9 or higher
- 8GB RAM minimum (16GB recommended for Llama models)
- GPU optional but recommended for Llama models

## Installation Steps

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

4. Configure your .env file with appropriate values:
- For OpenAI (GPT):
  - Set `LLM_TYPE=gpt`
  - Add your OpenAI API key to `OPENAI_API_KEY`

- For Claude:
  - Set `LLM_TYPE=claude`
  - Add your Anthropic API key to `ANTHROPIC_API_KEY`

- For Llama:
  - Set `LLM_TYPE=llama`
  - Download a Llama model (e.g., from Hugging Face)
  - Set `LLAMA_MODEL_PATH` to your model path

5. Choose vector store:
- For Chroma (recommended for most cases):
  ```
  VECTOR_STORE_TYPE=chroma
  CHROMA_PERSIST_DIR=./chroma_db
  ```

- For FAISS:
  ```
  VECTOR_STORE_TYPE=faiss
  CHROMA_PERSIST_DIR=./faiss_db
  ```

6. Set up Telegram bot:
- Create a new bot with @BotFather on Telegram
- Add the bot token to `TELEGRAM_BOT_TOKEN` in .env

## Running the Bot

1. Start the bot:
```bash
python app.py
```

2. Test the bot in Telegram:
- Send /start to begin
- Send /help for usage instructions

## Switching LLM Models

You can switch between models by changing the `LLM_TYPE` in your .env file:

### OpenAI GPT
```
LLM_TYPE=gpt
OPENAI_API_KEY=your_key_here
```

### Claude
```
LLM_TYPE=claude
ANTHROPIC_API_KEY=your_key_here
```

### Llama
```
LLM_TYPE=llama
LLAMA_MODEL_PATH=/path/to/model.gguf
```

## Vector Store Management

### Chroma
- Data is automatically persisted to `CHROMA_PERSIST_DIR`
- No additional setup required

### FAISS
- Indexes are saved to `CHROMA_PERSIST_DIR`
- Faster for larger datasets
- Requires more memory

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

### Getting Help

1. Check logs for detailed error messages
2. Verify all environment variables are set correctly
3. Ensure all dependencies are installed properly
4. Check API key validity for OpenAI/Claude

## Performance Optimization

1. For faster responses:
- Use FAISS instead of Chroma
- Reduce `top_k` in search operations
- Use local models when possible

2. For better accuracy:
- Use GPT-4 or Claude instead of local models
- Increase context size if using Llama
- Fine-tune embeddings model if needed
