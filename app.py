import os
import asyncio
import logging
import argparse
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import config
from utils.factories import create_llm_handler, create_vector_store
from utils.xml_loader import load_properties_from_xml

# Global handlers
llm_handler = None
vector_store = None

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def initialize_vector_store(force_reload: bool = False):
    """Initialize vector store and load properties if needed"""
    vector_store = create_vector_store(config)
    folder_path = os.path.dirname(__file__)
    xml_path = os.path.join(folder_path, "data", "properties.xml")
    
    if force_reload or vector_store.needs_loading():
        logger.info("Loading properties from XML and updating vector store...")
        properties = load_properties_from_xml(xml_path)
        vector_store.load_properties(properties)
        logger.info(f"Loaded {len(properties)} properties into vector store")
    else:
        logger.info("Using existing vector store")
    
    return vector_store

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /start command"""
    user = update.effective_user
    welcome_message = (
        f"Hi {user.mention_html()}! 👋\n\n"
        "I'm your real estate assistant powered by AI. I can help you find properties "
        "based on your specific requirements.\n\n"
        "Simply describe what you're looking for in natural language. For example:\n"
        "- 'Show me 2-bedroom apartments in Guardamar under 150000 euros'\n"
        "- 'I need a villa with a pool near the beach'\n"
        "- 'Find me properties with 3 bedrooms and 2 bathrooms in Torrevieja'\n\n"
        "Use /help to see more examples and tips."
    )
    await update.message.reply_html(welcome_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /help command"""
    help_text = r"""  # Added 'r' prefix for raw string
🏠 *Property Search Help*

You can ask me about properties using natural language\. Include details like:
- Location preferences
- Price range
- Number of bedrooms/bathrooms
- Property type \(apartment, villa, etc\.\)
- Special features \(pool, garage, etc\.\)
- Views or proximity to amenities

*Example Queries:*
1\. "Find me a 2\-bedroom apartment in Guardamar under 200,000 euros"
2\. "Show villas with pools in Torrevieja"
3\. "I need a property near the beach with at least 3 bedrooms"
4\. "What properties are available in Alicante with a garage?"

*Tips:*
- Be specific about your requirements
- You can mention multiple criteria
- Ask for clarification if needed
- Use /start to restart the conversation
    """
    await update.message.reply_markdown_v2(help_text)

async def handle_error(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Error handler"""
    logger.error(f"Update {update} caused error {context.error}")
    error_message = (
        "I apologize, but I encountered an error while processing your request. "
        "Please try again or use /help for guidance."
    )
    if update.effective_message:
        await update.effective_message.reply_text(error_message)

async def process_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle user messages"""
    if not update.message or not update.message.text:
        return

    user_query = update.message.text
    
    # Send typing indicator
    await update.message.chat.send_action("typing")
    
    try:
        # Search for matching properties
        matches = vector_store.search(user_query, top_k=5)
        
        if not matches:
            await update.message.reply_text(
                "I couldn't find any properties matching your requirements. "
                "Please try different criteria or use /help for guidance."
            )
            return

        # Generate response using LLM
        response = await llm_handler.generate_response(user_query, matches)
        await update.message.reply_text(response)
        
        # Send detailed information for top 3 matches
        for match in matches[:3]:
            await update.message.reply_text(match.property.to_display_text())
            
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        await update.message.reply_text(
            "I apologize, but something went wrong while processing your request. "
            "Please try again or contact support if the problem persists."
        )

def main() -> None:
    """Start the bot"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Real Estate Telegram Bot')
    parser.add_argument('--reload-vectors',
                       action='store_true',
                       help='Force reload of vector database from XML')
    args = parser.parse_args()

    # Initialize handlers
    global llm_handler, vector_store
    llm_handler = create_llm_handler(config)
    vector_store = initialize_vector_store(force_reload=args.reload_vectors)

    # Initialize bot
    application = (
        Application.builder()
        .token(config.telegram_token)
        .build()
    )

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_message))
    application.add_error_handler(handle_error)

    # Start bot
    logger.info("Starting bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
