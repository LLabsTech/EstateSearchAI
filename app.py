import os
import asyncio
import logging
import argparse
import shutil
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    CallbackQueryHandler,
    filters,
)
from typing import Optional, Dict
from config import config, StorageMode
from utils.factories import create_llm_handler, create_vector_store
from utils.xml_loader import load_properties_from_xml

# Global handlers
llm_handler = None
vector_store = None
effective_storage_mode = None

# Store property descriptions for callback handling
property_descriptions: Dict[str, str] = {}

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def determine_storage_mode(cmd_storage_mode: Optional[StorageMode] = None) -> StorageMode:
    """
    Determine storage mode using the following priority:
    1. Command line argument (if provided)
    2. Environment variable (from config)
    3. Auto-detection
    """
    global effective_storage_mode

    if effective_storage_mode is not None:
        # Use already determined mode
        return effective_storage_mode

    # Command line argument has the highest priority
    if cmd_storage_mode is not None:
        effective_storage_mode = cmd_storage_mode
        logger.info(f"Using storage mode from command line: {cmd_storage_mode.value}")
        return effective_storage_mode

    # Environment variable is the next priority
    if config.storage_mode:
        effective_storage_mode = config.storage_mode
        logger.info(f"Using storage mode from environment: {config.storage_mode.value}")
        return effective_storage_mode

    # Auto-detection has the lowest priority
    chroma_dir = os.path.abspath(config.chroma_persist_dir)
    chroma_db_exists = os.path.exists(os.path.join(chroma_dir, '.chroma', 'chroma.sqlite3'))
    effective_storage_mode = StorageMode.DISK if chroma_db_exists else StorageMode.MEMORY
    logger.info(f"Auto-detected storage mode: {effective_storage_mode.value}")
    return effective_storage_mode

def initialize_vector_store(force_reload: bool = False, storage_mode: Optional[StorageMode] = None):
    """Initialize vector store and load properties if needed"""
    # Determine effective storage mode
    effective_mode = determine_storage_mode(storage_mode)
    
    # Update config with final storage mode
    config.storage_mode = effective_mode
    
    # Create vector store instance
    # Delete chroma_db folder if force_reload is True
    if force_reload and effective_mode == StorageMode.DISK:
        chroma_dir = os.path.abspath(config.chroma_persist_dir)
        if os.path.exists(chroma_dir):
            logger.info("Force reload: removing existing vector store directory")
            shutil.rmtree(chroma_dir)
    
    vector_store = create_vector_store(config)
    
    if effective_mode == StorageMode.MEMORY:
        # For memory mode, we always need to load vectors
        logger.info("Memory mode: loading properties from XML...")
        folder_path = os.path.dirname(__file__)
        xml_path = os.path.join(folder_path, "data", "properties.xml")
        properties = load_properties_from_xml(xml_path)
        vector_store.load_properties(properties)
        logger.info(f"Loaded {len(properties)} properties into memory store")
    else:
        # For disk mode, only load if forced or needed
        if force_reload:
            logger.info("Force reload requested for persistent storage")
            folder_path = os.path.dirname(__file__)
            xml_path = os.path.join(folder_path, "data", "properties.xml")
            properties = load_properties_from_xml(xml_path)
            vector_store.load_properties(properties)
            logger.info(f"Reloaded {len(properties)} properties into persistent storage")
        elif vector_store.needs_loading():
            logger.info("No existing vector data found, loading from XML...")
            folder_path = os.path.dirname(__file__)
            xml_path = os.path.join(folder_path, "data", "properties.xml")
            properties = load_properties_from_xml(xml_path)
            vector_store.load_properties(properties)
            logger.info(f"Loaded {len(properties)} properties into persistent storage")
        else:
            # Initialize store without loading new data
            vector_store._initialize_store()
            logger.info("Using existing persistent vector store")
    
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
    help_text = r"""
🏠 *Property Search Help*

You can ask me about properties using natural language\. Include details like:
\- Location preferences
\- Price range
\- Number of bedrooms/bathrooms
\- Property type \(apartment, villa, etc\.\)
\- Special features \(pool, garage, etc\.\)
\- Views or proximity to amenities

*Example Queries:*
1\. "Find me a 2\-bedroom apartment in Guardamar under 200,000 euros"
2\. "Show villas with pools in Torrevieja"
3\. "I need a property near the beach with at least 3 bedrooms"
4\. "What properties are available in Alicante with a garage?"

*Tips:*
\- Be specific about your requirements
\- You can mention multiple criteria
\- Ask for clarification if needed
\- Use /start to restart the conversation
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
        await update.message.reply_text(error_message)

async def toggle_description(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle description toggle button press"""
    query = update.callback_query
    await query.answer()
    
    # Extract property ID and current state from callback data
    prop_id, show_desc = query.data.split('_')[1:]  # Format: "desc_propId_state"
    show_desc = show_desc == "1"
    
    # Get the property description
    description = property_descriptions.get(prop_id, "Description not available")
    if isinstance(description, dict) and 'es' in description:
        description = description['es']
    
    # Create property display text
    base_text = query.message.text.split('\n\nDescription:')[0]  # Get text before description
    if show_desc:
        new_text = f"{base_text}\n\nDescription: {description}"
        new_state = "0"
        button_text = "Hide Description"
    else:
        new_text = base_text
        new_state = "1"
        button_text = "Show Description"
    
    try:
        # Update keyboard
        keyboard = [[InlineKeyboardButton(button_text, callback_data=f"desc_{prop_id}_{new_state}")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.message.edit_text(new_text[:4096], reply_markup=reply_markup) # parse_mode='MarkdownV2'
    except Exception as e:
        logger.error(f"Error updating message: {str(e)}")
        await query.answer("Error updating message - text might be too long")

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

        # Generate high-level analysis using LLM
        response = await llm_handler.generate_response(user_query, matches)
        
        # Convert ** to * for markdown and escape special characters
        response = response.replace("**", "*")
        special_chars = ['_', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
        for char in special_chars:
            response = response.replace(char, f'\\{char}')
            
        await update.message.reply_text(response[:4096], parse_mode='MarkdownV2')
        
        # Send detailed information for each match
        for match in matches[:5]:
            prop = match.property
            
            # Store description for callback handling
            if isinstance(prop.desc, dict) and 'es' in prop.desc:
                property_descriptions[str(prop.id)] = prop.desc['es']
            else:
                property_descriptions[str(prop.id)] = "Description not available"
            
            # Get display text using property's method
            display_text = prop.to_display_text()
            
            # Create keyboard with description toggle button
            keyboard = [[InlineKeyboardButton(
                "Show Description",
                callback_data=f"desc_{prop.id}_1"
            )]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Send property details
            await update.message.reply_text(display_text, reply_markup=reply_markup, parse_mode='MarkdownV2')
            
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        await update.message.reply_text(
            "I apologize, but something went wrong while processing your request. "
            "Please try again or contact support if the problem persists."
        )

def main() -> None:
    """Start the bot"""
    global llm_handler, vector_store

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Real Estate Telegram Bot")
    parser.add_argument("--reload-vectors",
                        action="store_true",
                        help="Force reload of vector database from XML")
    parser.add_argument("--storage-mode",
                        choices=["memory", "disk"],
                        help="Storage mode for vector database (auto-detect if not specified)")
    args = parser.parse_args()

    # Initialize handlers
    llm_handler = create_llm_handler(config)
    vector_store = initialize_vector_store(
        force_reload=args.reload_vectors,
        storage_mode=StorageMode(args.storage_mode) if args.storage_mode else None
    )

    # Initialize and run bot
    application = Application.builder().token(config.telegram_token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_message))
    application.add_handler(CallbackQueryHandler(toggle_description, pattern="^desc_"))
    application.add_error_handler(handle_error)

    logger.info("Starting bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
