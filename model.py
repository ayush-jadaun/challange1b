import os
import logging
from sentence_transformers import SentenceTransformer
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Define the models to be downloaded
SENTENCE_MODEL_NAME = 'all-MiniLM-L6-v2'
SPACY_MODEL_NAME = 'en_core_web_sm'

# Define the local directory to save models
MODEL_DIR = 'models'
SENTENCE_MODEL_PATH = os.path.join(MODEL_DIR, SENTENCE_MODEL_NAME)
SPACY_MODEL_PATH = os.path.join(MODEL_DIR, SPACY_MODEL_NAME)

def download_and_save_models():
    """
    Downloads and saves the required machine learning models to a local directory
    for offline use.
    """
    logger.info("Starting model download and setup process...")

    # Create the target directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info(f"Models will be saved to: '{MODEL_DIR}' directory.")

    # --- Download and Save Sentence Transformer Model ---
    try:
        logger.info(f"Downloading Sentence Transformer model: '{SENTENCE_MODEL_NAME}'...")
        # Load the model from the hub
        model = SentenceTransformer(SENTENCE_MODEL_NAME)
        # Save the model to the specified local path
        model.save(SENTENCE_MODEL_PATH)
        logger.info(f"Successfully saved '{SENTENCE_MODEL_NAME}' to '{SENTENCE_MODEL_PATH}'")
    except Exception as e:
        logger.error(f"Failed to download or save Sentence Transformer model. Error: {e}")
        # Exit if the essential model fails to download
        exit(1)

    # --- Download and Save SpaCy Model ---
    try:
        logger.info(f"Downloading SpaCy model: '{SPACY_MODEL_NAME}'...")
        # Download the model package
        spacy.cli.download(SPACY_MODEL_NAME)
        # Load the downloaded model
        nlp = spacy.load(SPACY_MODEL_NAME)
        # Save it to the local models directory
        nlp.to_disk(SPACY_MODEL_PATH)
        logger.info(f"Successfully saved '{SPACY_MODEL_NAME}' to '{SPACY_MODEL_PATH}'")
    except Exception as e:
        logger.error(f"Failed to download or save SpaCy model. Error: {e}")
        # Exit if the essential model fails to download
        exit(1)

    logger.info("=" * 60)
    logger.info("All models have been downloaded and saved successfully.")
    logger.info("The application is now ready for offline execution.")
    logger.info("=" * 60)

if __name__ == "__main__":
    download_and_save_models()