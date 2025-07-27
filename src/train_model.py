import schedule
import time
from response_trainer import ResponseTrainer
import logging
import os
from datetime import datetime

# Configure logging
if not os.path.exists("logs"):
    os.makedirs("logs")
    
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/training_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("model_trainer")

def run_training_job():
    """Daily job to process feedback and export training data"""
    logger.info("Starting scheduled training job")
    
    try:
        trainer = ResponseTrainer()
        
        # Export training data
        count = trainer.export_training_data()
        logger.info(f"Exported {count} high-quality examples for training")
        
        # Additional training logic could be added here
        # For example, periodically fine-tuning a smaller model
        
        logger.info("Training job completed successfully")
    except Exception as e:
        logger.error(f"Error in training job: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting training scheduler")
    
    # Schedule daily training
    schedule.every().day.at("03:00").do(run_training_job)  # Run at 3 AM
    
    # Run once at startup
    run_training_job()
    
    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)