"""
Logger Utility

Centralized logging configuration for the evaluation pipeline.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str, 
    log_file: str = None, 
    level: str = "INFO"
) -> logging.Logger:
    """
    Set up a logger with console and file handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        # Create logs directory if needed
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_evaluation_result(result: dict, logger: logging.Logger):
    """
    Log evaluation result in a structured format.
    
    Args:
        result: Evaluation result dictionary
        logger: Logger instance
    """
    logger.info("=" * 60)
    logger.info("EVALUATION RESULT")
    logger.info("=" * 60)
    logger.info(f"Message ID: {result.get('message_id', 'N/A')}")
    logger.info(f"Timestamp: {result.get('timestamp', 'N/A')}")
    logger.info(f"Overall Score: {result.get('overall_score', 0):.3f}")
    logger.info("")
    logger.info("RELEVANCE METRICS:")
    if 'relevance' in result:
        for key, value in result['relevance'].items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.3f}")
            else:
                logger.info(f"  {key}: {value}")
    logger.info("")
    logger.info("HALLUCINATION METRICS:")
    if 'hallucination' in result:
        for key, value in result['hallucination'].items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.3f}")
            else:
                logger.info(f"  {key}: {value}")
    logger.info("")
    logger.info("PERFORMANCE METRICS:")
    if 'performance' in result:
        for key, value in result['performance'].items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  {key}: {value}")
    logger.info("=" * 60)