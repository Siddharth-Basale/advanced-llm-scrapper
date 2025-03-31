import os
import time
import logging
from flask import Flask, render_template, request, jsonify
from phi.agent import Agent
from phi.tools.crawl4ai_tools import Crawl4aiTools
from phi.model.google import Gemini
from dotenv import load_dotenv
import traceback
import re

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scraper_debug.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.critical("GOOGLE_API_KEY not found in .env file")
    raise ValueError("GOOGLE_API_KEY not found in .env file")

app = Flask(__name__)

# Constants
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB

def scrape_website(url, debug_id="DEBUG"):
    """Scrape website content using Crawl4aiTools"""
    try:
        logger.debug(f"[{debug_id}] Starting scrape with Crawl4aiTools")
        crawler = Crawl4aiTools(max_length=None)
        scraped_data = crawler.web_crawler(url)
        logger.debug(f"[{debug_id}] Scrape completed successfully")
        return scraped_data
    except Exception as e:
        logger.error(f"[{debug_id}] Scraping failed: {str(e)}")
        raise

def get_structured_gemini_response(content, prompt):
    """Analyze scraped content with Gemini and return cleaned response"""
    agent = Agent(model=Gemini(id="gemini-2.0-flash-exp", temperature=0.2))
    full_prompt = f"Based on this website content:\n\n{content}\n\n{prompt}"
    
    response = agent.run(full_prompt)  # Get raw response
    response_str = str(response)  # Ensure response is a string

    # Clean the response
    cleaned_content = response_str.split("content_type=")[0].strip()
    
    # Replace single asterisks used for emphasis with bold tags
    cleaned_content = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'**\1**', cleaned_content)
    
    # Replace newlines with markdown line breaks
    cleaned_content = cleaned_content.replace('\n', '  \n')  # Markdown needs two spaces for line breaks
    
    return cleaned_content

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/scrape", methods=["POST"])
def scrape_url():
    debug_id = f"SCRAPE-{time.time_ns()}"
    logger.debug(f"[{debug_id}] Starting scrape request")
    
    url = request.form.get("url", "").strip()
    if not url:
        logger.warning(f"[{debug_id}] Empty URL provided")
        return jsonify({
            "error": "URL is required",
            "status": "error",
            "debug_id": debug_id
        }), 400

    try:
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'
            logger.debug(f"[{debug_id}] Added https prefix: {url}")

        # Scrape content
        logger.debug(f"[{debug_id}] Scraping content...")
        scraped_content = scrape_website(url, debug_id)
        
        return jsonify({
            "content": scraped_content,
            "status": "success",
            "debug_id": debug_id
        })

    except Exception as e:
        logger.error(f"[{debug_id}] Scraping failed: {str(e)}")
        logger.debug(f"[{debug_id}] Full traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": f"Scraping failed: {str(e)}",
            "status": "error",
            "debug_id": debug_id
        }), 500

@app.route("/analyze", methods=["POST"])
def analyze_content():
    debug_id = f"ANALYZE-{time.time_ns()}"
    logger.debug(f"[{debug_id}] Starting analysis request")
    
    content = request.form.get("content", "").strip()
    prompt = request.form.get("prompt", "").strip()
    
    if not content:
        logger.warning(f"[{debug_id}] Empty content provided")
        return jsonify({
            "error": "Content is required",
            "status": "error",
            "debug_id": debug_id
        }), 400
        
    if not prompt:
        logger.warning(f"[{debug_id}] Empty prompt provided")
        return jsonify({
            "error": "Prompt is required",
            "status": "error",
            "debug_id": debug_id
        }), 400

    try:
        logger.debug(f"[{debug_id}] Getting structured Gemini response...")
        analysis = get_structured_gemini_response(content, prompt)
        
        # Return clean structured response
        return jsonify({
            "analysis": analysis,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"[{debug_id}] Analysis failed: {str(e)}")
        logger.debug(f"[{debug_id}] Full traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": f"Analysis failed: {str(e)}",
            "status": "error"
        }), 500

if __name__ == "__main__":
    logger.info("Starting Flask application")
    app.run(debug=True, host='0.0.0.0', port=5000)