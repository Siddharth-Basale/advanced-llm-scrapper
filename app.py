import os
import time
import logging
from flask import Flask, render_template, request, jsonify, send_file
import requests
from bs4 import BeautifulSoup, Comment
from dotenv import load_dotenv
from groq import Groq
from io import BytesIO
from playwright.sync_api import sync_playwright
import traceback
from urllib.parse import urlparse
import random

# Configure verbose logging
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
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    logger.critical("GROQ_API_KEY not found in .env file")
    raise ValueError("GROQ_API_KEY not found in .env file")

app = Flask(__name__)

# Constants
SCREENSHOT_DIR = "static/screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB
REQUEST_TIMEOUT = 30  # seconds
PLAYWRIGHT_TIMEOUT = 60000  # milliseconds

# User-Agents for rotation
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1'
]

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def clean(content, debug_id="DEBUG"):
    """Enhanced cleaning with detailed logging and proper Comment handling"""
    logger.debug(f"[{debug_id}] Cleaning content (length: {len(content)})")
    try:
        if not content:
            return ""
            
        soup = BeautifulSoup(content, "html.parser")
        
        # Remove unwanted tags
        removed_tags = set()
        tags_to_remove = ["script", "style", "noscript", "iframe", "nav", "footer", "header", "form"]
        for tag in soup(tags_to_remove):
            removed_tags.add(tag.name)
            tag.decompose()
        logger.debug(f"[{debug_id}] Removed tags: {', '.join(removed_tags) or 'None'}")
        
        # Remove comments
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        logger.debug(f"[{debug_id}] Found {len(comments)} comments")
        for comment in comments:
            comment.extract()
            
        # Clean whitespace and normalize
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        logger.debug(f"[{debug_id}] Final line count: {len(lines)}")
        
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"[{debug_id}] Cleaning failed: {str(e)}")
        logger.debug(f"[{debug_id}] Content sample (first 200 chars): {content[:200]}")
        raise

def fetch_page_content(url, debug_id="DEBUG"):
    """Enhanced fetch with rotating headers and better error handling"""
    try:
        headers = {
            'User-Agent': get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Referer': 'https://www.google.com/',
            'DNT': '1'
        }
        
        logger.debug(f"[{debug_id}] Fetching URL with headers: {headers}")
        
        # Add random delay to avoid rate limiting
        time.sleep(random.uniform(0.5, 2.0))
        
        response = requests.get(
            url,
            headers=headers,
            timeout=REQUEST_TIMEOUT,
            stream=True
        )
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' not in content_type:
            logger.warning(f"[{debug_id}] Unexpected content type: {content_type}")
            
        return response.text
        
    except requests.exceptions.SSLError:
        logger.warning(f"[{debug_id}] SSL error, trying without verification")
        response = requests.get(
            url,
            headers=headers,
            verify=False,
            timeout=REQUEST_TIMEOUT
        )
        return response.text
        
    except requests.exceptions.TooManyRedirects:
        logger.warning(f"[{debug_id}] Too many redirects, trying with allow_redirects=False")
        response = requests.get(
            url,
            headers=headers,
            allow_redirects=False,
            timeout=REQUEST_TIMEOUT
        )
        return response.text
        
    except Exception as e:
        logger.error(f"[{debug_id}] Fetch failed: {str(e)}")
        raise

def capture_screenshot(url, debug_id="DEBUG"):
    """Enhanced screenshot with multiple fallback strategies"""
    logger.debug(f"[{debug_id}] Starting screenshot for {url}")
    screenshot_path = None
    
    try:
        with sync_playwright() as p:
            # Launch browser with specific options
            browser = p.chromium.launch(
                headless=True,
                timeout=PLAYWRIGHT_TIMEOUT
            )
            context = browser.new_context(
                user_agent=get_random_user_agent(),
                viewport={'width': 1280, 'height': 720},
                java_script_enabled=True,
                ignore_https_errors=True
            )
            
            page = context.new_page()
            
            try:
                # Try to load page with multiple strategies
                logger.debug(f"[{debug_id}] Navigating to URL...")
                
                # First attempt with default settings
                try:
                    page.goto(
                        url,
                        timeout=PLAYWRIGHT_TIMEOUT,
                        wait_until="domcontentloaded"
                    )
                except:
                    # Fallback to networkidle if domcontentloaded fails
                    page.goto(
                        url,
                        timeout=PLAYWRIGHT_TIMEOUT,
                        wait_until="networkidle"
                    )
                
                # Additional wait for content
                page.wait_for_load_state("networkidle", timeout=PLAYWRIGHT_TIMEOUT/2)
                
                # Create screenshot filename
                timestamp = int(time.time())
                screenshot_path = os.path.join(SCREENSHOT_DIR, f"screenshot_{timestamp}.png")
                
                logger.debug(f"[{debug_id}] Taking screenshot...")
                page.screenshot(
                    path=screenshot_path,
                    full_page=True,
                    timeout=PLAYWRIGHT_TIMEOUT/2
                )
                return screenshot_path
                
            except Exception as e:
                logger.warning(f"[{debug_id}] Screenshot failed: {str(e)}")
                
                # Attempt to capture visible portion if full page fails
                try:
                    partial_path = os.path.join(SCREENSHOT_DIR, f"partial_{timestamp}.png")
                    page.screenshot(
                        path=partial_path,
                        full_page=False,
                        timeout=PLAYWRIGHT_TIMEOUT/2
                    )
                    return partial_path
                except Exception as e:
                    logger.warning(f"[{debug_id}] Partial screenshot failed: {str(e)}")
                    return None
                    
            finally:
                try:
                    context.close()
                    browser.close()
                except Exception as e:
                    logger.warning(f"[{debug_id}] Browser cleanup failed: {str(e)}")
                    
    except Exception as e:
        logger.error(f"[{debug_id}] Playwright initialization failed: {str(e)}")
        return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/scrape", methods=["POST"])
def scrape_url():
    """Main scraping endpoint with comprehensive error handling"""
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
        # Validate and normalize URL
        if not url.startswith(('http://', 'https://')):
            url = f'https://{url}'
            logger.debug(f"[{debug_id}] Added https prefix: {url}")

        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            logger.error(f"[{debug_id}] Invalid URL structure: {url}")
            raise ValueError("Invalid URL format")

        # Fetch content
        logger.debug(f"[{debug_id}] Fetching page content...")
        raw_content = fetch_page_content(url, debug_id)
        
        # Process content
        logger.debug(f"[{debug_id}] Processing content...")
        soup = BeautifulSoup(raw_content, "html.parser")
        body = soup.body or soup  # Fallback to entire document if no body
        cleaned_content = clean(str(body), debug_id)
        word_count = len(cleaned_content.split())
        
        # Attempt screenshot (non-blocking)
        screenshot_url = None
        try:
            screenshot_path = capture_screenshot(url, debug_id)
            if screenshot_path:
                screenshot_url = f"/{screenshot_path}"
        except Exception as e:
            logger.warning(f"[{debug_id}] Screenshot failed (non-critical): {str(e)}")

        return jsonify({
            "dom_content": cleaned_content,
            "screenshot": screenshot_url,
            "word_count": word_count,
            "status": "success",
            "debug_id": debug_id
        })

    except ValueError as e:
        logger.error(f"[{debug_id}] Validation error: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error",
            "debug_id": debug_id
        }), 400
        
    except Exception as e:
        logger.error(f"[{debug_id}] Scraping failed: {str(e)}")
        logger.debug(f"[{debug_id}] Full traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": f"Scraping failed: {str(e)}",
            "status": "error",
            "debug_id": debug_id,
            "suggestion": "Try a different URL or check if the site blocks scrapers"
        }), 500

@app.route("/summarize", methods=["POST"])
def summarize_content():
    debug_id = f"SUMMARY-{time.time_ns()}"
    content = request.form.get("content", "").strip()
    summary_length = request.form.get("length", "medium")
    
    if not content:
        logger.warning(f"[{debug_id}] Empty content provided")
        return jsonify({
            "error": "Content is required",
            "status": "error",
            "debug_id": debug_id
        }), 400
    
    try:
        logger.debug(f"[{debug_id}] Generating {summary_length} summary...")
        client = Groq(api_key=API_KEY)
        
        length_instructions = {
            "short": "Provide a very concise summary in 1-2 sentences.",
            "medium": "Provide a summary in about 3-5 sentences.",
            "long": "Provide a detailed summary in about 7-10 sentences."
        }
        
        prompt = (
            f"Please summarize the following content. {length_instructions[summary_length]}\n\n"
            f"Content:\n{content[:20000]}\n\n"  # Limit content to avoid token limits
            "Summary:"
        )
        
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
        )
        
        summary = completion.choices[0].message.content
        
        return jsonify({
            "summary": summary,
            "status": "success",
            "debug_id": debug_id
        })
        
    except Exception as e:
        logger.error(f"[{debug_id}] Summary generation failed: {str(e)}")
        return jsonify({
            "error": f"Summary generation failed: {str(e)}",
            "status": "error",
            "debug_id": debug_id
        }), 500

@app.route("/parse", methods=["POST"])
def parse_content():
    debug_id = f"PARSE-{time.time_ns()}"
    dom_content = request.form.get("dom_content", "").strip()
    parse_description = request.form.get("parse_description", "").strip()
    
    if not dom_content:
        logger.warning(f"[{debug_id}] Empty DOM content provided")
        return jsonify({
            "error": "DOM content is required",
            "status": "error",
            "debug_id": debug_id
        }), 400
        
    if not parse_description:
        logger.warning(f"[{debug_id}] Empty parse description provided")
        return jsonify({
            "error": "Parse description is required",
            "status": "error",
            "debug_id": debug_id
        }), 400

    try:
        logger.debug(f"[{debug_id}] Parsing content...")
        client = Groq(api_key=API_KEY)
        
        # Split content into chunks if too large
        max_chunk_size = 6000
        dom_chunks = [dom_content[i:i+max_chunk_size] for i in range(0, len(dom_content), max_chunk_size)]
        logger.debug(f"[{debug_id}] Split into {len(dom_chunks)} chunks")
        
        parsed_results = []
        for i, chunk in enumerate(dom_chunks):
            try:
                prompt = (
                    f"Extract specific information from this text based on: {parse_description}\n\n"
                    f"Text content:\n{chunk}\n\n"
                    "Instructions:\n"
                    "1. Extract only the requested information\n"
                    "2. Return only the data with no additional text\n"
                    "3. If nothing matches, return empty string"
                )
                
                completion = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1024,
                )
                
                result = completion.choices[0].message.content.strip()
                if result:
                    parsed_results.append(result)
                    
            except Exception as e:
                logger.warning(f"[{debug_id}] Chunk {i} parsing failed: {str(e)}")
                continue
                
        final_result = "\n\n".join(parsed_results) if parsed_results else "No matching content found"
        
        return jsonify({
            "parsed_results": final_result,
            "status": "success",
            "debug_id": debug_id
        })
        
    except Exception as e:
        logger.error(f"[{debug_id}] Parsing failed: {str(e)}")
        return jsonify({
            "error": f"Parsing failed: {str(e)}",
            "status": "error",
            "debug_id": debug_id
        }), 500

@app.route("/download", methods=["POST"])
def download_content():
    debug_id = f"DOWNLOAD-{time.time_ns()}"
    content = request.form.get("content", "").strip()
    
    if not content:
        logger.warning(f"[{debug_id}] Empty content provided for download")
        return jsonify({
            "error": "No content to download",
            "status": "error",
            "debug_id": debug_id
        }), 400
    
    try:
        # Create in-memory file
        mem_file = BytesIO()
        mem_file.write(content.encode('utf-8'))
        mem_file.seek(0)
        
        # Generate filename with timestamp
        filename = f"scraped_content_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        
        logger.debug(f"[{debug_id}] Preparing download: {filename}")
        
        return send_file(
            mem_file,
            as_attachment=True,
            download_name=filename,
            mimetype="text/plain"
        )
        
    except Exception as e:
        logger.error(f"[{debug_id}] Download failed: {str(e)}")
        return jsonify({
            "error": f"Download failed: {str(e)}",
            "status": "error",
            "debug_id": debug_id
        }), 500

if __name__ == "__main__":
    logger.info("Starting Flask application")
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.critical(f"Application failed to start: {str(e)}")
        raise