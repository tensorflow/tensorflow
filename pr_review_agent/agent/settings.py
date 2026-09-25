import os
from dotenv import load_dotenv

load_dotenv()

GITHUB_BASE_URL = "https://api.github.com"
GITHUB_GRAPHQL_URL = GITHUB_BASE_URL + "/graphql"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable not set")

OWNER = os.getenv("OWNER")
REPO = os.getenv("REPO")
PULL_REQUEST_NUMBER = os.getenv("PULL_REQUEST_NUMBER")

IS_INTERACTIVE = False
