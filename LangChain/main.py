from dotenv import load_dotenv
import os

load_dotenv() # 这会自动把 .env 里的内容注入到 os.environ

from flashrank import Ranker # 此时 flashrank 就能读到 Token 了