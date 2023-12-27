from linebot.v3.messaging import (
    Configuration,
    ApiClient,
    MessagingApi,
    MessagingApiBlob,
    RichMenuRequest,
    RichMenuArea,
    RichMenuSize,
    RichMenuBounds,
    URIAction,
    RichMenuSwitchAction,
    CreateRichMenuAliasRequest
)
from linebot.models import TextSendMessage
from linebot.exceptions import LineBotApiError
import os
import json
import urllib.request
from dotenv import load_dotenv

load_dotenv()
CHANNEL_ACCESS_TOKEN = os.environ['LINE_ACCESS_TOKEN']

req_header = {
    'Content-Type': 'application/json',
    'Authorization': f"Bearer {CHANNEL_ACCESS_TOKEN}",
}

url = "https://api.line.me/v2/bot/message/broadcast"

def send_message(text):
    req_data = json.dumps({
        "messages":[
            {
                "type":"text",
                "text":text,
            },
        ]
    })

    req = urllib.request.Request(url, data=req_data.encode(), method='POST', headers=req_header)
    try:
        with urllib.request.urlopen(req) as response:
            body = json.loads(response.read())
            headers = response.getheaders()
            status = response.getcode()

    except urllib.error.URLError as e:
        print(e.reason)

if __name__ == "__main__":
  text = "学習がおわったよ！！！"
  send_message(text)