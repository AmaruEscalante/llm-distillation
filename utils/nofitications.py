import os
import requests
from dotenv import load_dotenv

load_dotenv()


def send_telegram_message(message):
    """
    Send a message via Telegram Bot API using the requests library.

    :param bot_token: Your Telegram Bot API token
    :param chat_id: The chat ID to send the message to
    :param message: The message to send
    :return: True if successful, False otherwise
    """
    base_url = (
        f"https://api.telegram.org/bot{os.getenv('TELEGRAM_BOT_TOKEN')}/sendMessage"
    )

    payload = {"chat_id": os.getenv("TELEGRAM_CHAT_ID"), "text": message}

    try:
        response = requests.post(base_url, json=payload)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        result = response.json()
        return result.get("ok", False)
    except requests.RequestException as e:
        print(f"Error sending Telegram message: {e}")
        return False


if __name__ == "__main__":
    send_telegram_message(message="[TEST] Hello from your Python script!")
