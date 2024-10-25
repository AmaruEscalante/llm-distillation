from utils.nofitications import send_telegram_message

try:
    send_telegram_message(
        message="Hello from your Python script!",
    )
    print("Message sent successfully!")
except Exception as e:
    print(f"Error sending Telegram message: {str(e)}")
