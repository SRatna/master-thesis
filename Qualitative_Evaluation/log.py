import requests

TELEGRAM_TOKEN = '1684278746:AAHVJ7E3wzWg5ov2tV37J5f_cBSUtmK4uxM'
TELEGRAM_CHAT_ID = '1334399164'

def telegram_logger(msg):
	print(msg)
	payload = {
		'chat_id': TELEGRAM_CHAT_ID,
		'text': msg
	}
	requests.post("https://api.telegram.org/bot{token}/sendMessage".format(token=TELEGRAM_TOKEN),
							 data=payload)
