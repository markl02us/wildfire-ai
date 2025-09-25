import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SENDER_EMAIL = "your_alert_account@gmail.com"
SENDER_PASSWORD = "your_app_password"
RECEIVER_EMAIL = "markl02us@yahoo.com"

def send_email_alert(detections):
    subject = "🚨 Wildfire Alert – Smoke/Fire Detected"
    body = f"The Jetson/Pi node detected: {detections}"
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
    print(f"[EMAIL] Alert sent to {RECEIVER_EMAIL}")
