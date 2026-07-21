import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class EmailEngine:
    def __init__(self):
        # Load from environment variables if set, otherwise default to placeholders
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        try:
            self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        except (TypeError, ValueError):
            self.smtp_port = 587
        self.sender_email = os.getenv("SMTP_SENDER_EMAIL", "your-chatbot-emailmail.com")
        self.sender_password = os.getenv("SMTP_SENDER_PASSWORD", "your-app-password")

    def send_summary_email(self, recipient_email: str, subject: str, body_content: str) -> bool:
        """
        Backward-compatible dispatcher for simple text emails.
        """
        result = self.send_email_with_attachment(recipient_email, subject, body_content)
        return result.get("success", False)

    def send_email_with_attachment(self, recipient_email: str, subject: str, body_content: str, attachment_path: str = None) -> dict:
        """
        Dispatches a structured email with optional PDF attachments to the recipient.
        If SMTP server settings are default/placeholders, or if sending fails,
        this falls back to Mock Mode to log the request and prevent system errors.
        """
        is_placeholder = (
            self.sender_email == "your-chatbot-email@gmail.com" or
            self.sender_password == "your-app-password" or
            not self.sender_email or
            not self.sender_password
        )

        # Build message container
        msg = MIMEMultipart()
        msg['From'] = self.sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body_content, 'plain'))

        # Handle attachment if provided
        attachment_name = None
        if attachment_path and os.path.isfile(attachment_path):
            attachment_name = os.path.basename(attachment_path)
            try:
                with open(attachment_path, 'rb') as f:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename="{attachment_name}"',
                )
                msg.attach(part)
            except Exception as e:
                logger.error(f"[EMAIL] Failed to read attachment: {e}")

        # Check if we should use Mock Mode directly
        if is_placeholder:
            return self._dispatch_mock_email(recipient_email, subject, body_content, attachment_name, "Placeholder credentials in use")

        try:
            # Establish secure connection and login
            server = smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10)
            server.starttls()  # Upgrade connection to secure TLS
            server.login(self.sender_email, self.sender_password)
            
            # Send the email and terminate connection
            server.sendmail(self.sender_email, recipient_email, msg.as_string())
            server.quit()
            
            logger.info(f"[EMAIL] Successfully sent email to {recipient_email}")
            return {
                "success": True,
                "mode": "SMTP",
                "message": f"Successfully sent email to {recipient_email}",
                "recipient": recipient_email,
                "attachment": attachment_name
            }
            
        except Exception as e:
            logger.warning(f"[EMAIL] SMTP dispatch failed ({e}). Falling back to Mock Mode.")
            return self._dispatch_mock_email(
                recipient_email, subject, body_content, attachment_name, f"SMTP Error: {str(e)}"
            )

    def _dispatch_mock_email(self, recipient_email: str, subject: str, body_content: str, attachment_name: str = None, reason: str = "") -> dict:
        """Helper to output mock email file details in reports/mock_emails/"""
        try:
            reports_dir = os.path.join(os.path.dirname(__file__), "reports")
            mock_emails_dir = os.path.join(reports_dir, "mock_emails")
            os.makedirs(mock_emails_dir, exist_ok=True)
            
            mock_filename = f"mock_email_{uuid.uuid4().hex[:8]}.txt"
            mock_filepath = os.path.join(mock_emails_dir, mock_filename)
            
            email_contents = f"""MOCK EMAIL DISPATCH RECEIPT
========================================
Timestamp:    {datetime.now().isoformat()}
Reason:       {reason}
Sender:       {self.sender_email} (Server: {self.smtp_server}:{self.smtp_port})
Recipient:    {recipient_email}
Subject:      {subject}
Attachment:   {attachment_name or "None"}
========================================
Body Content:
{body_content}
"""
            with open(mock_filepath, "w", encoding="utf-8") as f:
                f.write(email_contents)
                
            logger.info(f"[EMAIL MOCK] Mock email written to {mock_filepath}")
            return {
                "success": True,
                "mode": "Mock Mode",
                "message": f"Report successfully emailed to {recipient_email} (Mock Mode active)",
                "recipient": recipient_email,
                "attachment": attachment_name,
                "mock_file": mock_filepath
            }
        except Exception as e:
            logger.error(f"[EMAIL ERROR] Failed to write mock email: {e}")
            return {
                "success": False,
                "mode": "Failed",
                "message": f"Failed to dispatch email or write mock file: {e}",
                "recipient": recipient_email,
                "attachment": attachment_name
            }