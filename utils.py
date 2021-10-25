from pathlib import Path
from smtplib import SMTPException, SMTP_SSL
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

logger = logging.getLogger("utils")

EMAIL_SENDER = ""
EMAIL_RECIEVER = ""
EMAIL_PW = ""

def email(content, subject, sender, reciever, pw=None):
    '''
    Sends an email from a gmail account.
    :param content:
    :param subject:
    :param sender:
    :param reciever:
    :param pw:
    :return:
    '''
    message = MIMEMultipart()
    message['Subject'] = subject
    message['From'] = sender
    message['To'] = reciever
    message.attach(MIMEText(content, "plain"))

    try:
        context = ssl.create_default_context()
        with SMTP_SSL(host="smtp.gmail.com", port=465, context=context) as server:
            server.login(sender, pw)
            server.sendmail(sender, reciever, message.as_string())
            server.quit()
    except SMTPException as e:
        logger.warning("Error while trying to send email:")
        logger.info(e)


def email_callback(args):
    """
    Callback function to provide a constant email sender, reciever, and pw.
    :param sender:
    :param reciever:
    :param pw:
    :return:
    """
    globals()
    EMAIL_SENDER = args['sender']
    EMAIL_RECIEVER = args['reciever']
    EMAIL_PW = args['pw']

    def send_email(subject, content=""):
        email(content=content, subject=subject, sender=EMAIL_SENDER, reciever=EMAIL_RECIEVER, pw=EMAIL_PW)

    return send_email

if __name__ == '__main__':
    print(f"Testing on module {Path.cwd()}")