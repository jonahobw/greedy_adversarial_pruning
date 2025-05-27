"""
Utility functions for experiment management.

Includes email notification tools, parameter permutation generators, and other
helpers for experiment orchestration.
"""

# pylint: disable=too-many-arguments, invalid-name, too-few-public-methods
import copy
import os
from itertools import product
from pathlib import Path
from smtplib import SMTPException, SMTP_SSL
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

logger = logging.getLogger("utils")


class Email_Sender:
    """
    Stores email sender, receiver, and password, and provides an email sending interface.
    """
    def __init__(
        self,
        sender: str = None,
        reciever: str = None,
        send: bool = True,
        **kwargs,
    ) -> None:
        """
        Store email parameters and set up the email sending function.

        Args:
            sender (str, optional): Email address of the sender.
            reciever (str, optional): Email address of the receiver.
            pw (str, optional): Path to file containing the sender's email password.
            send (bool): Whether to actually send emails (default: True).
        """
        self.sender = sender
        self.reciever = reciever
        self.pw = self.retrieve_pw()
        self.send = send
        if self.send is not None:
            logger.info("Email settings: send set to %s", send)
        # Dummy function in case an argument is not provided
        if None in (sender, reciever):
            logger.warning(
                "At least one of email sender or reciever was not"
                "specified, will not send any emails."
            )
            self.email = lambda subject, content: 0
        else:
            self.email = self._email

    def retrieve_pw(self) -> str:
        """
        Retrieves the gmail password from a "email_pw.txt" file.

        Returns:
            str: The password as a string, or None if not provided.

        Raises:
            FileNotFoundError: If the "email_pw.txt" file is not found.
        """
        with open(Path() / "src" / "email_pw.txt", "r") as pw_file:
            pw = pw_file.read()
        return pw

    def _email(self, subject, content=""):
        """
        Send an email using the stored credentials.

        Args:
            subject (str): Subject line of the email.
            content (str, optional): Body of the email.
        """
        email(
            content=content,
            subject=subject,
            sender=self.sender,
            reciever=self.reciever,
            pw=self.pw,
            send=self.send,
        )


def email(
    content: str,
    subject: str,
    sender: str,
    reciever: str,
    pw: str = None,
    send: bool = True,
) -> None:
    """
    Sends an email from a gmail account.

    Args:
        content (str): The message inside the email.
        subject (str): The subject line.
        sender (str): The sending email address.
        reciever (str): The destination email address.
        pw (str, optional): The gmail password for the sending email address.
        send (bool): Will only send an email if this is true.

    Returns:
        None
    """
    if not send:
        return

    message = MIMEMultipart()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = reciever
    message.attach(MIMEText(content, "plain"))

    try:
        context = ssl.create_default_context()
        with SMTP_SSL(host="smtp.gmail.com", port=465, context=context) as server:
            server.login(sender, pw)
            server.sendmail(sender, reciever, message.as_string())
            server.quit()
    except Exception as e:
        logger.warning("Error while trying to send email: \n%s", e)


def timer(time_in_s):
    """
    Converts a time in seconds to a string in HH:MM:SS format.

    Args:
        time_in_s (int or float): Time in seconds.
    Returns:
        str: Time formatted as HH:MM:SS.
    """
    hours, rem = divmod(time_in_s, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:0>2}".format(int(hours), int(minutes), int(seconds))


def format_path(path=None, directory=None):
    """
    Formats a path relative to the repository root and checks for existence.

    Args:
        path (str or Path, optional): Path to format and check.
        directory (str, optional): Directory to format path relative to.
    Returns:
        Path or None: Absolute Path object if path exists, else raises FileNotFoundError.
    """
    if path:
        repo_root = Path(__file__).parent.parent.absolute()
        if directory:
            repo_root = repo_root / directory
        path = repo_root / path
        if os.name != "nt":
            path = Path(path.as_posix())
        if not path.exists():
            raise FileNotFoundError(f"Model path {path} not found.")
        return path
    return None


def generate_permutations(list_args: dict) -> list:
    """
    Given a dict of several parameters values which each can take multiple values given as lists,
    return all permutations of the parameters as a list of dicts.

    Example input:
        {
            "model_type": ["vgg_bn_drop", "resnet20"],
            "prune_method": ["RandomPruning", "GlobalMagWeight"]
        }

    Example output:
        [
            {"model_type": "vgg_bn_drop", "prune_method": "RandomPruning"},
            {"model_type": "vgg_bn_drop", "prune_method": "GlobalMagWeight"},
            {"model_type": "resnet20", "prune_method": "RandomPruning"},
            {"model_type": "resnet20", "prune_method": "GlobalMagWeight"}
        ]

    Args:
        list_args (dict): Dictionary of parameters as {"parameter_name": [parameter_values]}.
    Returns:
        list: List of dictionaries, each a permutation of the possible parameter combinations.
    """
    return [dict(zip(list_args, v)) for v in product(*list_args.values())]


def find_recent_file(folder, prefix):
    """
    Find the most recent file in a folder that starts with a certain prefix.

    Args:
        folder (str or Path): Folder to search in.
        prefix (str): Prefix string to match files.
    Returns:
        Path or int: Path to the most recent file, or -1 if none found.
    """
    folder = Path(folder)
    if not folder.exists():
        return -1
    files_with_prefix = list(folder.glob(f"{prefix}*"))
    if len(files_with_prefix) == 0:
        # no files found
        return -1
    return max(files_with_prefix, key=lambda x: x.stat().st_ctime)


if __name__ == "__main__":
    print(f"Testing on module {Path.cwd()}")
    example = {
        "model_type": ["vgg_bn_drop", "resnet20"],
        "prune_method": ["RandomPruning"],
        "finetune_iterations": [10, 20, 40],
    }
    a = generate_permutations(example)
    import json

    print(json.dumps(a, indent=4))
